"""Chained-Flow drafter as a native vLLM speculative-decoding proposer (V1 `custom_class` hook).

    LLM(model=..., speculative_config={
        "method": "custom_class",
        "model": "chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
        "num_speculative_tokens": K})
    env: CF_DRAFTER_DIR=<ckpt dir>   [CF_SHORTLIST=<.pt of token ids>] [CF_WIDTH=4]

vLLM V1 verifies a LINEAR CHAIN (RejectionSampler over flat num_draft_tokens) — there is no tree-verify
hook. So we still build the draft TREE from the one flow pass and beam-search it with markov +
path-residual rescoring, then emit the highest-scoring PATH as the chain. That keeps the tree's
token-selection benefit; only the multi-branch acceptance is lost (that needs a forked verify).

Perf: one batched flow pass for the whole batch, beam search fully on-GPU (no per-token host syncs;
exactly one .tolist() at the end), and an optional shortlist lm_head (full-vocab head is ~40% of the
draft and is pure waste — the tree only ever uses the top candidates).
"""
from __future__ import annotations

import dataclasses
import json
import os
import time

import torch

_STASH: dict = {}


def _install_hidden_state_hook() -> None:
    """The stock custom_class hook passes only token ids; hidden states are in scope one frame up."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_cf_hooked", False):
        return
    orig = GPUModelRunner.propose_draft_token_ids

    def patched(self, scheduler_output, sampled_token_ids, sampling_metadata,
                hidden_states, sample_hidden_states, *args, **kwargs):
        _STASH["runner"] = self
        _STASH["sample_hidden_states"] = sample_hidden_states
        _STASH["hidden_states"] = hidden_states
        _STASH["spec_meta"] = args[1] if len(args) > 1 else None
        _STASH["sampling_metadata"] = sampling_metadata
        return orig(self, scheduler_output, sampled_token_ids, sampling_metadata,
                    hidden_states, sample_hidden_states, *args, **kwargs)

    GPUModelRunner.propose_draft_token_ids = patched
    GPUModelRunner._cf_hooked = True


class FlowDrafterProposer:
    def __init__(self, vllm_config):
        _install_hidden_state_hook()
        spec = vllm_config.speculative_config
        self.tree = os.environ.get("VLLM_SPEC_TREE", "0") == "1"
        # Tree mode needs ONE spare mamba state column so a node never
        # overwrites the request's initial GDN state while siblings still read
        # it (see vllm/v1/spec_decode/tree_gdn._colmap): configure K+1, emit K.
        self.tree_keep = int(os.environ.get("CF_TREE_KEEP", "4"))
        self.tree_topb = int(os.environ.get("CF_TREE_TOPB", "8"))
        self.tree_depth = int(os.environ.get("CF_TREE_DEPTH", "4"))
        self.branching = self.tree and os.environ.get("CF_TREE_BRANCH", "1") == "1"
        # `_beam_tree` can only emit min(tree_depth, draft_length-1) LEVELS: the flow returns
        # draft_length hiddens and depth 0 reconstructs the already-committed token. If N is computed
        # from an unclamped tree_depth we declare more draft slots than we emit, and vLLM's verify
        # indexes into slots that were never filled -> silently MALFORMED drafts (measured: depth 8
        # gave accept 1.60, BELOW the K=4 chain's 2.14, on both PIECEWISE and FULL cudagraph paths).
        self.tree_depth = self._clamp_depth(self.tree_depth)
        self.N = self.tree_keep * self.tree_depth
        if self.branching:
            # one spare mamba state column (see _colmap) => config N+1, emit N
            assert int(spec.num_speculative_tokens) == self.N + 1, (
                f"set num_speculative_tokens={self.N + 1} for "
                f"CF_TREE_KEEP={self.tree_keep} x CF_TREE_DEPTH={self.tree_depth}")
        self.K = int(spec.num_speculative_tokens) - (1 if self.tree else 0)
        self._draft_fn = self._beam_tree if self.branching else self._beam_chains
        self.width = int(os.environ.get("CF_WIDTH", "4"))
        self.ckd = os.environ.get("CF_DRAFTER_DIR")
        if not self.ckd:
            raise ValueError("set CF_DRAFTER_DIR to the drafter checkpoint dir or a HF repo id")
        if not os.path.isdir(self.ckd):
            # a HF repo id (e.g. selimaktas/Flow-Drafter-Qwen3.5-27B-v2): run the PUBLISHED weights
            from huggingface_hub import snapshot_download
            self.ckd = snapshot_download(
                self.ckd, allow_patterns=["model.safetensors", "chained_flow_tree_config.json"])
            print(f"[chained-flow] drafter from HF hub -> {self.ckd}", flush=True)
        self.shortlist_path = os.environ.get("CF_SHORTLIST")
        self.drafter = None
        self.ctx_hist: dict[str, torch.Tensor] = {}
        # CF_GPUCTX=1 (default): the per-request context history lives in ONE preallocated
        # GPU ring buffer [max_num_seqs, ctx_size+1, hidden] updated with index_copy_, and
        # every row index is computed on the GPU (or from host-resident vLLM metadata).
        # CF_GPUCTX=0 restores the legacy dict-of-tensors path (torch.cat per request in a
        # Python loop + torch.stack), which is what the ring replaces.
        self.gpuctx = os.environ.get("CF_GPUCTX", "1") == "1"
        self.max_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._pin: dict = {}
        self.prof = os.environ.get("CF_PROFILE", "0") == "1"
        self.use_cg = os.environ.get("CF_CUDAGRAPH", "1") == "1" and not self.prof
        # knob 2: torch.compile the flow net (predict_hidden). Measured 9.79 -> 5.26 ms (1.86x) at 27B.
        # "-no-cudagraphs" so inductor does not also try to own graph capture — we capture the whole
        # draft ourselves in _capture(); the compiled callable is replayed inside that graph.
        self.compile = os.environ.get("CF_COMPILE", "0") == "1"
        self.compile_mode = os.environ.get("CF_COMPILE_MODE", "max-autotune-no-cudagraphs")
        self.fuse_path = os.environ.get("CF_FUSE_PATH", "0") == "1"
        self.compile_beam = os.environ.get("CF_COMPILE_BEAM", "0") == "1"
        self.feedback = os.environ.get("CF_FEEDBACK", "0") == "1"
        if self.feedback:
            self.use_cg = False     # two flow passes + rebuilt context: eager for now
            self.gpuctx = False     # needs the raw per-request history tensors
        if self.gpuctx and os.environ.get("CF_CTXDIAG", "0") == "1":
            raise ValueError("CF_CTXDIAG inspects the per-request context tensor and only "
                             "works on the legacy path: set CF_GPUCTX=0")
        self._cg: dict[int, tuple] = {}          # batch-bucket -> (graph, ctx_buf, out_buf)
        self._buckets = [1, 2, 4, 8, 16, 32]
        self._t = {"n": 0, "ctx": 0.0, "flow": 0.0, "beam": 0.0, "sync": 0.0, "total": 0.0,
                   "gap": 0.0, "_last_exit": None}
        # CF_HPROF: HOST-ONLY segment timing. Unlike CF_PROFILE it inserts NO
        # torch.cuda.synchronize(), so it is cudagraph-capture-safe and can stay on in a
        # shipping-config run. Every segment is host wall time; a segment that blocks on
        # the GPU (a D2H copy of a not-yet-ready tensor) shows up as a large host segment,
        # which is exactly the glue we are hunting.
        self.hprof = os.environ.get("CF_HPROF", "0") == "1"
        self._h = {"n": 0, "_mark": None, "_exit": None}
        self._hsegs = ["pre", "rowmap", "ctx", "draft", "sync", "reg", "tail", "gap"]
        for s in self._hsegs:
            self._h[s] = 0.0
        _STASH["proposer"] = self       # so the bench harness can read per-set accept counters

    def _hm(self, seg):
        """Close the open host segment and attribute it to `seg`."""
        if not self.hprof:
            return
        now = time.perf_counter()
        if self._h["_mark"] is not None:
            self._h[seg] = self._h.get(seg, 0.0) + now - self._h["_mark"]
        self._h["_mark"] = now

    # ---------- lazy build (needs the loaded base model's embed / lm_head) ----------
    def _build(self):
        import torch.nn.functional as F
        from types import SimpleNamespace
        from safetensors.torch import load_file
        from chained_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig

        model = _STASH["runner"].model
        inner = model.language_model.model if hasattr(model, "language_model") else model.model
        embed_w = inner.embed_tokens.weight
        lm = model.language_model.lm_head if hasattr(model, "language_model") else model.lm_head
        lm_w = lm.weight
        self.dev, self.dtype = embed_w.device, embed_w.dtype
        H, V = embed_w.shape[1], embed_w.shape[0]

        class Emb:
            def __init__(s, w): s.w = w
            def __call__(s, i): return F.embedding(i, s.w)

        class Head(torch.nn.Module):
            def __init__(s, w): super().__init__(); s.weight = w
            def forward(s, h): return h.to(s.weight.dtype) @ s.weight.T

        class SM:
            def __init__(s):
                s.config = SimpleNamespace(hidden_size=H, vocab_size=V)
                s._lm, s._e = Head(lm_w), Emb(embed_w)
            @property
            def lm_head(s): return s._lm
            def get_input_embeddings(s): return s._e

        class Stub:
            def __init__(s): s.model = SM()
            def lm_head(s, h): return s.model.lm_head(h)

        cfgj = json.load(open(f"{self.ckd}/chained_flow_tree_config.json"))["model_args"]
        fields = {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}
        dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in fields})
        # INFERENCE-ONLY knob: number of Euler steps the flow-matching ODE solver takes
        # (TreeFlowDrafter.integrate loops `config.num_flow_steps` times).  Defaults to the
        # checkpoint's own value so nothing changes unless CF_FLOW_STEPS is set.  The drafter
        # holds `dcfg` BY REFERENCE as `self.config`, so overriding it here is what integrate()
        # actually reads -- verified below by printing the value off the built drafter.
        _fs_env = os.environ.get("CF_FLOW_STEPS")
        if _fs_env:
            _fs = int(_fs_env)
            if _fs < 1:
                raise ValueError(f"CF_FLOW_STEPS must be >= 1, got {_fs}")
            print(f"[chained-flow] CF_FLOW_STEPS: num_flow_steps {dcfg.num_flow_steps} -> {_fs}",
                  flush=True)
            dcfg.num_flow_steps = _fs
        d = TreeVAEFlowDrafter(Stub(), dcfg).to(self.dev).to(self.dtype).eval()
        assert d.config is dcfg and d.config.num_flow_steps == dcfg.num_flow_steps
        print(f"[chained-flow] integrator will run num_flow_steps="
              f"{d.config.num_flow_steps} (dt={1.0 / d.config.num_flow_steps:.4f}) x "
              f"num_drafter_layers={dcfg.num_drafter_layers}", flush=True)
        d._dtype = self.dtype
        sd = load_file(f"{self.ckd}/model.safetensors")
        _sub = {k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}
        _res = d.load_state_dict(_sub, strict=False)
        _own = dict(d.named_parameters())
        print(f"[chained-flow] ckpt keys={len(sd)} drafter-prefixed={len(_sub)} "
              f"model params={len(_own)} MISSING={len(_res.missing_keys)} UNEXPECTED={len(_res.unexpected_keys)}",
              flush=True)
        if _res.missing_keys:
            print(f"[chained-flow]   first missing: {_res.missing_keys[:6]}", flush=True)
        if _res.unexpected_keys:
            print(f"[chained-flow]   first unexpected: {_res.unexpected_keys[:6]}", flush=True)
        self.drafter, self.ctx_size, self.order = d, dcfg.context_size, dcfg.path_order
        if self.branching:
            emit = min(self.tree_depth, dcfg.draft_length - 1)
            if emit != self.tree_depth:
                raise ValueError(
                    f"CF_TREE_DEPTH={self.tree_depth} but the drafter can only emit "
                    f"{emit} levels (draft_length={dcfg.draft_length}). N/num_speculative_tokens "
                    f"were sized for the larger depth, so verify would index unfilled draft slots "
                    f"and silently corrupt the drafts. Set CF_TREE_DEPTH<={emit} and "
                    f"num_speculative_tokens={self.tree_keep * emit + 1}.")
        if self.gpuctx:
            self._init_ring(H)

        # knob 1: shortlist head (quality-free at full coverage; the tree only uses top candidates)
        self._sl = None
        self._hw = lm_w
        self._w2 = d.markov.w2.weight
        if self.shortlist_path:
            # fail loudly: a typo'd path silently benchmarking the full head is exactly how the
            # shortlist went unmeasured for a week.
            if not os.path.exists(self.shortlist_path):
                raise FileNotFoundError(f"CF_SHORTLIST={self.shortlist_path} does not exist")
            sl = torch.load(self.shortlist_path, map_location="cpu").flatten().long()
            sl = sl[(sl >= 0) & (sl < V)].unique().to(self.dev)
            self._sl = sl
            self._hw = lm_w[sl].contiguous()
            self._w2 = d.markov.w2.weight[sl].contiguous()
        # knob 2: fuse/compile the flow net. Both are pure-speed monkeypatches on the loaded drafter.
        if self.fuse_path or self.compile:
            from chained_flow.vllm_plugin import fused
            if self.fuse_path:
                fused.fuse_path_head(d)
            if self.compile:
                fused.compile_flow(d, mode=self.compile_mode,
                                   fullgraph=os.environ.get('CF_FULLGRAPH','0')=='1')
        if self.compile_beam:
            # MEASURED NEGATIVE: inductor's AOT metadata pass dies on the beam
            # ("RuntimeError: <weakref to UntypedStorage>") because the beam runs under
            # inference_mode over the persistent ring/shortlist buffers. Left behind the
            # flag with an explicit failure rather than a silent fallback -- and note the
            # beam is NOT where the draft's time goes anyway (see CF_DRAFTPROF: the four
            # shortlist-head GEMMs are at 96% of the memory roofline, the rest is the
            # flow's serial kernel chain).
            self._draft_fn = torch.compile(self._draft_fn, mode=self.compile_mode,
                                           dynamic=False, fullgraph=False)
        print(f"[chained-flow] drafter={os.path.basename(self.ckd)} K={self.K} width={self.width} "
              f"head={'shortlist ' + str(self._sl.numel()) if self._sl is not None else 'full ' + str(V)}"
              f" compile={self.compile_mode if self.compile else 'off'}"
              f" fuse_path={'on' if self.fuse_path else 'off'} cudagraph={'on' if self.use_cg else 'off'}",
              flush=True)

    # ================= GPU-resident context ring (replaces the per-request dict) =========
    #
    # The old path kept `ctx_hist: dict[req_id -> Tensor]` and every step did, per request,
    # `torch.cat([hist, h])[-ctx:]` in a Python loop followed by a `torch.stack`.  That is
    # O(batch) Python + O(batch) allocations per step and — worse — hands the drafter a
    # FRESHLY ALLOCATED context tensor every step, so no cudagraph can span the context
    # build.  The ring fixes both: one preallocated [max_num_seqs, ctx_size+1, hidden]
    # buffer at a STABLE address, updated with a single fixed-shape index_copy_.
    #
    # Layout: slot b holds its history circularly in rows 0..ctx_size-1; row `ctx_size` is a
    # TRASH row.  Masked-off writes are redirected there so the scatter always runs at the
    # full [B, Kmax] shape — a compacted scatter would need a D2H to learn its own size.
    def _init_ring(self, H: int):
        R, C = self.max_reqs, self.ctx_size
        self._ring = torch.zeros(R, C + 1, H, device=self.dev, dtype=self.dtype)
        self._ring_flat = self._ring.view(R * (C + 1), H)
        self._rpos = torch.zeros(R, dtype=torch.long, device=self.dev)   # next write position
        self._nval = torch.zeros(R, dtype=torch.long, device=self.dev)   # valid rows, <= C
        self._slot_req: list = [None] * R      # slot -> req_id currently occupying it
        self._req_slot: dict = {}              # req_id -> slot (detects slot migration)
        self._ring_H = H

    def _h2d(self, key, arr, dtype=torch.long, pad_to=0, pad_val=0):
        """numpy -> GPU with NO host sync.

        `torch.tensor(python_list, device="cuda")` and `t.to("cuda")` from pageable memory
        are both BLOCKING copies (they showed up in torch's sync debugger once per step).
        Staging through a persistent PINNED buffer makes the copy async.
        """
        n = int(arr.shape[0])
        m = max(n, pad_to)
        ent = self._pin.get(key)
        if ent is None or ent[0].shape[0] < m:
            cap = max(m, self.max_reqs * (self.K + 2))
            ent = (torch.empty(cap, dtype=dtype, pin_memory=True),
                   torch.empty(cap, dtype=dtype, device=self.dev))
            self._pin[key] = ent
        cpu, gpu = ent
        cpu[:n].copy_(torch.from_numpy(arr))
        if m > n:
            # Pad up to the cudagraph batch bucket. The graph is captured on a FIXED
            # [bucket] view of this persistent buffer, so the tail must be well-defined
            # (padding rows point at slot 0; their outputs are discarded).
            cpu[n:m].fill_(pad_val)
        gpu[:m].copy_(cpu[:m], non_blocking=True)
        return gpu[:m] if pad_to else gpu[:n]

    def _ring_write(self, sh, rows_g, starts_g, cnt_g, wmask_g, path_g, Kmax):
        """Append each request's newly-harvested hiddens to its ring slot.

        Row j of request b comes from `sample_hidden_states` row `start_b + srcoff[b, j]`,
        where srcoff is 0 for the previously-committed token and `1 + path[b, j-1]` for the
        j-th accepted draft.  In TREE mode the accepted path's rows are SCATTERED (node
        path[j], not node j), so a contiguous slice would feed the drafter the hidden states
        of REJECTED sibling nodes.  `path` stays on the GPU throughout — reading it on the
        host is what the two `.to("cpu")` calls in the fork's rejection sampler were for.
        """
        C = self.ctx_size
        B = rows_g.shape[0]
        ar = torch.arange(Kmax, device=self.dev)
        if path_g is None:
            srcoff = ar.unsqueeze(0).expand(B, Kmax)
        else:
            p = path_g.index_select(0, rows_g)[:, : Kmax - 1].long()
            srcoff = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=self.dev), 1 + p], 1)
        valid = (ar.unsqueeze(0) < cnt_g.unsqueeze(1)) & wmask_g.unsqueeze(1)
        srcrow = (starts_g.unsqueeze(1) + srcoff).clamp_(0, sh.shape[0] - 1)
        hrows = sh.index_select(0, torch.where(valid, srcrow, srcrow.new_zeros(())).reshape(-1))
        head_b = self._rpos.index_select(0, rows_g)
        dst = torch.remainder(head_b.unsqueeze(1) + ar.unsqueeze(0), C)
        dst = torch.where(valid, dst, dst.new_full((), C))          # -> the trash row
        self._ring_flat.index_copy_(
            0, (rows_g.unsqueeze(1) * (C + 1) + dst).reshape(-1), hrows.to(self.dtype))
        add = torch.where(wmask_g, cnt_g, cnt_g.new_zeros(()))
        self._rpos.index_copy_(0, rows_g, torch.remainder(head_b + add, C))
        self._nval.index_copy_(
            0, rows_g, torch.clamp(self._nval.index_select(0, rows_g) + add, max=C))

    def _ring_gather(self, rows_g):
        """Read each slot's history back IN ORDER, oldest first, front-padded to ctx_size.

        Reproduces `tree_flow._context` exactly: when fewer than ctx_size hiddens exist the
        window is front-padded by REPEATING THE OLDEST row.  All index math is on the GPU
        against stable buffers, so this whole function is cudagraph-capturable.
        """
        C, H = self.ctx_size, self._ring_H
        m = self._nval.index_select(0, rows_g).unsqueeze(1)          # [B, 1]
        head = self._rpos.index_select(0, rows_g).unsqueeze(1)       # [B, 1]
        j = torch.arange(C, device=self.dev).unsqueeze(0)            # [1, C]
        t = torch.minimum((j - (C - m)).clamp_(min=0), (m - 1).clamp_(min=0))
        phys = torch.remainder(head - m + t, C)                      # [B, C]
        flat = (rows_g.unsqueeze(1) * (C + 1) + phys).reshape(-1)
        return self._ring_flat.index_select(0, flat).view(-1, C, H)

    @staticmethod
    def _tree_ok(sm) -> bool:
        """Mirror the fork's OWN preconditions for the tree verify path
        (rejection_sampler.py: all_greedy + no logprobs + not _tree_needs_processors).

        Do NOT hand-roll this: an earlier version also rejected on `logitsprocs`, which is truthy
        even when inactive, so it returned False for ordinary GREEDY steps and took down the
        working path. Note stock never applies logitsprocs to target rows either -- only to the
        bonus row -- so they are deliberately not disqualifying here.
        """
        if sm is None:
            return False
        try:
            if not bool(getattr(sm, "all_greedy", False)):
                return False
            if getattr(sm, "max_num_logprobs", None) is not None:
                return False
            holder = getattr(sm, "thinking_budget_state_holder", None)
            if (getattr(sm, "bad_words_token_ids", None)
                    or not bool(getattr(sm, "no_penalties", True))
                    or getattr(sm, "allowed_token_ids_mask", None) is not None
                    or (holder is not None and holder.has_tracked_requests())):
                return False
        except Exception:
            return False
        return True

    def _clamp_depth(self, depth: int) -> int:
        """Clamp CF_TREE_DEPTH to draft_length-1, reading the checkpoint config if it is local.
        _build() re-checks authoritatively once the drafter is loaded."""
        try:
            ckd = os.environ.get("CF_DRAFTER_DIR", "")
            cfg_path = os.path.join(ckd, "chained_flow_tree_config.json")
            if os.path.isfile(cfg_path):
                dl = int(json.load(open(cfg_path))["model_args"]["draft_length"])
                if depth > dl - 1:
                    print(f"[chained-flow] CF_TREE_DEPTH={depth} exceeds draft_length-1={dl - 1}; "
                          f"clamping to {dl - 1} (the flow cannot emit more levels)", flush=True)
                    return dl - 1
        except Exception:
            pass
        return depth

    class _DS:
        def __init__(s, h): s.final_hidden = h

    def _head(self, h):
        return h.to(self._hw.dtype) @ self._hw.T

    def _bias(self, prev):
        return self.drafter.markov.w1(prev) @ self._w2.T

    def _map(self, local):
        return local if self._sl is None else self._sl[local]

    # ---------- batched tree beam search: one flow pass, no host syncs ----------
    @torch.inference_mode()
    def _beam_chains(self, ctx: torch.Tensor, known0: torch.Tensor) -> torch.Tensor:
        """`known0` is the token vLLM just committed. The context we get from vLLM ends at the hidden
        that PRODUCED known0 — vLLM has not yet run a forward pass with known0 as input — so the flow's
        depth-0 output predicts known0 itself, not the future. (The offline harness has that extra
        hidden because it drives generation itself and re-runs the model on each committed token.)
        Drafting depth 0 therefore re-proposes an already-committed token and every draft lands one
        position early: measured 855/950 of chain[0] at offset -1. So seed depth 0 with the known
        token and emit depths 1..K."""
        d = self.drafter
        B, W, K = ctx.shape[0], self.width, self.K
        if self.prof: torch.cuda.synchronize(); _t0 = time.perf_counter()
        # ANCHOR: an anchor-trained drafter MUST be given emb(committed token) at inference too --
        # `known0` is exactly that token. Omitting it evaluates the model off-distribution (measured:
        # tree accept collapsed 3.69 -> 1.83). anchor_embed() returns None unless anchor_token is set,
        # so unanchored checkpoints are unaffected.
        pred = d.predict_hidden(ctx.to(self.dtype), d.anchor_embed(known0))   # [B, Kd, D]
        if self.prof:
            torch.cuda.synchronize(); self._t["flow"] += time.perf_counter() - _t0
            _t0 = time.perf_counter()
        D = pred.shape[-1]
        lastp = torch.full((B, W, self.order), -1, dtype=torch.long, device=self.dev)
        toks: list[torch.Tensor] = []
        bps: list[torch.Tensor] = []
        if self.feedback:
            # context was repaired to end at h(t_c), so depth 0 is a genuine future token: draft it
            steps = min(K, pred.shape[1])
            lp0 = torch.log_softmax(self._head(pred[:, 0]).float(), dim=-1)
            cum, loc0 = lp0.topk(W, dim=-1)
            tok0 = self._map(loc0)
            toks.append(tok0)
            bps.append(torch.arange(W, device=self.dev).expand(B, W))
            lastp[:, :, 0] = tok0
            first = 1
        else:
            steps = min(K + 1, pred.shape[1])                  # depth 0 is consumed by the seed
            cum = torch.full((B, W), float("-inf"), device=self.dev)
            cum[:, 0] = 0.0                                    # one live beam: the known prefix
            lastp[:, :, 0] = known0.unsqueeze(1)
            first = 1

        for step in range(first, steps):
            res = d._residual_from_lastp(lastp.reshape(B * W, self.order))          # [B*W, D]
            h = pred[:, step].unsqueeze(1).expand(B, W, D).reshape(B * W, D)
            logits = self._head(h + res) + self._bias(lastp[:, :, 0].reshape(-1))   # [B*W, Vs]
            lp = torch.log_softmax(logits.float(), dim=-1)
            cv, ci = lp.topk(W, dim=-1)                                             # [B*W, W]
            cand = (cum.reshape(B * W, 1) + cv).reshape(B, W * W)
            cum, flat = cand.topk(W, dim=-1)                                        # [B, W]
            src = flat // W                                                         # parent beam
            newtok = self._map(ci.reshape(B, W * W).gather(1, flat))                # [B, W]
            lastp = lastp.gather(1, src.unsqueeze(-1).expand(B, W, self.order))
            lastp = torch.cat([newtok.unsqueeze(-1), lastp[:, :, :-1]], dim=-1)
            toks.append(newtok)
            bps.append(src)

        # walk back the best path (all on GPU; single sync at the end)
        idx = cum.argmax(dim=-1)                                                    # [B]
        seq = [None] * len(toks)
        for step in range(len(toks) - 1, -1, -1):
            seq[step] = toks[step].gather(1, idx.unsqueeze(1)).squeeze(1)
            idx = bps[step].gather(1, idx.unsqueeze(1)).squeeze(1)   # every level has a backpointer now
        _out = torch.stack(seq, dim=1)
        if self.prof: torch.cuda.synchronize(); self._t["beam"] += time.perf_counter() - _t0
        return _out                                                                 # [B, steps]

    # ---------- batched TREE draft: one flow pass -> a tree of N nodes, no host syncs ----------
    @torch.inference_mode()
    def _beam_tree(self, ctx: torch.Tensor, known0: torch.Tensor) -> torch.Tensor:
        """Return [B, 2N]: N draft tokens in BFS order followed by their parents.

        Same flow pass and markov/path-residual rescoring as `_beam_chains`, but
        every surviving beam extension is kept as a TREE NODE instead of being
        collapsed to the single best path.  Nodes are emitted depth by depth so
        parent index < child index, which is what the fork's tree verify
        (ancestor mask + per-node GDN state slots) requires.
        """
        d = self.drafter
        B = ctx.shape[0]
        W, TB, D = self.tree_keep, self.tree_topb, self.tree_depth
        pred = d.predict_hidden(ctx.to(self.dtype), d.anchor_embed(known0))  # [B, Kd, Dh]
        Dh = pred.shape[-1]
        fr = 1
        lastp = torch.full((B, fr, self.order), -1, dtype=torch.long, device=self.dev)
        lastp[:, :, 0] = known0.unsqueeze(1)                 # seed = the committed token
        cum = torch.zeros(B, fr, device=self.dev)
        prev_ids = torch.full((B, fr), -1, dtype=torch.long, device=self.dev)
        toks: list[torch.Tensor] = []
        pars: list[torch.Tensor] = []
        base = 0
        for step in range(1, min(D, pred.shape[1] - 1) + 1):
            res = d._residual_from_lastp(lastp.reshape(B * fr, self.order))
            h = pred[:, step].unsqueeze(1).expand(B, fr, Dh).reshape(B * fr, Dh)
            logits = self._head(h + res) + self._bias(lastp[:, :, 0].reshape(-1))
            lp = torch.log_softmax(logits.float(), dim=-1)
            cv, ci = lp.topk(TB, dim=-1)                     # [B*fr, TB]
            cand = (cum.reshape(B * fr, 1) + cv).reshape(B, fr * TB)
            k = min(W, fr * TB)
            cum, flat = cand.topk(k, dim=-1)                 # [B, k]
            src = flat // TB                                 # parent slot in frontier
            newtok = self._map(ci.reshape(B, fr * TB).gather(1, flat))
            toks.append(newtok)
            pars.append(prev_ids.gather(1, src))
            lastp = lastp.gather(1, src.unsqueeze(-1).expand(B, k, self.order))
            lastp = torch.cat([newtok.unsqueeze(-1), lastp[:, :, :-1]], dim=-1)
            prev_ids = base + torch.arange(k, device=self.dev).expand(B, k)
            base += k
            fr = k
        return torch.cat(toks + pars, dim=1)                 # [B, 2N]

    # ============ ONE cudagraph over the WHOLE draft ====================================
    # With CF_GPUCTX=1 the captured region is
    #     ring gather -> VAE encode -> flow (predict_hidden) -> tree/beam build -> out buf
    # i.e. everything from the context tensor onwards.  The only draft work left outside is
    # the ring APPEND, which reads `sample_hidden_states` — a tensor vLLM reallocates every
    # step, so its address cannot be baked into a graph.  Both graph inputs (`rows`, `k0`)
    # are fixed-length views of persistent pinned-staged buffers, so their addresses are
    # stable across replays.
    def _capture_ring(self, bucket: int, rows_g, k0_g):
        # torch.compile does its codegen on the FIRST call; doing that inside the capture
        # warmup stream trips inductor's buffer bookkeeping, so force it on the default
        # stream first.
        self._draft_from_ring(rows_g, k0_g)
        torch.cuda.synchronize()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._draft_from_ring(rows_g, k0_g)
        torch.cuda.current_stream().wait_stream(st)
        try:
            from vllm.platforms import current_platform
            pool = current_platform.get_global_graph_pool()
        except Exception:
            pool = None
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool) if pool is not None else torch.cuda.graph(g):
            out = self._draft_from_ring(rows_g, k0_g)
        self._cg[bucket] = (g, None, None, out)
        print(f"[chained-flow] draft cudagraph captured (ring->flow->beam, bucket {bucket})",
              flush=True)
        if os.environ.get("CF_DRAFTPROF", "0") == "1":
            self._draft_breakdown(g, rows_g, k0_g, bucket)

    def _draft_breakdown(self, g, rows_g, k0_g, bucket):
        """ONE-OFF (capture-time) GPU timing of the draft and its pieces.

        Runs after capture so it never adds a per-step sync; the numbers say whether the
        draft is launch-bound (graph << eager) or genuinely GPU-bound (graph ~= eager).
        """
        d = self.drafter

        def t(fn, n=30):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            for _ in range(n):
                fn()
            b.record()
            torch.cuda.synchronize()
            return a.elapsed_time(b) / n

        ctx = self._ring_gather(rows_g)
        cf = ctx.to(self.dtype)
        lat = d._encode(cf)
        z = d.integrate(lat, d.init_latents(lat))
        pred = d.predict_hidden(cf)
        parts = [
            ("graph replay (WHOLE draft)", lambda: g.replay()),
            ("  ring gather", lambda: self._ring_gather(rows_g)),
            ("  vae encode", lambda: d._encode(cf)),
            ("  flow integrate", lambda: d.integrate(lat, d.init_latents(lat))),
            ("  vae decode", lambda: d._decode(z)),
            ("  predict_hidden (enc+flow+dec)", lambda: d.predict_hidden(cf)),
            ("  beam/tree build", lambda: self._draft_fn(ctx, k0_g)),
            ("  one shortlist head", lambda: self._head(pred[:, 0])),
            ("eager whole draft", lambda: self._draft_from_ring(rows_g, k0_g)),
        ]
        print(f"[cf-draftprof] bucket={bucket} GPU ms (mean of 30):", flush=True)
        for name, fn in parts:
            print(f"[cf-draftprof]   {name:32s} {t(fn):7.3f}", flush=True)
        # kernel-level breakdown of the captured graph: is the 4.7 ms a few big kernels
        # (real work) or a long tail of tiny ones (latency-bound serial chain)?
        import collections
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False)
        with prof:
            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()
        agg, cnt = collections.Counter(), collections.Counter()
        for e in prof.key_averages():
            if e.self_device_time_total:
                agg[e.key] += e.self_device_time_total / 10.0
                cnt[e.key] += e.count / 10.0
        tot = sum(agg.values())
        print(f"[cf-draftprof] graph kernels: {sum(cnt.values()):.0f} launches/replay, "
              f"{tot:.1f} us total", flush=True)
        for k, v in agg.most_common(14):
            print(f"[cf-draftprof]   {v:8.1f} us  x{cnt[k]:5.1f}  {k[:70]}", flush=True)

    def _draft_from_ring(self, rows_g, k0_g):
        return self._draft_fn(self._ring_gather(rows_g), k0_g)

    def _ring_cg(self, rows_g, k0_g, B):
        bucket = rows_g.shape[0]
        if not self.use_cg or bucket not in self._buckets:
            return self._draft_from_ring(rows_g, k0_g)[:B]
        if bucket not in self._cg:
            self._capture_ring(bucket, rows_g, k0_g)
        self._cg[bucket][0].replay()
        return self._cg[bucket][3][:B]

    # ---------- legacy (CF_GPUCTX=0) cudagraph: beam search only ----------
    def _capture(self, bucket: int, ctx_like: torch.Tensor, k0_like: torch.Tensor):
        buf = torch.zeros(bucket, ctx_like.shape[1], ctx_like.shape[2],
                          device=self.dev, dtype=ctx_like.dtype)
        buf[: ctx_like.shape[0]].copy_(ctx_like)
        kbuf = torch.zeros(bucket, dtype=torch.long, device=self.dev)
        kbuf[: k0_like.shape[0]].copy_(k0_like)
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(3):
                self._draft_fn(buf, kbuf)
        torch.cuda.current_stream().wait_stream(st)
        try:
            from vllm.platforms import current_platform
            pool = current_platform.get_global_graph_pool()
        except Exception:
            pool = None
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool) if pool is not None else torch.cuda.graph(g):
            out = self._draft_fn(buf, kbuf)
        self._cg[bucket] = (g, buf, kbuf, out)
        print(f"[chained-flow] draft cudagraph captured (batch bucket {bucket})", flush=True)

    def _chains_cg(self, ctx: torch.Tensor, known0: torch.Tensor) -> torch.Tensor:
        B = ctx.shape[0]
        bucket = next((b for b in self._buckets if b >= B), None)
        if bucket is None:
            return self._draft_fn(ctx, known0)               # oversized batch: eager fallback
        if bucket not in self._cg:
            self._capture(bucket, ctx, known0)
        g, buf, kbuf, out = self._cg[bucket]
        buf.zero_(); buf[:B].copy_(ctx)
        kbuf.zero_(); kbuf[:B].copy_(known0)
        g.replay()
        return out[:B]

    def _rows_for_requests(self, sampled_token_ids):
        """With spec decode, sample_hidden_states has (num_draft+1) rows PER REQUEST (vLLM computes
        logits for every drafted position). Request i's committed hidden is its block start plus
        (accepted-1) — NOT row i. Without drafts (first step) it is one row per request."""
        smd = _STASH.get("spec_meta")
        if smd is None or not hasattr(smd, "cu_num_draft_tokens"):
            return {i: (i, 1) for i, t in enumerate(sampled_token_ids) if t}
        cu = smd.cu_num_draft_tokens
        cu = cu.tolist() if hasattr(cu, "tolist") else list(cu)
        rows = {}
        for i, t in enumerate(sampled_token_ids):
            if not t:
                continue
            start = (cu[i - 1] if i > 0 else 0) + i        # sum_{j<i}(num_draft[j]+1)
            rows[i] = (start, len(t))                       # ALL accepted positions
        return rows

    def _tree_row_offsets(self, i, cnt):
        """Local logits-row offsets of the hiddens for request i's committed run.

        Row 0 is the previously committed token; the j-th accepted draft is tree
        node ``path[j]``, i.e. row ``1 + path[j]``.  Returns None when this step
        was not a (branching) tree step.
        """
        if not self.branching:
            return None
        try:
            from vllm.v1.spec_decode import tree_state
        except Exception:
            return None
        ts = tree_state.get_current()
        if ts is None or "accepted_path_cpu" not in ts.extras:
            return None
        path = ts.extras["accepted_path_cpu"]
        alen = ts.extras["accepted_len_cpu"]
        if i >= path.shape[0]:
            return None
        L = int(alen[i])
        offs = [0] + [1 + int(path[i, j]) for j in range(L)]
        return offs[:cnt] if cnt <= len(offs) else None

    def _seed_prompt_ctx(self, i):
        """Harness prefill runs step_commit over EVERY prompt token, so the first draft sees 8 real
        consecutive hiddens. vLLM only hands us one row per request at prefill, so _context would pad
        one hidden 8x — a context that never occurs in training. Seed from the full hidden_states
        tensor instead: request i's tokens span [query_start_loc[i], query_start_loc[i+1])."""
        hs = _STASH.get("hidden_states")
        runner = _STASH.get("runner")
        if hs is None or runner is None:
            return None
        qsl = getattr(runner, "query_start_loc", None)
        if qsl is None:
            return None
        for attr in ("np", "cpu"):
            v = getattr(qsl, attr, None)
            if v is not None:
                try:
                    q = v.tolist() if hasattr(v, "tolist") else list(v)
                    break
                except Exception:
                    q = None
        else:
            return None
        if q is None or i + 1 >= len(q):
            return None
        end = int(q[i + 1]); start = int(q[i])
        if end <= start or end > hs.shape[0]:
            return None
        lo = max(start, end - self.ctx_size)
        return hs[lo:end].reshape(end - lo, -1).to(self.dtype)

    @torch.inference_mode()
    def propose(self, sampled_token_ids, num_tokens_no_spec, token_ids_cpu, slot_mappings=None, **kw):
        if self.prof:
            _tp = time.perf_counter()
            if self._t["_last_exit"] is not None:
                self._t["gap"] += _tp - self._t["_last_exit"]
        if self.hprof:
            _now = time.perf_counter()
            if self._h["_exit"] is not None:
                self._h["gap"] += _now - self._h["_exit"]
            self._h["_mark"] = _now
        if self.drafter is None:
            self._build()
        runner = _STASH["runner"]
        sh = _STASH.get("sample_hidden_states")
        out: list[list[int]] = [[] for _ in sampled_token_ids]
        if sh is None:
            return out
        req_ids = list(runner.input_batch.req_ids)
        if os.environ.get("CF_ACCEPT", "0") == "1":
            _n = sum(len(t) for t in sampled_token_ids if t)
            _r = sum(1 for t in sampled_token_ids if t)
            self._t["acc_tok"] = self._t.get("acc_tok", 0) + _n
            self._t["acc_req"] = self._t.get("acc_req", 0) + _r
            self._t["acc_n"] = self._t.get("acc_n", 0) + 1
            if self._t["acc_n"] % 25 == 0 and self._t["acc_req"]:
                print(f"[cf-accept] steps={self._t['acc_n']} "
                      f"mean tokens emitted per request-step = {self._t['acc_tok']/self._t['acc_req']:.3f} "
                      f"(K={self.K}, so max possible {self.K+1})", flush=True)
        if os.environ.get("CF_DIAG", "0") == "1" and sh is not None:
            # (a) is sh[i] REALLY the post-norm hidden the base lm_head consumes?
            #     argmax(lm_head(sh[i])) must equal the token the base model just sampled.
            # (b) did our previous draft predict the token the base model actually produced?
            dg = self._t.setdefault("diag", {"n": 0, "hid_ok": 0, "draft_ok": 0, "draft_n": 0})
            prev = self._t.setdefault("prev_draft", {})
            for i, t in enumerate(sampled_token_ids):
                if not t:
                    continue
                dg["n"] += 1
                _rmap = self._rows_for_requests(sampled_token_ids)
                _st, _cnt = _rmap.get(i, (i, 1))
                lg = self._head(sh[_st + _cnt - 1].reshape(1, -1).to(self.dtype))
                pred_now = int(self._map(lg[0].argmax()))
                if pred_now == int(t[0]):
                    dg["hid_ok"] += 1
                rid = req_ids[i] if i < len(req_ids) else str(i)
                pd = prev.get(rid)
                if pd:
                    dg["draft_n"] += 1
                    if pd[0] == int(t[0]):
                        dg["draft_ok"] += 1
            if dg["n"] and dg["n"] % 40 < 1:
                print(f"[cf-diag] hidden-state valid: {dg['hid_ok']}/{dg['n']} "
                      f"(argmax(lm_head(sh))==sampled token) | prev-draft[0] hit: "
                      f"{dg['draft_ok']}/{max(dg['draft_n'],1)}", flush=True)
        if os.environ.get("CF_ORACLE", "0") == "1":
            # Score our PREVIOUS drafts against the tokens vLLM actually committed, independently of
            # whether vLLM's rejection sampler accepted them. Splits the failure in half:
            #   high d0 hit + accept~1  -> vLLM is discarding correct drafts (integration bug)
            #   low  d0 hit             -> the drafts themselves are wrong
            # Offsets -1/0/+1 catch a phase error (draft aligned to the wrong committed position).
            orc = self._t.setdefault("orc", {"n": 0, "depth": [0] * 8,
                                             "off": {-1: 0, 0: 0, 1: 0}, "offn": 0})
            th = self._t.setdefault("orc_tok", {})
            pend = self._t.setdefault("orc_pend", {})
            for i, t in enumerate(sampled_token_ids):
                if not t:
                    continue
                rid = req_ids[i] if i < len(req_ids) else str(i)
                th.setdefault(rid, []).extend(int(x) for x in t)
                seq = th[rid]
                keep = []
                for (L, chain) in pend.get(rid, []):
                    if L + len(chain) > len(seq) and L < len(seq):
                        keep.append((L, chain))     # not yet resolvable
                        continue
                    if L >= len(seq):
                        keep.append((L, chain))
                        continue
                    orc["n"] += 1
                    d = 0
                    while d < len(chain) and L + d < len(seq) and int(chain[d]) == seq[L + d]:
                        d += 1
                    orc["depth"][min(d, 7)] += 1
                    for o in (-1, 0, 1):            # where does chain[0] actually land?
                        if 0 <= L + o < len(seq) and int(chain[0]) == seq[L + o]:
                            orc["off"][o] += 1
                    orc["offn"] += 1
                pend[rid] = keep
            if orc["n"] and orc["n"] % 50 < 3:
                dep = orc["depth"]; N = max(orc["n"], 1)
                mean = sum(k * v for k, v in enumerate(dep)) / N
                # Report an interval, not bare digits. Draft outcomes are consecutive decode steps
                # from a handful of requests, so they are autocorrelated: the naive SE below is a
                # LOWER BOUND on the true uncertainty (effective n < N).
                var = max(sum(k * k * v for k, v in enumerate(dep)) / N - mean * mean, 0.0)
                se = (var / N) ** 0.5
                print(f"[cf-oracle] drafts scored={orc['n']} | ORACLE accept={1+mean:.2f} "
                      f"+/-{se:.2f} (naive SE, autocorrelated -> optimistic) "
                      f"(d0 hit {100*sum(dep[1:])/N:.1f}%) | depth hist {dep[:6]} | "
                      f"chain[0] lands at offset -1/0/+1: "
                      f"{orc['off'][-1]}/{orc['off'][0]}/{orc['off'][1]} of {orc['offn']}", flush=True)

        rows = [i for i, t in enumerate(sampled_token_ids) if t]
        if not rows:
            return out

        self._hm("pre")
        if self.gpuctx:
            if os.environ.get("CF_RINGCHECK", "0") == "1":
                # Run the legacy per-request build alongside the ring and diff the
                # contexts. The ring is only allowed to be a PERFORMANCE change.
                _lctx, _lk0 = self._ctx_legacy(rows, req_ids, sh, sampled_token_ids)
            _rows_g, _k0, _B = self._ctx_gpu(rows, req_ids, sh, sampled_token_ids)
            if os.environ.get("CF_RINGCHECK", "0") == "1":
                _rc = self._t.setdefault("rc", {"n": 0, "bad": 0, "maxd": 0.0, "shown": 0,
                                                "k0bad": 0})
                _gctx = self._ring_gather(_rows_g[:_B])
                _rc["n"] += 1
                _d = (_gctx.float() - _lctx.float()).abs().max().item()
                _rc["maxd"] = max(_rc["maxd"], _d)
                _kb = int((_k0[:_B] != _lk0).sum())
                _rc["k0bad"] += _kb
                if _d > 0 or _kb:
                    _rc["bad"] += 1
                    if _rc["shown"] < 5:
                        _rc["shown"] += 1
                        _rowdiff = (_gctx.float() - _lctx.float()).abs().amax(dim=-1)
                        print(f"[cf-ringcheck] step {_rc['n']}: maxdiff {_d:.3g} k0bad {_kb} "
                              f"per-ctx-row maxdiff {_rowdiff[0].tolist()} "
                              f"nval {self._nval[_rows_g[:_B]].tolist()} "
                              f"rpos {self._rpos[_rows_g[:_B]].tolist()} "
                              f"cnt {[len(sampled_token_ids[i]) for i in rows]}", flush=True)
                if _rc["n"] % 50 == 0:
                    print(f"[cf-ringcheck] steps={_rc['n']} mismatching={_rc['bad']} "
                          f"maxdiff={_rc['maxd']:.3g} k0 mismatches={_rc['k0bad']}", flush=True)
            self._hm("ctx")
            # NON-GREEDY GUARD (see _tree_ok). A branching tree that reaches the stock linear
            # rejection sampler RAISES, so one temperature>0 request would kill the engine. Emit a
            # CHAIN for such steps -- exact under any sampling params, and cheap (27B b1: chain
            # 1.61x vs tree 1.68x). The captured graph is tree-shaped, so bypass it here.
            # NON-GREEDY: the fork's tree verify needs all_greedy + no logprobs + no penalties;
            # a branching tree reaching the stock linear sampler RAISES. A per-step CHAIN fallback
            # does NOT work: a chain is at most draft_length-1 (=7) tokens while the ring/cudagraph
            # are fixed-shape for N=keep*depth (=40), so switching shapes mid-process throws
            # "size of tensor a (41) must match tensor b (8)" in _ring_write. Making the draft width
            # dynamic is real surgery. Until then TREE MODE IS GREEDY-ONLY: CF_NONGREEDY_CHAIN=1
            # opts into the (shape-broken) fallback for experimentation; default is to fail fast
            # with an actionable message instead of a deep RuntimeError.
            self._branch_now = self.branching
            if self.branching and not self._tree_ok(_STASH.get("sampling_metadata")):
                if os.environ.get("CF_NONGREEDY_CHAIN", "0") == "1":
                    self._branch_now = False
                    _ch = self._beam_chains(self._ring_gather(_rows_g), _k0)
                elif True:
                    raise RuntimeError(
                        "chained-flow tree mode requires greedy sampling (temperature=0, no "
                        "logprobs, no penalties). Got non-greedy sampling params. Run with "
                        "VLLM_SPEC_TREE=0 (chain mode, 1.61x vs the tree's 1.68x at 27B) for "
                        "sampled decoding, or CF_NONGREEDY_CHAIN=1 to try the experimental "
                        "per-step chain fallback (currently shape-incompatible).")
            else:
                # _ring_cg drives self._draft_fn, which is _beam_chains in chain mode and
                # _beam_tree in tree mode -- correct for BOTH. Only the non-greedy-with-branching
                # case above diverts away from it.
                _ch = self._ring_cg(_rows_g, _k0, _B)
            self._hm("draft")
            if self.prof: _ts = time.perf_counter()
            return self._finish(out, rows, req_ids, _ch, sampled_token_ids,
                                _tp if self.prof else None)
        _ctx, _k0 = self._ctx_legacy(rows, req_ids, sh, sampled_token_ids)
        hists = self._legacy_hists
        if self.feedback:
            # Residual lag: training contexts END at h(last committed token); vLLM never computes
            # that hidden (the verify forward's rows are keyed on t_prev and the DRAFT tokens).
            # Substitute the flow's own depth-0 output, which is trained to be exactly that hidden,
            # then draft depths 0..K-1 off the repaired context as the offline harness does.
            p0 = self.drafter.predict_hidden(_ctx.to(self.dtype))[:, 1]        # [B, D] ~= h(t_c)
            rep = [torch.cat([hists[j], p0[j: j + 1].to(self.dtype)], 0)[-self.ctx_size:]
                   for j in range(len(hists))]
            _ctx = torch.stack([self.drafter._context(self._DS(r.unsqueeze(0)))[0] for r in rep], 0)
            _k0 = torch.full_like(_k0, -1)     # -1 => no seed, draft depth 0 (harness semantics)
            self._hm("ctx")
        # NON-GREEDY GUARD. The fork's tree verify requires all_greedy + no logprobs + no logits
        # processors; a BRANCHING tree that reaches the stock linear rejection sampler RAISES
        # (rejection_sampler.py). So a single temperature>0 request would take down the engine.
        # Emit a CHAIN for those steps instead -- it is exact under any sampling params, and costs
        # little (27B batch 1: chain 1.61x vs tree 1.68x).
        # instance attr, not a local: the tree-registration block lives in _finish()
        self._branch_now = branch_now = self.branching and (
            self._tree_ok(_STASH.get("sampling_metadata"))
            or os.environ.get("CF_NONGREEDY_CHAIN", "0") != "1")
        if branch_now:
            _ch = self._chains_cg(_ctx, _k0) if self.use_cg else self._draft_fn(_ctx, _k0)
        else:
            _ch = self._beam_chains(_ctx, _k0)      # eager: shape differs from the captured graph
        self._hm("draft")
        if self.prof: _ts = time.perf_counter()
        return self._finish(out, rows, req_ids, _ch, sampled_token_ids, _tp if self.prof else None)

    # ---------- legacy (CF_GPUCTX=0) per-request Python context build ----------
    @torch.inference_mode()
    def _ctx_legacy(self, rows, req_ids, sh, sampled_token_ids):
        rowmap = self._rows_for_requests(sampled_token_ids)
        self._hm("rowmap")
        ctxs = []
        hists: list[torch.Tensor] = []
        k0: list[int] = []
        for i in rows:
            rid = req_ids[i] if i < len(req_ids) else str(i)
            _st, _cnt = rowmap.get(i, (i, 1))
            _off = self._tree_row_offsets(i, _cnt)
            if _off is None:
                h = sh[_st:_st + _cnt].reshape(_cnt, -1).to(self.dtype)  # accepted hiddens, in order
            else:
                # TREE h(t) harvest: the accepted path's rows are scattered
                # (row 1+path[j]); taking rows 0..L-1 would feed the drafter the
                # hidden states of REJECTED sibling nodes.
                h = sh[[_st + o for o in _off]].reshape(len(_off), -1).to(self.dtype)
            hist = self.ctx_hist.get(rid)
            if hist is None:
                # seed ALREADY ends at the current sampled position -> do NOT append h again
                # (double-counting it shifted the window and evicted the true oldest slot)
                seed = self._seed_prompt_ctx(i)
                # .clone() is LOAD-BEARING: `_seed_prompt_ctx` slices vLLM's `hidden_states`
                # and `.to(self.dtype)` is a no-op when the dtypes already match, so `seed`
                # was a VIEW into a buffer vLLM overwrites on the very next step. Stashing it
                # in ctx_hist therefore silently turned the 8-row prompt seed into whatever
                # the next forward wrote there. (Found by diffing this path against the GPU
                # ring, which copies. Same hazard for `h`, which aliases
                # `sample_hidden_states`.)
                hist = (seed if seed is not None else h).clone()
            else:
                hist = torch.cat([hist, h], 0)[-self.ctx_size:]
            self.ctx_hist[rid] = hist
            hists.append(hist)
            ctxs.append(self.drafter._context(self._DS(hist.unsqueeze(0)))[0])
            k0.append(int(sampled_token_ids[i][-1]))   # last committed token: the flow's depth-0 target
        if self.prof: torch.cuda.synchronize()
        self._legacy_hists = hists
        return torch.stack(ctxs, 0), torch.tensor(k0, dtype=torch.long, device=self.dev)

    # ---------- GPU context build: preallocated ring, every index computed on device ------
    @torch.inference_mode()
    def _ctx_gpu(self, rows, req_ids, sh, sampled_token_ids):
        import numpy as np
        C, R = self.ctx_size, self.max_reqs
        # --- host-resident vLLM metadata only: NO tensor is read back from the device ---
        # `num_draft_tokens` is a plain python list on SpecDecodeMetadata, so the flat row
        # offset of each request in `sample_hidden_states` is host arithmetic. The legacy
        # path instead called `.tolist()` on the GPU `cu_num_draft_tokens`, one sync/step.
        smd = _STASH.get("spec_meta")
        nreq = len(sampled_token_ids)
        if smd is not None and getattr(smd, "num_draft_tokens", None) is not None:
            nd = np.asarray(smd.num_draft_tokens, dtype=np.int64)
            if nd.shape[0] < nreq:
                nd = np.concatenate([nd, np.zeros(nreq - nd.shape[0], dtype=np.int64)])
            starts_all = np.cumsum(nd + 1) - (nd + 1)     # sum_{j<i}(num_draft[j] + 1)
        else:
            starts_all = np.arange(nreq, dtype=np.int64)  # first step: one row per request
        rows_np = np.asarray(rows, dtype=np.int64)
        cnt_np = np.asarray([len(sampled_token_ids[i]) for i in rows], dtype=np.int64)
        k0_np = np.asarray([sampled_token_ids[i][-1] for i in rows], dtype=np.int64)

        # --- slot lifecycle -------------------------------------------------------------
        # A vLLM input-batch index is a SLOT, not a request. `InputBatch.condense()` moves a
        # LIVE request down to a lower index when an earlier one finishes, and a finished
        # request's index is later handed to a new request. The ring is keyed by slot, so
        # both cases must be handled explicitly:
        #   migration -> MOVE the history (re-seeding would collapse the context to the one
        #                decode hidden `_seed_prompt_ctx` returns outside prefill);
        #   recycle   -> RESET, so the new request cannot inherit a dead one's history.
        # Both are decided from req_ids (host strings), so neither costs a sync.
        fresh: list[int] = []
        migrate: list[tuple[int, int]] = []          # (new slot, old slot)
        for i in rows:
            rid = req_ids[i] if i < len(req_ids) else str(i)
            if self._slot_req[i] == rid:
                continue
            old = self._req_slot.get(rid)
            if old is not None and old != i and self._slot_req[old] == rid:
                migrate.append((i, old))
            else:
                fresh.append(i)
            self._slot_req[i] = rid
            self._req_slot[rid] = i
        _live = {req_ids[j] for j in range(len(req_ids))}
        for _k in [k for k in self._req_slot if k not in _live]:
            self._req_slot.pop(_k, None)
        if migrate:
            # snapshot first: two requests can swap slots within one condense()
            _snap, _sp, _sn = self._ring.clone(), self._rpos.clone(), self._nval.clone()
            for _new, _old in migrate:
                self._ring[_new].copy_(_snap[_old])
            _ng = self._h2d("mig_new", np.asarray([m[0] for m in migrate], dtype=np.int64))
            _og = self._h2d("mig_old", np.asarray([m[1] for m in migrate], dtype=np.int64))
            self._rpos.index_copy_(0, _ng, _sp.index_select(0, _og))
            self._nval.index_copy_(0, _ng, _sn.index_select(0, _og))

        wmask_np = np.ones(rows_np.shape[0], dtype=np.int64)
        seeded: dict[int, torch.Tensor] = {}
        recycled: list[int] = []
        for i in fresh:
            s = self._seed_prompt_ctx(i)
            if s is None:
                recycled.append(i)      # no prompt hiddens: start empty, this step's h seeds it
            else:
                seeded[i] = s
        if recycled:
            _rg = self._h2d("reset_i", np.asarray(recycled, dtype=np.int64))
            self._rpos.index_fill_(0, _rg, 0)
            self._nval.index_fill_(0, _rg, 0)
        if seeded:
            # The prefill seed ALREADY ends at the current sampled position, so this step's
            # harvested hidden must NOT also be appended (double-counting it shifts the
            # window and evicts the true oldest slot).
            for j, i in enumerate(rows):
                if i in seeded:
                    wmask_np[j] = 0
            # Batched: one index_copy_ per pointer. Doing it per slot with FIXED _h2d keys
            # would let the host overwrite a pinned staging buffer whose async H2D from the
            # previous slot may still be in flight.
            _sl, _hd, _nv = [], [], []
            for i, s in seeded.items():
                L = min(int(s.shape[0]), C)
                self._ring[i, :L].copy_(s[-L:].to(self.dtype))
                _sl.append(i); _hd.append(L % C); _nv.append(L)
            _sg = self._h2d("seed_i", np.asarray(_sl, dtype=np.int64))
            self._rpos.index_copy_(0, _sg, self._h2d("seed_h", np.asarray(_hd, dtype=np.int64)))
            self._nval.index_copy_(0, _sg, self._h2d("seed_n", np.asarray(_nv, dtype=np.int64)))

        B = rows_np.shape[0]
        bucket = next((b for b in self._buckets if b >= B), B)
        rows_g = self._h2d("rows", rows_np, pad_to=bucket)
        starts_g = self._h2d("starts", starts_all[rows_np])
        cnt_g = self._h2d("cnt", cnt_np)
        k0_g = self._h2d("k0", k0_np, pad_to=bucket)
        wmask_g = self._h2d("wmask", wmask_np).bool()
        path_g = self._accepted_path_gpu()
        # The ring write touches ONLY the real rows; the padded tail exists purely so the
        # graph sees a fixed shape.
        self._ring_write(sh, rows_g[:B], starts_g, cnt_g, wmask_g, path_g, self.K + 1)
        return rows_g, k0_g, B

    def _accepted_path_gpu(self):
        """The tree verify's accepted node path, as a GPU tensor [num_reqs, N].

        `None` in chain mode, where the j-th accepted draft IS node j.
        """
        if not self.branching:
            return None
        try:
            from vllm.v1.spec_decode import tree_state
        except Exception:
            return None
        ts = tree_state.get_current()
        if ts is None or ts.accepted_path is None:
            return None
        return ts.accepted_path

    @torch.inference_mode()
    def _finish(self, out, rows, req_ids, _ch, sampled_token_ids, _tp):
        _ts = time.perf_counter()
        chains = _ch.tolist()      # <- the one sync
        self._hm("sync")
        if self.prof: self._t["sync"] += time.perf_counter() - _ts
        if getattr(self, "_branch_now", False):
            N = self.N
            depths = [d for d in range(self.tree_depth) for _ in range(self.tree_keep)]
            from vllm.v1.spec_decode import tree_state
            for j, i in enumerate(rows):
                row = chains[j]
                toks, pars = row[:N], row[N:]
                out[i] = toks
                rid = req_ids[i] if i < len(req_ids) else str(i)
                tree_state.register(rid, toks, pars, depths)
            tree_state.drop(req_ids)
            chains = [c[:N] for c in chains]
        else:
            for j, i in enumerate(rows):
                out[i] = chains[j]
            if self.tree:
                # Stage A: publish the chain AS A TREE (parents = [-1,0,1,...]) so
                # the fork's tree-verify plumbing runs on a degenerate tree and
                # must reproduce today's accept exactly.
                from vllm.v1.spec_decode import tree_state
                for j, i in enumerate(rows):
                    rid = req_ids[i] if i < len(req_ids) else str(i)
                    n = len(chains[j])
                    tree_state.register(rid, chains[j], list(range(-1, n - 1)),
                                        list(range(n)))
                tree_state.drop(req_ids)

        self._hm("reg")
        if os.environ.get("CF_DIAG", "0") == "1":
            pv = self._t.setdefault("prev_draft", {})
            for j, i in enumerate(rows):
                pv[req_ids[i] if i < len(req_ids) else str(i)] = chains[j]
        if os.environ.get("CF_ORACLE", "0") == "1":
            # chain[k] predicts the committed token at index len(seq)+k (seq already includes this step)
            _th = self._t.setdefault("orc_tok", {})
            _pend = self._t.setdefault("orc_pend", {})
            for j, i in enumerate(rows):
                rid = req_ids[i] if i < len(req_ids) else str(i)
                _pend.setdefault(rid, []).append((len(_th.get(rid, [])), list(chains[j])))
        live = set(req_ids)
        for k in list(self.ctx_hist):
            if k not in live:
                self.ctx_hist.pop(k, None)
        if self.prof:
            self._t["n"] += 1
            self._t["total"] += time.perf_counter() - _tp
            self._t["_last_exit"] = time.perf_counter()
            if self._t["n"] % 20 == 0:
                n = self._t["n"]
                print(f"[cf-prof] n={n} | per-step ms: flow {self._t['flow']/n*1000:6.2f} "
                      f"beam {self._t['beam']/n*1000:6.2f} ctx {self._t['ctx']/n*1000:6.2f} "
                      f"sync {self._t['sync']/n*1000:6.2f} | propose TOTAL {self._t['total']/n*1000:6.2f} "
                      f"| OUTSIDE propose (vLLM) {self._t['gap']/n*1000:7.2f}", flush=True)
        if self.hprof:
            self._hm("tail")
            self._h["_exit"] = self._h["_mark"]
            self._h["_mark"] = None
            self._h["n"] += 1
            n = self._h["n"]
            if n % 100 == 0:
                ins = sum(self._h[s] for s in self._hsegs if s != "gap")
                print("[cf-hprof] n=%d | HOST ms/step: " % n
                      + " ".join(f"{s} {self._h[s]/n*1000:5.2f}" for s in self._hsegs)
                      + f" | in-propose {ins/n*1000:5.2f} | STEP {(ins+self._h['gap'])/n*1000:6.2f}",
                      flush=True)
        return out
