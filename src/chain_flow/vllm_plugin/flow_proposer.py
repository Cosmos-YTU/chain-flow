"""Chained-Flow drafter as a native vLLM speculative-decoding proposer (V1 `custom_class` hook).

    LLM(model=..., speculative_config={
        "method": "custom_class",
        "model": "chain_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
        "num_speculative_tokens": K})
    env: CF_DRAFTER_DIR=<ckpt dir>   [CF_SHORTLIST=<.pt of token ids>] [CF_WIDTH=4]

STOCK vLLM V1 verifies a LINEAR CHAIN (RejectionSampler over flat num_draft_tokens) — there is no
tree-verify hook. So we still build the draft TREE from the one flow pass and beam-search it with
markov + path-residual rescoring, then emit the highest-scoring PATH as the chain. That keeps the
tree's token-selection benefit; only the multi-branch acceptance is lost. This CHAIN path is the
DEFAULT SHIPPING TARGET and needs no patching.

The branching TREE path (`VLLM_SPEC_TREE=1`) recovers that acceptance and needs the forked vLLM
(`chain_flow/patches/vllm-0.25.1-chain-flow-tree.patch`): a tree-aware verify, a tree-shaped
GDN recurrence, and the `vllm.v1.spec_decode.tree_state` hand-off. Every use of that module in
this file is behind a `branching` / `self.tree` gate or a try/except, so the file imports and runs
unchanged on unmodified vLLM — see `_install_async_hooks`.

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
# CF_ASYNC_SPEC shape-only stand-in for "this request emitted at least one token". Never a
# real token id -- see `propose`.
_ASYNC_ROW = (-1,)


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


def _install_uniform_decode_guard() -> None:
    """Stop vLLM from mistaking a PREFILL of exactly ``num_speculative_tokens + 1`` tokens
    for a uniform spec-decode batch.

    THE BUG (upstream vLLM, not this proposer -- stock `method="ngram"` reproduces it
    byte-for-byte).  `GPUModelRunner._is_uniform_decode` is::

        max_num_scheduled_tokens == 1 + num_speculative_tokens
        and num_tokens == max_num_scheduled_tokens * num_reqs

    which is a statement about SHAPE ONLY.  A single request prefilling exactly K+1 prompt
    tokens schedules K+1 tokens in one row and satisfies it exactly, so the dispatcher hands
    the batch to the **FULL (decode) cudagraph** instead of PIECEWISE -- `uniform=True` in the
    BatchDescriptor.  The attention METADATA is still built correctly (`num_prefills=1`), but
    a FULL graph has the kernel selection baked in at capture time, and on a HYBRID model
    (Qwen3.5 = GDN linear-attention + full attention) prefill and decode are different
    kernels: the captured graph holds the recurrent/spec-decode gated-delta-rule step, not
    the chunked prefill scan.  Replaying it computes the wrong linear-attention output AND
    leaves a wrong recurrent state, so the request is corrupt from its very first token and
    the generation degenerates (measured: repeated token id 0).

    K-dependence is the signature: the collision is at prompt length K+1 for every K, which
    is exactly what `uniform_decode_query_len = 1 + num_speculative_tokens` predicts.  Our
    7-domain benchmark cannot see it because those prompts are long -- but the README
    Quickstart prompt is 6 tokens and the default K is 5.

    THE FIX.  `uniform_decode` must be a statement about PHASE, not shape.  A row is still
    in prefill iff it has not yet computed its whole prompt (`num_computed_tokens <
    num_prompt_tokens`); that is exactly the vLLM-side condition for "this step runs prompt
    tokens through the model", it covers chunked-prefill chunks that happen to be K+1 long,
    and it covers the mixed case (a K+1 prefill batched with real spec decodes, where
    `num_tokens == max * num_reqs` also holds).  When any such row is present we pass
    `force_uniform_decode=False`, which is the SAME public lever vLLM already uses at its
    cudagraph-CAPTURE call site to stop a capture batch from being misread as a uniform
    decode.  Speculation is NOT disabled: only this one step's cudagraph mode changes, from
    FULL to PIECEWISE, which is what every other prompt length already does.

    `force_uniform_decode is not None` is left alone, which is what makes this a strict
    no-op on the two paths that already decide for themselves: vLLM's own cudagraph capture,
    and the fork's `CF_TREE_FULLCG` tree dispatch, which passes `_cf_force_uniform` on EVERY
    step (`_graphable`, a genuine tree-decode predicate).  That is also why the TREE arm was
    never exposed to this bug -- measured, not assumed.

    MEASURED: 0 dispatch decisions changed over the whole 7-domain sweep at 4B chain
    (1120 steps) and 4B/27B tree (776/689 steps); outputs bit-identical, pooled tok/s
    unchanged.  CF_PREFILL_GUARD=0 disables the patch.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_cf_uniform_guard", False):
        return
    if os.environ.get("CF_PREFILL_GUARD", "1") == "0":
        return
    orig = getattr(GPUModelRunner, "_determine_batch_execution_and_padding", None)
    if orig is None:
        # A vLLM that dispatches cudagraphs somewhere else entirely: say so rather than
        # silently shipping the corruption.
        print("[chain-flow] WARNING: this vLLM has no "
              "_determine_batch_execution_and_padding; the prefill/uniform-decode guard is "
              "NOT installed. A prompt of exactly num_speculative_tokens+1 tokens may "
              "produce garbage on hybrid models.", flush=True)
        return

    def patched(self, *a, **kw):
        if kw.get("force_uniform_decode") is None:
            ib = self.input_batch
            n = int(kw.get("num_reqs", ib.num_reqs))
            # numpy views, no device sync: `num_computed_tokens_cpu` is the KV the request
            # already has, `num_prompt_tokens` the length of its prompt.
            if bool((ib.num_computed_tokens_cpu[:n] < ib.num_prompt_tokens[:n]).any()):
                kw["force_uniform_decode"] = False
        return orig(self, *a, **kw)

    GPUModelRunner._determine_batch_execution_and_padding = patched
    GPUModelRunner._cf_uniform_guard = True


def _install_early_hook() -> None:
    """CF_DRAFT_EARLY: issue the draft on a SIDE STREAM the moment the tree verify has been
    enqueued, instead of after `_bookkeeping_sync`.

    Why this is the only overlap available (see the module docstring in the report): the draft
    reads `sample_hidden_states` rows selected by the ACCEPTED PATH, so it cannot start before
    this step's target forward AND its verify. It is genuinely serial with both. What it does
    NOT depend on is anything vLLM does *after* the verify: the tree KV compaction / GDN state
    finalize (`_tree_postprocess`), and — the part that actually costs wall time — the ~1-1.5 ms
    of pure HOST work in `_bookkeeping_sync` (RejectionSampler.parse_output, the token_ids_cpu
    loop, output construction) during which the GPU has drained and sits idle.

    Stock order:  [fwd|verify] --gpu idle-- host bookkeeping --> [draft]        (draft not started)
    Early order:  [fwd|verify] [draft on side stream] || host bookkeeping       (draft overlapped)

    The default stream's D2H of the sampled tokens is NOT pushed behind the draft (that is the
    whole point of the side stream — putting the draft on the default stream ahead of the D2H
    just makes the host block longer and gains exactly nothing).

    MEASURED, 4B batch 1, keep 8 x depth 5, 7-domain sweep: 125.7 -> 130.1 tok/s (+3.5%), step
    19.4 -> 18.4 ms, in-propose `sync` 5.00 -> 4.58 ms and `ctx` 0.34 -> 0.00 ms. Accept is
    BIT-IDENTICAL on all 7 domains (3.526 / 3.821 / 2.154 / 2.520 / 2.116 / 1.882 / 2.315),
    which is the real correctness evidence: `cnt`/`k0` are read off the verify's own GPU
    tensors instead of the CPU token lists, so an off-by-one would move accept immediately.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_cf_early_hooked", False):
        return
    orig_st = GPUModelRunner.sample_tokens

    def sample_tokens(self, *a, **k):
        # `execute_model_state` is cleared inside the original before `_sample` runs, so the
        # hidden states have to be grabbed here — one frame up — not in `_sample`.
        _STASH["early_state"] = self.execute_model_state
        _STASH["runner"] = self
        return orig_st(self, *a, **k)

    GPUModelRunner.sample_tokens = sample_tokens

    orig_sample = GPUModelRunner._sample

    def _sample(self, logits, spec_decode_metadata):
        out = orig_sample(self, logits, spec_decode_metadata)
        p = _STASH.get("proposer")
        if p is not None and getattr(p, "early", False):
            p._prelaunch(self, out, spec_decode_metadata)
        return out

    GPUModelRunner._sample = _sample

    orig_take = GPUModelRunner.take_draft_token_ids

    def take_draft_token_ids(self):
        # CF_DRAFT_DEFER: this is the FIRST place vLLM reads the draft token values on the host
        # (engine core post_step -> scheduler.update_draft_token_ids). Join here, not in propose().
        p = _STASH.get("proposer")
        if p is not None and getattr(p, "_defer", None) is not None:
            p._materialize()
        return orig_take(self)

    GPUModelRunner.take_draft_token_ids = take_draft_token_ids
    GPUModelRunner._cf_early_hooked = True


def _install_async_hooks(tree: bool) -> None:
    """CF_ASYNC_SPEC=1: make this `custom_class` proposer satisfy vLLM's ASYNC-SCHEDULING
    contract, so the engine can overlap `scheduler.schedule()` + `_prepare_inputs` of step
    N+1 with step N's GPU work (worth +10.5% / +5.5% / +1.6% at 4B / 9B / 27B -- see
    docs/BENCHMARKING.md).

    `tree=False` is the FORK-FREE CHAIN build, and it is the reason this function takes an
    argument at all.  Requirements (1)-(3) below are properties of vLLM's async contract and
    hold for any proposer; requirement (4) is about the TREE SHAPE and lives entirely in
    `vllm.v1.spec_decode.tree_state`, which exists ONLY in the Chained-Flow vLLM fork.  This
    module used to import it unconditionally, so `CF_ASYNC_SPEC=1` on stock vLLM raised
    ImportError at proposer construction and took the whole engine down -- which is why the
    fork-free arm had never been measured with async on.  A chain draft has no tree to
    register, so on stock vLLM there is simply nothing to join.

    With `async_scheduling=True`, `_bookkeeping_sync` takes the branch at
    gpu_model_runner.py:4149 and `valid_sampled_token_ids` is `[]`; `token_ids_cpu` /
    `output_token_ids` are filled with -1 placeholders.  A proposer that reads those lists
    sees zero rows every step and silently drafts nothing.  Four things have to change:

      1. cnt / k0 must come off the VERIFY's own GPU tensors, never the CPU lists.
         `_prelaunch` already did this for the steady-state fast path; under CF_ASYNC_SPEC
         it publishes them for EVERY step (`_STASH["gpu_cnt"] / ["gpu_k0"]`) so the slow
         path (`_ctx_gpu`) is GPU-token-native too.
      2. `_draft_token_ids` must be a GPU TENSOR [num_reqs, N].  vLLM's async path never
         copies drafts to the host: `_prepare_input_ids` SCATTERS them straight into
         `input_ids.gpu` from that tensor.  A list-of-lists is silently ignored there.
      3. `valid_sampled_token_count_gpu` + `prev_sampled_token_ids` must be published, or
         `update_num_computed_tokens_for_batch_change` never runs and `num_computed_tokens`
         stays at the scheduler's OPTIMISTIC value (all drafts accepted) -- KV corruption,
         not merely a lost speedup.  Both are exactly the `cnt`/`k0` of (1).
      4. the TREE SHAPE still has to reach the host for `_prepare_inputs` (ancestor mask +
         GDN chain).  It is joined at the TOP of `_prepare_inputs` instead of inside
         `propose()`, which is the latest possible point: everything between the two --
         bookkeeping, output construction, `update_from_output`, detokenise,
         `scheduler.schedule()` -- then overlaps the GPU.
         TREE ONLY.  A chain draft's "shape" is the identity (node j's parent is j-1), so
         there is nothing to hand over and nothing to join; stock vLLM's own
         `_prepare_inputs` already knows how to lay out a flat chain.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_cf_async_hooked", False):
        return

    if tree:
        # JOIN POINT (4).  The PRIMARY join is `tree_state.join_pending()`, called from inside
        # `_prepare_inputs` immediately before the tree registry is read -- i.e. after the
        # state update, the positions, the slot mapping and `_prepare_input_ids`, all of which
        # then overlap the GPU.  Putting it at the top of `_prepare_inputs` instead costs ~1 ms
        # of GPU idle per step (MEASURED: it drains the device and every host instruction after
        # it runs against an empty queue), which is most of what async scheduling is worth.
        #
        # `tree_state` is a FORK-ONLY module.  The import is inside this branch, not at the top
        # of the function, because `tree=False` is the stock-vLLM shipping path and an
        # unconditional import there is an ImportError that kills the engine at proposer
        # construction.
        from vllm.v1.spec_decode import tree_state as _ts_mod

        _ts_mod.PENDING_JOIN = lambda: (
            _STASH["proposer"]._async_register() if "proposer" in _STASH else None
        )

        orig_prep = GPUModelRunner._prepare_inputs

        def _prepare_inputs(self, *a, **k):
            try:
                return orig_prep(self, *a, **k)
            finally:
                # INVARIANT: a device hand-off is valid for exactly the ONE `_prepare_inputs`
                # that follows it. Dropping it unconditionally here means the only entry a
                # step can ever see is one published since the previous step -- a stale tree
                # would pass the req_id check whenever the batch is unchanged, and describe
                # the wrong draft.
                _ts_mod.drop_gpu_tree()

        GPUModelRunner._prepare_inputs = _prepare_inputs

        # The tree registry validates its shape against the tokens the scheduler actually
        # scheduled.  Under async those are -1 placeholders, so that check has to become a
        # length check or every step falls back to chain semantics.
        _ts_mod.PLACEHOLDER_TOKENS = True

    orig_bk = GPUModelRunner._bookkeeping_sync

    def _bookkeeping_sync(self, *a, **k):
        # (3).  `execute_model` clears `valid_sampled_token_count_gpu` and
        # `prev_sampled_token_ids` AFTER `_sample` (gpu_model_runner.py:4964-4965), so
        # `_prelaunch` cannot publish them itself -- they would be wiped.  Publish here,
        # which is after the clear and before the async branch of `_bookkeeping_sync`
        # asserts `sampled_token_ids.shape[-1] == 1` on a None `prev_sampled_token_ids`.
        p = _STASH.get("proposer")
        if p is not None and p.async_spec:
            p._publish_gpu_counts(self)
        return orig_bk(self, *a, **k)

    GPUModelRunner._bookkeeping_sync = _bookkeeping_sync

    # The AsyncScheduler pads every running request with `num_spec_tokens_to_schedule`
    # placeholder (-1) spec tokens and the worker fills the real values on the GPU.  Our
    # tree configures `num_speculative_tokens = N + 1` (one SPARE mamba state column, see
    # tree_gdn._colmap) but only ever EMITS N draft tokens, so the stock padding would
    # schedule one query row per step that nothing ever writes.  In CHAIN mode the two
    # already agree (`draft_width == K == num_speculative_tokens`), so this is a no-op there
    # -- installed anyway so the one code path covers both.
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler

        if not getattr(AsyncScheduler, "_cf_async_hooked", False):
            _orig_uas = AsyncScheduler._update_after_schedule

            def _update_after_schedule(self, scheduler_output):
                n = _STASH.get("draft_width")
                if n is not None:
                    scheduler_output.num_spec_tokens_to_schedule = int(n)
                return _orig_uas(self, scheduler_output)

            AsyncScheduler._update_after_schedule = _update_after_schedule
            AsyncScheduler._cf_async_hooked = True
    except Exception as e:      # pragma: no cover - keeps a stock vLLM importable
        print(f"[cf-async] could not patch AsyncScheduler: {e!r}", flush=True)

    GPUModelRunner._cf_async_hooked = True


class FlowDrafterProposer:
    def __init__(self, vllm_config):
        _install_hidden_state_hook()
        # Must be installed before the first forward (this runs during engine init, so it
        # is) -- see the docstring: without it a prompt of exactly K+1 tokens is dispatched
        # to the FULL decode cudagraph and the request is corrupt from token 0.
        _install_uniform_decode_guard()
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
        # Depth 1 expands ONE frontier node (the committed seed), so it can only ever hold
        # min(tree_keep, tree_topb) nodes -- `k = min(W, fr * TB)` in _beam_tree with fr=1.
        # N assumes every level is full, and _finish slices `row[:N]`, so keep > topb would
        # hand vLLM a short/misaligned draft: the same silent-corruption failure as an
        # over-large CF_TREE_DEPTH (see _clamp_depth). Fail loudly instead.
        if self.branching and self.tree_keep > self.tree_topb:
            raise ValueError(
                f"CF_TREE_KEEP={self.tree_keep} > CF_TREE_TOPB={self.tree_topb}: depth 1 expands "
                f"a single node, so it can hold at most {self.tree_topb} children while "
                f"N=keep*depth assumes {self.tree_keep}. Raise CF_TREE_TOPB to >= CF_TREE_KEEP.")
        self.N = self.tree_keep * self.tree_depth
        if self.branching:
            # one spare mamba state column (see _colmap) => config N+1, emit N
            assert int(spec.num_speculative_tokens) == self.N + 1, (
                f"set num_speculative_tokens={self.N + 1} for "
                f"CF_TREE_KEEP={self.tree_keep} x CF_TREE_DEPTH={self.tree_depth}")
        self.K = int(spec.num_speculative_tokens) - (1 if self.tree else 0)
        self._draft_fn = self._beam_tree if self.branching else self._beam_chains
        # How many draft tokens a step actually EMITS, i.e. the column count of the tensor
        # handed to vLLM.  A tree emits N = keep*depth nodes (out of N+1 configured columns);
        # a chain emits exactly its K.  `self.N` is the TREE width and is meaningless in chain
        # mode -- slicing `ch[:, :self.N]` there would over-read a [B, K] draft.
        self.draft_width = self.N if self.branching else self.K
        self.width = int(os.environ.get("CF_WIDTH", "4"))
        self.ckd = os.environ.get("CF_DRAFTER_DIR")
        if not self.ckd:
            raise ValueError("set CF_DRAFTER_DIR to the drafter checkpoint dir or a HF repo id")
        if not os.path.isdir(self.ckd):
            # a HF repo id (e.g. ytu-ce-cosmos/Flow-Drafter-Qwen3.5-27B): run the PUBLISHED weights
            from huggingface_hub import snapshot_download
            # `shortlist.pt` is in the allow-list so that a drafter repo which ships one is
            # actually picked up.  Documenting "drop shortlist.pt next to the checkpoint" while
            # never downloading it -- and then testing `os.path.isdir` on a repo id, which is
            # never true -- made that instruction unreachable for the documented usage.
            self.ckd = snapshot_download(
                self.ckd, allow_patterns=["model.safetensors", "chained_flow_tree_config.json",
                                          "shortlist.pt"])
            print(f"[chain-flow] drafter from HF hub -> {self.ckd}", flush=True)
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
        # The VALUE is resolved in the defaults block below (it is a gated default now, see
        # `defaults.finalize_compile`); only the mode is read here.
        self.compile_mode = os.environ.get("CF_COMPILE_MODE", "max-autotune-no-cudagraphs")
        # ---- the capability-gated DEFAULTS (chain_flow.defaults) ----------------------
        # Every flag below now reads a value that `defaults.apply()` has already proposed at
        # package-import time.  The reads stay bare `os.environ` lookups on purpose: the env is
        # the single channel that also reaches the parts of the stack we do not own (the forked
        # vLLM reads its own flags lazily and cannot be handed a Python object).  `defaults`
        # writes a gate failure BACK into the env, so what is read here is the decision, not the
        # request, and `CF_X=0` still wins everywhere.
        from chain_flow import defaults as _cfd
        _cfd.apply()
        self.fuse_path = _cfd.truthy(os.environ.get("CF_FUSE_PATH"))
        # CF_COMPILE: proposed by the table, GATED in `_build` (finalize_compile) before the
        # wrap happens.  It used to default to 0 here while bench_cf.sh's chain and tree arms
        # hard-coded 1, so every published number came from a path no pip user was on and no
        # line in the log mentioned it.
        self.compile = _cfd.truthy(os.environ.get("CF_COMPILE"))
        # CF_TWOPASS_M: two-pass candidate head (see _candidates). 0 = OFF (one-pass).
        self.twopass_m = int(os.environ.get("CF_TWOPASS_M", "0") or 0)
        # CF_TWOPASS_SEED conditions pass A on the seed path's residual+markov bias. MEASURED
        # WORSE at every M (tree accept -0.02 to -0.03) and slower -- the seed conditional is the
        # depth-1 distribution and drags the deeper depths' candidate sets toward it, while the
        # bare marginal covers more paths. Kept as a flag, default OFF.
        self.twopass_seed = os.environ.get("CF_TWOPASS_SEED", "0") == "1"
        # CF_PATH_TRIM: skip the PathHead offsets that provably have no ancestor (see _nlive).
        # BIT-EXACT; DEFAULT ON, no gate.
        self.path_trim = _cfd.truthy(os.environ.get("CF_PATH_TRIM"))
        # CF_RING_TRIM: narrow the ring append to the slots a TREE step can actually fill
        # (see _wmax).  BIT-EXACT; DEFAULT ON, gated to branching-tree mode by `_wmax` itself
        # (which is also what the summary prints -- `ring_slots=6 of 41`).
        self.ring_trim = _cfd.truthy(os.environ.get("CF_RING_TRIM"))
        # CF_TWOPASS_SHARED: one candidate set for all depths (gather once, not per depth).
        self.twopass_shared = _cfd.truthy(os.environ.get("CF_TWOPASS_SHARED"))
        self.compile_beam = os.environ.get("CF_COMPILE_BEAM", "0") == "1"
        self.feedback = os.environ.get("CF_FEEDBACK", "0") == "1"
        if self.feedback:
            self.use_cg = False     # two flow passes + rebuilt context: eager for now
            self.gpuctx = False     # needs the raw per-request history tensors
        if self.gpuctx and os.environ.get("CF_CTXDIAG", "0") == "1":
            raise ValueError("CF_CTXDIAG inspects the per-request context tensor and only "
                             "works on the legacy path: set CF_GPUCTX=0")
        self._cg: dict[int, tuple] = {}          # batch-bucket -> (graph, ctx_buf, out_buf)
        # THE LADDER REACHES `max_num_seqs`, NOT 32.  It used to stop at 32, and the consequence
        # was measured rather than reasoned about: profiling a 4B chain server at a decode batch
        # of 64, 1901 of 2000 steps ran the draft with NO CUDAGRAPH AT ALL, and every distinct
        # batch size in 33..64 (33, 34, 35, 37, 38, 40, 42, ..., 64) was its own `dynamic=False`
        # max-autotune compile -- the ~850-autotune-event storm and the 70-80 s per-bucket stall.
        # Both are the same missing rung. See `_bucket_ladder`.
        self._buckets = self._bucket_ladder()
        self._t = {"n": 0, "ctx": 0.0, "flow": 0.0, "beam": 0.0, "sync": 0.0, "total": 0.0,
                   "gap": 0.0, "_last_exit": None}
        # CF_HPROF: HOST-ONLY segment timing. Unlike CF_PROFILE it inserts NO
        # torch.cuda.synchronize(), so it is cudagraph-capture-safe and can stay on in a
        # shipping-config run. Every segment is host wall time; a segment that blocks on
        # the GPU (a D2H copy of a not-yet-ready tensor) shows up as a large host segment,
        # which is exactly the glue we are hunting.
        # The mean alone lies: `gap` also spans llm.generate() boundaries (teardown +
        # tokenize + the next prompt's PREFILL), and the first generate carries drafter
        # compile. Averaging those into a per-decode-step figure inflated `gap` to 13.68 ms
        # against a real steady-state value. So: discard CF_HPROF_SKIP steps, and report the
        # gap's median and max next to the mean -- a mean far above the median IS the artifact.
        self.hprof = os.environ.get("CF_HPROF", "0") == "1"
        self._hskip = int(os.environ.get("CF_HPROF_SKIP", "40"))
        self._hevery = int(os.environ.get("CF_HPROF_EVERY", "50"))
        self._h = {"n": 0, "_mark": None, "_exit": None, "_seen": 0}
        self._hgaps = []
        self._hsegs = ["pre", "rowmap", "ctx", "draft", "sync", "reg", "tail", "gap"]
        for s in self._hsegs:
            self._h[s] = 0.0
        # CF_DRAFT_EARLY (default OFF): overlap the draft with vLLM's post-verify HOST work by
        # issuing it on a side stream right after `_sample`. See `_install_early_hook`.
        self.early = _cfd.truthy(os.environ.get("CF_DRAFT_EARLY"))
        # CF_DRAFT_DEFER (needs CF_DRAFT_EARLY): don't join the draft inside propose() either.
        # vLLM does not READ the draft token values until `take_draft_token_ids()` in the engine
        # core's post_step, so the join can slide past the whole bookkeeping tail / kv-connector
        # finalize / ModelRunnerOutput construction.
        # MEASURED (4B, batch 1, keep 8 x depth 5): NOT WORTH IT -- 128.9 vs 130.2 tok/s for
        # CF_DRAFT_EARLY alone, and the `sync` segment only moved 4.58 -> 4.55 ms. That window
        # is ~0.03 ms of host work; it is kept only because that null result is the PROOF that
        # the ~2.2 ms of GPU idle still left in the step lives entirely DOWNSTREAM of
        # `take_draft_token_ids` (scheduler.schedule + _prepare_inputs), i.e. it cannot be
        # reclaimed until the draft TOKENS and the tree PARENTS stop having to reach the host.
        self.defer = self.early and os.environ.get("CF_DRAFT_DEFER", "0") == "1"
        self._defer = None
        self._pre = None                 # (rows, out_tensor, event) staged by _prelaunch
        self._dstream = None
        self._ones_b = None
        self._pre_n = 0
        self._pre_skip = 0
        # CF_BATCH_AUDIT (DEFAULT OFF): per-step tally of the DECODE BATCH the drafter actually
        # ran at.  The `[cf-defaults]` line is a BUILD-TIME statement -- it is printed once,
        # before a single request has arrived, and it cannot express a flag that is gated on the
        # batch of an individual step.  Two of ours are:
        #   * the fused block kernel (`chunked_flow._cf_fused_runner`) engages only for
        #     `x.shape[0] == 1`, and the drafter's x.shape[0] is exactly the cudagraph BUCKET
        #     below -- so `bucket==1` steps ran the CUDA kernel and every other step silently
        #     ran the PyTorch stack.  CF_CUDA_PAIR rides on it and disengages with it.
        #   * a bucket of `None` (batch > 32) means the draft cudagraph was skipped entirely.
        # So this histogram is the per-REQUEST answer that the startup banner cannot give.
        # Host-side dict arithmetic once per step, no sync, no GPU work; still default off
        # because a benchmark should not pay for its own instrumentation unasked.
        self._audit = os.environ.get("CF_BATCH_AUDIT", "0") == "1"
        self._audit_t = {"steps": 0, "bucket": {}, "B": {}, "nocg": 0, "pre_hit": 0,
                         "cut": 0, "drafted": 0}
        # CF_SPEC_MAX_BATCH (DEFAULT ON where the threshold was MEASURED): stop drafting above a
        # decode batch. The half of the cutoff that saves the DRAFT; `vllm_plugin.batch_cutoff`
        # patches the scheduler, which is the half that saves the VERIFY. See that module for why
        # both are needed and why a skip must return a full-width tensor rather than a short one.
        # `requested()`, not `max_batch()`: under CF_SPEC_MAX_BATCH=auto the NUMBER is not known
        # until `_build()` has the target's hidden size, and this runs before that. Latching a 0
        # here would leave the drafter half permanently off while the scheduler half was on --
        # the exact half-configured state whose symptom is "the cutoff does not work".
        from chain_flow.vllm_plugin import batch_cutoff as _cf_cut
        self._cut_on = _cf_cut.requested() and _cf_cut.drafter_half()
        # CF_WARM_BUCKETS (DEFAULT OFF): compile + capture the draft graph for every batch bucket
        # at BUILD time instead of the first time each bucket is touched. See `_warm_buckets`.
        self.warm_buckets = os.environ.get("CF_WARM_BUCKETS", "0") == "1"
        # CF_ASYNC_SPEC (default OFF): become GPU-token-native so vLLM's async scheduling
        # can be enabled for this proposer.  See `_install_async_hooks`.  Requires the
        # `_sample` hook, so it implies CF_DRAFT_EARLY.
        # The gate is the ENGINE's resolved `scheduler_config.async_scheduling`, not the env var:
        # config/vllm.py only keeps async ON for a custom_class proposer when CF_ASYNC_SPEC was
        # already set when the config was built.  If it was not, the engine is synchronous and
        # going GPU-token-native would hand a GPU draft tensor to a path that silently ignores it.
        self.async_spec = _cfd.finalize_async(self, vllm_config)
        if self.async_spec:
            self.early = True
            self.defer = False           # the join moved to _prepare_inputs, see _async_register
            _STASH["draft_width"] = self.draft_width
        # node depths are a pure function of the tree SHAPE (level-major BFS), so they
        # never have to travel with the data.
        self._areg_depths = [d for d in range(self.tree_depth) for _ in range(self.tree_keep)]
        self._areg = None                # pending host tree registration (pinned, event, ...)
        self._areg_pin = None
        self._areg_ev = None
        self._ptree = None               # persistent [max_reqs, N] GPU tree PARENTS
        self._areg_n = 0                 # steps that offered the device hand-off
        self._skip_n = 0                 # ... of which took it (i.e. skipped the join)
        self._dtok = None                # persistent [max_reqs, N] GPU draft-token tensor
        # CF_ASYNC_PROBE: how long the tree-shape join at the top of `_prepare_inputs`
        # actually blocks the host, and what the host-side registration costs.
        self._aprobe = ([0.0, 0.0, 0]
                        if os.environ.get("CF_ASYNC_PROBE", "0") == "1" else None)
        # CF_ASYNC_NOJOIN: MEASUREMENT ONLY, DELIBERATELY INCORRECT.  Skips the join, so
        # the tree shape handed to `_prepare_inputs` may be a step stale.  Its only purpose
        # is to price the join: it is the upper bound a fully GPU-resident tree could reach.
        self._nojoin = os.environ.get("CF_ASYNC_NOJOIN", "0") == "1"
        self._acc_gpu = None             # [2] i64 (tokens emitted, request-steps): accept
        # without ever reading `cnt` on the host mid-run
        self._acc_host = (0, 0)
        if self.early:
            _install_early_hook()
        if self.async_spec:
            _install_async_hooks(self.branching)
        _STASH["proposer"] = self       # so the bench harness can read per-set accept counters

    def _hm(self, seg):
        """Close the open host segment and attribute it to `seg`."""
        if not self.hprof:
            return
        now = time.perf_counter()
        if self._h["_mark"] is not None and self._h["_seen"] > self._hskip:
            self._h[seg] = self._h.get(seg, 0.0) + now - self._h["_mark"]
        self._h["_mark"] = now

    def _hbail(self):
        """Early return out of propose(): stamp _exit so the NEXT gap sample measures one
        step, not two merged ones, and do not count this step in n."""
        if self.hprof:
            self._h["_exit"] = time.perf_counter()
            self._h["_mark"] = None

    # ---------- lazy build (needs the loaded base model's embed / lm_head) ----------
    def _build(self):
        import torch.nn.functional as F
        from types import SimpleNamespace
        from safetensors.torch import load_file
        from chain_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig

        model = _STASH["runner"].model
        inner = model.language_model.model if hasattr(model, "language_model") else model.model
        embed_w = inner.embed_tokens.weight
        lm = model.language_model.lm_head if hasattr(model, "language_model") else model.lm_head
        lm_w = lm.weight
        self.dev, self.dtype = embed_w.device, embed_w.dtype
        H, V = embed_w.shape[1], embed_w.shape[0]
        # Read off the LOADED embedding, not off a config: it is the one number here that cannot
        # be stale or overridden. `CF_SPEC_MAX_BATCH=auto` keys its threshold on it.
        self.target_hidden = int(H)

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
            print(f"[chain-flow] CF_FLOW_STEPS: num_flow_steps {dcfg.num_flow_steps} -> {_fs}",
                  flush=True)
            dcfg.num_flow_steps = _fs
        d = TreeVAEFlowDrafter(Stub(), dcfg).to(self.dev).to(self.dtype).eval()
        assert d.config is dcfg and d.config.num_flow_steps == dcfg.num_flow_steps
        print(f"[chain-flow] integrator will run num_flow_steps="
              f"{d.config.num_flow_steps} (dt={1.0 / d.config.num_flow_steps:.4f}) x "
              f"num_drafter_layers={dcfg.num_drafter_layers}", flush=True)
        d._dtype = self.dtype
        sd = load_file(f"{self.ckd}/model.safetensors")
        _sub = {k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}
        _res = d.load_state_dict(_sub, strict=False)
        _own = dict(d.named_parameters())
        print(f"[chain-flow] ckpt keys={len(sd)} drafter-prefixed={len(_sub)} "
              f"model params={len(_own)} MISSING={len(_res.missing_keys)} UNEXPECTED={len(_res.unexpected_keys)}",
              flush=True)
        if _res.missing_keys:
            print(f"[chain-flow]   first missing: {_res.missing_keys[:6]}", flush=True)
        if _res.unexpected_keys:
            print(f"[chain-flow]   first unexpected: {_res.unexpected_keys[:6]}", flush=True)
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
        else:
            # The CHAIN equivalent of the check above, and it was missing.  A chain emits one
            # token per flow depth and depth 0 reconstructs the already-committed token, so the
            # ceiling is `draft_length - 1` exactly as for the tree.  Asking for more used to
            # build fine, load the model, and then die on the first decode step inside
            # `_async_draft_tensor` with "draft is (32, 7) but 8 columns were declared to vLLM"
            # -- a shape assertion about an internal buffer, several minutes after the mistake,
            # naming neither the flag to change nor the value to change it to. (Found by the
            # K x prompt-length sweep at K=8 against a draft_length=8 drafter.)
            emit = dcfg.draft_length - 1
            if self.K > emit:
                raise ValueError(
                    f"num_speculative_tokens={self.K} but this drafter can only emit {emit} "
                    f"chain tokens (draft_length={dcfg.draft_length}; depth 0 reconstructs the "
                    f"token that was already committed). vLLM has been told to expect "
                    f"{self.K} draft columns and a short draft cannot be widened -- doing so "
                    f"would scatter the previous step's tokens into input_ids. Set "
                    f"num_speculative_tokens<={emit}.")
        if self.gpuctx:
            self._init_ring(H)

        # knob 1: shortlist head (quality-free at full coverage; the tree only uses top candidates)
        #
        # Resolved HERE rather than at import time for two reasons, both of which were bugs:
        # `self.ckd` is a real local directory only after the HF snapshot download above, and the
        # VOCAB GUARD needs `V`, which only exists once the target model is loaded.
        self._sl = None
        self._sl_src = "none"
        self._sl_why = ("FULL HEAD -- ~43-47% of the draft is wasted; "
                        "build one with `chain-flow build-shortlist`")
        self._hw = lm_w
        self._w2 = d.markov.w2.weight
        from chain_flow import defaults as _cfd_sl
        from chain_flow import shortlist as _slmod

        if _cfd_sl.was_explicit("CF_SHORTLIST") and self.shortlist_path:
            # fail loudly: a typo'd path silently benchmarking the full head is exactly how the
            # shortlist went unmeasured for a week.
            if not os.path.exists(self.shortlist_path):
                raise FileNotFoundError(f"CF_SHORTLIST={self.shortlist_path} does not exist")
            cands = [(self.shortlist_path, "CF_SHORTLIST")]
        elif _cfd_sl.was_explicit("CF_SHORTLIST"):
            cands = []                       # CF_SHORTLIST="" means "full head, deliberately"
            self._sl_why = "explicit CF_SHORTLIST='' -- FULL HEAD by request"
        else:
            cands = _cfd_sl.shortlist_candidates(self.ckd)

        rejected: list[str] = []
        for path, src in cands:
            ids, meta = _slmod.load(path)
            # THE GUARD.  A shortlist is a list of integers; against a different vocabulary every
            # id names a different token, so the list is not suboptimal, it is nonsense -- and the
            # only symptom is a quietly lower accept.  The old code filtered `ids < V` and carried
            # on, which is exactly the silent path.  Refuse, name the reason, try the next one.
            bad = _slmod.check(ids, meta, V)
            if bad:
                rejected.append(f"{src} {path}: {bad}")
                continue
            sl = ids[(ids >= 0) & (ids < V)].unique().to(self.dev)
            # IDENTITY CHECK. A checkpoint may declare which shortlist it was trained and measured
            # with; if resolution landed on a different one, that is a silent accept loss (-0.41 at
            # 27B, -0.70 at 9B, invisible without an A/B) and never something to proceed through.
            # Keyed on a CONTENT hash of the sorted ids, not the file bytes: the same list saved by
            # two code paths differs byte-wise while naming identical tokens, and a file-hash check
            # would raise on that.
            decl = self._declared_shortlist()
            if decl:
                got = _slmod.ids_content_sha(sl.detach().cpu())
                want_rows, want_sha = decl.get("rows"), decl.get("ids_sha256")
                if (want_rows and int(want_rows) != sl.numel()) or (want_sha and want_sha != got):
                    raise RuntimeError(
                        "shortlist does not match what this drafter declares.\n"
                        f"  resolved : {path}  [{src}]\n"
                        f"             {sl.numel()} rows, ids-sha256 {got}\n"
                        f"  declared : {want_rows} rows, ids-sha256 {want_sha}\n"
                        "  This drafter was trained and measured against the declared list; a\n"
                        "  different one silently costs acceptance. Fix by letting the drafter's\n"
                        "  own shortlist.pt resolve (it ships in the model repo), or set\n"
                        "  CF_SHORTLIST to the declared list if you know what you are doing.")
            self._sl, self._sl_src = sl, f"{src} {os.path.basename(path)}"
            self._hw = lm_w[sl].contiguous()
            self._w2 = d.markov.w2.weight[sl].contiguous()
            print(f"[chain-flow] shortlist head: {sl.numel()} of {V} rows "
                  f"({V / max(sl.numel(), 1):.2f}x less head traffic) from {path} [{src}]",
                  flush=True)
            # A list that did NOT come with the drafter is a guess, even when it loads cleanly.
            # Published drafters ship their own, so this should be unreachable for them; say so
            # rather than letting an unrelated list pass silently, which was the original bug.
            if src not in ("drafter checkpoint", "CF_SHORTLIST"):
                print(f"[chain-flow] WARNING: this shortlist did NOT come from the drafter "
                      f"({src}). The drafter did not ship one and declares no identity, so this "
                      f"is a guess; acceptance may be silently below what the model can do.",
                      flush=True)
            break
        if rejected:
            # Loud, and loud even when a later candidate DID load: "the packaged list was
            # refused" is a fact about this model that the user needs, and the fallback list is
            # short enough that this can never become noise.
            self._sl_why = ("FULL HEAD -- every candidate refused: " + " | ".join(rejected)) \
                if self._sl is None else self._sl_why
            for r in rejected:
                print(f"[chain-flow] WARNING: shortlist REFUSED -- {r}", flush=True)
            if self._sl is None:
                print("[chain-flow] WARNING: running the FULL "
                      f"{V}-row lm_head at every draft depth (~43-47% of the draft). Build a "
                      "matching one with `chain-flow build-shortlist --tokenizer <target "
                      "model> --ids ...`.", flush=True)
        if self.twopass_m >= self._hw.shape[0]:
            # A two-pass head wider than the head it narrows would gather more rows than exist
            # and buy nothing.  If the value was ASKED FOR, refuse loudly; if it is only the
            # DEFAULT meeting a head it does not fit, gate it off (a default must never be able
            # to stop an otherwise-valid configuration from starting).  `finalize_proposer`
            # records which of the two happened.
            from chain_flow import defaults as _cfd0
            if _cfd0.was_explicit("CF_TWOPASS_M"):
                raise ValueError(f"CF_TWOPASS_M={self.twopass_m} >= head rows {self._hw.shape[0]}")
            self.twopass_m, self.twopass_shared = 0, False
        if self.twopass_m:
            print(f"[chain-flow] TWO-PASS candidate head ON: M={self.twopass_m} of "
                  f"{self._hw.shape[0]} rows, shared={'on' if self.twopass_shared else 'off'}, "
                  f"seed_cond={'on' if self.twopass_seed else 'off'}", flush=True)
        # knob 2: fuse/compile the flow net. Both are pure-speed monkeypatches on the loaded drafter.
        # The compile GATE runs first, so a box without an inductor backend never reaches the wrap
        # (the failure would otherwise surface inside the first draft, i.e. inside a cudagraph
        # capture, which the engine does not survive).
        from chain_flow import defaults as _cfd_c
        self.compile = _cfd_c.finalize_compile(self)
        if self.fuse_path or self.compile:
            from chain_flow.vllm_plugin import fused
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
        print(f"[chain-flow] drafter={os.path.basename(self.ckd)} K={self.K} width={self.width} "
              f"head={'shortlist ' + str(self._sl.numel()) if self._sl is not None else 'full ' + str(V)}"
              f" compile={self.compile_mode if self.compile else 'off'}"
              f" fuse_path={'on' if self.fuse_path else 'off'} cudagraph={'on' if self.use_cg else 'off'}"
              # ENGAGEMENT PROOF for the two trims: print the widths actually used, not the
              # env vars. A flag that silently fails a shape gate is how CF_CUDA_BLOCK became a
              # no-op at two model sizes for hours.
              f" | path_trim={'on' if self.path_trim else 'off'}"
              f" ring_slots={self._wmax()} of {self.K + 1}"
              # The bucket ladder is the one part of this line that used to be a silent
              # constant; the batch it STOPS at is the batch above which the draft ran eager,
              # so it belongs next to `cudagraph=on`.
              f" draft_buckets={self._buckets} (max_num_seqs={self.max_reqs})",
              flush=True)
        # ---- resolve the remaining gates and print ONE summary of what is actually on ----
        # Deliberately the LAST thing _build does and the FIRST forward's precondition: the
        # cuda-extension probe must complete before `_cf_fused_off` latches, and every value in
        # the line below is read off the built objects, never off the environment.
        from chain_flow import defaults as _cfd
        _cfd.finalize_cuda_block(d)
        _cfd.finalize_proposer(self)
        _cfd.finalize_tree(self)
        if not self.use_cg and _cfd.truthy(os.environ.get("CF_CUDA_BLOCK")):
            print("[cf-defaults] NOTE: CF_CUDA_BLOCK is on with cudagraph OFF -- the kernel is "
                  "nondeterministic outside capture (~2.7% of tree nodes flip run to run). "
                  "Bit-exactness checks in this configuration must use CF_CUDA_BLOCK=0.",
                  flush=True)
        _cfd.print_summary()
        from chain_flow.vllm_plugin import batch_cutoff as _cf_cut2
        if _cf_cut2.requested():
            # RESOLVE the threshold HERE, and only here: this is the first moment the target's
            # hidden size is known, and both halves of the cutoff read the resolved number out of
            # the module (scheduler and proposer are the same process, the engine core).
            # This covers BOTH `=auto` and the unset default -- `batch_cutoff.resolve()` is what
            # knows the difference (the default refuses to guess for an unladdered combination
            # and resolves to no cutoff at all), so there is no mode test on this side to drift.
            if _cf_cut2._mode() in ("auto", "default"):
                _h = int(getattr(self, "target_hidden", 0) or 0)
                # `draft_width + 1` is the number of query positions the scheduler hands the
                # target per request per step -- 6 for a chain, 42 for the 8x5 tree -- and it is
                # the second axis of the threshold, not a detail. Read off the built proposer,
                # not off CF_K, so a tree that silently fell back to a chain is keyed correctly.
                _w = int(getattr(self, "draft_width", 0) or 0) + 1
                _n, _why = (_cf_cut2.resolve(_h, _w) if _h
                            else (0, "target hidden size unavailable"))
                _cf_cut2.set_resolved(_n, f"{_why}; hidden_size={_h}, verify width={_w}")
                # The K-SCHEDULE resolves from the same two axes at the same moment.  It is a
                # separate call rather than folded into `set_resolved` so that the cliff's
                # provenance line and the ladder's stay independently auditable -- the ladder can
                # be empty (today's default) while the cliff is not.
                _rungs, _lwhy = (_cf_cut2.auto_ladder(_h, _w) if _h
                                 else ((), "target hidden size unavailable"))
                _cf_cut2.set_resolved_ladder(_rungs, _lwhy)
            print(f"[cf-plugin] K-schedule: {_cf_cut2.describe_ladder()}", flush=True)
            # Both halves, on one line, from STATE.  The proposer half is this object; the
            # scheduler half is a class patch installed by a different module, and "the drafter
            # stopped but the target is still verifying zeros" is precisely the half-configured
            # state that would read as "the cutoff does not work" -- so it says which halves are
            # live rather than which flag was set.
            _n = _cf_cut2.max_batch()
            print(f"[chain-flow] CF_SPEC_MAX_BATCH: drafter "
                  f"{'skips any decode batch > ' + str(_n) if (self._cut_on and _n) else 'HALF OFF'}"
                  f" | scheduler cutoff {_cf_cut2.status()}", flush=True)
        if self.warm_buckets:
            self._warm_buckets()

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
                    print(f"[chain-flow] CF_TREE_DEPTH={depth} exceeds draft_length-1={dl - 1}; "
                          f"clamping to {dl - 1} (the flow cannot emit more levels)", flush=True)
                    return dl - 1
        except Exception:
            pass
        return depth

    def _declared_shortlist(self) -> dict:
        """`shortlist` block from the drafter's own config, if it declares one.

        Optional by design: checkpoints published before this field existed simply do not have it,
        and must keep loading. Its presence turns a silent mismatch into a hard failure.
        """
        try:
            with open(os.path.join(self.ckd, "chained_flow_tree_config.json")) as f:
                return json.load(f).get("shortlist") or {}
        except Exception:
            return {}

    class _DS:
        def __init__(s, h): s.final_hidden = h

    def _head(self, h):
        return h.to(self._hw.dtype) @ self._hw.T

    def _bias(self, prev):
        return self.drafter.markov.w1(prev) @ self._w2.T

    def _map(self, local):
        return local if self._sl is None else self._sl[local]

    def _nlive(self, depth: int):
        """CF_PATH_TRIM: how many PathHead offsets can have an ancestor at this draft depth.

        A node at depth d has exactly d path tokens (the d-1 drafted ones plus the committed seed),
        so offset_proj[d..order-1] see only the -1 fill and are multiplied by a zero mask.  Skipping
        them is BIT-EXACT and removes (order-d) x D x D fp16 of weight traffic per depth: at
        order=8, depth=5 that is 40 -> 15 projections, 500 -> 196 MB.  None = keep all (default)."""
        return depth if self.path_trim else None

    # ===================== AUDIT: the rest of the draft path ==========================
    #
    # CF_PATH_TRIM generalises to exactly one more site (_wmax, below).  Everything else was
    # measured and ruled out; the numbers are here so the question stays closed.
    #
    # WHY PathHead was special: its padded dimension (the path OFFSET) indexes a SEPARATE
    # Linear(D, D) per slot, so a dead slot costs a whole weight matrix.  Every other padded
    # dimension in the draft is a ROW of a batched activation, and at batch 1 with S <= 8 rows
    # the drafter is weight-streaming-bound -- a dead row costs FLOPs but no bytes.  Measured
    # row slopes (4B, D=640, 8 blocks, cudagraphed, CF_CUDA_BLOCK=1):
    #   fused block stack   S=4 200.6 us  ->  S=8 279.4 us   =  19.7 us per row (7% each)
    #   VAE decode          5 rows 251.9 us -> 8 rows 254.9 us  =  1.0 us per row (0.4% each)
    #   VAE encode          5 rows 265.9 us -> 8 rows 269.2 us  =  1.1 us per row
    #
    # Sites checked and CLOSED:
    #
    # 1. beam/tree frontier.  `_beam_tree` already carries the LIVE frontier width `fr`
    #    (1 at depth 1, then tree_keep), not a padded one, so the markov/shortlist rescoring
    #    never scores a dead node.  Nothing to exploit.  (`_beam_chains` does run W-1 dead
    #    -inf beams at its first depth, but its head GEMM is weight-bound, so those rows are
    #    free -- see the row slope above.)
    #
    # 2. flow / VAE over draft_length=8 when only depths 1..tree_depth are emitted.  Draft
    #    positions 6,7 (and the depth-0 output) are never READ by the tree -- but they are not
    #    dead: the VAE decoder is a BIDIRECTIONAL TransformerEncoder over all draft_length
    #    latents, so every flow position feeds the 5 rows that are read.  MEASURED: perturbing
    #    latent rows 6,7 moves decoded rows 1..5 by 0.226; decoding rows 0..5 alone moves them
    #    by 0.322.  So trimming the flow to S=6 is NOT bit-exact, and it is worth only
    #    2 rows x 19.7 us x 2 Euler steps = 79 us of a 3.13 ms draft (0.4% of the step) --
    #    an accept risk for nothing.  NOT IMPLEMENTED.
    #
    # 3. masked self-attention inside csrc/fused_block.cu.  The draft mask is causal, so 28 of
    #    the 64 (row, key) pairs at S=8 are -inf.  But BOTH attentions together are only 57.9 us
    #    of the 279.4 us S=8 kernel (20.7%; measured with CF_CUDA_BLOCK_STAGES=16), the cross
    #    attention half of that is unmasked, and small_attn's phase 1 is lane-parallel over keys
    #    so masked keys already cost nothing.  Ceiling on skipping them: ~13 us per expert-call,
    #    ~26 us per draft (0.8%).  Not worth touching the hot inner loop.  NOT IMPLEMENTED.
    #
    # 4. cudagraph BATCH BUCKET padding (`_buckets`).  A batch of 5 replays the bucket-8 graph,
    #    i.e. 3 whole drafts on padding rows.  This is the one site where the padding is large,
    #    but it is zero at batch 1 (the shipping config) and the draft barely scales with batch
    #    anyway because the flow is weight-bound -- see probe numbers in the report.  Fixing it
    #    means more buckets (more capture time + memory), not less compute.
    #    That last sentence is also why the ladder now REACHES `max_num_seqs` (`_bucket_ladder`):
    #    padding is cheap for exactly the reason above, so a coarse rung at 64 is much better
    #    than no rung -- the alternative was not "an exact shape", it was an eager draft and a
    #    fresh `dynamic=False` compile per batch size.
    #
    # 5. context ring / ctx_size=8 window.  Every one of the 8 context rows is consumed by the
    #    VAE encoder and the flow's cross-attention.  The front-padding repeats the OLDEST real
    #    hidden, so those rows are inputs, not dead fill.  Nothing to trim.

    def _wmax(self) -> int:
        """CF_RING_TRIM: how many ring slots one step's commit can actually fill.

        `_ring_write` runs at a fixed [B, Kmax] shape and redirects masked-off slots to the
        ring's trash row.  Kmax has always been K+1 because a CHAIN can commit that many
        tokens -- but a BRANCHING tree cannot: the accepted path is at most `tree_depth`
        nodes deep, so a request commits at most tree_depth+1 tokens and slots
        tree_depth+2..K+1 are masked on EVERY step.  Dropping them is BIT-EXACT.
        MEASURED: 41 -> 6 slots takes the (launch-bound, 12-kernel) ring write from
        90.5 to 88.4 us, i.e. ~2 us/step.  Kept because it is free and correct, not because
        it is worth anything.

        CF_NONGREEDY_CHAIN opts a step out of branching and emits a CHAIN, which can commit
        up to draft_length tokens -- more than tree_depth+1.  A too-small Kmax would then
        SILENTLY DROP accepted hiddens from the ring (`add = cnt` still advances _rpos, so the
        window would gain stale rows) and quietly cost accept.  Refuse to trim in that mode.
        """
        if (self.ring_trim and self.branching
                and os.environ.get("CF_NONGREEDY_CHAIN", "0") != "1"):
            return self.tree_depth + 1
        return self.K + 1

    # ================= TWO-PASS CANDIDATE HEAD (CF_TWOPASS_M) ==========================
    #
    # The one-pass head re-reads the WHOLE shortlist weight (62642 x 2560 fp16 = 306 MB) at
    # EVERY depth -- 5 depths = 1.5 GB, 47% of the draft's entire memory traffic -- even though
    # each depth only ever consumes the top-8 rows per beam.  Two passes instead:
    #
    #   pass A (once):  score the `nstep` MARGINAL flow hiddens against the full shortlist in ONE
    #                   GEMM (the weight is read once, amortised over all depths) and keep the
    #                   top-M ids per depth.
    #   pass B (per depth): GATHER those M rows and rescore only them with the path residual and
    #                   the markov bias -- M x 2560 fp16 = 2.6 MB at M=512 instead of 306 MB.
    #
    # NUMERICS CHANGE (deliberately): the per-depth softmax is normalised over the M survivors,
    # and a token whose score only becomes competitive AFTER the path residual / markov bias can
    # fall out of the candidate set.  That is safe for a DRAFTER -- tree verify re-checks every
    # proposed token against the target, so output cannot be corrupted; the only risk is accept,
    # which is why M is swept (scripts/sweep_twopass_m.py).
    #
    # MEASURED at 4B, keep8/depth5, 7-domain in-engine sweep, all with CF_PATH_TRIM=1 (accept is
    # exactly reproducible in-engine, so these accept numbers are exact, not sampled):
    #   M       shared   draft ms   mean accept   tok/s
    #   off        -       4.05       2.619       134.4
    #   1024       0       3.05       2.538       136.8  <- coverage CLIFF (summ 2.116 -> 1.627)
    #   2048       0       2.85       2.607       142.6
    #   8192       0       3.33       2.623       140.1
    #   2048       1       2.80       2.597       141.9
    #   8192       1       3.14       2.623       141.9  <- BEST: lossless and fastest
    # The offline window sweep badly understates the risk (it saw only -0.010 at M=1024): a greedy
    # run amplifies one lost candidate over a whole trajectory.  Pick M in-engine, not offline.
    def _candidates(self, pred, lo, hi, seed_res=None, seed_prev=None):
        """[B, hi-lo, M] shortlist-LOCAL ids: the top-M of one full-shortlist GEMM per depth.

        `seed_res` [B, Dh] / `seed_prev` [B] are the SEED path's residual and parent token: adding
        them makes the candidate set come from a distribution close to the one pass B scores,
        instead of the bare marginal (CF_TWOPASS_SEED)."""
        B, Dh = pred.shape[0], pred.shape[-1]
        n = hi - lo
        hm = pred[:, lo:hi]                                          # [B, n, Dh]
        if seed_res is not None:
            hm = hm + seed_res.view(B, 1, Dh)
        M = self.twopass_m
        logits = self._head(hm.reshape(B * n, Dh))                   # [B*n, Vs]  <- one weight read
        if seed_prev is not None:
            logits = (logits.view(B, n, -1) + self._bias(seed_prev).unsqueeze(1)).view(B * n, -1)
        if self.twopass_shared:
            # ONE candidate set for every depth: rank tokens by their best per-depth log-prob
            # (log_softmax first, so depths with different logit scales are comparable). The M
            # rows are then gathered ONCE for the whole draft instead of once per depth, which is
            # the other half of the head's remaining traffic -- at M=8192, depth 5, that is
            # 420 MB of gather -> 84 MB.
            lp = torch.log_softmax(logits.view(B, n, -1).float(), dim=-1).amax(dim=1)  # [B, Vs]
            return lp.topk(M, dim=-1).indices.unsqueeze(1).expand(B, n, M)
        return logits.topk(M, dim=-1).indices.view(B, n, M)

    def _gather_cand(self, cand_d):
        """Materialise pass B's gathered head rows for one candidate set: ([B,M,Dh], [B,M,rank])."""
        B, M = cand_d.shape
        flat = cand_d.reshape(-1)
        return (self._hw.index_select(0, flat).view(B, M, -1),
                self._w2.index_select(0, flat).view(B, M, -1))

    def _rescore(self, x, gathered, prev):
        """Pass B. x [B, F, Dh] conditioned hiddens, `gathered` the (head, markov) rows for this
        depth's candidate set, prev [B, F] parent tokens -> log-probs [B, F, M].

        Normalising over M rather than the whole shortlist is the one numerics change
        (see _candidates)."""
        wc, w2c = gathered
        logits = torch.bmm(x.to(wc.dtype), wc.transpose(1, 2))       # [B, F, M]
        mb = self.drafter.markov.w1(prev)                            # [B, F, rank]
        logits = logits + torch.bmm(mb.to(w2c.dtype), w2c.transpose(1, 2))
        return torch.log_softmax(logits.float(), dim=-1)

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

        cands = res0 = gath = None
        if self.twopass_m and steps > first:
            res0 = d._residual_from_lastp(lastp.reshape(B * W, self.order), self._nlive(1))
            sr, sp = ((res0.view(B, W, D)[:, 0], lastp[:, 0, 0])
                      if self.twopass_seed else (None, None))
            cands = self._candidates(pred, first, steps, seed_res=sr, seed_prev=sp)
        for step in range(first, steps):
            res = res0 if (step == first and res0 is not None) else \
                d._residual_from_lastp(lastp.reshape(B * W, self.order),
                                       self._nlive(step - first + 1))               # [B*W, D]
            h = pred[:, step].unsqueeze(1).expand(B, W, D).reshape(B * W, D)
            if cands is not None:
                cd = cands[:, step - first]                                         # [B, M]
                if gath is None or not self.twopass_shared:
                    gath = self._gather_cand(cd)                                    # shared: once
                lp = self._rescore((h + res).view(B, W, D), gath, lastp[:, :, 0])
                cv, ii = lp.topk(W, dim=-1)                                         # [B, W, W]
                cv = cv.reshape(B * W, W)
                ci = cd.gather(1, ii.reshape(B, W * W))                             # [B, W*W]
            else:
                logits = self._head(h + res) + self._bias(lastp[:, :, 0].reshape(-1))  # [B*W, Vs]
                lp = torch.log_softmax(logits.float(), dim=-1)
                cv, ci = lp.topk(W, dim=-1)                                         # [B*W, W]
                ci = ci.reshape(B, W * W)
            cand = (cum.reshape(B * W, 1) + cv).reshape(B, W * W)
            cum, flat = cand.topk(W, dim=-1)                                        # [B, W]
            src = flat // W                                                         # parent beam
            newtok = self._map(ci.gather(1, flat))                                  # [B, W]
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
        nstep = min(D, pred.shape[1] - 1)
        cands = res0 = gath = None
        if self.twopass_m:
            # pass A: ONE full-shortlist GEMM over the nstep marginal hiddens (see _candidates).
            # res0 is the seed path's residual -- needed for step 1 anyway, so reuse it.
            res0 = d._residual_from_lastp(lastp.reshape(B * fr, self.order), self._nlive(1))
            sr, sp = (res0, known0) if self.twopass_seed else (None, None)
            cands = self._candidates(pred, 1, nstep + 1, seed_res=sr, seed_prev=sp)
        for step in range(1, nstep + 1):
            res = res0 if (step == 1 and res0 is not None) else \
                d._residual_from_lastp(lastp.reshape(B * fr, self.order), self._nlive(step))
            h = pred[:, step].unsqueeze(1).expand(B, fr, Dh).reshape(B * fr, Dh)
            if cands is not None:
                cd = cands[:, step - 1]                      # [B, M]
                if gath is None or not self.twopass_shared:
                    gath = self._gather_cand(cd)             # shared set => gather once
                lp = self._rescore((h + res).view(B, fr, Dh), gath, lastp[:, :, 0])
                cv, ii = lp.topk(TB, dim=-1)                 # [B, fr, TB] (ids are LOCAL to cd)
                cv = cv.reshape(B * fr, TB)
                ci = cd.gather(1, ii.reshape(B, fr * TB))    # [B, fr*TB] shortlist-local
            else:
                logits = self._head(h + res) + self._bias(lastp[:, :, 0].reshape(-1))
                lp = torch.log_softmax(logits.float(), dim=-1)
                cv, ci = lp.topk(TB, dim=-1)                 # [B*fr, TB]
                ci = ci.reshape(B, fr * TB)
            cand = (cum.reshape(B * fr, 1) + cv).reshape(B, fr * TB)
            k = min(W, fr * TB)
            cum, flat = cand.topk(k, dim=-1)                 # [B, k]
            src = flat // TB                                 # parent slot in frontier
            newtok = self._map(ci.gather(1, flat))
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
        print(f"[chain-flow] draft cudagraph captured (ring->flow->beam, bucket {bucket})",
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

    def _audit_step(self, bucket: int, B: int, drafted: bool = True) -> None:
        """CF_BATCH_AUDIT: record the batch this step's draft actually ran at, and say what
        that implies for the batch-1-gated flags.  Printed periodically because a server has
        no exit to report at.

        `drafted=False` is a step `CF_SPEC_MAX_BATCH` skipped.  Its BATCH is still recorded --
        that is the histogram the cutoff has to be judged against -- but it must NOT count
        toward the bucket/kernel tallies, because on that step no draft ran at all: folding it
        in would have the report claim the fused kernel "fell back to the PyTorch block stack"
        on steps where the PyTorch block stack did not run either.
        """
        t = self._audit_t
        t["steps"] += 1
        t["B"][B] = t["B"].get(B, 0) + 1
        if drafted:
            t["drafted"] += 1
            t["bucket"][bucket] = t["bucket"].get(bucket, 0) + 1
            if bucket not in self._buckets:
                t["nocg"] += 1
        if t["steps"] % 500:
            return
        n, d = t["steps"], max(t["drafted"], 1)
        fused = t["bucket"].get(1, 0)
        print(f"[cf-batch-audit] steps={n} ({t['drafted']} drafted, {t['cut']} skipped by "
              f"CF_SPEC_MAX_BATCH) | decode batch B hist "
              f"{dict(sorted(t['B'].items()))} | draft cudagraph bucket hist "
              f"{dict(sorted(t['bucket'].items()))} | fused block kernel (needs bucket==1) ran "
              f"on {fused}/{t['drafted']} = {100.0 * fused / d:.1f}% of DRAFTED steps -- the rest "
              f"fell back to the PyTorch block stack | no-cudagraph drafted steps "
              f"(B above the top bucket {self._buckets[-1]}) "
              f"{t['nocg']} | CF_DRAFT_EARLY prelaunch: {t['pre_hit']} of {n} steps took the "
              f"side-stream fast path (guard bail-outs are steps where a request joined/left "
              f"the batch)", flush=True)

    # ---------- CF_SPEC_MAX_BATCH: don't draft for a batch that cannot pay for it ----------
    def _cut(self, B: int) -> bool:
        """Is this step over the cutoff?  `B` is the number of DRAFTABLE rows this step.

        Delegates so the drafter half and the scheduler half can never drift apart on where the
        boundary is; `self._cut_on` is only the "is it on at all" short-circuit.
        """
        if not self._cut_on:
            return False
        from chain_flow.vllm_plugin import batch_cutoff as _cf_cut
        return _cf_cut.should_cut(B)

    @torch.inference_mode()
    def _no_draft(self, out, nreq: int):
        """The "no draft this step" return value, in whichever shape the caller owes vLLM.

        SYNCHRONOUS path: the list-of-empty-lists vLLM already understands -- it becomes
        `request.spec_token_ids = []`, so the scheduler allocates no spec slots next step and the
        cutoff needs no scheduler patch at all.  (Under ASYNC scheduling it never reads these,
        which is exactly why `vllm_plugin.batch_cutoff` exists.)

        ASYNC path: a `[num_reqs, draft_width]` GPU tensor of ZEROS.  Not `None`, and not a
        narrower tensor: `_prepare_input_ids` scatters `_draft_token_ids.flatten()[...]` into
        `input_ids` whenever the scheduler DID allocate spec slots, and it neither checks the
        width nor tolerates a missing tensor.  A short row would leave the tail of the persistent
        buffer holding the previous step's draft, which is scattered as if it were this step's --
        silent corruption.  Zeros are simply drafts that lose, which rejection sampling handles.
        This mirrors vLLM's own drafter-skip (`not input_fits_in_drafter` -> zeros of the full
        declared width).
        """
        if self._audit:
            self._audit_t["cut"] += 1
        if not self.async_spec:
            return out
        N = self.draft_width
        if self._dtok is None:
            self._dtok = torch.zeros((self.max_reqs, N), dtype=torch.int32, device=self.dev)
        v = self._dtok[:nreq]
        v.zero_()
        if self.branching:
            # No tree was staged, so nothing may be left in the registry claiming to describe
            # this step -- `_async_register` treats `self._areg is None` as "clear it and take
            # the documented fallback", which is what we want here.
            self._areg = None
            try:
                from vllm.v1.spec_decode import tree_state
                tree_state.drop_gpu_tree()
            except Exception:                                # noqa: BLE001 - fork-only module
                pass
        return v

    def _bucket_ladder(self) -> list[int]:
        """The draft cudagraph's batch buckets, up to `max_num_seqs`.

        `CF_DRAFT_BUCKETS="1,2,4,8,16,32,64"` overrides it; unset gives powers of two capped at
        `max_num_seqs`, with `max_num_seqs` itself appended when it is not one (so the top of the
        range a server can actually reach is always covered, never merely approached).

        WHY THE TOP RUNGS ARE FREE, AND WHY THE PADDING THEY COST IS NOT THE ISSUE.  A batch of
        33 replays the bucket-64 graph, i.e. 31 whole drafts on padding rows -- which sounds
        expensive and is not, because the flow net is WEIGHT-BANDWIDTH-BOUND and every row of a
        batch shares those weights.  Measured eager, no engine, 4B: 10.13 ms at B=2 against
        10.22 ms at B=32 -- FLAT.  So a padded replay costs about what an exact-shape one costs,
        while an unbucketed batch costs a fresh `torch.compile` plus the launch overhead of an
        eager draft, every step.  The trade the old ceiling made was backwards.

        The cost of a rung is compile + capture (~70-80 s each, once) and one graph's worth of
        pool memory.  That is why the ladder is coarse above 32 rather than one rung per batch
        size: two extra rungs replace ~25 compiles.
        """
        env = (os.environ.get("CF_DRAFT_BUCKETS", "") or "").replace(",", " ").split()
        cap = max(int(self.max_reqs), 1)
        if env:
            b = sorted({int(x) for x in env if int(x) > 0})
            return [x for x in b if x <= cap] or [1]
        b, n = [], 1
        while n < cap:
            b.append(n)
            n *= 2
        b.append(cap)                       # ... and always the top, power of two or not
        return b

    def _warm_buckets(self) -> None:
        """CF_WARM_BUCKETS: pay the per-bucket compile+capture at BUILD time, not in the server.

        `CF_COMPILE` wraps the flow net with `torch.compile(..., dynamic=False)`, so inductor
        re-codegens AND re-autotunes for every new draft batch shape.  Under `vllm serve` the
        shapes are discovered one at a time as concurrency changes, and each discovery is a
        70-80 s engine stall in the middle of live traffic (measured; a first guidellm pass at
        rate 64 logged 850 autotune events as concurrency decayed through the buckets).  The work
        is unavoidable -- it is what makes the draft fast -- but WHEN it is paid is a choice, and
        paying it during startup is strictly better than paying it during a request.

        Only buckets the cutoff can actually reach are warmed: with `CF_SPEC_MAX_BATCH=8` the
        drafter never sees a batch above 8, so buckets 16 and 32 would be pure startup cost for
        shapes that can never occur.

        Uses the REAL capture path (`_capture_ring`) against the freshly zeroed ring, so the
        graphs it leaves behind are the ones serving will replay -- `_h2d` allocates each staging
        buffer once at full capacity, so the addresses the capture bakes in are the addresses the
        first real step writes to.  The drafts produced here are meaningless and discarded.
        """
        if not (self.gpuctx and self.use_cg) or self.feedback:
            return
        import numpy as np
        buckets = [b for b in self._buckets if not self._cut(b)]
        t0 = time.perf_counter()
        for b in buckets:
            rows_g = self._h2d("rows", np.zeros(b, dtype=np.int64), pad_to=b)
            k0_g = self._h2d("k0", np.zeros(b, dtype=np.int64), pad_to=b)
            self._capture_ring(b, rows_g, k0_g)
        # The ring was only ever read here, but `_rpos`/`_nval` are left exactly as `_init_ring`
        # made them, so the first real step seeds its slots normally.
        print(f"[chain-flow] draft buckets warmed at startup: {buckets} in "
              f"{time.perf_counter() - t0:.1f}s (CF_WARM_BUCKETS=1) -- these compiles and "
              f"captures will NOT happen mid-request", flush=True)

    def _draft_from_ring(self, rows_g, k0_g):
        return self._draft_fn(self._ring_gather(rows_g), k0_g)

    def _ring_cg(self, rows_g, k0_g, B):
        bucket = rows_g.shape[0]
        if self._audit:
            self._audit_step(bucket, B)
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
        print(f"[chain-flow] draft cudagraph captured (batch bucket {bucket})", flush=True)

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
            self._h["_seen"] += 1
            if self._h["_exit"] is not None and self._h["_seen"] > self._hskip:
                _g = _now - self._h["_exit"]
                self._h["gap"] += _g
                self._hgaps.append(_g)
            self._h["_mark"] = _now
        if self.drafter is None:
            self._build()
        runner = _STASH["runner"]
        sh = _STASH.get("sample_hidden_states")
        if self.async_spec:
            # ASYNC SCHEDULING: `sampled_token_ids` is `[]` -- vLLM writes -1 placeholders
            # into `token_ids_cpu` / `output_token_ids` and defers the real D2H to
            # `AsyncGPUModelRunnerOutput.get_output()`.  Reconstruct the ROW SET from
            # host-resident metadata (every scheduled request emits at least the bonus
            # token unless it is discarded); the row CONTENTS are never needed, they are
            # in `_STASH["gpu_counts"]`.  `_ASYNC_ROW` is a shape-only sentinel: any code
            # that reads a token value out of it is a bug, and it is -1 so that it is.
            _nreq = int(runner.input_batch.num_reqs)
            _dm = runner.discard_request_mask.np[:_nreq]
            sampled_token_ids = [[] if _dm[i] else _ASYNC_ROW for i in range(_nreq)]
        out: list[list[int]] = [[] for _ in sampled_token_ids]
        if sh is None:
            self._hbail()
            return out
        req_ids = list(runner.input_batch.req_ids)
        if os.environ.get("CF_ACCEPT", "0") == "1" and not self.async_spec:
            _n = sum(len(t) for t in sampled_token_ids if t)
            _r = sum(1 for t in sampled_token_ids if t)
            self._t["acc_tok"] = self._t.get("acc_tok", 0) + _n
            self._t["acc_req"] = self._t.get("acc_req", 0) + _r
            self._t["acc_n"] = self._t.get("acc_n", 0) + 1
            if self._t["acc_n"] % 25 == 0 and self._t["acc_req"]:
                print(f"[cf-accept] steps={self._t['acc_n']} "
                      f"mean tokens emitted per request-step = {self._t['acc_tok']/self._t['acc_req']:.3f} "
                      f"(K={self.K}, so max possible {self.K+1})", flush=True)
        if os.environ.get("CF_DIAG", "0") == "1" and sh is not None and not self.async_spec:
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
        if os.environ.get("CF_ORACLE", "0") == "1" and not self.async_spec:
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
            self._hbail()
            return out

        _pre = self._pre
        if _pre is not None:
            # The draft for THIS step was issued on the side stream back in `_sample`, from the
            # verify's own GPU tensors. All that is left is to join and read it back.
            self._pre = None
            _prows, _pch, _pev = _pre
            # The prelaunch guard implies every request emits >= 1 token, so the row set it
            # assumed must be exactly what the CPU lists say. If it ever is not, the ring has
            # already been appended for the wrong rows and there is no silent recovery: fail.
            assert rows == _prows, (
                f"CF_DRAFT_EARLY row-set mismatch: prelaunched {_prows}, actual {rows}. "
                f"The steady-state guard in _prelaunch is wrong; disable CF_DRAFT_EARLY.")
            self._hm("pre")
            if self.async_spec:
                # Nothing to read back: vLLM scatters the draft TENSOR into the next step's
                # `input_ids` on the GPU, and `_prelaunch` already staged the tree shape's
                # D2H.  The default stream still has to ORDER AFTER the side stream (that
                # is a GPU-side wait, not a host sync).
                torch.cuda.current_stream().wait_event(_pev)
                self._hm("draft")
                return self._async_draft_tensor(_pch, rows, len(sampled_token_ids))
            if self.defer:
                # Hand vLLM the (still empty) `out` lists it will store as
                # `_draft_token_ids`; `_materialize` fills those SAME list objects in place
                # once the engine core actually asks for the values.
                self._defer = (out, rows, req_ids, _pch, _pev, sampled_token_ids,
                               _tp if self.prof else None)
                return out
            torch.cuda.current_stream().wait_event(_pev)
            self._hm("draft")
            return self._finish(out, rows, req_ids, _pch, sampled_token_ids,
                                _tp if self.prof else None)

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
            # CF_SPEC_MAX_BATCH: over the cutoff, skip the flow net. Deliberately AFTER
            # `_ctx_gpu`: the ring append is what keeps every slot's context history current, and
            # it is cheap (one fixed-shape index_copy_). Skipping it too would leave the history
            # a hole `ctx_size` steps wide, so the first drafts after the batch drops back below
            # the cutoff would be drawn from a stale window -- correct (they are verified) but
            # worthless, for as long as it takes the ring to refill.
            if self._cut(len(rows)):
                if self._audit:
                    self._audit_step(int(_rows_g.shape[0]), _B, drafted=False)
                self._hm("draft")
                return self._no_draft(out, len(sampled_token_ids))
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
                        "chain-flow tree mode requires greedy sampling (temperature=0, no "
                        "logprobs, no penalties). Got non-greedy sampling params. Run with "
                        "VLLM_SPEC_TREE=0 (chain mode, 1.61x vs the tree's 1.68x at 27B) for "
                        "sampled decoding, or CF_NONGREEDY_CHAIN=1 to try the experimental "
                        "per-step chain fallback (currently shape-incompatible). "
                        "UNDER `vllm serve` THIS RAISE KILLS THE ENGINE CORE for every client, "
                        "permanently -- it was designed for the offline LLM() path, where it "
                        "reaches the caller. Set CF_TREE_GREEDY_GUARD=1 to have the API server "
                        "reject such requests with a 4xx before they ever get here.")
            else:
                # _ring_cg drives self._draft_fn, which is _beam_chains in chain mode and
                # _beam_tree in tree mode -- correct for BOTH. Only the non-greedy-with-branching
                # case above diverts away from it.
                _ch = self._ring_cg(_rows_g, _k0, _B)
            self._hm("draft")
            if self.prof: _ts = time.perf_counter()
            if self.async_spec:
                # SLOW path (prefill, fresh/migrated slot, graph not yet captured): the
                # draft ran on the DEFAULT stream, so no ordering fixup is needed -- just
                # stage the same D2H `_prelaunch` would have staged.
                self._stage_registration(_ch, rows, req_ids)
                return self._async_draft_tensor(_ch, rows, len(sampled_token_ids))
            return self._finish(out, rows, req_ids, _ch, sampled_token_ids,
                                _tp if self.prof else None)
        if self._cut(len(rows)):
            # Legacy (CF_GPUCTX=0) path.  This bails BEFORE the context build, so unlike the ring
            # path above it does leave `ctx_hist` a hole `ctx_size` steps wide and the first
            # drafts after the batch drops back under the cutoff are drawn from a stale window --
            # still correct (every draft is verified), just worthless until it refills.  Accepted
            # because the legacy build is an O(batch) Python loop of `torch.cat`s, i.e. the one
            # thing on this path that is genuinely expensive at the batches this branch runs at,
            # and CF_GPUCTX=0 is a diagnostic fallback, not a shipping configuration.
            self._hm("draft")
            return self._no_draft(out, len(sampled_token_ids))
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
        # CF_ASYNC_SPEC: `sampled_token_ids` carries only the SHAPE (see `propose`); the
        # values live on the GPU, published by `_publish_counts_from_verify`.
        _gc = _STASH.get("gpu_counts") if self.async_spec else None
        if _gc is None:
            cnt_np = np.asarray([len(sampled_token_ids[i]) for i in rows], dtype=np.int64)
            k0_np = np.asarray([sampled_token_ids[i][-1] for i in rows], dtype=np.int64)
        else:
            cnt_np = k0_np = None

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
        if _gc is None:
            cnt_g = self._h2d("cnt", cnt_np)
            k0_g = self._h2d("k0", k0_np, pad_to=bucket)
        else:
            _k0_all, _cnt_all, _ = _gc
            cnt_g = _cnt_all.index_select(0, rows_g[:B]).long()
            # k0 must land in the buffer whose ADDRESS the draft graph baked at capture
            # time, so allocate/zero it through _h2d and then overwrite from the device.
            k0_g = self._h2d("k0", np.zeros(B, dtype=np.int64), pad_to=bucket)
            k0_g[:B].copy_(_k0_all.index_select(0, rows_g[:B]).long())
        wmask_g = self._h2d("wmask", wmask_np).bool()
        path_g = self._accepted_path_gpu()
        # The ring write touches ONLY the real rows; the padded tail exists purely so the
        # graph sees a fixed shape.
        self._ring_write(sh, rows_g[:B], starts_g, cnt_g, wmask_g, path_g, self._wmax())
        return rows_g, k0_g, B

    # ---------- CF_DRAFT_EARLY: side-stream draft, issued right after the verify ----------
    @torch.inference_mode()
    def _prelaunch(self, runner, sampler_output, smd) -> None:
        """Write the ring and replay the draft graph on a side stream, from GPU-ONLY inputs.

        CORRECTNESS, in order:
          * inputs.  The normal path takes `cnt` and `k0` off the CPU token lists, which do not
            exist until `_bookkeeping_sync` has synced.  Both are already on the GPU as the tree
            verify's own outputs: `accepted_len[i]` is the number of accepted DRAFT NODES, so the
            request emits `accepted_len+1` tokens (path + bonus) and the last committed token is
            `output_token_ids[i, accepted_len[i]]`.  That is *the same tensor* RejectionSampler.
            parse_output later turns into the python lists, so the values are identical by
            construction, not by luck.  `propose()` asserts the row set agrees.
          * ordering.  The side stream waits on the default stream first, so it cannot read
            `sample_hidden_states` / `accepted_path` before the forward and the descent kernel
            have produced them.  Nothing on the default stream writes those buffers again inside
            this step, and `propose()` joins before the step ends, so the next step's forward
            cannot overwrite them while the draft is in flight.  The ring / `_rpos` / `_nval` are
            written by nobody but this function.
          * exactly-once.  The ring append MUST NOT happen twice.  `propose()` consumes
            `self._pre` and skips its own ctx build whenever it is set.
          * scope.  Steady-state decode only: every request spec'd, none discarded, every slot
            holding the SAME request as last step (so no seeding, no slot migration, no recycle),
            greedy, and the graph for this bucket already captured.  Anything else falls through
            to the normal path, which handles those cases.  This is a guard, not a check: it is
            all host-resident metadata, so it costs no sync.
        """
        import numpy as np
        if self.async_spec:
            # CF_ASYNC_SPEC: publish (cnt, k0) for EVERY step, before any of the fast-path
            # guards below can bail out.  Under async scheduling the CPU token lists are
            # `[]`, so these GPU tensors are the ONLY source for
            #   * `_ctx_gpu`'s cnt/k0 on the slow path (prefill, fresh/migrated slot, ...),
            #   * vLLM's `valid_sampled_token_count` (see `_publish_gpu_counts`),
            #   * the accept counter.
            # The rejection sampler writes the emitted tokens contiguously from column 0
            # and pads with -1 (this is the SAME invariant stock vLLM relies on in
            # `_update_states_after_model_execute`), so the count of non-(-1) entries is
            # the number of tokens this request emitted -- accepted drafts plus the bonus
            # -- for the tree exactly as for a chain.
            self._publish_counts_from_verify(runner, sampler_output)
        if self._defer is not None:
            # Belt and braces: normally `take_draft_token_ids` has already drained this.
            self._materialize()
        if self._pre is not None:
            # A prelaunch that propose() never consumed (e.g. vLLM zeroed the drafts because the
            # request no longer fits the drafter). Its side-stream writes to _rpos/_nval were
            # never joined, so join them now before anything reads those buffers again.
            torch.cuda.current_stream().wait_event(self._pre[2])
            self._pre = None
        if not (self.gpuctx and self.branching and self.use_cg) or self.feedback:
            return
        try:
            if smd is None or getattr(smd, "num_draft_tokens", None) is None:
                return
            nd = list(smd.num_draft_tokens)
            nreq = runner.input_batch.num_reqs
            if nreq <= 0 or len(nd) != nreq or min(nd) <= 0:
                return
            if self._cut(nreq):
                # CF_SPEC_MAX_BATCH: over the cutoff. Falling through to the normal path (rather
                # than skipping here and letting the side-stream draft run) is the point --
                # `propose()` still runs `_ctx_gpu`, so the ring stays current, and it takes the
                # same cutoff branch a step later without the draft ever being issued.
                return
            if bool(runner.discard_request_mask.np[:nreq].any()):
                return
            req_ids = list(runner.input_batch.req_ids)
            if len(req_ids) != nreq:
                return
            if any(self._slot_req[i] != req_ids[i] for i in range(nreq)):
                return                                  # fresh / migrated / recycled slot
            if not self._tree_ok(runner.input_batch.sampling_metadata):
                return
            bucket = next((b for b in self._buckets if b >= nreq), None)
            if bucket is None or bucket not in self._cg:
                return                                  # graph not captured yet
            st = _STASH.get("early_state")
            sh = None if st is None else st.sample_hidden_states
            if sh is None:
                return
            from vllm.v1.spec_decode import tree_state
            ts = tree_state.get_current()
            if ts is None or ts.accepted_path is None or ts.accepted_len is None:
                return
            tok = sampler_output.sampled_token_ids
            if tok is None or tok.shape[0] < nreq:
                return
        except Exception:
            self._pre_skip += 1
            return

        if self._dstream is None:
            self._dstream = torch.cuda.Stream()
            self._dev_ev = torch.cuda.Event()
        s = self._dstream
        s.wait_stream(torch.cuda.current_stream())
        # These two live on the default stream; tell the caching allocator not to hand their
        # memory to a later default-stream allocation until the side stream is past them.
        sh.record_stream(s)
        ts.accepted_path.record_stream(s)
        with torch.cuda.stream(s):
            ndarr = np.asarray(nd, dtype=np.int64)
            starts_all = np.cumsum(ndarr + 1) - (ndarr + 1)
            rows_g = self._h2d("rows", np.arange(nreq, dtype=np.int64), pad_to=bucket)
            starts_g = self._h2d("starts", starts_all)
            alen = ts.accepted_len[:nreq].long()
            cnt_g = alen + 1
            # k0 must land in the SAME persistent buffer whose address the draft graph baked in
            # at capture time (`_h2d("k0", ..., pad_to=bucket)`), so write through that view.
            k0_g = self._pin["k0"][1][:bucket]
            k0_g.zero_()
            k0_g[:nreq].copy_(tok[:nreq].long().gather(1, alen.unsqueeze(1)).squeeze(1))
            if self._ones_b is None or self._ones_b.shape[0] < nreq:
                self._ones_b = torch.ones(max(nreq, self.max_reqs), dtype=torch.bool,
                                          device=self.dev)
            self._ring_write(sh, rows_g[:nreq], starts_g, cnt_g, self._ones_b[:nreq],
                             ts.accepted_path, self._wmax())
            if self._audit:
                # The prelaunch fast path replays the graph HERE and `propose()` then consumes
                # `self._pre` without ever reaching `_ring_cg`, so the audit has to count this
                # step itself or the histogram silently omits every steady-state tree step --
                # i.e. exactly the steps the report is about.
                self._audit_step(bucket, nreq)
                self._audit_t["pre_hit"] += 1
            self._cg[bucket][0].replay()
            out = self._cg[bucket][3][:nreq]
            if self.async_spec:
                # Stage the tree SHAPE for the host on this same side stream and record a
                # second event.  Nothing joins it until the top of the next step's
                # `_prepare_inputs`, so the whole post-step tail + `scheduler.schedule()`
                # runs while these kernels are still in flight.
                self._stage_registration(out, list(range(nreq)), req_ids, s)
            self._dev_ev.record(s)
        self._branch_now = True
        self._pre = (list(range(nreq)), out, self._dev_ev)
        self._pre_n += 1
        if self._pre_n == 1 or self._pre_n % 400 == 0:
            print(f"[cf-early] draft issued on side stream after verify (n={self._pre_n})",
                  flush=True)

    # ---------- CF_ASYNC_SPEC: GPU-token-native plumbing ----------
    @torch.inference_mode()
    def _publish_counts_from_verify(self, runner, sampler_output) -> None:
        """(cnt, k0) for every request, from the verify's own GPU output. No host read.

        cnt[i] = number of tokens request i just emitted (accepted drafts + bonus);
        k0[i]  = the LAST of those, i.e. the token the next draft must condition on.
        Discarded requests emit nothing, so their cnt is forced to 0 -- matching the
        `valid_sampled_token_ids[i].clear()` the synchronous path does on the host.
        """
        tok = getattr(sampler_output, "sampled_token_ids", None)
        if tok is None or tok.dim() != 2:
            _STASH.pop("gpu_counts", None)
            return
        nreq = int(runner.input_batch.num_reqs)
        if nreq <= 0 or tok.shape[0] < nreq:
            _STASH.pop("gpu_counts", None)
            return
        t = tok[:nreq]
        cnt = (t != -1).sum(dim=1)
        dm = runner.discard_request_mask.gpu[:nreq]
        cnt = torch.where(dm.bool(), torch.zeros_like(cnt), cnt)
        k0 = t.gather(1, (cnt - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
        _STASH["gpu_counts"] = (k0, cnt, nreq)
        if self._acc_gpu is None:
            # `self.dev` is only set by the lazy `_build()`, and this hook fires from
            # `_sample` which precedes the first `propose()`.
            self._acc_gpu = torch.zeros(2, dtype=torch.long, device=cnt.device)
        # CF_ACCEPT without a host read: accumulate on the GPU, drain once per prompt set.
        self._acc_gpu[0] += cnt.sum()
        self._acc_gpu[1] += (cnt > 0).sum()

    def _publish_gpu_counts(self, runner) -> None:
        """Hand vLLM the two tensors its async spec-decode path needs.

        `valid_sampled_token_count_gpu` drives `update_num_computed_tokens_for_batch_change`
        (gpu_model_runner.py:2340), which is the ONLY thing that walks back the scheduler's
        optimistic "every draft was accepted" `num_computed_tokens`.  Without it the KV /
        position bookkeeping silently runs ahead of reality.
        `prev_sampled_token_ids` is the source `_prepare_input_ids` scatters into the next
        step's `input_ids` for the committed token of each request.
        """
        g = _STASH.get("gpu_counts")
        if g is None or runner.valid_sampled_token_count_event is None:
            return
        k0, cnt, _n = g
        runner._copy_valid_sampled_token_count(
            k0.to(runner.input_ids.gpu.dtype), cnt.to(torch.int32))

    @torch.inference_mode()
    def _async_draft_tensor(self, ch, rows, nreq):
        """The draft tokens as a GPU tensor [num_reqs, N], indexed by INPUT-BATCH ROW.

        vLLM's async path never brings drafts to the host: `_prepare_input_ids` reads this
        tensor with `flatten()[prev_index * prev_num_spec_tokens + j]`, so a row per batch
        slot (not per drafted request) is required and a list-of-lists is ignored outright.
        Copied into a persistent buffer because `ch` is the draft cudagraph's OUTPUT buffer,
        which the next replay overwrites.
        """
        import numpy as np
        # `draft_width`, not `self.N`: a chain's draft is [B, K] and K != keep*depth.
        N = self.draft_width
        # A draft narrower than the width we declared would leave the tail of `_dtok` holding
        # the PREVIOUS step's tokens, and vLLM scatters the whole row into `input_ids` without
        # ever looking at it. That is a stale-draft corruption with no error, which is the exact
        # failure shape of the CF_TREE_DEPTH / CF_TREE_KEEP bugs -- so check it here, where the
        # message can say what happened.
        assert int(ch.shape[1]) >= N, (
            f"draft is {tuple(ch.shape)} but {N} columns were declared to vLLM "
            f"({'tree keep*depth' if self.branching else 'chain K'}). Widening _dtok from a "
            f"short draft would scatter the previous step's tokens into input_ids silently.")
        if self._dtok is None:
            self._dtok = torch.zeros((self.max_reqs, N), dtype=torch.int32, device=ch.device)
        v = self._dtok[:nreq]
        if len(rows) == nreq:
            v.copy_(ch[:nreq, :N])
        else:
            v.zero_()
            v.index_copy_(0, self._h2d("dtokrows", np.asarray(rows, dtype=np.int64)),
                          ch[: len(rows), :N].to(v.dtype))
        return v

    @torch.inference_mode()
    def _stage_registration(self, ch, rows, req_ids, stream=None) -> None:
        """Start the (tokens ‖ parents) D2H that the next `_prepare_inputs` needs.

        This is the one host dependency the draft path cannot shed: `_prepare_inputs` builds
        the ancestor mask and the GDN `chain` from the tree parents on the CPU.  Making it a
        NON-BLOCKING pinned copy with its own event moves the cost from "serialize the whole
        post-step tail" to "join once, as late as possible".

        NOTHING TO DO IN CHAIN MODE, and that is the whole point of the fork-free build: a
        chain's parents are implicit (node j's parent is j-1), so vLLM's own `_prepare_inputs`
        lays it out with no help from us and `tree_state` -- a fork-only module -- is never
        imported.  `_install_async_hooks(tree=False)` correspondingly installs no join.
        """
        if not self.branching:
            return
        B, W = int(ch.shape[0]), int(ch.shape[1])
        N = self.N
        if self._areg_pin is None or self._areg_pin.shape != (self.max_reqs, W):
            self._areg_pin = torch.empty((self.max_reqs, W), dtype=ch.dtype,
                                         pin_memory=True)
            self._areg_ev = torch.cuda.Event()
        self._areg_pin[:B].copy_(ch, non_blocking=True)
        self._areg_ev.record(stream or torch.cuda.current_stream())
        self._areg = (B, list(rows), list(req_ids))
        # The DEVICE hand-off, which is what lets the canonical step skip the join
        # entirely.  Only offered when the draft covers every input-batch row in order --
        # `_prepare_inputs` indexes the parents by input-batch row, and it re-checks the
        # req_ids anyway, but offering a partial row set could only ever be rejected.
        if list(rows) == list(range(len(req_ids))) and W == 2 * N:
            from vllm.v1.spec_decode import tree_state
            if self._ptree is None or self._ptree.shape != (self.max_reqs, N):
                self._ptree = torch.empty((self.max_reqs, N), dtype=ch.dtype,
                                          device=ch.device)
            # Copied out of the draft cudagraph's OUTPUT buffer: that buffer is
            # overwritten by the next replay, and `_prepare_inputs` reads the parents one
            # step later.
            self._ptree[:B].copy_(ch[:, N:])
            tree_state.publish_gpu_tree(parents=self._ptree[:B], req_ids=req_ids,
                                        n=N, discard=self._async_discard)
            self._areg_n += 1
            if self._aprobe is not None and self._areg_n % 200 == 0:
                print(f"[cf-async-probe] device tree offered {self._areg_n} | taken "
                      f"{self._skip_n} (= joins skipped) | host joins "
                      f"{self._aprobe[2]}", flush=True)
        else:
            from vllm.v1.spec_decode import tree_state
            tree_state.drop_gpu_tree()

    def _async_discard(self) -> None:
        """`_prepare_inputs` took the GPU-resident tree, so the join was skipped.

        Deliberately does NOT drop `self._areg`.  The host staging stays pending and is
        overwritten by the next step's D2H (same side stream, so the two copies cannot
        overlap).  That is what keeps the FALLBACK correct: when a later step is not
        canonical it joins the staging of the step IMMEDIATELY BEFORE it, which is exactly
        the tree those draft tokens came from.  Discarding here would leave the registry
        empty and silently degrade the next non-canonical step to chain semantics.
        """
        self._skip_n += 1

    @torch.inference_mode()
    def _async_register(self) -> None:
        """Join the staged D2H and publish the tree shape for THIS step's `_prepare_inputs`.

        Called from the top of `_prepare_inputs`, i.e. after `scheduler.schedule()`.  The
        GPU has been busy with step N's forward/verify/draft for the whole interval, so the
        wait here is what is left of the draft, not of the host.
        """
        from vllm.v1.spec_decode import tree_state
        a = self._areg
        if a is None:
            # Nothing has been staged since the last join, so whatever is in the registry
            # describes an OLDER step.  Under PLACEHOLDER_TOKENS the lookup is a length
            # check and cannot tell the difference -- exactly the silent-malformed-draft
            # shape of the CF_TREE_DEPTH / CF_TREE_KEEP bugs.  Clear it and let the caller
            # take the documented stale-tree fallback instead.
            tree_state.clear_registry()
            return
        self._areg = None
        B, rows, req_ids = a
        if self._aprobe is not None:
            _t0 = time.perf_counter()
            if not self._nojoin:
                self._areg_ev.synchronize()
            _t1 = time.perf_counter()
            self._aprobe[0] += _t1 - _t0
        elif not self._nojoin:
            self._areg_ev.synchronize()
        N = self.N
        depths = self._areg_depths
        chains = self._areg_pin[:B].tolist()
        for j, i in enumerate(rows):
            row = chains[j]
            rid = req_ids[i] if i < len(req_ids) else str(i)
            tree_state.register(rid, row[:N], row[N:], depths)
        tree_state.drop(req_ids)
        if self._aprobe is not None:
            self._aprobe[1] += time.perf_counter() - _t1
            self._aprobe[2] += 1
            if self._aprobe[2] % 200 == 0:
                n = self._aprobe[2]
                print(f"[cf-async-probe] n={n} | join wait {self._aprobe[0]/n*1000:6.3f} ms/step "
                      f"| host register {self._aprobe[1]/n*1000:6.3f} ms/step"
                      + ("  [NOJOIN: timing probe only, drafts are STALE]"
                         if self._nojoin else ""), flush=True)

    @torch.inference_mode()
    def _materialize(self) -> None:
        """CF_DRAFT_DEFER: join the side stream and fill the draft lists vLLM already holds."""
        d = self._defer
        if d is None:
            return
        self._defer = None
        out, rows, req_ids, _pch, _pev, sti, _tp = d
        torch.cuda.current_stream().wait_event(_pev)
        self._hm("draft")
        self._finish(out, rows, req_ids, _pch, sti, _tp)   # mutates `out` in place

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
            if self._h["_seen"] > self._hskip:
                self._h["n"] += 1
            n = self._h["n"]
            if n and n % self._hevery == 0:
                ins = sum(self._h[s] for s in self._hsegs if s != "gap")
                g = sorted(self._hgaps)
                med = g[len(g) // 2] * 1000 if g else float("nan")
                print("[cf-hprof] n=%d | HOST ms/step: " % n
                      + " ".join(f"{s} {self._h[s]/n*1000:5.2f}" for s in self._hsegs)
                      + f" | gap med {med:5.2f} max {max(g) * 1000 if g else float('nan'):6.2f}"
                      + f" | in-propose {ins/n*1000:5.2f} | STEP {(ins+self._h['gap'])/n*1000:6.2f}",
                      flush=True)
        return out
