"""Faithful tree speculative decoding on vLLM's real Qwen3.5 modules.

Drives vLLM's actually-loaded decoder-layer modules (``in_proj_qkvz``,
``_project_qkv_gate``, ``RMSNormGated``, ``mlp`` ...) so the forward is
bit-faithful to vLLM's own greedy decode (validated 120/120 token match on a
single-token step). Tree attention uses a tree-ancestor mask; the linear
attention (Gated-DeltaNet) state is parent-routed with the fused tree-scan
kernels in ``tree_ssm.py`` (bit-exact to vLLM's ``packed_decode``). Acceptance is
greedy longest-path (deterministic), so no rejection sampler or custom attention
backend is required — the verify forward + accept run as torch orchestration over
vLLM's real kernels.

Runs inside the model worker, e.g.::

    def _run(model):
        dec = NativeTreeSpecDecoder(model, drafter)
        return dec.generate(prompt_ids, max_new=256)
    llm.apply_model(_run)

The verify has an eager path (``tree_verify``) and a CUDA-graphed path
(``tree_verify_graphed``, static tree topology + padded/masked KV) for wall-clock
speedup; the graphed path uses SDPA for the tree attention core (graph-friendly;
matches FlashInfer to ~2e-4, negligible next to the batched-vs-single bf16 that
every speculative verify already carries).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from chained_flow.vllm_tree.tree_ssm import (
    tree_conv_op,
    tree_conv_op_v2,
    tree_gdn_op,
    tree_gdn_op_v2,
    tree_scan_causal_conv1d,
    tree_scan_causal_conv1d_v2,
    tree_scan_gdn_recurrence,
    tree_scan_gdn_recurrence_v2,
)


def _ancestor_mask(par, n, device):
    """Boolean [n,n] where anc[i,j]=True iff j is on i's root->i path (incl. i). Vectorized pointer-jump
    closure — replaces the per-element Python `anc[i,j]=True` loop that stalled the verify (~1.5ms bubble
    of GPU-idle host/sync work per call). `par` is a list of parent node-indices (-1 for roots)."""
    pt = torch.tensor(par[:n], device=device)
    self_idx = torch.arange(n, device=device)
    jump = torch.where(pt < 0, self_idx, pt)             # roots point to self
    anc = torch.eye(n, dtype=torch.bool, device=device)
    for _ in range(8):                                   # pointer DOUBLING: reaches depth 2^8 (>> any tree)
        anc = anc | anc[jump]
        jump = jump[jump]
    return anc


def _depth_starts_from_depths(dep, n):
    """Static per-depth node-index boundaries for the depth-parallel tree scan.
    ``dep`` must be BFS-ordered (it is, straight out of build_tree_fast)."""
    maxd = max(dep[:n]) if n else 0
    ds = [0]
    for d in range(maxd + 1):
        ds.append(ds[-1] + sum(1 for x in dep[:n] if x == d))
    return ds


class NativeTreeSpecDecoder:
    def __init__(self, model, drafter, *, max_nodes=64, max_prefix=1024,
                 top_b=8, tree_max_nodes=8, max_depth=5, eos_id=None):
        self.model = model
        self.drafter = drafter
        self.dev = "cuda"
        lang = model.language_model
        self.inner = lang.model
        self.layers = self.inner.layers
        self.embed = self.inner.embed_tokens.weight
        self.dtype = self.embed.dtype
        self.nl = len(self.layers)
        self.lin = [L for L in range(self.nl) if hasattr(self.layers[L], "linear_attn")]
        self.att = [L for L in range(self.nl) if not hasattr(self.layers[L], "linear_attn")]
        la0 = self.layers[self.lin[0]].linear_attn
        self.HV, self.Vd, self.Kd = la0.num_v_heads, la0.head_v_dim, la0.head_k_dim
        self.keyd, self.vald = la0.key_dim, la0.value_dim
        self.convdim = self.keyd * 2 + self.vald
        self.NGC = max_nodes
        self.SG = max_prefix
        self.top_b, self.tree_max_nodes, self.max_depth = top_b, tree_max_nodes, max_depth
        self.eos_id = eos_id
        # persistent tree GDN state buffers (slot 0 sentinel, 1 prefix, 2.. nodes)
        self._gssm = {L: torch.zeros(max_nodes + 2, self.HV, self.Vd, self.Kd, device=self.dev, dtype=self.dtype) for L in self.lin}
        self._gconv = {L: torch.zeros(max_nodes + 2, self.convdim, 4, device=self.dev, dtype=self.dtype) for L in self.lin}
        self._vg = None  # cuda-graph state (lazily built)

    # ---- eager norms (compile-friendly; used only in the compiled verify) ----
    def _rms(self, x, w, gemma=True):
        x32 = x.float()
        o = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6))
        return (o * ((1.0 + w.float()) if gemma else w.float())).to(x.dtype)

    def _rope(self, x, pos):  # x [.., heads, HD], pos [N] int
        rd = self._rot_dim
        freqs = pos.float()[:, None] * self._inv_freq
        cos = torch.cat([freqs.cos(), freqs.cos()], -1)[:, None, :]
        sin = torch.cat([freqs.sin(), freqs.sin()], -1)[:, None, :]
        xr, xp = x[..., :rd], x[..., rd:]
        half = rd // 2
        rot = torch.cat([-xr[..., half:], xr[..., :half]], -1)
        return torch.cat([(xr * cos + rot * sin).to(x.dtype), xp], -1)

    # ---- state -----------------------------------------------------------
    def new_state(self):
        return {
            "conv": {L: torch.zeros(2, self.convdim, 4, device=self.dev, dtype=self.dtype) for L in self.lin},
            "ssm": {L: torch.zeros(2, self.HV, self.Vd, self.Kd, device=self.dev, dtype=self.dtype) for L in self.lin},
            "kv": {L: None for L in self.att},
            "pos": 0, "hctx": None,
        }

    # ---- faithful single-token decode (vLLM real modules) ----------------
    @torch.inference_mode()
    def step(self, tokid, st):
        import flashinfer
        from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as cu
        from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode as pd
        dev = self.dev
        h = self.embed[tokid].unsqueeze(0)
        pos = torch.tensor([st["pos"]], device=dev, dtype=torch.int32)
        idx1 = torch.tensor([1], device=dev, dtype=torch.int32)
        for L, ly in enumerate(self.layers):
            resid = h
            hn = ly.input_layernorm(h)
            if L in self._gssm:
                g = ly.linear_attn
                mq, _ = g.in_proj_qkvz(hn); ba, _ = g.in_proj_ba(hn)
                mixed, z = mq.split([self.keyd * 2 + self.vald, self.vald], dim=-1)
                z = z.reshape(z.size(0), -1, self.Vd)
                b, a = g.split_ba(ba); b = b.contiguous(); a = a.contiguous()
                xi = mixed.clone()
                cu(xi, st["conv"][L], g.conv1d.weight.view(self.convdim, -1), g.conv1d.bias, "silu", conv_state_indices=idx1)
                o = torch.zeros(1, 1, self.HV, self.Vd, device=dev, dtype=self.dtype)
                pd(xi.contiguous(), a.contiguous(), b.contiguous(), g.A_log.float(), g.dt_bias.float(),
                   self.Kd ** -0.5, st["ssm"][L], o, idx1, use_qk_l2norm_in_kernel=True)
                h = resid + g.out_proj(g.norm(o[:, 0], z).reshape(1, -1))[0]
            else:
                s = ly.self_attn
                qkv, _ = s.qkv_proj(hn)
                q, k, v, gate = s._project_qkv_gate(qkv, pos)
                q = q.view(1, s.num_heads, s.head_dim); k = k.view(1, s.num_kv_heads, s.head_dim); v = v.view(1, s.num_kv_heads, s.head_dim)
                kv = st["kv"][L]
                kc = k if kv is None else torch.cat([kv[0], k], 0)
                vc = v if kv is None else torch.cat([kv[1], v], 0)
                st["kv"][L] = (kc, vc)
                att = flashinfer.single_prefill_with_kv_cache(q, kc, vc, causal=False, sm_scale=s.head_dim ** -0.5).reshape(1, -1)
                if gate is not None:
                    att = att * torch.sigmoid(gate)
                h = resid + s.o_proj(att)[0]
            h = h + ly.mlp(ly.post_attention_layernorm(h))
        st["pos"] += 1
        hf = self.inner.norm(h)
        prev = st["hctx"]
        st["hctx"] = hf if prev is None else torch.cat([prev, hf], 0)[-8:]
        return (hf @ self.embed.T)[0]

    # ---- eager tree verify (vLLM real modules + SDPA tree mask) ----------
    @torch.inference_mode()
    def tree_verify(self, st, toks, par, dep):
        dev = self.dev
        Nn = len(toks); S = st["pos"]
        nt = torch.tensor(toks, device=dev)
        ns = torch.arange(2, Nn + 2, device=dev)
        pslot = torch.tensor([1 if p < 0 else p + 2 for p in par], device=dev)
        depth_starts = _depth_starts_from_depths(dep, Nn)     # depth-parallel tree scan
        npos = torch.tensor([S + d for d in dep], device=dev, dtype=torch.int32)
        anc = _ancestor_mask(par, Nn, dev)                    # vectorized closure (no per-element GPU writes)
        am = torch.cat([torch.ones(Nn, S, dtype=torch.bool, device=dev), anc], 1)
        h = self.embed[nt]; vkv = {}
        for L, ly in enumerate(self.layers):
            resid = h
            hn = ly.input_layernorm(h)
            if L in self._gssm:
                g = ly.linear_attn
                mq, _ = g.in_proj_qkvz(hn); ba, _ = g.in_proj_ba(hn)
                mixed, z = mq.split([self.keyd * 2 + self.vald, self.vald], dim=-1)
                z = z.reshape(z.size(0), -1, self.Vd)
                b, a = g.split_ba(ba)
                cs = self._gconv[L]; cs[1].copy_(st["conv"][L][1])
                co = tree_scan_causal_conv1d_v2(mixed.contiguous(), cs, g.conv1d.weight.view(self.convdim, -1), g.conv1d.bias, ns, pslot, depth_starts)
                ss = self._gssm[L]; ss[1].copy_(st["ssm"][L][1])
                core = tree_scan_gdn_recurrence_v2(co, a.contiguous(), b.contiguous(), g.A_log.float(), g.dt_bias.float(), ss, ns, pslot, depth_starts, self.Kd ** -0.5)
                h = resid + g.out_proj(g.norm(core, z).reshape(Nn, -1))[0]
            else:
                s = ly.self_attn
                qkv, _ = s.qkv_proj(hn)
                q, k, v, gate = s._project_qkv_gate(qkv, npos)
                q = q.view(Nn, s.num_heads, s.head_dim); k = k.view(Nn, s.num_kv_heads, s.head_dim); v = v.view(Nn, s.num_kv_heads, s.head_dim)
                ck, cv = st["kv"][L]
                kc = torch.cat([ck, k], 0); vc = torch.cat([cv, v], 0); vkv[L] = (k, v)
                rep = s.num_heads // s.num_kv_heads
                att = F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0),
                    kc.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0),
                    vc.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0),
                    attn_mask=am[None, None], scale=s.head_dim ** -0.5)[0].transpose(0, 1).reshape(Nn, -1)
                if gate is not None:
                    att = att * torch.sigmoid(gate)
                h = resid + s.o_proj(att)[0]
            h = h + ly.mlp(ly.post_attention_layernorm(h))
        hf = self.inner.norm(h)
        st["_vkv"] = vkv; st["_vh"] = hf
        return hf @ self.embed.T

    # ---- CUDA-graphed tree verify (static topology + padded/masked KV) ---
    #  Same math as tree_verify (vLLM real modules + tree-scan GDN + SDPA tree
    #  attn), but static-shaped so the whole N-node forward replays as one graph
    #  — eliminates the per-layer launch overhead that makes eager orchestration
    #  ~7x slower than vLLM's compiled decode.
    def _ensure_vgraph(self):
        if self._vg is not None:
            return
        dev, dt = self.dev, self.dtype
        NGC, SG = self.NGC, self.SG
        s0 = self.layers[self.att[0]].self_attn
        NKV, NQ, HD = s0.num_kv_heads, s0.num_heads, s0.head_dim
        G = dict(NKV=NKV, NQ=NQ, HD=HD)
        re = s0.rotary_emb
        self._rot_dim = getattr(re, "rotary_dim", HD)
        theta = float(getattr(re, "base", 1e7))
        self._inv_freq = (1.0 / (theta ** (torch.arange(0, self._rot_dim, 2, device=dev).float() / self._rot_dim)))
        G["nt"] = torch.zeros(NGC, dtype=torch.long, device=dev)
        G["node_pos"] = torch.zeros(NGC, dtype=torch.int32, device=dev)
        G["node_slots"] = torch.arange(2, NGC + 2, device=dev)
        G["parent_slots"] = torch.ones(NGC, dtype=torch.long, device=dev)
        G["amask"] = torch.zeros(NGC, SG + NGC, dtype=torch.bool, device=dev)
        G["ssm_in"] = {L: torch.zeros(self.HV, self.Vd, self.Kd, device=dev, dtype=dt) for L in self.lin}
        G["conv_in"] = {L: torch.zeros(self.convdim, 4, device=dev, dtype=dt) for L in self.lin}
        G["ck"] = {L: torch.zeros(SG, NKV, HD, device=dev, dtype=dt) for L in self.att}
        G["cv"] = {L: torch.zeros(SG, NKV, HD, device=dev, dtype=dt) for L in self.att}
        # STATIC per-depth boundaries for the depth-parallel tree scan. The proposer emits a canonical
        # BFS tree: top_b nodes at the root, then max_depth depths of tree_max_nodes each. Any NGC padding
        # beyond that is one final group (parent = prefix slot) so every out-row is written, exactly like v1.
        ds = [0, self.top_b]
        for _ in range(self.max_depth):
            ds.append(ds[-1] + self.tree_max_nodes)
        if ds[-1] < NGC:
            ds.append(NGC)
        self._vg_depth_starts = ds
        self._vg = G
        # torch.compile the fixed-shape verify: tree-scan ops are torch.library custom ops (no graph
        # break) so inductor fuses the dense/norm/residual work into memory-bound regions.
        self._vg_compiled = torch.compile(self._verify_body, mode="reduce-overhead", fullgraph=False)
        for _ in range(4):
            self._vg_compiled()

    def _verify_body(self):
        G = self._vg; NGC = self.NGC; SG = self.SG
        ns, pslot = G["node_slots"], G["parent_slots"]
        amask = G["amask"]; npos = G["node_pos"]
        h = self.embed[G["nt"]]
        vkv = {}
        for L, ly in enumerate(self.layers):
            resid = h
            hn = self._rms(h, ly.input_layernorm.weight)
            if L in self._gssm:
                g = ly.linear_attn
                mq, _ = g.in_proj_qkvz(hn); ba, _ = g.in_proj_ba(hn)
                mixed, z = mq.split([self.keyd * 2 + self.vald, self.vald], dim=-1)
                z = z.reshape(z.size(0), -1, self.Vd)
                b, a = g.split_ba(ba)
                co = tree_conv_op_v2(mixed.contiguous(), self._gconv[L], g.conv1d.weight.view(self.convdim, -1), g.conv1d.bias, ns, pslot, self._vg_depth_starts)
                core = tree_gdn_op_v2(co, a.contiguous(), b.contiguous(), g.A_log.float(), g.dt_bias.float(), self._gssm[L], ns, pslot, self._vg_depth_starts, self.Kd ** -0.5, True)
                normed = (self._rms(core, g.norm.weight, gemma=False) * F.silu(z.float()).to(core.dtype))
                h = resid + g.out_proj(normed.reshape(NGC, -1))[0]
            else:
                s = ly.self_attn
                NQ, NKV, HD = s.num_heads, s.num_kv_heads, s.head_dim
                qkv, _ = s.qkv_proj(hn)
                qg, k, v = qkv.split([NQ * HD * 2, NKV * HD, NKV * HD], dim=-1)
                q, gate = qg.view(NGC, NQ, HD * 2).chunk(2, dim=-1)
                q = self._rope(self._rms(q, s.q_norm.weight), npos)
                k = self._rope(self._rms(k.view(NGC, NKV, HD), s.k_norm.weight), npos)
                v = v.view(NGC, NKV, HD); vkv[L] = (k, v)
                rep = NQ // NKV
                kc = torch.cat([G["ck"][L], k], 0).repeat_interleave(rep, 1)
                vc = torch.cat([G["cv"][L], v], 0).repeat_interleave(rep, 1)
                att = F.scaled_dot_product_attention(
                    q.transpose(0, 1).unsqueeze(0), kc.transpose(0, 1).unsqueeze(0), vc.transpose(0, 1).unsqueeze(0),
                    attn_mask=amask[None, None], scale=HD ** -0.5)[0].transpose(0, 1).reshape(NGC, -1)
                att = att * torch.sigmoid(gate.reshape(NGC, -1))
                h = resid + s.o_proj(att)[0]
            h = h + ly.mlp(self._rms(h, ly.post_attention_layernorm.weight))
        hf = self._rms(h, self.inner.norm.weight)
        self._vkv = vkv; self._vh = hf
        return hf @ self.embed.T

    @torch.inference_mode()
    def tree_verify_graphed(self, st, toks, par, dep):
        self._ensure_vgraph()
        G = self._vg; NGC = self.NGC; SG = self.SG; dev = self.dev
        N = len(toks); S = st["pos"]
        if N > NGC or S >= SG:
            return self.tree_verify(st, toks, par, dep)
        nt = torch.zeros(NGC, dtype=torch.long); nt[:N] = torch.tensor(toks); G["nt"].copy_(nt.to(dev))
        ps = torch.ones(NGC, dtype=torch.long)
        pt = torch.tensor(par); ps[:N] = torch.where(pt < 0, torch.ones(N, dtype=torch.long), pt + 2); G["parent_slots"].copy_(ps.to(dev))
        npos = torch.zeros(NGC, dtype=torch.int32); npos[:N] = S + torch.tensor(dep, dtype=torch.int32); G["node_pos"].copy_(npos.to(dev))
        am = torch.zeros(NGC, SG + NGC, dtype=torch.bool); am[:, :S] = True
        am[:N, SG:SG + N] = _ancestor_mask(par, N, "cpu")     # vectorized (built on host, copied below)
        di = torch.arange(NGC); am[di, SG + di] = True
        G["amask"].copy_(am.to(dev))
        for L in self.lin:                                  # pre-seed prefix state in slot 1 (outside compiled body)
            self._gssm[L][1].copy_(st["ssm"][L][1]); self._gconv[L][1].copy_(st["conv"][L][1])
        for L in self.att:
            ck, cv = st["kv"][L]
            G["ck"][L][:S].copy_(ck); G["ck"][L][S:].zero_()
            G["cv"][L][:S].copy_(cv); G["cv"][L][S:].zero_()
        out = self._vg_compiled()
        st["_vkv"] = self._vkv; st["_vh"] = self._vh
        return out[:N]

    # ---- greedy tree accept + free commit --------------------------------
    @torch.inference_mode()
    def accept_tree(self, root_logits, node_logits, toks, par):
        ch = {}
        for i, p in enumerate(par):
            ch.setdefault(p, []).append(i)
        g = int(root_logits.argmax()); cur = -1; acc = []; path = []
        while True:
            m = next((c for c in ch.get(cur, []) if toks[c] == g), None)
            if m is None:
                break
            acc.append(g); path.append(m); cur = m; g = int(node_logits[cur].argmax())
        return acc, g, path

    @torch.inference_mode()
    def free_commit(self, st, path, bonus):
        if path:
            leaf = path[-1] + 2
            pn = torch.tensor(path, device=self.dev)
            for L in self.lin:
                st["ssm"][L][1] = self._gssm[L][leaf]
                st["conv"][L][1] = self._gconv[L][leaf]
            for L in self.att:
                ck, cv = st["kv"][L]; nk, nv = st["_vkv"][L]
                st["kv"][L] = (torch.cat([ck, nk[pn]], 0), torch.cat([cv, nv[pn]], 0))
            st["pos"] += len(path)
            ah = st["_vh"][pn]; prev = st["hctx"]
            st["hctx"] = ah if prev is None else torch.cat([prev, ah], 0)[-8:]
        return self.step(bonus, st)

    # ---- drafter proposal (flow pass CUDA-graphed) -----------------------
    def _setup_draft_graph(self):
        if getattr(self, "_fg", None) is not None:
            return
        cs = self.drafter.config.context_size
        hd = self.embed.shape[1]
        self._fg_ctx = torch.zeros(1, cs, hd, device=self.dev, dtype=self.drafter._dtype)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self.drafter.predict_hidden(self._fg_ctx)
        torch.cuda.current_stream().wait_stream(s)
        self._fg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._fg):
            self._fg_out = self.drafter.predict_hidden(self._fg_ctx)
        self._fg_cs = cs
        self._orig_predict = self.drafter.predict_hidden
        def graphed(c):
            if c.shape[1] != self._fg_cs:
                return self._orig_predict(c)
            self._fg_ctx.copy_(c); self._fg.replay(); return self._fg_out
        self.drafter.predict_hidden = graphed

    @torch.inference_mode()
    def propose(self, st):
        if st["hctx"] is None:
            return [], [], []
        self._setup_draft_graph()
        class _DS:
            def __init__(s, h): s.final_hidden = h.unsqueeze(0)
        tr = self.drafter.build_tree_fast(_DS(st["hctx"]), top_b=self.top_b,
                                          max_nodes=self.tree_max_nodes, max_depth=self.max_depth)
        return tr.tokens, tr.parents, tr.depths

    # ---- generation ------------------------------------------------------
    @torch.inference_mode()
    def generate(self, prompt_ids, max_new, *, free=True):
        st = self.new_state(); lg = None
        for t in prompt_ids:
            lg = self.step(int(t), st)
        out = []; acc = 0; steps = 0
        while len(out) < max_new:
            tk, pa, dp = self.propose(st)
            if not tk or len(tk) > self.NGC or st["pos"] >= self.SG:
                nt = int(lg.argmax()); out.append(nt); lg = self.step(nt, st)
                if nt == self.eos_id:
                    break
                continue
            nl = self.tree_verify_graphed(st, tk, pa, dp)
            a, b, path = self.accept_tree(lg, nl, tk, pa)
            out += a + [b]; acc += len(a); steps += 1
            if free:
                lg = self.free_commit(st, path, b)
            else:
                for t in a + [b]:
                    lg = self.step(t, st)
            if self.eos_id is not None and self.eos_id in a + [b]:
                break
        return out[:max_new], (acc / steps if steps else 0.0)
