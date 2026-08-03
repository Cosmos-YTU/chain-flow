"""Standalone Qwen3.5 decode forward built from the HF weights + vLLM's own kernels.

Used to (a) validate the assembled forward token-for-token against vLLM's greedy decode, and (b) be the
substrate for tree verification (the decode path *is* the per-node tree forward, with parent-routed
state). Decode-only (processes the prompt token-by-token) so it reuses exactly the validated tree
primitives — packed_decode + causal_conv1d_update — no separate chunk-prefill path.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode as _pd
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as _conv_upd

HV = 16          # num linear value heads
V = K = 128      # head_v_dim / head_k_dim
N_KV_HEADS, N_Q_HEADS, HEAD_DIM = 2, 8, 256
ROT_DIM = 64
ROPE_THETA = 1e7
EPS = 1e-6


def _load_weights(model_id="Qwen/Qwen3.5-0.8B", device="cuda:0", dtype=torch.bfloat16):
    from huggingface_hub import snapshot_download
    d = snapshot_download(model_id, allow_patterns=["*.safetensors", "*.index.json"])
    sd = {}
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        for k, v in load_file(f, device="cpu").items():
            if "language_model" in k:
                sd[k.replace("model.language_model.", "")] = v.to(device=device, dtype=dtype)
    return sd


class TreeBackbone:
    def __init__(self, model_id="Qwen/Qwen3.5-0.8B", device="cuda:0", dtype=torch.bfloat16):
        self.dev, self.dt = torch.device(device), dtype
        self.w = _load_weights(model_id, device, dtype)
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_id, trust_remote_code=True).text_config
        self.layer_types = c.layer_types
        self.nlayers = c.num_hidden_layers
        self.embed = self.w["embed_tokens.weight"]
        # rotary table
        inv = 1.0 / (ROPE_THETA ** (torch.arange(0, ROT_DIM, 2, device=device).float() / ROT_DIM))
        self.inv_freq = inv
        # persistent PER-LAYER tree-verify state buffers (slot 0 sentinel); per-layer so the accepted
        # path's state can be extracted for a FREE commit (no sequential re-decode).
        self._maxn = 320
        lin = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        att = [i for i, t in enumerate(self.layer_types) if t != "linear_attention"]
        self._vssm = {L: torch.zeros(self._maxn, HV, V, K, device=device, dtype=dtype) for L in lin}
        self._vconv = {L: torch.zeros(self._maxn, 6144, 4, device=device, dtype=dtype) for L in lin}
        self._vkv: dict = {L: None for L in att}

    def _rms(self, x, w):  # Qwen3.5 RMSNorm: weight is zero-centred -> (1 + weight)
        x32 = x.float()
        out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
        return (out * (1.0 + w.float())).to(x.dtype)

    def _gated_norm(self, x, gate, w):
        x32 = x.float()
        n = (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(x.dtype) * w
        return n * F.silu(gate.float()).to(x.dtype)

    def _rope(self, x, pos):  # x [.., n_heads, HEAD_DIM], pos [..] int
        freqs = pos.float()[..., None] * self.inv_freq                    # [.., ROT_DIM/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], -1)[..., None, :]      # [.., 1, ROT_DIM]
        sin = torch.cat([freqs.sin(), freqs.sin()], -1)[..., None, :]
        xr, xp = x[..., :ROT_DIM], x[..., ROT_DIM:]
        half = ROT_DIM // 2
        rot = torch.cat([-xr[..., half:], xr[..., :half]], -1)
        return torch.cat([(xr * cos + rot * sin).to(x.dtype), xp], -1)

    def new_state(self):
        gdn = sum(1 for t in self.layer_types if t == "linear_attention")
        return {
            "conv": torch.zeros(self.nlayers, 2, 6144, 4, device=self.dev, dtype=self.dt),   # slot0 sentinel/1 live
            "ssm": torch.zeros(self.nlayers, 2, HV, V, K, device=self.dev, dtype=self.dt),
            "kv": {i: None for i in range(self.nlayers)},   # per full-attn layer: (k[seq,2,256], v[...])
            "pos": 0,
        }

    @torch.inference_mode()
    def step(self, token: int, st: dict) -> torch.Tensor:
        """One decode step for a single sequence. Returns logits [vocab]. Mutates st in place."""
        w = self.w
        h = self.embed[token].unsqueeze(0)          # [1, 1024]
        pos = torch.tensor([st["pos"]], device=self.dev)
        idx1 = torch.tensor([1], device=self.dev, dtype=torch.int32)
        for L in range(self.nlayers):
            p = f"layers.{L}."
            resid = h
            hn = self._rms(h, w[p + "input_layernorm.weight"])
            if self.layer_types[L] == "linear_attention":
                q = p + "linear_attn."
                mixed = hn @ w[q + "in_proj_qkv.weight"].T           # [1, 6144]
                z = hn @ w[q + "in_proj_z.weight"].T                 # [1, 2048]
                b = hn @ w[q + "in_proj_b.weight"].T                 # [1, 16]
                a = hn @ w[q + "in_proj_a.weight"].T                 # [1, 16]
                conv_w = w[q + "conv1d.weight"].view(6144, 4)
                conv_b = w.get(q + "conv1d.bias")
                xi = mixed.clone()
                _conv_upd(xi, st["conv"][L], conv_w, conv_b, "silu", conv_state_indices=idx1)
                o = torch.zeros(1, 1, HV, V, device=self.dev, dtype=self.dt)
                _pd(xi.contiguous(), a.contiguous(), b.contiguous(),
                    w[q + "A_log"].float(), w[q + "dt_bias"].float(), K ** -0.5,
                    st["ssm"][L], o, idx1, use_qk_l2norm_in_kernel=True)
                core = o[0]                                          # [HV, V]
                normed = self._gated_norm(core, z.view(HV, V), w[q + "norm.weight"])
                h = resid + normed.reshape(1, -1) @ w[q + "out_proj.weight"].T
            else:
                q = p + "self_attn."
                qg = (hn @ w[q + "q_proj.weight"].T).view(1, N_Q_HEADS, HEAD_DIM * 2)
                query, gate = qg.chunk(2, dim=-1)                    # each [1, 8, 256]
                gate = gate.reshape(1, -1)                           # [1, 2048]
                qs = self._rms(query, w[q + "q_norm.weight"])
                ks = self._rms((hn @ w[q + "k_proj.weight"].T).view(1, N_KV_HEADS, HEAD_DIM), w[q + "k_norm.weight"])
                vs = (hn @ w[q + "v_proj.weight"].T).view(1, N_KV_HEADS, HEAD_DIM)
                qs = self._rope(qs, pos); ks = self._rope(ks, pos)
                kv = st["kv"][L]
                kcat = ks if kv is None else torch.cat([kv[0], ks], 0)
                vcat = vs if kv is None else torch.cat([kv[1], vs], 0)
                st["kv"][L] = (kcat, vcat)
                krep = kcat.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)   # [S,8,256]
                vrep = vcat.repeat_interleave(N_Q_HEADS // N_KV_HEADS, dim=1)
                att = F.scaled_dot_product_attention(
                    qs.transpose(0, 1).unsqueeze(0), krep.transpose(0, 1).unsqueeze(0),
                    vrep.transpose(0, 1).unsqueeze(0), scale=HEAD_DIM ** -0.5)[0].transpose(0, 1)
                att = att.reshape(1, -1) * torch.sigmoid(gate)       # gated attention
                h = resid + att @ w[q + "o_proj.weight"].T
            r2 = h
            hn2 = self._rms(h, w[p + "post_attention_layernorm.weight"])
            gate = F.silu(hn2 @ w[p + "mlp.gate_proj.weight"].T) * (hn2 @ w[p + "mlp.up_proj.weight"].T)
            h = r2 + gate @ w[p + "mlp.down_proj.weight"].T
        st["pos"] += 1
        hf = self._rms(h, w["norm.weight"])
        prev = st.get("hctx")
        st["hctx"] = hf if prev is None else torch.cat([prev, hf], 0)[-8:]   # rolling last-8 POST-norm ctx
        return (hf @ self.embed.T)[0]                # tied lm_head

    @torch.inference_mode()
    def tree_verify(self, st: dict, node_tokens, parent, depth) -> torch.Tensor:
        """Score a draft tree against the committed state. Returns per-node logits [N, vocab].
        Reuses the validated forward with parent-routed SSM state + tree-attn. Does not mutate st."""
        from chained_flow.vllm_tree.tree_ssm import tree_gated_delta_recurrence, tree_causal_conv1d
        w = self.w
        N = len(node_tokens)
        dev = self.dev
        nt = torch.tensor(node_tokens, device=dev)
        par = torch.tensor(parent, device=dev)
        dep = torch.tensor(depth, device=dev)
        S = st["pos"]
        node_slots = torch.arange(2, N + 2, device=dev)
        parent_slots = torch.where(par < 0, torch.ones_like(par), par + 2)
        node_pos = S + dep                                   # [N] position of each node
        maxd = int(dep.max().item()) if N else -1
        groups = [(dep == d).nonzero(as_tuple=True)[0] for d in range(maxd + 1)]   # precompute ONCE
        h = self.embed[nt]                                   # [N, 1024]
        # tree-attn mask: node i sees all prefix (S) + ancestors(i)+self among nodes
        anc = torch.zeros(N, N, device=dev, dtype=torch.bool)
        for i in range(N):
            j = i
            while j >= 0:
                anc[i, j] = True
                j = parent[j]
        amask = torch.cat([torch.ones(N, S, device=dev, dtype=torch.bool), anc], dim=1)  # [N, S+N]
        for L in range(self.nlayers):
            p = f"layers.{L}."
            resid = h
            hn = self._rms(h, w[p + "input_layernorm.weight"])
            if self.layer_types[L] == "linear_attention":
                q = p + "linear_attn."
                mixed = hn @ w[q + "in_proj_qkv.weight"].T
                z = hn @ w[q + "in_proj_z.weight"].T
                b = hn @ w[q + "in_proj_b.weight"].T
                a = hn @ w[q + "in_proj_a.weight"].T
                conv_state = self._vconv[L][:N + 2]
                conv_state[1] = st["conv"][L][1]
                conv_out = tree_causal_conv1d(mixed.clone(), conv_state, w[q + "conv1d.weight"].view(6144, 4),
                                              w.get(q + "conv1d.bias"), node_slots, parent_slots, groups)
                ssm_state = self._vssm[L][:N + 2]
                ssm_state[1] = st["ssm"][L][1]
                core = tree_gated_delta_recurrence(conv_out, a, b, w[q + "A_log"].float(), w[q + "dt_bias"].float(),
                                                   ssm_state, node_slots, parent_slots, groups, K ** -0.5)
                normed = self._gated_norm(core, z.view(N, HV, V), w[q + "norm.weight"])
                h = resid + normed.reshape(N, -1) @ w[q + "out_proj.weight"].T
            else:
                q = p + "self_attn."
                qg = (hn @ w[q + "q_proj.weight"].T).view(N, N_Q_HEADS, HEAD_DIM * 2)
                query, gate = qg.chunk(2, dim=-1); gate = gate.reshape(N, -1)
                qs = self._rms(query, w[q + "q_norm.weight"])
                ks = self._rms((hn @ w[q + "k_proj.weight"].T).view(N, N_KV_HEADS, HEAD_DIM), w[q + "k_norm.weight"])
                vs = (hn @ w[q + "v_proj.weight"].T).view(N, N_KV_HEADS, HEAD_DIM)
                qs = self._rope(qs, node_pos); ks = self._rope(ks, node_pos)
                self._vkv[L] = (ks, vs)                        # save per-node KV for free commit
                ck, cv = st["kv"][L]                          # committed prefix KV [S,2,256]
                Kf = torch.cat([ck, ks], 0).repeat_interleave(N_Q_HEADS // N_KV_HEADS, 1)   # [S+N,8,256]
                Vf = torch.cat([cv, vs], 0).repeat_interleave(N_Q_HEADS // N_KV_HEADS, 1)
                att = F.scaled_dot_product_attention(
                    qs.transpose(0, 1).unsqueeze(0), Kf.transpose(0, 1).unsqueeze(0), Vf.transpose(0, 1).unsqueeze(0),
                    attn_mask=amask[None, None], scale=HEAD_DIM ** -0.5)[0].transpose(0, 1)
                att = att.reshape(N, -1) * torch.sigmoid(gate)
                h = resid + att @ w[q + "o_proj.weight"].T
            r2 = h
            hn2 = self._rms(h, w[p + "post_attention_layernorm.weight"])
            g = F.silu(hn2 @ w[p + "mlp.gate_proj.weight"].T) * (hn2 @ w[p + "mlp.up_proj.weight"].T)
            h = r2 + g @ w[p + "mlp.down_proj.weight"].T
        hf = self._rms(h, w["norm.weight"])
        self._vhidden = hf                                    # [N, 1024] POST-norm hidden (for ctx on commit)
        return hf @ self.embed.T                              # [N, vocab]

    # ------------------------------------------------------------------ #
    #  CUDA-graphed tree verify (static topology + padded/masked KV).      #
    #  The launch-bound SSM loop (18 GDN layers x depths) dominates the     #
    #  eager verify; a graph replays it in ~half the time. Bit-exact with   #
    #  the eager path (same kernels, same math; padding masked out).        #
    # ------------------------------------------------------------------ #
    def _ensure_vgraph(self, NGC=56, SG=160, DG=7, WG=8, NMAX=48):
        if getattr(self, "_vg", None) is not None:
            return
        dev, dt = self.dev, self.dt
        lin = [i for i, t in enumerate(self.layer_types) if t == "linear_attention"]
        att = [i for i, t in enumerate(self.layer_types) if t != "linear_attention"]
        G = dict(NGC=NGC, SG=SG, DG=DG, WG=WG, NMAX=NMAX, lin=lin, att=att)
        G["node_slots"] = torch.arange(2, NGC + 2, device=dev)              # const: node i -> slot i+2
        G["nt"] = torch.zeros(NGC, dtype=torch.long, device=dev)
        G["parent_slots"] = torch.ones(NGC, dtype=torch.long, device=dev)   # default -> prefix slot 1
        G["node_pos"] = torch.zeros(NGC, dtype=torch.long, device=dev)
        G["groups"] = [torch.full((WG,), NGC - 1, dtype=torch.long, device=dev) for _ in range(DG)]
        G["amask"] = torch.zeros(NGC, SG + NGC, dtype=torch.bool, device=dev)
        G["ssm_in"] = {L: torch.zeros(HV, V, K, device=dev, dtype=dt) for L in lin}
        G["conv_in"] = {L: torch.zeros(6144, 4, device=dev, dtype=dt) for L in lin}
        G["ck"] = {L: torch.zeros(SG, N_KV_HEADS, HEAD_DIM, device=dev, dtype=dt) for L in att}
        G["cv"] = {L: torch.zeros(SG, N_KV_HEADS, HEAD_DIM, device=dev, dtype=dt) for L in att}
        self._vg = G
        # torch.compile the fixed-shape verify body (tree-scan ops are custom ops -> no graph break;
        # inductor fuses the dense/norm/residual work into memory-bound regions + its own CUDA graph).
        self._vg_compiled = torch.compile(self._verify_body, mode="reduce-overhead", fullgraph=False)
        for _ in range(4):
            self._vg_compiled()

    def _verify_body(self):
        from chained_flow.vllm_tree.tree_ssm import tree_gdn_op as _tgd, tree_conv_op as _tcv
        G = self._vg; NGC = G["NGC"]; w = self.w
        node_slots, parent_slots = G["node_slots"], G["parent_slots"]
        amask = G["amask"]; node_pos = G["node_pos"]
        h = self.embed[G["nt"]]                                             # [NGC, 1024]
        vkv = {}
        for L in range(self.nlayers):
            p = f"layers.{L}."
            resid = h
            hn = self._rms(h, w[p + "input_layernorm.weight"])
            if self.layer_types[L] == "linear_attention":
                q = p + "linear_attn."
                mixed = hn @ w[q + "in_proj_qkv.weight"].T
                z = hn @ w[q + "in_proj_z.weight"].T
                b = hn @ w[q + "in_proj_b.weight"].T
                a = hn @ w[q + "in_proj_a.weight"].T
                conv_out = _tcv(mixed, self._vconv[L], w[q + "conv1d.weight"].view(6144, 4),
                                w.get(q + "conv1d.bias"), node_slots, parent_slots)   # prefix pre-seeded in slot 1
                core = _tgd(conv_out, a, b, w[q + "A_log"].float(), w[q + "dt_bias"].float(),
                            self._vssm[L], node_slots, parent_slots, K ** -0.5, True)   # custom-op tree scan (compile-safe)
                normed = self._gated_norm(core, z.view(NGC, HV, V), w[q + "norm.weight"])
                h = resid + normed.reshape(NGC, -1) @ w[q + "out_proj.weight"].T
            else:
                q = p + "self_attn."
                qg = (hn @ w[q + "q_proj.weight"].T).view(NGC, N_Q_HEADS, HEAD_DIM * 2)
                query, gate = qg.chunk(2, dim=-1); gate = gate.reshape(NGC, -1)
                qs = self._rms(query, w[q + "q_norm.weight"])
                ks = self._rms((hn @ w[q + "k_proj.weight"].T).view(NGC, N_KV_HEADS, HEAD_DIM), w[q + "k_norm.weight"])
                vs = (hn @ w[q + "v_proj.weight"].T).view(NGC, N_KV_HEADS, HEAD_DIM)
                qs = self._rope(qs, node_pos); ks = self._rope(ks, node_pos)
                vkv[L] = (ks, vs)
                Kf = torch.cat([G["ck"][L], ks], 0).repeat_interleave(N_Q_HEADS // N_KV_HEADS, 1)
                Vf = torch.cat([G["cv"][L], vs], 0).repeat_interleave(N_Q_HEADS // N_KV_HEADS, 1)
                att = F.scaled_dot_product_attention(
                    qs.transpose(0, 1).unsqueeze(0), Kf.transpose(0, 1).unsqueeze(0), Vf.transpose(0, 1).unsqueeze(0),
                    attn_mask=amask[None, None], scale=HEAD_DIM ** -0.5)[0].transpose(0, 1)
                att = att.reshape(NGC, -1) * torch.sigmoid(gate)
                h = resid + att @ w[q + "o_proj.weight"].T
            r2 = h
            hn2 = self._rms(h, w[p + "post_attention_layernorm.weight"])
            gg = F.silu(hn2 @ w[p + "mlp.gate_proj.weight"].T) * (hn2 @ w[p + "mlp.up_proj.weight"].T)
            h = r2 + gg @ w[p + "mlp.down_proj.weight"].T
        hf = self._rms(h, w["norm.weight"])
        self._vkv = vkv; self._vhidden = hf
        return hf @ self.embed.T                                            # [NGC, vocab]

    @torch.inference_mode()
    def tree_verify_g(self, st, node_tokens, parent, depth):
        """Graphed equivalent of tree_verify; falls back to eager if the tree exceeds graph capacity."""
        self._ensure_vgraph()
        G = self._vg; NGC = G["NGC"]; SG = G["SG"]
        N = len(node_tokens); S = st["pos"]; dev = self.dev
        # tree-scan handles any depth/width; only graph capacity (NGC nodes, SG prefix) constrains.
        if N > NGC or S >= SG:
            return self.tree_verify(st, node_tokens, parent, depth)
        nt = torch.zeros(NGC, dtype=torch.long); nt[:N] = torch.tensor(node_tokens); G["nt"].copy_(nt.to(dev))
        par = torch.tensor(parent); ps = torch.ones(NGC, dtype=torch.long)
        ps[:N] = torch.where(par < 0, torch.ones(N, dtype=torch.long), par + 2); G["parent_slots"].copy_(ps.to(dev))
        npos = torch.zeros(NGC, dtype=torch.long); npos[:N] = S + torch.tensor(depth); G["node_pos"].copy_(npos.to(dev))
        am = torch.zeros(NGC, SG + NGC, dtype=torch.bool); am[:, :S] = True
        anc = torch.zeros(N, N, dtype=torch.bool)
        for i in range(N):
            j = i
            while j >= 0:
                anc[i, j] = True; j = parent[j]
        am[:N, SG:SG + N] = anc
        di = torch.arange(NGC); am[di, SG + di] = True                      # self-attend all rows (no NaN on pad)
        G["amask"].copy_(am.to(dev))
        for L in G["lin"]:                                   # pre-seed prefix state in slot 1 (outside compiled body)
            self._vssm[L][1].copy_(st["ssm"][L][1]); self._vconv[L][1].copy_(st["conv"][L][1])
        for L in G["att"]:
            ck, cv = st["kv"][L]
            G["ck"][L][:S].copy_(ck); G["ck"][L][S:].zero_()
            G["cv"][L][:S].copy_(cv); G["cv"][L][S:].zero_()
        out = self._vg_compiled()
        return out[:N]

    @torch.inference_mode()
    def accept_tree(self, root_logits, node_logits, tokens, parents):
        """Longest correct root-to-leaf path: follow the backbone's greedy token through the tree
        while a child matches it. node_logits ARE the truth (tree_verify == sequential decode), so the
        accepted tokens are exactly greedy -> lossless. Returns (accepted[list], bonus)."""
        children: dict[int, list[int]] = {}
        for i, par in enumerate(parents):
            children.setdefault(par, []).append(i)
        g = int(root_logits.argmax()); cur = -1; accepted: list[int] = []; path: list[int] = []
        while True:
            m = next((c for c in children.get(cur, []) if tokens[c] == g), None)
            if m is None:
                break
            accepted.append(g); path.append(m); cur = m; g = int(node_logits[cur].argmax())
        return accepted, g, path                 # path = accepted node indices (root..leaf)

    @torch.inference_mode()
    def free_commit(self, st, path, bonus_token):
        """Advance committed state along the accepted path using state ALREADY in tree_verify buffers
        (NO re-decode), then step the bonus token once. path = accepted node indices."""
        if path:
            leaf = path[-1] + 2
            pn = torch.tensor(path, device=self.dev)
            for L in self._vssm:
                st["ssm"][L][1] = self._vssm[L][leaf]
                st["conv"][L][1] = self._vconv[L][leaf]
            for L in self._vkv:
                ck, cv = st["kv"][L]
                nk, nv = self._vkv[L]
                st["kv"][L] = (torch.cat([ck, nk[pn]], 0), torch.cat([cv, nv[pn]], 0))
            st["pos"] += len(path)
            prev = st.get("hctx")
            ah = self._vhidden[pn]
            st["hctx"] = ah if prev is None else torch.cat([prev, ah], 0)[-8:]
        return self.step(bonus_token, st)

    @torch.inference_mode()
    def spec_generate(self, prompt_ids, max_new, propose, free=True, stop=None):
        st, logits = self.prefill(prompt_ids)
        out: list[int] = []; steps = 0; acc = 0
        while len(out) < max_new:
            tokens, parents, depth = propose(self, st, logits)
            if not tokens:
                nt = int(logits.argmax()); out.append(nt); logits = self.step(nt, st)
                if nt == stop: break
                continue
            nl = self.tree_verify_g(st, tokens, parents, depth)
            accepted, bonus, path = self.accept_tree(logits, nl, tokens, parents)
            out += accepted + [bonus]; acc += len(accepted); steps += 1
            if free:
                logits = self.free_commit(st, path, bonus)
            else:
                for t in accepted + [bonus]:
                    logits = self.step(t, st)
            if stop is not None and stop in (accepted + [bonus]):
                break
        return out[:max_new], (acc / steps if steps else 0.0)

    @torch.inference_mode()
    def prefill(self, prompt_ids):
        st = self.new_state()
        logits = None
        for t in prompt_ids:
            logits = self.step(int(t), st)
        return st, logits

    @torch.inference_mode()
    def greedy(self, prompt_ids, max_new=32, stop=None):
        st = self.new_state()
        logits = None
        for t in prompt_ids:
            logits = self.step(int(t), st)
        out = []
        for _ in range(max_new):
            nt = int(logits.argmax()); out.append(nt)
            if nt == stop: break
            logits = self.step(nt, st)
        return out
