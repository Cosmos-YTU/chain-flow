import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys; sys.path.insert(0, "/home/shadeform/chained-flow/src")
import time, json, dataclasses, torch
import numpy as _np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
CKD = "/home/shadeform/chained-flow/out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8"
tok = AutoTokenizer.from_pretrained(MODEL)
# structured (high-accept) + prose prompts
import json as _json
_pf = os.environ.get("PROMPTS_FILE")
if _pf:
    _rows = [_json.loads(l) for l in open(_pf) if l.strip()]
    DOMAINS = [r["domain"] for r in _rows]
    def _ct(t):
        try: return tok.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError: return tok.apply_chat_template([{"role":"user","content":t}], tokenize=False, add_generation_prompt=True)
    PTEXT = [_ct(r["prompt"]) for r in _rows]
else:
    PTEXT = ["Q: What is 15*23? A: Let's think step by step.", "def fibonacci(n):",
             "The capital of France is", "Write a short paragraph about the ocean."]
    DOMAINS = ["math","code","short","prose"]
PIDS = [tok(t)["input_ids"] for t in PTEXT]

print("== load 4B (full VL model) ==", flush=True)
llm = LLM(model=MODEL, gpu_memory_utilization=0.55, max_model_len=2048, dtype="float16", enforce_eager=False)
sp = SamplingParams(temperature=0, max_tokens=48)
VOUT = [list(o.outputs[0].token_ids) for o in llm.generate([{"prompt_token_ids": p} for p in PIDS], sp)]
t0 = time.time(); llm.generate([{"prompt_token_ids": p} for p in PIDS], sp); tnat = time.time() - t0
print(f"[native vLLM] {sum(len(v) for v in VOUT)} tok / {tnat:.2f}s = {sum(len(v) for v in VOUT)/tnat:.0f} tok/s", flush=True)


def run(worker):
    import torch, torch.nn.functional as F
    from vllm.forward_context import set_forward_context, BatchDescriptor
    from vllm.config import CUDAGraphMode
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as cu
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode as pd
    from chained_flow.vllm_tree.tree_ssm import tree_gdn_op, tree_conv_op
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dev = "cuda"; model = worker.get_model(); runner = worker.model_runner
    lang = model.language_model; inner = lang.model; layers = inner.layers
    embed = inner.embed_tokens.weight; lmw = lang.lm_head.weight
    HIDDEN, VOCAB = embed.shape[1], embed.shape[0]
    g0 = [l for l in layers if hasattr(l, "linear_attn")][0].linear_attn
    HV, Vd, Kd = g0.num_v_heads, g0.head_v_dim, g0.head_k_dim
    keyd, vald = g0.key_dim, g0.value_dim; convdim = keyd * 2 + vald
    s0 = [l for l in layers if hasattr(l, "self_attn")][0].self_attn
    NQ, NKV, HD = s0.num_heads, s0.num_kv_heads, s0.head_dim
    LIN = [L for L, ly in enumerate(layers) if hasattr(ly, "linear_attn")]
    ATT = [L for L, ly in enumerate(layers) if hasattr(ly, "self_attn")]
    MAXN = 64; NG = 48; SG = 512
    print(f"[struct] hidden={HIDDEN} vocab={VOCAB} lin={len(LIN)} att={len(ATT)} HV={HV} Vd={Vd} Kd={Kd} NQ={NQ} NKV={NKV} HD={HD}", flush=True)

    def SDPA():
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

    def BD(n):  # 0.25.1 BatchDescriptor gained num_reqs
        try:    return BatchDescriptor(num_tokens=n, uniform=False)
        except TypeError: return BatchDescriptor(num_tokens=n, num_reqs=n, uniform=False)

    class CTS:
        def __init__(s):
            s.ssm = {L: torch.zeros(MAXN + 2, HV, Vd, Kd, device=dev, dtype=embed.dtype) for L in LIN}
            s.conv = {L: torch.zeros(MAXN + 2, convdim, 4, device=dev, dtype=embed.dtype) for L in LIN}
            s.kv = {L: None for L in ATT}; s.idx1 = torch.tensor([1], device=dev, dtype=torch.int32)
            s.mode = "decode"; s.captured = False; s.hctx = None
            s.g_ns = torch.arange(2, NG + 2, device=dev); s.g_ps = torch.ones(NG, device=dev, dtype=torch.long)
            s.g_amask = torch.zeros(NG, SG + NG, dtype=torch.bool, device=dev); s._cp = {}
            s.g_kbuf = {L: torch.zeros(SG + NG, NKV, HD, device=dev, dtype=embed.dtype) for L in ATT}
            s.g_vbuf = {L: torch.zeros(SG + NG, NKV, HD, device=dev, dtype=embed.dtype) for L in ATT}
            s.g_nodek = {L: torch.zeros(NG, NKV, HD, device=dev, dtype=embed.dtype) for L in ATT}
            s.g_nodev = {L: torch.zeros(NG, NKV, HD, device=dev, dtype=embed.dtype) for L in ATT}
            s._install()
        def _install(s):
            for L, ly in enumerate(layers):
                if hasattr(ly, "linear_attn"): ly.linear_attn._forward_core = s._gdn(L, ly.linear_attn)
                else: ly.self_attn.attn.impl.forward = s._attn(L)
        def _gdn(s, L, g):
            cw = g.conv1d.weight.view(convdim, -1); cb = g.conv1d.bias; Al = g.A_log.float(); db = g.dt_bias.float()
            def fc(mixed_qkv, b, a, core_attn_out):
                if s.mode == "verify":
                    co = tree_conv_op(mixed_qkv.contiguous(), s.conv[L], cw, cb, s.g_ns, s.g_ps)
                    core = tree_gdn_op(co, a.contiguous(), b.contiguous(), Al, db, s.ssm[L], s.g_ns, s.g_ps, Kd ** -0.5, True)
                    core_attn_out.copy_(core)
                else:
                    xi = mixed_qkv.clone(); cu(xi, s.conv[L], cw, cb, "silu", conv_state_indices=s.idx1)
                    o = torch.zeros(mixed_qkv.shape[0], 1, HV, Vd, device=dev, dtype=mixed_qkv.dtype)
                    pd(xi.contiguous(), a.contiguous(), b.contiguous(), Al, db, Kd ** -0.5, s.ssm[L], o, s.idx1, use_qk_l2norm_in_kernel=True)
                    core_attn_out.copy_(o[:, 0])
            return fc
        def _attn(s, L):
            def af(layer, query, key, value, kv_cache, attn_metadata, output=None, **kw):
                q, k, v = query, key, value; rep = NQ // NKV
                if s.mode == "verify":
                    s.g_kbuf[L][SG:SG + NG].copy_(k); s.g_vbuf[L][SG:SG + NG].copy_(v)
                    s.g_nodek[L].copy_(k); s.g_nodev[L].copy_(v)
                    kc = s.g_kbuf[L].repeat_interleave(rep, 1); vc = s.g_vbuf[L].repeat_interleave(rep, 1)
                    att = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), kc.transpose(0, 1).unsqueeze(0), vc.transpose(0, 1).unsqueeze(0), attn_mask=s.g_amask[None, None], scale=HD ** -0.5)[0].transpose(0, 1)
                else:
                    pv = s.kv[L]; kc = k if pv is None else torch.cat([pv[0], k], 0); vc = v if pv is None else torch.cat([pv[1], v], 0); s.kv[L] = (kc, vc)
                    att = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), kc.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0), vc.repeat_interleave(rep, 1).transpose(0, 1).unsqueeze(0), scale=HD ** -0.5)[0].transpose(0, 1)
                output.copy_(att)
            return af
        @torch.inference_mode()
        def step(s, tokid, pos):
            s.mode = "decode"; posn = torch.tensor([pos], device=dev).view(1, 1).expand(3, 1).contiguous()
            with set_forward_context(None, runner.vllm_config, num_tokens=1, cudagraph_runtime_mode=CUDAGraphMode.NONE):
                hs = model(input_ids=None, positions=posn, inputs_embeds=inner.embed_tokens(torch.tensor([tokid], device=dev)))
            return hs
        @torch.inference_mode()
        def step_commit(s, tokid, pos):
            hs = s.step(tokid, pos)
            s.hctx = hs if s.hctx is None else torch.cat([s.hctx, hs], 0)[-8:]
            return model.compute_logits(hs)[-1]
        @torch.inference_mode()
        def verify(s, toks, par, dep, N):
            s.mode = "verify"; S = s.kv[ATT[0]][0].shape[0]
            ps = [1 if p < 0 else p + 2 for p in par]; s.g_ps.copy_(torch.tensor(ps, device=dev))
            # mask built on CPU (one H2D copy) instead of ~N*depth eager per-element GPU writes
            _anc = _np.zeros((N, N), dtype=bool)
            for i in range(N):
                j = i
                while j >= 0: _anc[i, j] = True; j = par[j]
            _am = _np.zeros((NG, SG + NG), dtype=bool)
            _am[:N, :S] = True; _am[:N, SG:SG + N] = _anc
            _di = _np.arange(NG); _am[_di, SG + _di] = True
            s.g_amask.copy_(torch.from_numpy(_am))
            # incremental KV refill: copy only NEW context rows; no zeroing (those cols are masked out)
            for L in ATT:
                ck, cv = s.kv[L]
                c = s._cp.get(L, 0)
                if c > S: c = 0
                if S > c:
                    s.g_kbuf[L][c:S].copy_(ck[c:S]); s.g_vbuf[L][c:S].copy_(cv[c:S])
                s._cp[L] = S
            emb = inner.embed_tokens(torch.tensor(toks, device=dev))
            npos = torch.tensor([S + d for d in dep], device=dev).view(1, NG).expand(3, NG).contiguous()
            runner.inputs_embeds.gpu[:NG].copy_(emb); runner.mrope_positions.gpu[:, :NG].copy_(npos)
            bd = BD(NG)
            def _call():
                with set_forward_context(None, runner.vllm_config, num_tokens=NG, cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=bd):
                    return runner.model(input_ids=None, positions=runner.mrope_positions.gpu[:, :NG], inputs_embeds=runner.inputs_embeds.gpu[:NG])
            if not s.captured:
                set_cudagraph_capturing_enabled(True)
                with graph_capture(torch.device(dev)): _call()
                set_cudagraph_capturing_enabled(False); s.captured = True
            hs = _call(); s._vh = hs
            return model.compute_logits(hs)[:N]
        @torch.inference_mode()
        def free_commit(s, path, bonus, pos):
            if path:
                leaf = path[-1] + 2; pn = torch.tensor(path, device=dev)
                for L in LIN: s.ssm[L][1] = s.ssm[L][leaf]; s.conv[L][1] = s.conv[L][leaf]
                for L in ATT:
                    ck, cv = s.kv[L]; s.kv[L] = (torch.cat([ck, s.g_nodek[L][pn]], 0), torch.cat([cv, s.g_nodev[L][pn]], 0))
                ah = s._vh[pn]; s.hctx = ah if s.hctx is None else torch.cat([s.hctx, ah], 0)[-8:]
            return s.step_commit(bonus, pos)
        @torch.inference_mode()
        def prefill(s, prompt):
            lg = None
            for i, t in enumerate(prompt): lg = s.step_commit(int(t), i)
            return lg

    # ---- 4B joint-VAE drafter (stub w/ real untied lm_head) ----
    from chained_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig
    from safetensors.torch import load_file
    from types import SimpleNamespace
    class Emb:
        def __init__(s, w): s.w = w
        def __call__(s, i): return F.embedding(i, s.w)
    class Head(torch.nn.Module):
        def __init__(s, w): super().__init__(); s.weight = w
        def forward(s, h): return h.to(s.weight.dtype) @ s.weight.T
    class SM:
        def __init__(s): s.config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB); s._lm = Head(lmw); s._e = Emb(embed)
        @property
        def lm_head(s): return s._lm
        def get_input_embeddings(s): return s._e
    class Stub:
        def __init__(s): s.model = SM()
        def lm_head(s, h): return s.model.lm_head(h)
    cfgj = json.load(open(f"{CKD}/chained_flow_tree_config.json"))["model_args"]
    dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}})
    drafter = TreeVAEFlowDrafter(Stub(), dcfg).to(dev).to(embed.dtype).eval(); drafter._dtype = embed.dtype
    sd = load_file(f"{CKD}/model.safetensors")
    drafter.load_state_dict({k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}, strict=False)
    class DS:
        def __init__(s, h): s.final_hidden = h.unsqueeze(0)

    # ---- cudagraph the full draft (flow + tree-build); tensor core, validated bit-exact ----
    order_ = dcfg.path_order; Kd_ = dcfg.draft_length
    def draft_core(context, top_b=8, max_nodes=8, max_depth=5):
        pred_hidden = drafter.predict_hidden(context)[0]
        base0 = drafter.lm_head(pred_hidden[:1])[0]
        b0 = min(top_b, base0.shape[-1])
        vals, idx = torch.log_softmax(base0.float(), dim=-1).topk(b0)
        tokens = idx.clone(); parents = torch.full((b0,), -1, dtype=torch.long, device=dev)
        depths = torch.zeros(b0, dtype=torch.long, device=dev); cum = vals.clone()
        frontier = torch.arange(b0, device=dev)
        lastp = torch.full((b0, order_), -1, dtype=torch.long, device=dev); lastp[:, 0] = idx
        for d in range(1, min(Kd_, max_depth + 1)):
            res = drafter._residual_from_lastp(lastp)
            logits = drafter.lm_head(pred_hidden[d].unsqueeze(0) + res) + drafter.markov.bias(lastp[:, 0])
            cvals, cidx = torch.log_softmax(logits.float(), dim=-1).topk(top_b, dim=-1)
            cand_cum = (cum[frontier].unsqueeze(1) + cvals).reshape(-1)
            keep = min(max_nodes, cand_cum.shape[0]); topv, topi = cand_cum.topk(keep)
            src_row = topi // top_b; new_tokens = cidx.reshape(-1)[topi]; new_parents = frontier[src_row]
            start = tokens.shape[0]
            tokens = torch.cat([tokens, new_tokens]); parents = torch.cat([parents, new_parents])
            depths = torch.cat([depths, torch.full((keep,), d, dtype=torch.long, device=dev)]); cum = torch.cat([cum, topv])
            lastp = torch.cat([new_tokens.unsqueeze(1), lastp[src_row][:, :-1]], dim=1)
            frontier = torch.arange(start, start + keep, device=dev)
        return tokens, parents, depths
    draft_state = {"cg": None}
    def setup_draft_cg(sample_ctx):
        cbuf = sample_ctx.clone()
        stx = torch.cuda.Stream(); stx.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stx):
            for _ in range(3): draft_core(cbuf)
        torch.cuda.current_stream().wait_stream(stx)
        from vllm.platforms import current_platform
        _pool = current_platform.get_global_graph_pool()
        gG = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gG, pool=_pool):
            gt, gp, gd = draft_core(cbuf)
        def _cg(newctx):
            cbuf.copy_(newctx); gG.replay()
            return gt.tolist(), gp.tolist(), gd.tolist()
        draft_state["cg"] = _cg
        print("[draft-cudagraph] captured", flush=True)

    cts = CTS()
    def accept_tree(rl, nl, toks, par):
        ch = {}
        for i, p in enumerate(par): ch.setdefault(p, []).append(i)
        g = int(rl.argmax()); cur = -1; acc = []; path = []
        while True:
            m = next((c for c in ch.get(cur, []) if toks[c] == g), None)
            if m is None: break
            acc.append(g); path.append(m); cur = m; g = int(nl[cur].argmax())
        return acc, g, path

    @torch.inference_mode()
    def spec_generate(prompt, mx, free=True):
        cts.kv = {L: None for L in ATT}
        for L in LIN: cts.ssm[L].zero_(); cts.conv[L].zero_()
        cts.hctx = None
        lg = cts.prefill(prompt); out = []; acc = 0; steps = 0; pos = len(prompt)
        while len(out) < mx:
            if cts.hctx is None: tk = []
            else:
                with SDPA():
                    ctx = drafter._context(DS(cts.hctx)).to(drafter._dtype)
                    if draft_state["cg"] is None: setup_draft_cg(ctx)
                    tk, pa, dp = draft_state["cg"](ctx)
            N = len(tk)
            if N == 0 or N > NG:
                nt = int(lg.argmax()); out.append(nt); lg = cts.step_commit(nt, pos); pos += 1
                if nt == tok.eos_token_id: break
                continue
            tkp = tk + [0] * (NG - N); pap = pa + [0] * (NG - N); dpp = dp + [0] * (NG - N)
            nl = cts.verify(tkp, pap, dpp, N); a, b, path = accept_tree(lg, nl, tk, pa)
            out += a + [b]; acc += len(a); steps += 1
            if free:
                pos += len(a); lg = cts.free_commit(path, b, pos); pos += 1
            else:
                for t in a + [b]: lg = cts.step_commit(t, pos); pos += 1
            if tok.eos_token_id in a + [b]: break
        return out[:mx], (acc / steps if steps else 0)

    results = []
    for pi, pids in enumerate(PIDS):
        # AR baseline through the compiled forward (step loop)
        cts.kv = {L: None for L in ATT}
        for L in LIN: cts.ssm[L].zero_(); cts.conv[L].zero_()
        cts.hctx = None
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = cts.prefill(pids); ar = []
        for _ in range(48):
            nt = int(lg.argmax()); ar.append(nt)
            if nt == tok.eos_token_id: break
            lg = cts.step_commit(nt, len(pids) + len(ar) - 1)
        torch.cuda.synchronize(); t_ar = time.perf_counter() - t0
        # spec free=False (lossless) then free=True (timed)
        spF, _ = spec_generate(pids, 48, free=False)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sp, ac = spec_generate(pids, 48, free=True)
        torch.cuda.synchronize(); t_sp = time.perf_counter() - t0
        m = sum(x == y for x, y in zip(VOUT[pi], spF)); n = min(len(VOUT[pi]), len(spF))
        results.append(dict(pi=pi, ar_tps=len(ar) / t_ar, sp_tps=len(sp) / t_sp, acc=ac,
                            lossless_vs_native=f"{m}/{n}", captured=cts.captured))
    return results


res = llm.llm_engine.collective_rpc(run)
if res and isinstance(res[0], list): res = res[0]
print("\n==== COMPILED-FORWARD FULL-CUDAGRAPH RESULTS ====")
import statistics as S
for r in res:
    print(f"  [{r['pi']}] AR {r['ar_tps']:.0f} t/s | spec {r['sp_tps']:.0f} t/s ({r['sp_tps']/r['ar_tps']:.2f}x) "
          f"| accept {r['acc']:.2f} | lossless(vs native) {r['lossless_vs_native']} | cudagraph={r['captured']}")
print(f"\n  MEAN AR {S.mean(r['ar_tps'] for r in res):.0f} t/s | spec {S.mean(r['sp_tps'] for r in res):.0f} t/s "
      f"| SPEEDUP {S.mean(r['sp_tps'] for r in res)/S.mean(r['ar_tps'] for r in res):.2f}x | accept {S.mean(r['acc'] for r in res):.2f}")
print(f"  vLLM-native reference: {sum(len(v) for v in VOUT)/tnat:.0f} tok/s")

# ---- per-domain aggregation (eval-prompt mode) ----
try:
    from collections import defaultdict as _dd
    _sp=_dd(list); _ar=_dd(list); _ac=_dd(list); _ll=_dd(lambda:[0,0])
    for r in res:
        _d = DOMAINS[r['pi']] if r['pi'] < len(DOMAINS) else "?"
        _m,_n = r['lossless_vs_native'].split('/')
        _sp[_d].append(r['sp_tps']); _ar[_d].append(r['ar_tps']); _ac[_d].append(r['acc'])
        _ll[_d][0]+=int(_m); _ll[_d][1]+=int(_n)
    print("\n==== PER-DOMAIN (eval prompts) ====")
    for _d in _sp:
        _sx = S.mean(_sp[_d])/S.mean(_ar[_d])
        print(f"  DOMAIN {_d}: speedup {_sx:.2f}x | accept {S.mean(_ac[_d]):.2f} | lossless {_ll[_d][0]}/{_ll[_d][1]} | n={len(_sp[_d])}")
except Exception as _e:
    print("per-domain agg skipped:", _e)
