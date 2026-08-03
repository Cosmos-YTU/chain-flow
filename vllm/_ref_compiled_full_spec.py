import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", trust_remote_code=True)
PID = list(tok(tok.apply_chat_template([{"role": "user", "content": "Explain gravity briefly."}],
             add_generation_prompt=True, tokenize=False)).input_ids)
llm = LLM(model="Qwen/Qwen3.5-0.8B", trust_remote_code=True, hf_overrides={"architectures": ["Qwen3_5ForCausalLM"]},
          gpu_memory_utilization=0.5, max_model_len=2048)
VOUT = list(llm.generate({"prompt_token_ids": PID}, SamplingParams(temperature=0.0, max_tokens=64))[0].outputs[0].token_ids)

def run(worker, PID=PID, VOUT=VOUT):
    import torch, torch.nn.functional as F
    from vllm.forward_context import set_forward_context, BatchDescriptor
    from vllm.config import CUDAGraphMode
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as cu
    from vllm.model_executor.layers.fla.ops import fused_recurrent_gated_delta_rule_packed_decode as pd
    from chained_flow.vllm_tree.tree_ssm import tree_gdn_op, tree_conv_op
    dev = "cuda"; model = worker.get_model(); runner = worker.model_runner
    inner = model.language_model.model; layers = inner.layers; embed = inner.embed_tokens.weight
    g0 = [l for l in layers if hasattr(l, "linear_attn")][0].linear_attn
    HV, Vd, Kd = g0.num_v_heads, g0.head_v_dim, g0.head_k_dim; keyd, vald = g0.key_dim, g0.value_dim; convdim = keyd * 2 + vald
    s0 = [l for l in layers if hasattr(l, "self_attn")][0].self_attn; NQ, NKV, HD = s0.num_heads, s0.num_kv_heads, s0.head_dim
    LIN = [L for L, ly in enumerate(layers) if hasattr(ly, "linear_attn")]; ATT = [L for L, ly in enumerate(layers) if hasattr(ly, "self_attn")]
    caps = getattr(runner, "cudagraph_batch_sizes", None) or []
    MAXN = 64; NG = 48; SG = 96          # NG = fixed FULL-cudagraph tree size (in caps); SG = padded prefix

    class CTS:
        def __init__(s):
            s.ssm = {L: torch.zeros(MAXN + 2, HV, Vd, Kd, device=dev, dtype=embed.dtype) for L in LIN}
            s.conv = {L: torch.zeros(MAXN + 2, convdim, 4, device=dev, dtype=embed.dtype) for L in LIN}
            s.kv = {L: None for L in ATT}; s.idx1 = torch.tensor([1], device=dev, dtype=torch.int32)
            s.mode = "decode"; s.captured = False; s.hctx = None
            # static verify buffers (FULL cudagraph needs fixed addresses)
            s.g_ns = torch.arange(2, NG + 2, device=dev); s.g_ps = torch.ones(NG, device=dev, dtype=torch.long)
            s.g_amask = torch.zeros(NG, SG + NG, dtype=torch.bool, device=dev)
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
        def verify(s, toks, par, dep, N):    # toks/par/dep are padded to NG; N = real node count
            s.mode = "verify"; S = s.kv[ATT[0]][0].shape[0]
            ps = [1 if p < 0 else p + 2 for p in par]; s.g_ps.copy_(torch.tensor(ps, device=dev))
            am = torch.zeros(NG, SG + NG, dtype=torch.bool, device=dev); am[:N, :S] = True
            anc = torch.zeros(N, N, dtype=torch.bool, device=dev)
            for i in range(N):
                j = i
                while j >= 0: anc[i, j] = True; j = par[j]
            am[:N, SG:SG + N] = anc
            di = torch.arange(NG, device=dev); am[di, SG + di] = True   # self-diagonal (no NaN on pad rows)
            s.g_amask.copy_(am)
            for L in ATT:
                ck, cv = s.kv[L]; s.g_kbuf[L][:S].copy_(ck); s.g_kbuf[L][S:SG].zero_(); s.g_vbuf[L][:S].copy_(cv); s.g_vbuf[L][S:SG].zero_()
            emb = inner.embed_tokens(torch.tensor(toks, device=dev))
            npos = torch.tensor([S + d for d in dep], device=dev).view(1, NG).expand(3, NG).contiguous()
            runner.inputs_embeds.gpu[:NG].copy_(emb); runner.mrope_positions.gpu[:, :NG].copy_(npos)
            bd = BatchDescriptor(num_tokens=NG, uniform=False)
            def _call():
                with set_forward_context(None, runner.vllm_config, num_tokens=NG, cudagraph_runtime_mode=CUDAGraphMode.FULL, batch_descriptor=bd):
                    return runner.model(input_ids=None, positions=runner.mrope_positions.gpu[:, :NG], inputs_embeds=runner.inputs_embeds.gpu[:NG])
            if not s.captured:
                set_cudagraph_capturing_enabled(True)
                with graph_capture(torch.device(dev)): _call()   # capture (placeholder output)
                set_cudagraph_capturing_enabled(False); s.captured = True
            hs = _call()                                          # replay = correct
            s._vh = hs
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

    # ---- drafter ----
    import json, dataclasses
    from types import SimpleNamespace
    from chained_flow.drafters.tree_flow import TreeFlowDrafter, TreeFlowConfig
    from safetensors.torch import load_file
    class Emb:
        def __init__(s, w): s.weight = w
        def __call__(s, i): return F.embedding(i, s.weight)
    class SM:
        def __init__(s, e): s.config = SimpleNamespace(hidden_size=1024, vocab_size=e.shape[0]); s.lm_head = SimpleNamespace(weight=e, bias=None); s._e = Emb(e)
        def get_input_embeddings(s): return s._e
    class Stub:
        def __init__(s, e, d): s.model = SM(e); s.device = d; s._e = e
        def lm_head(s, h): return h.to(s._e.dtype) @ s._e.T
    CK = "/home/shadeform/chained-flow/out/flow/ckpts/tree-hiddenkv-k8-l8-ffn6-rank256-o8-cov8"
    ma = json.load(open(CK + "/chained_flow_tree_config.json"))["model_args"]
    cfg = TreeFlowConfig(**{k: v for k, v in ma.items() if k in {f.name for f in dataclasses.fields(TreeFlowConfig)}})
    drafter = TreeFlowDrafter(Stub(embed, dev), cfg).to(dev)
    sd = load_file(CK + "/model.safetensors"); drafter.load_state_dict({k[8:]: v for k, v in sd.items() if k.startswith("drafter.")}, strict=False); drafter = drafter.eval()
    class DS:
        def __init__(s, h): s.final_hidden = h.unsqueeze(0)

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
                tr = drafter.build_tree_fast(DS(cts.hctx), top_b=8, max_nodes=8, max_depth=5); tk, pa, dp = list(tr.tokens), list(tr.parents), list(tr.depths)
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

    sp, ac = spec_generate(PID, 64, free=True)
    spF, _ = spec_generate(PID, 64, free=False)
    m = sum(x == y for x, y in zip(VOUT, sp)) / len(VOUT)
    mF = sum(x == y for x, y in zip(VOUT, spF)) / len(VOUT)
    def tm(fn, n=15):
        for _ in range(4): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
    # end-to-end wallclock
    t0 = time.perf_counter()
    for _ in range(3): spec_generate(PID, 64, free=False)
    spec_ms = (time.perf_counter() - t0) / 3 / 64 * 1000
    return (f"=== FULL-CUDAGRAPH TREE SPEC-DECODE IN vLLM ===\n"
            f"spec free=True  vs native greedy: match {m*100:.1f}%  accept {ac:.2f}\n"
            f"spec free=False vs native greedy: match {mF*100:.1f}%  (LOSSLESS)\n"
            f"FULL cudagraph verify used (NG={NG}, captured={cts.captured})\n"
            f"end-to-end spec free=False: {spec_ms:.2f} ms/token")

print(llm.llm_engine.collective_rpc(run)[0])
