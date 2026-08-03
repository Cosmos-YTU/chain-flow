import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import sys; sys.path.insert(0, "/home/shadeform/chained-flow/src")
import time, json, dataclasses, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen3.5-4B"
CKD = "/home/shadeform/chained-flow/out/flow/ckpts/tree-vae-joint-4b-640-k8-l8"
tok = AutoTokenizer.from_pretrained(MODEL)
PROMPTS = [
    "The capital of France is",
    "Q: What is 15 * 23? A: Let's think step by step.",
    "def fibonacci(n):",
    "Write a short paragraph about the ocean.",
]

print("== loading 4B in vLLM (GPU visible=whatever CUDA_VISIBLE_DEVICES set) ==", flush=True)
llm = LLM(model=MODEL, gpu_memory_utilization=0.55, max_model_len=2048, dtype="float16", enforce_eager=False)

# vLLM-native greedy reference (tok/s + reference tokens)
sp = SamplingParams(temperature=0, max_tokens=64)
t0 = time.time(); ref = llm.generate(PROMPTS, sp); t_native = time.time() - t0
ntok = sum(len(o.outputs[0].token_ids) for o in ref)
ref_ids = [list(o.outputs[0].token_ids) for o in ref]
print(f"[native vLLM] {ntok} tok / {t_native:.2f}s = {ntok/t_native:.0f} tok/s", flush=True)


def run(model):
    from chained_flow.vllm_tree.native import NativeTreeSpecDecoder
    from chained_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig
    from safetensors.torch import load_file

    lang = model.language_model
    inner = lang.model
    nlin = sum(1 for l in inner.layers if hasattr(l, "linear_attn"))
    embw = inner.embed_tokens.weight
    lmw = lang.lm_head.weight
    dev, dt = embw.device, embw.dtype
    hidden, vocab = embw.shape[1], embw.shape[0]
    print(f"[struct] layers={len(inner.layers)} linear_attn={nlin} hidden={hidden} vocab={vocab} "
          f"untied_lm_head={lmw.data_ptr()!=embw.data_ptr()}", flush=True)

    # stub frozen_lm exposing embeddings + the REAL untied lm_head
    class Emb:
        def __init__(s, w): s.w = w
        def __call__(s, ids): return s.w[ids]
    class Head(torch.nn.Module):
        def __init__(s, w): super().__init__(); s.weight = w
        def forward(s, h): return h.to(s.weight.dtype) @ s.weight.T
    class Cfg:  hidden_size = hidden; vocab_size = vocab
    class M:
        def __init__(s): s.config = Cfg(); s._lm = Head(lmw); s._emb = Emb(embw)
        @property
        def lm_head(s): return s._lm
        def get_input_embeddings(s): return s._emb
    class Stub:
        def __init__(s): s.model = M()
        def lm_head(s, h): return s.model.lm_head(h)

    cfgj = json.load(open(f"{CKD}/chained_flow_tree_config.json"))["model_args"]
    fields = {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}
    dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in fields})
    drafter = TreeVAEFlowDrafter(Stub(), dcfg).to(dev).to(dt).eval()
    drafter._dtype = dt
    sd = load_file(f"{CKD}/model.safetensors")
    sd = {k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}
    miss, unexp = drafter.load_state_dict(sd, strict=False)
    print(f"[drafter] loaded | missing={len(miss)} unexpected={len(unexp)}", flush=True)

    dec = NativeTreeSpecDecoder(model, drafter, eos_id=tok.eos_token_id)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    def SDPA():  # fresh single-use context each call
        return sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

    results = []
    for pi, ptext in enumerate(PROMPTS):
        pids = tok(ptext)["input_ids"]
        with torch.inference_mode(), SDPA():
            # warm
            dec.generate(pids, 4)
            torch.cuda.synchronize()
            # in-framework autoregressive baseline (plugin's own step loop)
            t0 = time.time()
            st = dec.new_state(); lg = None
            for t in pids: lg = dec.step(int(t), st)
            ar_out = []
            for _ in range(64):
                nt = int(lg.argmax()); ar_out.append(nt)
                if nt == tok.eos_token_id: break
                lg = dec.step(nt, st)
            torch.cuda.synchronize(); t_ar = time.time() - t0
            # spec decode (lossless variant free=False for correctness) + timed free=True
            t0 = time.time(); sp_lossless, acc_l = dec.generate(pids, 64, free=False)
            torch.cuda.synchronize(); t_specF = time.time() - t0
            t0 = time.time(); sp_fast, acc_f = dec.generate(pids, 64, free=True)
            torch.cuda.synchronize(); t_specT = time.time() - t0
        # compare lossless spec vs the plugin's own AR
        m = sum(1 for a, b in zip(sp_lossless, ar_out) if a == b)
        n = min(len(sp_lossless), len(ar_out))
        results.append(dict(pi=pi, ar_tps=len(ar_out)/t_ar, specF_tps=len(sp_lossless)/t_specF,
                            specT_tps=len(sp_fast)/t_specT, acc=acc_f,
                            lossless=f"{m}/{n}", nout=len(sp_fast)))
    return results


res = llm.apply_model(run)
if res and isinstance(res[0], list):  # apply_model returns a per-rank list
    res = res[0]
print("\n==== RESULTS ====")
import statistics as S
for r in res:
    print(f"  [{r['pi']}] AR {r['ar_tps']:.0f} t/s | spec {r['specT_tps']:.0f} t/s "
          f"({r['specT_tps']/r['ar_tps']:.2f}x) | accept {r['acc']:.2f} | lossless(vs AR) {r['lossless']}")
ar = S.mean(r["ar_tps"] for r in res); spt = S.mean(r["specT_tps"] for r in res)
print(f"\n  MEAN in-framework AR {ar:.0f} t/s | spec {spt:.0f} t/s | SPEEDUP {spt/ar:.2f}x | "
      f"mean accept {S.mean(r['acc'] for r in res):.2f}")
print(f"  (vLLM-native reference was {ntok/t_native:.0f} t/s — fully-optimized, not the plugin's eager path)")
