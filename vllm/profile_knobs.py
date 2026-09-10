"""Microbenchmark: does lightening the draft (knob 1 shortlist head, 2 shorter depth, 3 smaller tree)
reduce draft time? 27B-v2 drafter, random embed/head (timing only). Non-destructive — inference knobs."""
import sys; sys.path.insert(0, "/home/shadeform/chain-flow/src")
import json, dataclasses, time, torch, torch.nn.functional as F
from types import SimpleNamespace
from chain_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
dev, dt = "cuda", torch.float16
CKD = "/home/shadeform/chain-flow/out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8"
HIDDEN, VOCAB = 5120, 248320
embed = torch.randn(VOCAB, HIDDEN, device=dev, dtype=dt) * 0.02
class Emb:
    def __init__(s, w): s.w = w
    def __call__(s, i): return F.embedding(i, s.w)
class Head(torch.nn.Module):
    def __init__(s, w): super().__init__(); s.weight = w
    def forward(s, h): return h.to(s.weight.dtype) @ s.weight.T
class SM:
    def __init__(s): s.config = SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB); s._lm = Head(embed); s._e = Emb(embed)
    @property
    def lm_head(s): return s._lm
    def get_input_embeddings(s): return s._e
class Stub:
    def __init__(s): s.model = SM()
    def lm_head(s, h): return s.model.lm_head(h)
cfgj = json.load(open(f"{CKD}/chained_flow_tree_config.json"))["model_args"]
dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}})
drafter = TreeVAEFlowDrafter(Stub(), dcfg).to(dev).to(dt).eval(); drafter._dtype = dt
sd = load_file(f"{CKD}/model.safetensors")
drafter.load_state_dict({k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}, strict=False)
class DS:
    def __init__(s, h): s.final_hidden = h
hctx = torch.randn(1, 8, HIDDEN, device=dev, dtype=dt)
SDPA = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])

def tm(fn, n=50):
    with torch.inference_mode(), SDPA():
        for _ in range(8): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000

print(f"=== 27B-v2 draft knobs (HIDDEN={HIDDEN}, VOCAB={VOCAB}) ===")
# tree-build under knobs 2 (max_depth) and 3 (top_b/max_nodes) — native params
configs = [
    ("BASELINE (topb8, nodes8, depth5)",   dict(top_b=8, max_nodes=8, max_depth=5)),
    ("knob2 shorter depth (depth3)",        dict(top_b=8, max_nodes=8, max_depth=3)),
    ("knob3 smaller tree (topb4, nodes4)",  dict(top_b=4, max_nodes=4, max_depth=5)),
    ("knob2+3 (topb4,nodes4,depth3)",       dict(top_b=4, max_nodes=4, max_depth=3)),
]
base_ms = None
for name, kw in configs:
    ms = tm(lambda: drafter.build_tree_fast(DS(hctx), **kw))
    if base_ms is None: base_ms = ms
    print(f"  {name:38} {ms:6.2f} ms   ({ms/base_ms*100:4.0f}% of baseline)")

# knob 1: head matmul full-vocab vs shortlist, at the batch sizes used in the tree
print("\n=== knob1: lm_head matmul cost (the ~40% of draft that's wasted on full vocab) ===")
for B in [1, 8, 32]:
    h = torch.randn(B, HIDDEN, device=dev, dtype=dt)
    full = tm(lambda: h @ embed.T, n=100)
    for K in [8192, 16384]:
        subw = embed[:K]
        short = tm(lambda: h @ subw.T, n=100)
        if K == 8192:
            print(f"  B={B:2d}: full-vocab {full:6.3f} ms | shortlist-{K} {short:6.3f} ms  ({short/full*100:.0f}%, {full/short:.1f}x cheaper)")
# flow (predict_hidden) — fixed cost, not reduced by these knobs
with torch.inference_mode(), SDPA():
    ctx = drafter._context(DS(hctx))
print(f"\n  predict_hidden (flow, fixed): {tm(lambda: drafter.predict_hidden(ctx)):.2f} ms")
