import sys; sys.path.insert(0,"/home/shadeform/chain-flow/src")
import json, dataclasses, time, torch, torch.nn.functional as F
from types import SimpleNamespace
from chain_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig
from safetensors.torch import load_file
from torch.nn.attention import SDPBackend, sdpa_kernel
dev="cuda"; dt=torch.float16
CKD="/home/shadeform/chain-flow/out/flow/ckpts/tree-vae-joint-4b-640-k8-l8"
HIDDEN=2560; VOCAB=248320
# stub with random embed/lm_head (profiling timing only; weights don't affect timing)
embed=torch.randn(VOCAB, HIDDEN, device=dev, dtype=dt)*0.02
class Emb:
    def __init__(s,w): s.w=w
    def __call__(s,i): return F.embedding(i,s.w)
class Head(torch.nn.Module):
    def __init__(s,w): super().__init__(); s.weight=w
    def forward(s,h): return h.to(s.weight.dtype)@s.weight.T
class SM:
    def __init__(s): s.config=SimpleNamespace(hidden_size=HIDDEN,vocab_size=VOCAB); s._lm=Head(embed); s._e=Emb(embed)
    @property
    def lm_head(s): return s._lm
    def get_input_embeddings(s): return s._e
class Stub:
    def __init__(s): s.model=SM()
    def lm_head(s,h): return s.model.lm_head(h)
cfgj=json.load(open(f"{CKD}/chained_flow_tree_config.json"))["model_args"]
dcfg=TreeVAEFlowConfig(**{k:v for k,v in cfgj.items() if k in {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}})
drafter=TreeVAEFlowDrafter(Stub(), dcfg).to(dev).to(dt).eval(); drafter._dtype=dt
sd=load_file(f"{CKD}/model.safetensors")
drafter.load_state_dict({k[len("drafter."):]:v for k,v in sd.items() if k.startswith("drafter.")}, strict=False)
class DS:
    def __init__(s,h): s.final_hidden=h
hctx=torch.randn(1,8,HIDDEN,device=dev,dtype=dt)
SDPA=lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH])
def tm(fn,n=30):
    with torch.inference_mode(), SDPA():
        for _ in range(5): fn()
        torch.cuda.synchronize(); t=time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter()-t)/n*1000
with torch.inference_mode(), SDPA():
    ctx=drafter._context(DS(hctx))
print("=== draft component timings (4B joint-VAE) ===")
print(f"  full build_tree_fast:   {tm(lambda: drafter.build_tree_fast(DS(hctx), top_b=8, max_nodes=8, max_depth=5)):.2f} ms")
print(f"  predict_hidden (flow):  {tm(lambda: drafter.predict_hidden(ctx)):.2f} ms")
# _context (encode prep)
print(f"  _context:               {tm(lambda: drafter._context(DS(hctx))):.2f} ms")

# ---- cudagraph the flow pass (predict_hidden) ----
print("\n=== cudagraph predict_hidden ===")
with torch.inference_mode(), SDPA():
    ctx_buf = drafter._context(DS(hctx)).clone()   # static input buffer
    # warmup on a side stream
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): out_ref = drafter.predict_hidden(ctx_buf)
    torch.cuda.current_stream().wait_stream(s)
    eager_out = drafter.predict_hidden(ctx_buf).clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        graph_out = drafter.predict_hidden(ctx_buf)
    def run_cg(newctx):
        ctx_buf.copy_(newctx); g.replay(); return graph_out
    # correctness
    newctx = drafter._context(DS(torch.randn(1,8,HIDDEN,device=dev,dtype=dt)))
    cg = run_cg(newctx).clone()
    eg = drafter.predict_hidden(newctx)
    md = (cg.float()-eg.float()).abs().max().item()
    print(f"  cudagraph vs eager max|diff|: {md:.2e}  (bit-close={md<1e-2})")
    print(f"  predict_hidden eager:     {tm(lambda: drafter.predict_hidden(ctx)):.2f} ms")
    tc = tm(lambda: run_cg(ctx))
    print(f"  predict_hidden cudagraph: {tc:.2f} ms")

# ---- what dominates the tree-build (post-flow) 8.4ms? ----
print("\n=== tree-build breakdown ===")
with torch.inference_mode(), SDPA():
    ph = drafter.predict_hidden(ctx)   # [1, K, hidden]
    K = ph.shape[1]
    # lm_head over the predicted hiddens (per-depth logits) — the big vocab matmul
    print(f"  lm_head(pred_hidden) [{K}x{HIDDEN}->{VOCAB}]: {tm(lambda: drafter.lm_head(ph)):.2f} ms")
    # a single topk over vocab
    lg = drafter.lm_head(ph).float()
    print(f"  topk(8) over vocab x{K}:                 {tm(lambda: lg.topk(8, dim=-1)):.2f} ms")
    # markov head (per prev token)
    prev = torch.zeros(1, K, dtype=torch.long, device=dev)
    print(f"  markov.bias:                             {tm(lambda: drafter.markov.bias(prev)):.2f} ms")

# ---- cudagraph the WHOLE draft (flow + tree-build), tensor core ----
print("\n=== cudagraph the FULL draft (flow + tree-build) ===")
cfg = drafter.config; order, Kd_ = cfg.path_order, cfg.draft_length
def draft_core(context, top_b=8, max_nodes=8, max_depth=5):
    pred_hidden = drafter.predict_hidden(context)[0]
    dev_ = context.device
    base0 = drafter.lm_head(pred_hidden[:1])[0]
    b0 = min(top_b, base0.shape[-1])
    vals, idx = torch.log_softmax(base0.float(), dim=-1).topk(b0)
    tokens = idx.clone(); parents = torch.full((b0,), -1, dtype=torch.long, device=dev_)
    depths = torch.zeros(b0, dtype=torch.long, device=dev_); cum = vals.clone()
    frontier = torch.arange(b0, device=dev_)
    lastp = torch.full((b0, order), -1, dtype=torch.long, device=dev_); lastp[:, 0] = idx
    for d in range(1, min(Kd_, max_depth + 1)):
        res = drafter._residual_from_lastp(lastp)
        logits = drafter.lm_head(pred_hidden[d].unsqueeze(0) + res) + drafter.markov.bias(lastp[:, 0])
        cvals, cidx = torch.log_softmax(logits.float(), dim=-1).topk(top_b, dim=-1)
        cand_cum = (cum[frontier].unsqueeze(1) + cvals).reshape(-1)
        keep = min(max_nodes, cand_cum.shape[0]); topv, topi = cand_cum.topk(keep)
        src_row = topi // top_b; new_tokens = cidx.reshape(-1)[topi]; new_parents = frontier[src_row]
        start = tokens.shape[0]
        tokens = torch.cat([tokens, new_tokens]); parents = torch.cat([parents, new_parents])
        depths = torch.cat([depths, torch.full((keep,), d, dtype=torch.long, device=dev_)]); cum = torch.cat([cum, topv])
        lastp = torch.cat([new_tokens.unsqueeze(1), lastp[src_row][:, :-1]], dim=1)
        frontier = torch.arange(start, start + keep, device=dev_)
    return tokens, parents, depths
with torch.inference_mode(), SDPA():
    cbuf = drafter._context(DS(hctx)).clone()
    st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3): draft_core(cbuf)
    torch.cuda.current_stream().wait_stream(st)
    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2):
        gt, gp, gd = draft_core(cbuf)
    def full_draft_cg(newctx):
        cbuf.copy_(newctx); g2.replay()
        return gt.tolist(), gp.tolist(), gd.tolist()
    # correctness vs eager build_tree_fast tokens
    nc = drafter._context(DS(hctx))
    t_cg, p_cg, d_cg = full_draft_cg(nc)
    tr = drafter.build_tree_fast(DS(hctx), top_b=8, max_nodes=8, max_depth=5)
    match = (list(tr.tokens) == t_cg and list(tr.parents) == p_cg)
    print(f"  full-draft cudagraph tokens==eager: {match} ({len(t_cg)} nodes)")
    print(f"  eager build_tree_fast:  {tm(lambda: drafter.build_tree_fast(DS(hctx), top_b=8, max_nodes=8, max_depth=5)):.2f} ms")
    print(f"  cudagraph full draft:   {tm(lambda: full_draft_cg(nc)):.2f} ms")
