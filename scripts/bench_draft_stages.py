"""Per-stage profile + roofline accounting of ONE draft, in the shipping configuration.

The drafter ships compiled (max-autotune) + captured in a CUDA graph with CF_CUDA_BLOCK=1, so that
-- never eager -- is the baseline every number is quoted against.  For each stage we report GPU ms
(cudagraph-replayed) and the BYTES it must stream, against the measured DRAM read bandwidth, i.e.
what fraction of the memory roofline the stage achieves.

  CUDA_VISIBLE_DEVICES=5 python scripts/bench_draft_stages.py \
      --ckd out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8 --model Qwen/Qwen3.5-4B \
      --shortlist out/flow/shortlist_q3527b.pt --keep 8 --depth 5
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def bench(fn, n=200, warmup=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def graphed(fn, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g.replay, out


def dram_read_bw():
    n = 1 << 28
    x = torch.empty(n, device="cuda", dtype=torch.float16)
    ms = bench(lambda: torch.sum(x), n=20, warmup=5)
    return n * 2 / (ms * 1e-3) / 1e9


def nbytes(*mods_or_tensors):
    tot = 0
    for m in mods_or_tensors:
        if m is None:
            continue
        if isinstance(m, torch.Tensor):
            tot += m.numel() * m.element_size()
        else:
            tot += sum(p.numel() * p.element_size() for p in m.parameters())
    return tot


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--shortlist", default="out/flow/shortlist_q3527b.pt")
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--topb", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--twopass", type=int, default=0, help="CF_TWOPASS_M (0 = off)")
    ap.add_argument("--path_trim", action="store_true", help="CF_PATH_TRIM (bit-exact)")
    ap.add_argument("--shared", action="store_true", help="CF_TWOPASS_SHARED")
    ap.add_argument("--no-cuda-block", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CF_CUDA_BLOCK", "0" if args.no_cuda_block else "1")
    os.environ["CF_TWOPASS_M"] = str(args.twopass)
    os.environ["CF_PATH_TRIM"] = "1" if args.path_trim else "0"
    os.environ["CF_TWOPASS_SHARED"] = "1" if args.shared else "0"

    from diff_plugin_vs_harness import build_proposer

    dev, dtype = "cuda", torch.float16
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    embed_w = m.get_input_embeddings().weight
    lm_w = m.lm_head.weight
    p, dcfg = build_proposer(args.ckd, embed_w, lm_w, args.keep * args.depth, 4, dev, dtype)
    d = p.drafter
    # keep only the two tables the drafter needs; drop the rest of the target model
    keep_embed, keep_lm = embed_w, lm_w
    del m
    torch.cuda.empty_cache()

    V, H = keep_lm.shape
    p.tree_keep, p.tree_topb, p.tree_depth = args.keep, args.topb, args.depth
    p.branching = True
    p.twopass_m = args.twopass
    p.path_trim = args.path_trim
    p.twopass_shared = args.shared
    if args.shortlist:
        sl = torch.load(args.shortlist, map_location="cpu").flatten().long()
        sl = sl[(sl >= 0) & (sl < V)].unique().to(dev)
        p._sl, p._hw = sl, keep_lm[sl].contiguous()
        p._w2 = d.markov.w2.weight[sl].contiguous()
    Vs = p._hw.shape[0]

    # build_proposer wraps integrate/flow_velocity in python counters; those mutate a dict and
    # blow dynamo's recompile limit, silently leaving the flow EAGER. Undo them before compiling.
    for name in ("integrate", "flow_velocity"):
        if name in d.__dict__:
            del d.__dict__[name]

    from chain_flow.vllm_plugin import fused
    fused.fuse_path_head(d)
    fused.compile_flow(d, mode="max-autotune-no-cudagraphs")

    # CONFIRM the fused CUDA block stack actually engages (a shape gate returning None made
    # CF_CUDA_BLOCK a silent no-op at 9B/27B for hours).
    from chain_flow import cuda_block as _cb
    _ex = next(iter(d._chunk_experts()))
    _S = dcfg.chunk_size
    _x = torch.randn(1, _S, dcfg.latent_size, device=dev, dtype=dtype)
    _c = torch.randn(1, p.ctx_size, dcfg.latent_size, device=dev, dtype=dtype)
    _r = None if _ex._cf_fused_off else _ex._cf_fused_runner(_x, _c)
    print(f"[flag] CF_CUDA_BLOCK={os.environ['CF_CUDA_BLOCK']} enabled={_cb.enabled()} "
          f"runner for S={_S},C={p.ctx_size}: {'ENGAGED' if _r is not None else 'NOT engaged'}")

    B, C = args.batch, p.ctx_size
    ctx = torch.randn(B, C, H, device=dev, dtype=dtype)
    k0 = torch.randint(0, 1000, (B,), device=dev)

    bw = dram_read_bw()
    print(f"\nckpt={os.path.basename(args.ckd)} model={args.model} hidden={H} vocab={V} "
          f"shortlist={Vs} latent={dcfg.latent_size}")
    print(f"tree keep={args.keep} depth={args.depth} topb={args.topb} batch={B}  "
          f"CF_CUDA_BLOCK={os.environ['CF_CUDA_BLOCK']} CF_TWOPASS_M={args.twopass} "
          f"CF_PATH_TRIM={int(args.path_trim)}")
    print(f"measured DRAM stream-read bandwidth: {bw:.0f} GB/s\n")

    # ---- byte accounting -------------------------------------------------------------
    nd = min(args.depth, dcfg.draft_length - 1)         # depths actually emitted
    hw_b = p._hw.numel() * 2
    w2_b = p._w2.numel() * 2
    off_b = nbytes(*[d.path_head.offset_proj[j] for j in range(d.config.path_order)])
    mlp_b = nbytes(d.path_head.mlp, d.path_head.norm)
    emb_row = H * 2
    enc_b = nbytes(d.vae.encoder_in, d.vae.encoder, d.vae.encoder_norm, d.vae.mu)
    dec_b = nbytes(d.vae.decoder_in, d.vae.decoder, d.vae.decoder_norm, d.vae.decoder_out)
    from chain_flow import cuda_block
    lay = cuda_block.Layout(dcfg.latent_size, dcfg.ffn_multiplier)
    passes = d.num_chunks * dcfg.num_drafter_layers * dcfg.num_flow_steps
    flow_b = lay.wstride * 2 * passes
    flow_res = lay.wstride * 2 * dcfg.num_drafter_layers * d.num_chunks   # resident weight set

    if args.twopass:
        M = args.twopass
        ng = 1 if args.shared else nd               # candidate-set gathers per draft
        head_b = hw_b + ng * M * H * 2 * 2          # one full pass + gather (read + write)
        mk_b = ng * M * d.markov.w2.weight.shape[1] * 2 * 2
    else:
        head_b = nd * hw_b
        mk_b = nd * w2_b
    # PathHead: with CF_PATH_TRIM depth d only touches d of the `order` offset projections
    off1 = off_b // d.config.path_order
    nproj = sum(min(j, d.config.path_order) for j in range(1, nd + 1)) if args.path_trim \
        else nd * d.config.path_order
    path_b = nproj * off1 + nd * mlp_b
    tot_b = head_b + mk_b + path_b + enc_b + dec_b + flow_b

    # ---- stage timings (each cudagraph-captured) --------------------------------------
    cf = ctx.to(d._dtype)
    lat = d._encode(cf)
    z0 = d.init_latents(lat)
    z = d.integrate(lat, z0)
    pred = d.predict_hidden(cf, d.anchor_embed(k0))
    lastp = torch.zeros(B * args.keep, d.config.path_order, dtype=torch.long, device=dev)

    stages = [
        ("vae encode", lambda: d._encode(cf), enc_b),
        ("flow integrate", lambda: d.integrate(lat, d.init_latents(lat)), flow_b),
        ("vae decode", lambda: d._decode(z), dec_b),
        ("predict_hidden (enc+flow+dec)", lambda: d.predict_hidden(cf, d.anchor_embed(k0)),
         enc_b + flow_b + dec_b),
        ("  x1 path residual (all 8)", lambda: d._residual_from_lastp(lastp), off_b + mlp_b),
        ("  x1 path residual (1 live)", lambda: d._residual_from_lastp(lastp, 1), off1 + mlp_b),
        ("  x1 shortlist head", lambda: p._head(pred[:, 0]), hw_b),
        ("  x1 markov bias", lambda: p._bias(lastp[:, 0]), w2_b),
        ("WHOLE DRAFT (flow+beam)", lambda: p._beam_tree(ctx, k0), tot_b),
    ]
    print(f"{'stage':<32} {'ms':>8} {'MB':>9} {'GB/s':>8} {'%roofline':>10}")
    print("-" * 72)
    results = {}
    for name, fn, by in stages:
        if name in ("vae encode", "flow integrate", "vae decode"):
            fn = torch.compile(fn, mode="max-autotune-no-cudagraphs", dynamic=False)
        rep, _ = graphed(fn)
        ms = bench(rep, n=200)
        results[name] = ms
        ach = by / (ms * 1e-3) / 1e9
        print(f"{name:<32} {ms:>8.3f} {by / 2**20:>9.1f} {ach:>8.0f} {ach / bw * 100:>9.1f}%")

    beam_ms = results["WHOLE DRAFT (flow+beam)"] - results["predict_hidden (enc+flow+dec)"]
    beam_b = head_b + mk_b + path_b
    print(f"{'beam/tree build (by difference)':<32} {beam_ms:>8.3f} {beam_b / 2**20:>9.1f} "
          f"{beam_b / (beam_ms * 1e-3) / 1e9:>8.0f} "
          f"{beam_b / (beam_ms * 1e-3) / 1e9 / bw * 100:>9.1f}%")

    print("-" * 72)
    print(f"traffic itemisation (MB, {nd} emitted depths):")
    print(f"  lm_head shortlist            {head_b / 2**20:9.1f}")
    print(f"  markov w2 shortlist          {mk_b / 2**20:9.1f}")
    print(f"  PathHead offset_proj         {nproj * off1 / 2**20:9.1f}   ({nproj} of "
          f"{nd * d.config.path_order} projections)")
    print(f"  PathHead mlp                 {nd * mlp_b / 2**20:9.1f}")
    print(f"  flow expert weights          {flow_b / 2**20:9.1f}   (resident set "
          f"{flow_res / 2**20:.0f} MB, {passes} block-passes)")
    print(f"  VAE encode                   {enc_b / 2**20:9.1f}")
    print(f"  VAE decode                   {dec_b / 2**20:9.1f}")
    print(f"  TOTAL                        {tot_b / 2**20:9.1f}   roofline "
          f"{tot_b / (bw * 1e9) * 1e3:.3f} ms")


def _whole(p, ctx, k0):
    return p._beam_tree(ctx, k0)


if __name__ == "__main__":
    main()
