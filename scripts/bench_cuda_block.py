"""Benchmark + numerics check for the fused CUDA block kernel (CF_CUDA_BLOCK).

The drafter SHIPS compiled (max-autotune) + captured in a CUDA graph, so that -- not eager --
is the baseline every number here is quoted against.  Reports, for the real checkpoint:

  * the block stack in isolation, flag off vs on, both compiled+cudagraphed
  * integrate() end to end, flag off vs on, both compiled+cudagraphed
  * achieved fraction of the weight-streaming roofline
  * max abs / rel deviation of the fused block stack against the PyTorch one

  python scripts/bench_cuda_block.py --ckd out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8
  python scripts/bench_cuda_block.py --ckd <hf-snapshot-dir> --sweep
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


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
    """Capture `fn` in a CUDA graph and return a replay callable."""
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


def count_kernels(fn, iters=5):
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    ev = prof.key_averages()
    return sum(e.count for e in ev if e.self_device_time_total > 0) / iters


def dram_read_bw():
    """Measured peak streaming-read bandwidth, GB/s -- the roofline denominator."""
    n = 1 << 28  # 512 MB of halves
    x = torch.empty(n, device="cuda", dtype=torch.float16)
    f = lambda: torch.sum(x)
    ms = bench(f, n=20, warmup=5)
    return n * 2 / (ms * 1e-3) / 1e9


def set_flag(experts, on: bool, drafter=None, pair: bool = False, batch: bool = True):
    os.environ["CF_CUDA_BLOCK"] = "1" if on else "0"
    os.environ["CF_CUDA_PAIR"] = "1" if pair else "0"
    os.environ["CF_CUDA_BLOCK_BATCH"] = "1" if batch else "0"
    for ex in experts:
        ex.__dict__.pop("_cf_fused_off_cached", None)
        ex.__dict__.pop("_cf_batch_off_cached", None)
        ex.__dict__.pop("_cf_static_ok", None)
    if drafter is not None:
        drafter.__dict__.pop("_cf_pair_off_cached", None)


def batch_ladder(d, experts, D, C, L, passes, wbytes, bw, dcfg, mode, batches):
    """us/block-pass and integrate() across a batch ladder, all compiled+cudagraphed.

    THE POINT OF THE WHOLE EXERCISE.  The fused kernel used to be gated on ``x.shape[0] == 1``,
    so every rung but the first measured the cutlass/PyTorch fallback.  Both arms are quoted
    per SLICE (divided by B) so the rungs are directly comparable: a flat per-slice number means
    batching is free, which is what the shared weight stream should buy."""
    import torch as _t

    # Measure the KERNEL, not the economics gate: `cuda_block.batch_limit()` exists to hand
    # batches above the crossover back to cutlass, and leaving it on here would print the
    # fallback's numbers in the fused column and hide the crossover we are trying to locate.
    os.environ["CF_CUDA_BLOCK_MAXB"] = "0"
    dev = "cuda"
    S = dcfg.chunk_size * 2 if d.num_chunks > 1 else dcfg.chunk_size
    ex0 = experts[0]
    from chain_flow import cuda_block

    mask = _t.triu(_t.full((S, S), float("-inf"), device=dev), diagonal=1)
    print(f"\nbatch ladder, block stack in isolation (S={S}, C={C}, {L} blocks), "
          f"us/block-pass PER SLICE:")
    print(f"{'B':>4} {'G/slice':>8} {'PyTorch':>9} {'fused':>9} {'speedup':>8} "
          f"{'fused total':>12} {'roofline%':>10}")
    fb = cuda_block.attach(ex0)
    for B in batches:
        x0 = _t.randn(B, S, D, device=dev, dtype=_t.float16)
        ctxb = _t.randn(B, C, D, device=dev, dtype=_t.float16)

        def py_stack():
            h = x0
            for b in ex0.blocks:
                h = b(h, ctxb, attn_mask=mask)
            return h

        rep, _ = graphed(torch.compile(py_stack, mode=mode, dynamic=False))
        t_py = bench(rep, n=100) / L * 1000 / B
        if not fb.can_run(S, C, B):
            print(f"{B:>4} {'--':>8} {t_py:9.2f} {'REFUSED':>9} {'':>8} {'':>12} {'':>10}")
            continue

        def run_fused():
            fb._kv_cache = None
            return fb.forward(x0, ctxb, mask)

        rep_f, _ = graphed(run_fused)
        tot = bench(rep_f, n=100) / L * 1000
        t_f = tot / B
        print(f"{B:>4} {fb.grid_for(S, C, B):>8} {t_py:9.2f} {t_f:9.2f} {t_py / t_f:7.2f}x "
              f"{tot:11.1f}  {wbytes / (bw * 1e9) * 1e6 / t_f * 100:9.1f}")

    print("\nbatch ladder, integrate() end to end (compiled + cudagraphed), ms:")
    print(f"{'B':>4} {'PyTorch':>9} {'fused':>9} {'+pair':>9} {'best x':>8} "
          f"{'us/pass/slice':>14} {'rel_l2':>10}")
    for B in batches:
        ctxb = _t.randn(B, C, D, device=dev, dtype=_t.float16)
        z0 = d.init_latents(ctxb)
        row, ref = [], None
        for on, pair in ((False, False), (True, False), (True, True)):
            set_flag(experts, on, d, pair)
            for ex in experts:
                ex.__dict__.pop("_cf_fused", None)
            cuda_block._PAIR_GRIDS.clear()
            try:
                rep2, out = graphed(torch.compile(lambda: d.integrate(ctxb, z0), mode=mode,
                                                  dynamic=False))
                row.append(bench(rep2, n=100))
                rep2()
                _t.cuda.synchronize()
                cur = out.float().clone()
                ref = cur if ref is None else ref
                if on and pair:
                    rl2 = float((cur - ref).norm() / ref.norm())
            except Exception as e:                                   # noqa: BLE001
                row.append(float("nan"))
                print(f"    B={B} on={on} pair={pair}: {type(e).__name__}: {e}")
        best = min(v for v in row[1:] if v == v)
        print(f"{B:>4} {row[0]:9.3f} {row[1]:9.3f} {row[2]:9.3f} {row[0] / best:7.2f}x "
              f"{best / passes * 1000 / B:13.2f} {rl2:10.3e}")


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--mode", default="max-autotune-no-cudagraphs")
    ap.add_argument("--sweep", action="store_true", help="sweep grid size / threads on the stack")
    ap.add_argument("--pair-sweep", action="store_true", help="sweep CF_CUDA_PAIR_G on integrate()")
    ap.add_argument("--no-numerics", action="store_true")
    ap.add_argument("--batches", default="", help="e.g. 1,2,4,8,16,32,64: run the batch ladder")
    ap.add_argument("--batch-only", action="store_true", help="skip the batch-1 sections")
    args = ap.parse_args()

    import dataclasses
    import json

    from safetensors.torch import load_file

    from chain_flow import cuda_block
    from chain_flow.drafters.tree_vae_flow import TreeVAEFlowConfig, TreeVAEFlowDrafter

    dev, dtype = "cuda", torch.float16
    cfgj = json.load(open(f"{args.ckd}/chained_flow_tree_config.json"))["model_args"]
    fields = {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}
    dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in fields})

    d = TreeVAEFlowDrafter.__new__(TreeVAEFlowDrafter)
    torch.nn.Module.__init__(d)
    d.frozen_lm = None
    d.config = dcfg
    d.hidden_size = cfgj["expert_dim"]
    d.latent_size = dcfg.latent_size
    d.num_chunks = dcfg.draft_length // dcfg.chunk_size
    from chain_flow.drafters.chunked_flow import HiddenKVFlowExpert

    def mk():
        return HiddenKVFlowExpert(
            hidden_size=dcfg.latent_size, context_size=dcfg.context_size,
            draft_length=dcfg.draft_length, chunk_size=dcfg.chunk_size,
            expert_dim=dcfg.latent_size, num_heads=dcfg.num_heads,
            ffn_multiplier=dcfg.ffn_multiplier, num_layers=dcfg.num_drafter_layers,
        )

    d.expert = mk()
    d.extra_experts = torch.nn.ModuleList([mk() for _ in range(d.num_chunks - 1)])
    d.anchor_in = None
    d._dtype = dtype
    sd = load_file(f"{args.ckd}/model.safetensors")
    sub = {k[len("drafter."):]: v for k, v in sd.items()
           if k.startswith("drafter.") and (".expert." in k or k.startswith("drafter.extra_experts"))}
    d.load_state_dict(sub, strict=False)
    d = d.to(dev).to(dtype).eval()
    d._dtype = dtype

    D = dcfg.latent_size
    FM = dcfg.ffn_multiplier
    L = dcfg.num_drafter_layers
    C = dcfg.context_size
    lay = cuda_block.Layout(D, FM)
    passes = d.num_chunks * L * dcfg.num_flow_steps
    wbytes = lay.wstride * 2
    experts = list(d._chunk_experts())
    print(f"ckpt={os.path.basename(args.ckd)} D={D} heads={dcfg.num_heads} ffn=x{FM} layers={L} "
          f"chunks={d.num_chunks} chunk_size={dcfg.chunk_size} draft_len={dcfg.draft_length} "
          f"ctx={C} flow_steps={dcfg.num_flow_steps} -> {passes} block-passes/draft")
    print(f"weights/block-pass = {wbytes / 2**20:.2f} MiB   whole draft = "
          f"{wbytes * passes / 2**20:.0f} MiB   resident set = "
          f"{wbytes * L * len(experts) / 2**20:.0f} MiB")
    bw = dram_read_bw()
    print(f"measured stream-read bandwidth: {bw:.0f} GB/s -> roofline "
          f"{wbytes / (bw * 1e9) * 1e6:.2f} us/block-pass")

    ctx = torch.randn(1, C, D, device=dev, dtype=dtype)
    z0 = d.init_latents(ctx)

    if args.batch_only:
        batch_ladder(d, experts, D, C, L, passes, wbytes, bw, dcfg, args.mode,
                     [int(v) for v in args.batches.split(",")])
        return

    # ---- numerics: fused block stack vs PyTorch block stack, per expert / S / mask -------
    if not args.no_numerics:
        set_flag(experts, True)
        print("\nnumerics (fused block stack vs PyTorch, per expert):")
        print(f"{'expert':>7} {'S':>3} {'mask':>5} {'max_abs':>10} {'mean_abs':>10} {'rel_l2':>9} "
              f"{'|ref|max':>9}")
        for ei, ex in enumerate(experts):
            fb = cuda_block.attach(ex)
            for S in cuda_block.SUPPORTED_S:
                for use_mask in (False, True):
                    x = torch.randn(1, S, D, device=dev, dtype=dtype)
                    m = (torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
                         if use_mask else None)
                    r = x
                    for b in ex.blocks:
                        r = b(r, ctx, attn_mask=m)
                    fb._kv_cache = None
                    o = fb.forward(x, ctx, m)
                    dd = (o.float() - r.float()).abs()
                    rl2 = dd.norm() / r.float().norm()
                    print(f"{ei:>7} {S:>3} {str(use_mask):>5} {dd.max():>10.3e} {dd.mean():>10.3e} "
                          f"{rl2:>9.3e} {r.float().abs().max():>9.3f}")

    # ---- isolated block stack: compiled+cudagraphed PyTorch vs the fused kernel ----------
    S = dcfg.chunk_size * 2 if d.num_chunks > 1 else dcfg.chunk_size
    ex0 = experts[0]
    mask = torch.triu(torch.full((S, S), float("-inf"), device=dev), diagonal=1)
    x0 = torch.randn(1, S, D, device=dev, dtype=dtype)

    def py_stack():
        h = x0
        for b in ex0.blocks:
            h = b(h, ctx, attn_mask=mask)
        return h

    print(f"\nblock stack in isolation (S={S}, C={C}, {L} blocks):")
    cpy = torch.compile(py_stack, mode=args.mode, dynamic=False)
    rep, _ = graphed(cpy)
    t_py = bench(rep) / L * 1000
    print(f"  {'PyTorch compiled+cudagraph':>30}: {t_py:7.2f} us/block-pass  "
          f"({wbytes / (bw * 1e9) * 1e6 / t_py * 100:5.1f}% of roofline)")

    set_flag(experts, True)
    fb = cuda_block.attach(ex0)
    fb._kv_cache = None
    if not fb.can_run(S, C):
        print("  fused kernel: UNSUPPORTED for this shape -> falls back to PyTorch")
        return

    def run_fused():
        fb._kv_cache = None
        return fb.forward(x0, ctx, mask)

    rep_f, _ = graphed(run_fused)
    t_f = bench(rep_f) / L * 1000
    print(f"  {'CUDA fused block cudagraph':>30}: {t_f:7.2f} us/block-pass  "
          f"({wbytes / (bw * 1e9) * 1e6 / t_f * 100:5.1f}% of roofline)   "
          f"speedup {t_py / t_f:.2f}x")

    if args.sweep:
        print("\n  sweep (G x threads), us/block-pass:")
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        best = (1e9, None)
        for th in (128, 256, 512):
            row = []
            for G in (32, 48, 64, 96, 128, 160, sm):
                if D % (th // 32):
                    row.append("   n/a")
                    continue
                os.environ["CF_CUDA_BLOCK_T"] = str(th)
                os.environ["CF_CUDA_BLOCK_G"] = str(G)
                ex0.__dict__.pop("_cf_fused", None)
                f2 = cuda_block.attach(ex0)
                if not f2.can_run(S, C):
                    row.append("   n/a")
                    continue

                def r2():
                    f2._kv_cache = None
                    return f2.forward(x0, ctx, mask)

                rp, _ = graphed(r2)
                v = bench(rp, n=100) / L * 1000
                row.append(f"{v:6.2f}")
                if v < best[0]:
                    best = (v, (G, th))
            print(f"    t={th:4d}: " + " ".join(f"G={g}:{v}" for g, v in
                                                zip((32, 48, 64, 96, 128, 160, sm), row)))
        print(f"    best: {best[0]:.2f} us/block-pass at G={best[1][0]} threads={best[1][1]} "
              f"({wbytes / (bw * 1e9) * 1e6 / best[0] * 100:.1f}% of roofline)")
        os.environ.pop("CF_CUDA_BLOCK_T", None)
        os.environ.pop("CF_CUDA_BLOCK_G", None)
        ex0.__dict__.pop("_cf_fused", None)

    # ---- integrate() end to end ---------------------------------------------------------
    # CF_CUDA_PAIR runs the two chunk-experts concurrently on two streams.  It changes the grid
    # size of each expert's kernel, so its atomicAdd order (and therefore its fp16 rounding)
    # differs from the serial path -- deviation against the PyTorch reference is the check that
    # matters, not bit-exactness against the serial fused path.
    print("\nintegrate() (compiled + cudagraphed, the shipping configuration):")
    ref = None
    for on, pair in ((False, False), (True, False), (True, True)):
        set_flag(experts, on, d, pair)
        for ex in experts:
            ex.__dict__.pop("_cf_fused", None)
        f = lambda: d.integrate(ctx, z0)
        cf = torch.compile(f, mode=args.mode, dynamic=False)
        rep2, out = graphed(cf)
        ms = bench(rep2)
        nk = count_kernels(rep2)
        name = ("CUDA fused" + (" +pair" if pair else "")) if on else "PyTorch"
        rep2()
        torch.cuda.synchronize()
        cur = out.float().clone()
        if ref is None:
            ref, dev_s = cur, ""
        else:
            dev_s = (f"   vs PyTorch: max_abs {(cur - ref).abs().max():.3e} "
                     f"rel_l2 {(cur - ref).norm() / ref.norm():.3e}")
        print(f"  {name:>17}: {ms:7.3f} ms   {ms / passes * 1000:6.2f} us/block-pass   "
              f"{nk:6.0f} kernels/replay{dev_s}")

    if args.pair_sweep:
        # The two grids share one machine, so only the SPLIT is free -- the total is pinned by
        # the resident-block cap (a bigger pair deadlocks or, benignly, serialises).
        print("\n  CF_CUDA_PAIR_G sweep on integrate(), ms:")
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        set_flag(experts, True, d, True)
        best = (1e9, None)
        for f0 in (0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.60):
            g0 = int(sm * f0)
            os.environ["CF_CUDA_PAIR_G"] = f"{g0},{sm - g0}"
            for ex in experts:
                ex.__dict__.pop("_cf_fused", None)
            cuda_block._PAIR_GRIDS.clear()
            cf = torch.compile(lambda: d.integrate(ctx, z0), mode=args.mode, dynamic=False)
            rep3, _ = graphed(cf)
            v = bench(rep3)
            print(f"    G=({g0:3d},{sm - g0:3d}): {v:7.3f} ms")
            if v < best[0]:
                best = (v, (g0, sm - g0))
        print(f"    best {best[1]} {best[0]:.3f} ms")
        os.environ.pop("CF_CUDA_PAIR_G", None)
        cuda_block._PAIR_GRIDS.clear()

    if args.batches:
        batch_ladder(d, experts, D, C, L, passes, wbytes, bw, dcfg, args.mode,
                     [int(v) for v in args.batches.split(",")])


if __name__ == "__main__":
    main()
