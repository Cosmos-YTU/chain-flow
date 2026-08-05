"""Kernel-level profile of the draft path — where do the ~971 kernels actually go?

We know: not launch-bound (cudagraph 6.19 -> 4.70 ms), not tuning-bound (fullgraph=True is
byte-identical), ~16x off the memory roofline. So the floor is kernel COUNT x per-kernel latency.
This attributes both to the stages that produce them.

Profiles EAGER on purpose: inside a cudagraph replay the whole draft collapses to one node and the
per-kernel structure is invisible. Graph replay time is measured separately for reference.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_plugin_vs_harness import build_proposer  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.float16
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    p, dcfg = build_proposer(args.ckd, m.get_input_embeddings().weight, m.lm_head.weight,
                             8, 4, dev, dtype)
    del m
    torch.cuda.empty_cache()
    d = p.drafter
    B, C, H = args.batch, p.ctx_size, d.hidden_size
    print(f"ckpt={os.path.basename(args.ckd)} B={B} ctx={C} hidden={H} latent={dcfg.latent_size} "
          f"layers={dcfg.num_drafter_layers} flow_steps={dcfg.num_flow_steps} "
          f"chunks={dcfg.draft_length // dcfg.chunk_size}", flush=True)

    hist = torch.randn(B, C, H, device=dev, dtype=dtype)
    k0 = torch.randint(0, 1000, (B,), device=dev)

    def one_draft():
        with record_function("1_context"):
            ctx = d._context(p._DS(hist))
        with record_function("2_vae_encode"):
            ctx_lat = d._encode(ctx.to(d._dtype))
        with record_function("3_init_latents"):
            z0 = d.init_latents(ctx_lat)
        with record_function("4_flow_integrate"):
            z = d.integrate(ctx_lat, z0)
        with record_function("5_vae_decode"):
            pred = d._decode(z)
        with record_function("6_beam_tree"):
            _ = p._beam_chains(d._context(p._DS(hist)), k0)
        return pred

    for _ in range(5):
        one_draft()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        one_draft()
    torch.cuda.synchronize()
    eager_ms = (time.perf_counter() - t0) / args.iters * 1000
    print(f"\neager draft: {eager_ms:.2f} ms/iter", flush=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        for _ in range(args.iters):
            one_draft()
        torch.cuda.synchronize()

    ev = prof.key_averages()
    tot_cuda = sum(e.self_device_time_total for e in ev) / args.iters / 1000.0
    n_launch = sum(e.count for e in ev if e.self_device_time_total > 0) / args.iters
    print(f"total self-CUDA {tot_cuda:.2f} ms/iter across ~{n_launch:.0f} kernel launches/iter")

    # per-stage attribution from the record_function ranges
    print(f"\n{'stage':<18} {'CUDA ms':>9} {'kernels':>9} {'us/kernel':>10}")
    print("-" * 50)
    stages = defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.key.startswith(("1_", "2_", "3_", "4_", "5_", "6_")):
            stages[e.key] = [e.device_time_total / args.iters / 1000.0, 0]
    for k in sorted(stages):
        print(f"{k:<18} {stages[k][0]:9.3f}")

    print(f"\ntop kernels by self-CUDA time (per iter):")
    print(f"{'kernel':<58} {'ms':>8} {'count':>7} {'us/call':>9}")
    print("-" * 86)
    rows = sorted([e for e in ev if e.self_device_time_total > 0],
                  key=lambda e: -e.self_device_time_total)[:18]
    for e in rows:
        ms = e.self_device_time_total / args.iters / 1000.0
        cnt = e.count / args.iters
        print(f"{e.key[:57]:<58} {ms:8.3f} {cnt:7.1f} {ms * 1000 / max(cnt, 1e-9):9.2f}")

    tiny = [e for e in ev if e.self_device_time_total > 0
            and (e.self_device_time_total / max(e.count, 1)) < 8.0]
    tiny_ms = sum(e.self_device_time_total for e in tiny) / args.iters / 1000.0
    tiny_n = sum(e.count for e in tiny) / args.iters
    print(f"\nkernels averaging <8 us: {tiny_n:.0f} launches, {tiny_ms:.2f} ms/iter "
          f"({100 * tiny_ms / max(tot_cuda, 1e-9):.0f}% of CUDA time) "
          f"-- this is the fusion headroom")


if __name__ == "__main__":
    main()
