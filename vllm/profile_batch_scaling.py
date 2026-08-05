#!/usr/bin/env python3
"""PHASE 3 EVIDENCE: what does the draft cost as the DECODE BATCH grows, and what does the
batch-1-only fused block kernel actually leave on the table under continuous batching?

`chunked_flow._cf_fused_runner` refuses any `x.shape[0] != 1`, so under `vllm serve` every
step with more than one decoding request runs the PyTorch block stack instead.  The
`[cf-defaults]` line still says `cuda_block(...)` because its gate is a BUILD-TIME shape
check.  This measures the difference directly, on the real drafter, with no engine around it:

    for B in 1,2,4,8,16,32:  time drafter.predict_hidden(context[B])   with CF_CUDA_BLOCK on/off

Three numbers come out of it and each answers a different Phase-3 question:
  * B=1 fused vs eager      -- the win the kernel is worth (and which serve loses at B>1).
  * eager(B)/eager(1)       -- is the PyTorch path amortising the batch at all?
  * fused(1)*B vs eager(B)  -- the crude ceiling for a batched kernel that simply ran B
                               independent grids; the real ceiling is better, because the
                               kernel is weight-bandwidth-bound and B drafts share weights.

    profile_batch_scaling.py --size 4b [--batches 1,2,4,8,16,32]

SCOPE, so the numbers are not over-read: this runs the drafter EAGER -- no `CF_COMPILE`
wrap and no cudagraph, because there is no engine here to capture into.  So the absolute
milliseconds are well above the in-engine draft, and the `B=1 fused vs eager` ratio
OVERSTATES what the kernel is worth end-to-end (the in-engine A/B, `CF_CUDA_BLOCK=0` against
a live 4B chain server, says +10.7%).  What the numbers are good for is the SHAPE: whether
the PyTorch path amortises the batch, and by how much the fused path's batch-1 specialisation
costs at each B.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, "/home/shadeform/chained-flow/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from types import SimpleNamespace  # noqa: E402

CKPTS = {
    "4b": ("selimaktas--Flow-Drafter-4B-v2", 2560),
    "9b": ("selimaktas--Flow-Drafter-9B-v2", 4096),
    "27b": ("selimaktas--Flow-Drafter-Qwen3.5-27B-v2", 5120),
}
VOCAB = 248320


def _snapshot(repo: str) -> str:
    import glob
    g = glob.glob(f"/root/.cache/huggingface/hub/models--{repo}/snapshots/*")
    if not g:
        raise SystemExit(f"no local snapshot for {repo}")
    return g[0]


def build(size: str, dev="cuda", dt=torch.float16):
    from chained_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig
    from safetensors.torch import load_file

    repo, hidden = CKPTS[size]
    ckd = _snapshot(repo)
    # Random embed/head: this measures the FLOW NET, and the head is a separate (already
    # shortlisted) GEMM whose cost is not what the fused kernel touches.
    embed = torch.randn(VOCAB, hidden, device=dev, dtype=dt) * 0.02

    class Head(torch.nn.Module):
        def __init__(s, w):
            super().__init__()
            s.weight = w

        def forward(s, h):
            return h.to(s.weight.dtype) @ s.weight.T

    class SM:
        def __init__(s):
            s.config = SimpleNamespace(hidden_size=hidden, vocab_size=VOCAB)
            s._lm = Head(embed)
            s._e = SimpleNamespace(__call__=lambda i: F.embedding(i, embed))

        @property
        def lm_head(s):
            return s._lm

        def get_input_embeddings(s):
            return lambda i: F.embedding(i, embed)

    class Stub:
        def __init__(s):
            s.model = SM()

        def lm_head(s, h):
            return s.model.lm_head(h)

    cfgj = json.load(open(f"{ckd}/chained_flow_tree_config.json"))["model_args"]
    fields = {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}
    cfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in fields})
    d = TreeVAEFlowDrafter(Stub(), cfg).to(dev).to(dt).eval()
    d._dtype = dt
    sd = load_file(f"{ckd}/model.safetensors")
    d.load_state_dict({k[len("drafter."):]: v for k, v in sd.items()
                       if k.startswith("drafter.")}, strict=False)
    return d, hidden, cfg


def tm(fn, n=30, warm=8):
    with torch.inference_mode():
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b", choices=list(CKPTS))
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    args = ap.parse_args()
    Bs = [int(x) for x in args.batches.split(",")]

    from chained_flow import cuda_block

    d, hidden, cfg = build(args.size)
    ctx_size = getattr(cfg, "context_size", 8)
    D = d._chunk_experts()[0].hidden_size if hasattr(d, "_chunk_experts") else None
    ok, err = cuda_block.available()
    print(f"=== {args.size.upper()} drafter: hidden={hidden} expert D={D} ctx={ctx_size} "
          f"| extension built={ok} {err or ''}")

    rows = []
    for B in Bs:
        ctx = torch.randn(B, ctx_size, hidden, device="cuda", dtype=torch.float16)
        res = {}
        for mode in ("1", "0"):
            os.environ["CF_CUDA_BLOCK"] = mode
            # `_cf_fused_off` is memoised per expert; clear it so the flag actually flips.
            for ex in d._chunk_experts():
                ex.__dict__.pop("_cf_fused_off_cached", None)
            try:
                res[mode] = tm(lambda: d.predict_hidden(ctx))
            except Exception as e:                                # noqa: BLE001
                res[mode] = float("nan")
                print(f"  B={B} mode={mode}: {e!r}")
        rows.append((B, res["1"], res["0"]))
        print(f"  B={B:<3} fused-requested {res['1']:7.3f} ms   eager {res['0']:7.3f} ms   "
              f"ratio {res['0'] / res['1']:5.2f}x   per-request {res['1'] / B:6.3f} ms")
    os.environ["CF_CUDA_BLOCK"] = "1"

    f1, e1 = rows[0][1], rows[0][2]
    print(f"\n  fused win at B=1: {e1:.3f} -> {f1:.3f} ms  ({e1 / f1:.2f}x)")
    print("  B    served-by  ms      vs B=1 fused x B   (a batched kernel that merely ran "
          "B independent grids would land at the second column)")
    for B, on, off in rows:
        served = "FUSED" if B == 1 else "eager"
        print(f"  {B:<4} {served:<9} {on:7.3f}   {f1 * B:7.3f}   "
              f"{'LOSS ' + format(on - f1 * B, '+.3f') + ' ms/step' if B > 1 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
