#!/usr/bin/env python3
"""Measure the SAME offline draft data under BOTH acceptance estimators, and write the result.

Why this script exists
----------------------
`scripts/diff_plugin_vs_harness.py` reports acceptance as the mean run length over EVERY token
POSITION of the generation -- a sliding window that scores position i, then i+1, then i+2.

vLLM reports acceptance as `1 + num_accepted_tokens / num_drafts`, where `num_drafts` counts draft
STEPS.  Steps do not land on every position: a step that accepts n tokens commits n+1 and the next
step starts at i + n + 1.  Steps therefore land preferentially where the previous run BROKE.

Those two denominators are not the same number.  Uniform position sampling is length-biased toward
easy stretches: a 50-token run of highly predictable JSON contributes 50 windows each with a large
run length, while the server crosses that same stretch in ~8 steps.  Measured on the 4B Turkish
holdouts the difference is -0.40 mean acceptance, which is 20x larger than the K=8 -> K=5
truncation the cards used to blame it on, and larger than the entire serving overhead.

So: same `_beam_chains` output, same lagged plugin context, two estimators.  The `renewal` column
is the one that is comparable to a served `vllm:spec_decode_*` number.  The `uniform` column is
the one comparable to the offline tables.  Never compare across the two.

  .venv/bin/python scripts/accept_estimator_tr.py --K 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diff_plugin_vs_harness import build_proposer  # noqa: E402


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4btr-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--states", default="teacher_states/bench-4btr-*")
    ap.add_argument("--shortlist", default="out/flow/shortlist_q3527b_tr_ids.pt")
    ap.add_argument("--K", type=int, default=5, help="the SERVED num_speculative_tokens")
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="out/flow/4btr_estimator.json")
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM

    dev, dtype, K = "cuda", torch.float16, args.K
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    p, dcfg = build_proposer(args.ckd, m.get_input_embeddings().weight, m.lm_head.weight,
                             K, args.width, dev, dtype)
    V = m.lm_head.weight.shape[0]
    sl = torch.load(args.shortlist, map_location="cpu").flatten().long()
    sl = sl[(sl >= 0) & (sl < V)].unique().to(dev)
    p._sl = sl
    p._hw = m.lm_head.weight[sl].contiguous()
    p._w2 = p.drafter.markov.w2.weight[sl].contiguous()
    p.feedback = False              # the vLLM constraint: context ends at h(t_{i-1}), seed t_i
    C = p.ctx_size
    print(f"K={K} ctx={C} shortlist={sl.numel()} twopass_M={p.twopass_m}", flush=True)
    print(f"{'domain':<14}{'uniform':>9}{'renewal':>9}{'windows':>9}{'steps':>8}", flush=True)

    doms: dict[str, dict] = {}
    for path in sorted(glob.glob(args.states)):
        dom = os.path.basename(path).split("-", 2)[-1]
        ds = load_from_disk(path)
        su = sn = sr = st = 0
        rows = 0
        per_row: list[int] = []
        for ex in ds:
            ids = ex["input_ids"]
            fh = torch.tensor(ex["final_hidden"], dtype=dtype)
            T = min(len(ids), fh.shape[0])
            start = max(C, int(ex.get("prompt_length", 0)))
            idxs = list(range(start, T - K - 1))
            rows += 1
            per_row.append(len(idxs))
            if not idxs:
                continue
            L: dict[int, int] = {}
            for b0 in range(0, len(idxs), args.batch):
                bb = idxs[b0: b0 + args.batch]
                H = torch.stack([fh[i - C: i + 1] for i in bb]).to(dev)
                k0 = torch.tensor([ids[i] for i in bb], device=dev)
                ctx = torch.stack([p.drafter._context(p._DS(H[j:j + 1, :-1]))[0]
                                   for j in range(H.shape[0])], 0)
                out = p._beam_chains(ctx, k0).tolist()
                for j, i in enumerate(bb):
                    t = ids[i + 1: i + 1 + K]
                    d = 0
                    while d < len(out[j]) and d < len(t) and int(out[j][d]) == int(t[d]):
                        d += 1
                    L[i] = d
            su += sum(L.values())
            sn += len(L)
            # the renewal walk: a step at i commits L[i]+1 tokens, the next step starts after them
            i = idxs[0]
            while i <= idxs[-1]:
                d = L[i]
                sr += d
                st += 1
                i += d + 1
        # How many CONVERSATIONS a 600-window sample actually covers. The published offline
        # tables say "600 windows/domain", which sounds like broad coverage and is not: these
        # rows generate 150-210 scorable windows each, so 600 windows is the first 3-4
        # conversations of a 50-150 row holdout.
        c = r600 = 0
        for k in per_row:
            c += k
            r600 += 1
            if c >= 600:
                break
        doms[dom] = {"uniform": 1 + su / sn, "renewal": 1 + sr / st,
                     "windows": sn, "steps": st, "rows": rows,
                     "windows_per_row": sn / max(rows, 1), "rows_in_first_600_windows": r600}
        print(f"{dom:<14}{doms[dom]['uniform']:>9.3f}{doms[dom]['renewal']:>9.3f}"
              f"{sn:>9d}{st:>8d}", flush=True)

    n = len(doms)
    res = {
        "K": K, "ckd": args.ckd, "model": args.model, "shortlist_rows": int(sl.numel()),
        "arm": "plugin_chain (lagged context, the vLLM constraint)",
        "domains": doms,
        "mean_uniform": sum(v["uniform"] for v in doms.values()) / n,
        "mean_renewal": sum(v["renewal"] for v in doms.values()) / n,
    }
    res["estimator_delta"] = res["mean_renewal"] - res["mean_uniform"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\nmean uniform {res['mean_uniform']:.3f}  mean renewal {res['mean_renewal']:.3f}  "
          f"({res['estimator_delta']:+.3f})\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
