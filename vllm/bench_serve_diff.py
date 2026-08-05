#!/usr/bin/env python3
"""Diff two bench_serve runs' COMPLETIONS before anyone compares their throughputs.

Two arms that decoded different text did not measure the same work, so nothing downstream of
the divergence is comparable -- not accept, not tok/s (docs/BENCHMARKING.md section 3).

BUT: **a raw mismatch count is not the signal.**  Greedy decoding of an fp16-logit model is
ill-posed -- 0.3-0.9% of tokens sit within 1 ULP of a tie -- so over a 256-token generation a
double-digit percentage of sequences diverge for reasons that have nothing to do with the
drafter.  Under `vllm serve` it is worse than offline, because **vLLM is not batch-invariant**:
measured here, the BASE arm agrees with its own concurrency-1 run on only 60/70 sequences at
concurrency 16, with no speculation anywhere in the picture.

So this prints a RATE, and `--null R` is the rate you measured for a comparison that is known
lossless (base vs base, same concurrency, different process).  Above that rate is a signal;
at or below it is the tie lottery.  With no `--null`, it reports and does not judge.

    bench_serve_diff.py logs/bench_serve/4b_base/serve_bench.json \
                        logs/bench_serve/4b_tree/serve_bench.json [--null 0.15]
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    null = None
    argv = list(sys.argv[1:])
    if "--null" in argv:
        i = argv.index("--null")
        null = float(argv[i + 1])
        del argv[i:i + 2]
    a = json.load(open(argv[0]))
    b = json.load(open(argv[1]))
    bad = 0
    rates = []
    for pa, pb in zip(a["phases"], b["phases"]):
        ta, tb = pa["texts"], pb["texts"]
        if len(ta) != len(tb):
            print(f"  {pa['tag']}: DIFFERENT REQUEST COUNT {len(ta)} vs {len(tb)}")
            bad += 1
            continue
        diff = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        # Where the two texts first differ, in characters -- a divergence at char 900 of 1000
        # is a late fp16 tie; one at char 0 is a different configuration.
        first = None
        if diff:
            x, y = ta[diff[0]], tb[diff[0]]
            first = next((k for k in range(min(len(x), len(y))) if x[k] != y[k]),
                         min(len(x), len(y)))
        rate = len(diff) / max(len(ta), 1)
        rates.append(rate)
        over = null is not None and rate > null
        if over:
            bad += 1
        print(f"  {pa['tag']:>10}: {len(ta) - len(diff)}/{len(ta)} identical"
              + (f"  |  {len(diff)} differ ({100 * rate:.1f}%)" if diff else "")
              + ("  <-- ABOVE THE NULL" if over else ""))
        if diff:
            x, y = ta[diff[0]], tb[diff[0]]
            k = first or 0
            print(f"      req {diff[0]} A: ...{x[max(0, k - 40):k + 40]!r}")
            print(f"      req {diff[0]} B: ...{y[max(0, k - 40):k + 40]!r}")
    mean = sum(rates) / max(len(rates), 1)
    print(f"[diff] {argv[0]}\n[diff] {argv[1]}")
    print(f"[diff] mean divergence {100 * mean:.1f}% of sequences"
          + (f", null {100 * null:.1f}% -> "
             + ("SIGNAL: some phase is worse than the tie lottery" if bad
                else "consistent with the fp16 tie lottery, not with a lossless failure")
             if null is not None else
             " -- pass --null <rate from a known-lossless pair> to judge it"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
