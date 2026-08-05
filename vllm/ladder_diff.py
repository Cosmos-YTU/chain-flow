#!/usr/bin/env python3
"""Token-level agreement between two ladders, against a NULL measured the same way.

    ladder_diff.py 4b base_lad6 chain_cut4 --null base_lad base_lad6

`--null A B` names two runs whose difference is known to be *nothing but noise* -- two BASE
servers, no speculation in either, same config, different processes. That rate is the floor:
greedy decoding of an fp16-logit model is ill-posed at ~0.3-0.9% of tokens, and vLLM is not
batch-invariant, so at concurrency the same engine disagrees with itself. A raw mismatch count
without that floor cannot distinguish "the drafter broke something" from "the batch was
different", and this repo has been wrong that way before (docs/BENCHMARKING.md section 3).

Prints per concurrency, because the null is not flat: at 4B it is 0% up to concurrency 8 and
only becomes nonzero at 16, so a number that looks harmless at 64 would be damning at 1.

CHOOSE THE NULL TO MATCH THE COMPARISON.  A base-vs-base pair at the SAME concurrency is a
*batch-shape-preserving* null, and it is the right floor only when the arm under test also
preserves the batch shape.  A speculative arm does not: it hands the target `K+1` query
positions per request instead of 1, which is different kernels, which is a different fp16 tie
lottery -- so base-vs-chain sits above a same-concurrency base null (measured: 11-17% against
0-8.6%) with nothing wrong.  For a spec arm the honest null is base-vs-base ACROSS concurrency
(docs/BENCHMARKING.md).

The same-concurrency null IS the right floor for the two comparisons the batch cutoff needs:

  * cutoff arm vs uncut spec arm BELOW the cutoff -- must be ~0%, it is the same code path;
  * cutoff arm vs BASE arm ABOVE the cutoff -- with speculation fully off, the cutoff arm is
    doing exactly what the base arm is doing, so it should sit inside the base-vs-base band.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/shadeform/chained-flow/logs/bench_serve"


def texts(size: str, tag: str, c: int) -> list[str] | None:
    p = os.path.join(ROOT, f"{size}_{tag}", f"c{c}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["phases"][0]["texts"]


def rate(a, b):
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    if not n:
        return None
    d = sum(1 for i in range(n) if a[i] != b[i])
    return d, n, 100.0 * d / n


CONCS = [1, 2, 4, 8, 16, 32, 64]


def null_band(size: str, base_tag: str) -> tuple[float, float, int]:
    """The batch-shape-changing null: ONE base run against ITSELF at every other concurrency.

    Self-contained on purpose -- it needs no second base server, and it varies exactly the thing
    a speculative (or de-speculating) arm varies, which is the batch shape the target forward
    sees.  A same-concurrency base-vs-base pair is a tighter but WRONG floor here: measured at
    4B it is 0.0% up to concurrency 8, which would call any kernel change a regression.
    """
    import itertools
    vals = []
    for a, b in itertools.combinations(CONCS, 2):
        r = rate(texts(size, base_tag, a), texts(size, base_tag, b))
        if r is not None:
            vals.append(r[2])
    return (min(vals), max(vals), len(vals)) if vals else (0.0, 0.0, 0)


def main() -> int:
    argv = list(sys.argv[1:])
    null_tag = None
    if "--null" in argv:
        i = argv.index("--null")
        null_tag = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) < 3:
        print(__doc__)
        return 2
    size, a, b = argv[0], argv[1], argv[2]

    lo = hi = None
    if null_tag:
        lo, hi, npairs = null_band(size, null_tag)
        print(f"null band, {null_tag} vs itself across concurrency ({npairs} pairs): "
              f"{lo:.1f}% .. {hi:.1f}%")
    print(f"{size.upper()}  {a}  vs  {b}")
    verdicts = []
    for c in CONCS:
        r = rate(texts(size, a, c), texts(size, b, c))
        if r is None:
            continue
        line = f"  c={c:<3} {r[0]:>4}/{r[1]:<4} = {r[2]:5.1f}%"
        if hi is not None:
            ok = r[2] <= hi
            line += f"   {'inside the null band' if ok else 'ABOVE THE NULL BAND'}"
            verdicts.append(ok)
        print(line)
    if verdicts:
        print("\n" + ("every level is inside the null band -- no evidence of a correctness "
                      "failure, and equally, n this small could not have found a small one"
                      if all(verdicts) else
                      "AT LEAST ONE LEVEL IS ABOVE THE NULL BAND -- investigate before quoting "
                      "anything from this arm"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
