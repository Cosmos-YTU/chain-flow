#!/usr/bin/env python3
"""Concurrency ladders side by side, as speedups against a NAMED baseline arm.

    ladder_report.py 4b base_lad chain_lad chain_cut8

Reads `logs/bench_serve/<size>_<tag>/c<N>.json` (what `serve_ladder.sh` writes) and prints one
row per concurrency.  The baseline is the FIRST tag; every other column is `tok/s` and the ratio
against it at the same concurrency.

The ratio is the whole point and it is the thing that has been got wrong here before, so the
table refuses to compute one when the two arms did not run the same number of requests at that
concurrency -- a ladder level driven with a different request count is a different measurement,
not a comparable one.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = "/home/shadeform/chained-flow/logs/bench_serve"


def load(size: str, tag: str) -> dict[int, dict]:
    d = os.path.join(ROOT, f"{size}_{tag}")
    out: dict[int, dict] = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if not (f.startswith("c") and f.endswith(".json")):
            continue
        try:
            j = json.load(open(os.path.join(d, f)))
        except Exception:                                    # noqa: BLE001 - partial run
            continue
        for p in j.get("phases", []):
            out[int(p["concurrency"])] = p
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    size, tags = sys.argv[1], sys.argv[2:]
    arms = {t: load(size, t) for t in tags}
    base_tag = tags[0]
    concs = sorted({c for a in arms.values() for c in a})

    w = 26
    print(f"{size.upper()}  baseline = {base_tag}")
    print(f"{'conc':>5} " + "".join(f"{t:>{w}}" for t in tags))
    for c in concs:
        row = f"{c:>5} "
        b = arms[base_tag].get(c)
        for t in tags:
            p = arms[t].get(c)
            if p is None:
                row += f"{'-':>{w}}"
                continue
            cell = f"{p['tps']:8.1f}"
            if p.get("accept"):
                cell += f" a={p['accept']:.2f}"
            if p["nerrors"]:
                cell += f" ERR{p['nerrors']}"
            elif t != base_tag and b is not None:
                if b["requests"] != p["requests"]:
                    cell += f" (n {p['requests']}v{b['requests']})"
                else:
                    cell += f" {p['tps'] / b['tps']:.2f}x"
            row += f"{cell:>{w}}"
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
