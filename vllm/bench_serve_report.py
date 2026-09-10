#!/usr/bin/env python3
"""Assemble every bench_serve run into one table, in the shape bench_cf.sh reports.

    bench_serve_report.py [logs/bench_serve]

Prints, per model size, one row per (arm, concurrency): server throughput, per-request
throughput, acceptance length, mean/p95 TTFT, and the speedup over the base arm at the SAME
concurrency -- which is the only speedup that means anything, because base and spec scale
differently with the decode batch.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ORDER = {"base": 0, "chain": 1, "tree": 2}


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/home/shadeform/chain-flow/logs/bench_serve"
    # ONLY the canonical `<size>_<arm>` directories. A tagged run (`4b_chain_noblock`,
    # `4b_tree_guard`, ...) is a probe with a deliberately different configuration; folding it
    # in under the same key silently replaced the real arm with the probe, which is precisely
    # the kind of quiet substitution this repo keeps getting bitten by. Tagged runs are listed
    # separately, unmixed.
    runs: dict[tuple[str, str], dict] = {}
    variants: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "serve_bench.json"))):
        name = os.path.basename(os.path.dirname(f))
        parts = name.split("_")
        if len(parts) == 2:
            runs[(parts[0], parts[1])] = json.load(open(f))
        else:
            variants[name] = json.load(open(f))
    if not runs:
        print(f"no runs under {root}")
        return 1

    for size in sorted({k[0] for k in runs}):
        print(f"\n=== {size.upper()}  (RedHatAI/speculator_benchmarks subset, 70 prompts x 7 "
              f"domains, 256 tok, ignore_eos, greedy) ===")
        base = runs.get((size, "base"))
        basetps = {p["concurrency"]: p["tps"] for p in base["phases"]} if base else {}
        print(f"{'arm':<6}{'conc':>5}{'tok/s server':>14}{'tok/s/req':>11}{'accept':>8}"
              f"{'drafted':>9}{'TTFT ms':>9}{'p95 TTFT':>10}{'vs base':>9}")
        for arm in sorted({k[1] for k in runs if k[0] == size}, key=lambda a: ORDER.get(a, 9)):
            for p in runs[(size, arm)]["phases"]:
                c = p["concurrency"]
                sp = p["tps"] / basetps[c] if basetps.get(c) else None
                print(f"{arm:<6}{c:>5}{p['tps']:>14.1f}{p['tps_per_req']:>11.1f}"
                      + (f"{p['accept']:>8.3f}" if p["accept"] else f"{'-':>8}")
                      + (f"{p['drafted']:>9.1f}" if p["drafted"] else f"{'-':>9}")
                      + (f"{p['ttft_mean'] * 1000:>9.0f}" if p["ttft_mean"] else f"{'-':>9}")
                      + (f"{p['ttft_p95'] * 1000:>10.0f}" if p["ttft_p95"] else f"{'-':>10}")
                      + (f"{sp:>8.2f}x" if sp else f"{'-':>9}")
                      + ("   ERRORS!" if p["nerrors"] else ""))

    if variants:
        print("\n=== TAGGED PROBES (different configuration -- NOT the arms above) ===")
        for name, d in sorted(variants.items()):
            for p in d["phases"]:
                print(f"{name:<22}{p['concurrency']:>5}{p['tps']:>14.1f}"
                      f"{p['tps_per_req']:>11.1f}"
                      + (f"{p['accept']:>8.3f}" if p["accept"] else f"{'-':>8}")
                      + ("   ERRORS!" if p["nerrors"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
