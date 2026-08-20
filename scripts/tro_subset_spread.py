"""Derive the ladder's selection floor from the data, per comparison, on PAIRED differences.

The offline eval is deterministic (init_mode=delta, fixed-step Euler, no sampling in the inference
path), so re-running is bit-identical and there is no run-to-run noise to threshold against. The
only open question is whether the scored subset represents Turkish generally -- which is measurable.

TWO THINGS THIS GETS RIGHT, both of which change the answer by large factors in OPPOSITE directions:

1. CLUSTER on rows, not windows. Windows inside one sequence are strongly correlated (a predictable
   passage yields many high-accept windows together). Resampling windows independently would treat
   those as independent draws and UNDERSTATE the spread.

2. Bootstrap the PAIRED DIFFERENCE, not one checkpoint's mean. The ladder never asks "how uncertain
   is checkpoint A's accept" -- it asks "is mean_A - mean_B real". Both checkpoints are scored on
   the SAME rows and the SAME windows, so between-row variance -- which is enormous here and
   dominates SD(mean_A) -- CANCELS in the difference. Two warm-started checkpoints a few hundred
   steps apart are highly correlated per row. Using SD(mean_A) as the floor would be conservative by
   a large unknown factor, would reject every advance in the ladder, and would present the collapse
   to the earliest checkpoint as a principled result when it is a statistical artifact. A floor that
   rejects everything looks like rigour, which is what makes that failure hard to spot.

The floor is therefore COMPARISON-SPECIFIC: the spread of (1500 - 900) need not equal that of
(2100 - 1500).

  python scripts/tro_subset_spread.py --dumps tr900=out/flow/tro_win_tr900.json \\
      tr1500=out/flow/tro_win_tr1500.json --compare tr1500:tr900
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict


def load(path: str) -> dict[str, dict[int, list[int]]]:
    """{corpus: {row_idx: [accepted_len per window, in order]}}"""
    raw = json.load(open(path))
    out: dict[str, dict[int, list[int]]] = {}
    for dom, pairs in raw.items():
        rows: dict[int, list[int]] = defaultdict(list)
        for r, d in pairs:
            rows[int(r)].append(int(d))
        out[dom] = dict(rows)
    return out


def check_aligned(a: dict[int, list[int]], b: dict[int, list[int]], dom: str, na: str, nb: str):
    """Differencing misaligned rows would INFLATE the spread and silently invalidate the floor."""
    if set(a) != set(b):
        raise SystemExit(f"{dom}: row sets differ between {na} and {nb} "
                         f"({len(set(a) ^ set(b))} rows differ) -- cannot pair.")
    bad = [r for r in a if len(a[r]) != len(b[r])]
    if bad:
        raise SystemExit(f"{dom}: {len(bad)} rows have different window counts between {na} and "
                         f"{nb} (e.g. row {bad[0]}: {len(a[bad[0]])} vs {len(b[bad[0]])}) "
                         f"-- windows do not correspond, cannot pair.")


def boot(rows: list[tuple[list[int], list[int]]], n_boot: int, seed: int) -> dict:
    """Cluster bootstrap over rows. Returns spreads of mean_A, and of the PAIRED mean_A - mean_B."""
    rng = random.Random(seed)
    nrow = len(rows)
    sa = sum(sum(a) for a, _ in rows)
    sb = sum(sum(b) for _, b in rows)
    nw = sum(len(a) for a, _ in rows)
    obs_a, obs_b = 1.0 + sa / nw, 1.0 + sb / nw
    ma, md = [], []
    for _ in range(n_boot):
        ta = tb = cnt = 0
        for _ in range(nrow):
            a, b = rows[rng.randrange(nrow)]
            ta += sum(a); tb += sum(b); cnt += len(a)
        if cnt:
            ma.append(1.0 + ta / cnt)
            md.append((ta - tb) / cnt)          # paired: the +1 cancels

    def sd(xs):
        mu = sum(xs) / len(xs)
        return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    md_sorted = sorted(md)
    return {"obs_a": round(obs_a, 4), "obs_b": round(obs_b, 4),
            "obs_diff": round(obs_a - obs_b, 4),
            "sd_unpaired_A": round(sd(ma), 4), "sd_paired_diff": round(sd(md), 4),
            "diff_p05": round(md_sorted[int(0.05 * len(md))], 4),
            "diff_p95": round(md_sorted[int(0.95 * len(md)) - 1], 4),
            "rows": nrow, "windows": nw}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True, help="name=path ...")
    ap.add_argument("--compare", nargs="+", required=True, help="A:B ... (A is the candidate)")
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dumps = {}
    for spec in args.dumps:
        name, _, path = spec.partition("=")
        dumps[name] = load(path)

    res: dict = {}
    for spec in args.compare:
        ca, _, cb = spec.partition(":")
        if ca not in dumps or cb not in dumps:
            raise SystemExit(f"--compare {spec}: need dumps for both {ca} and {cb}")
        print(f"\n=== {ca} vs {cb} ===")
        print(f"{'corpus':<22} {'A':>7} {'B':>7} {'diff':>7} {'sd(paired)':>11} "
              f"{'sd(unpaired)':>13} {'verdict':>10}")
        res[spec] = {}
        for dom in sorted(set(dumps[ca]) & set(dumps[cb])):
            a, b = dumps[ca][dom], dumps[cb][dom]
            check_aligned(a, b, dom, ca, cb)
            rows = [(a[r], b[r]) for r in sorted(a)]
            st = boot(rows, args.n_boot, args.seed)
            passes = st["obs_diff"] > st["sd_paired_diff"]
            st["passes_floor"] = passes
            res[spec][dom] = st
            print(f"{dom:<22} {st['obs_a']:>7.3f} {st['obs_b']:>7.3f} {st['obs_diff']:>+7.3f} "
                  f"{st['sd_paired_diff']:>11.4f} {st['sd_unpaired_A']:>13.4f} "
                  f"{'PASS' if passes else 'fail':>10}")
        allp = all(v["passes_floor"] for v in res[spec].values())
        print(f"  -> {ca} {'ADVANCES over' if allp else 'TIES with'} {cb} "
              f"(floor requires the diff to exceed sd(paired) in BOTH corpora)")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
