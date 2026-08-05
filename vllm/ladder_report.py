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

IT ALSO REPORTS THE LARGEST DECODE BATCH EACH ARM ACTUALLY REACHED, because "concurrency" in
this table is OFFERED LOAD and the thing `CF_SPEC_MAX_BATCH` thresholds is the DECODE BATCH, and
on a spec engine the two come apart badly.  Measured: the 27B tree arm run at nominal concurrency
8 had the same decode-batch histogram as at nominal 4 -- {1, 2, 3}, never a 4 -- because
`--speculative-config` had left it 27,185 KV tokens and it could not admit a fourth request.  Its
0.56x at nominal 8 is therefore the admission queue, not a batch-size crossing, and reading a
threshold off the concurrency column would have recorded a batch-4 measurement that never
happened.  Needs `CF_BATCH_AUDIT=1` on the run; silent when the arm has no audit.
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


def max_decode_batch(size: str, tag: str) -> int | None:
    """The largest decode batch the engine ever ran, from `CF_BATCH_AUDIT`'s cumulative histogram.

    `serve_ladder.sh` saves the last 20 audit lines; each is cumulative over the whole ladder, so
    the last one covers every level.  None when the run had no audit.
    """
    import re
    p = os.path.join(ROOT, f"{size}_{tag}", "batch_audit.txt")
    if not os.path.exists(p):
        return None
    hist = re.findall(r"decode batch B hist \{([^}]*)\}", open(p).read())
    if not hist:
        return None
    return max(int(kv.split(":")[0]) for kv in hist[-1].split(",") if ":" in kv)


def draft_config(size: str, tag: str) -> str | None:
    """What the DRAFTER was configured with, read back off the arm's own log.

    Two things that used to be constants are now per-arm variables, and both change the meaning
    of the row above: the cudagraph bucket ladder (`CF_DRAFT_BUCKETS`) and the K-schedule
    (`CF_SPEC_K_SCHEDULE`). A ladder table that does not say which arm ran which is a table whose
    rows cannot be compared -- the same failure mode as reading a threshold off the offered
    concurrency. `nocg` is the one number that says whether the ladder actually covered the run:
    it counts drafted steps that found NO bucket and therefore ran eager, each one also a fresh
    `dynamic=False` compile.
    """
    import re
    d = os.path.join(ROOT, f"{size}_{tag}")
    log, bits = os.path.join(d, "server.log"), []
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        m = re.findall(r"draft_buckets=(\[[^\]]*\])", txt)
        if m:
            bits.append(f"buckets {m[-1]}")
        m = re.findall(r"K-SCHEDULE: ([^.]*)\.", txt) or re.findall(r"K-schedule: (.*)", txt)
        if m and "no K-schedule" not in m[-1]:
            bits.append(f"K-schedule {m[-1].strip()}")
    p = os.path.join(d, "batch_audit.txt")
    if os.path.exists(p):
        m = re.findall(r"no-cudagraph drafted steps[^)]*\)\s*(\d+)", open(p).read())
        if m:
            bits.append(f"{m[-1]} drafted steps with NO cudagraph")
    return " | ".join(bits) or None


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

    top = max(concs) if concs else 0
    seen = {t: max_decode_batch(size, t) for t in tags}
    if any(v is not None for v in seen.values()):
        print(f"\n{'max B':>5} " + "".join(
            f"{('-' if seen[t] is None else str(seen[t])):>{w}}" for t in tags))
        print("  largest decode batch the engine actually reached (CF_BATCH_AUDIT). Where this "
              "is\n  below the top concurrency the arm could not admit that many requests, so "
              "those\n  levels measured the ADMISSION QUEUE -- read no batch threshold off them.")
        for t in tags:
            b = seen[t]
            if b is not None and b < top:
                print(f"    {t}: reached only B={b} against a ladder to {top}")

    cfg = {t: draft_config(size, t) for t in tags}
    if any(cfg.values()):
        print("\ndrafter config, per arm (both of these are now variables, so a ladder table "
              "that\ndoes not state them is not comparable):")
        for t in tags:
            print(f"    {t}: {cfg[t] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
