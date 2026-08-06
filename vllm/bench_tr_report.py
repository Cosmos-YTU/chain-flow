#!/usr/bin/env python3
"""Pool the Turkish serve runs into the per-prompt-set table.

Reads logs/bench_tr/<size>_<arm>/<set>_r<rep>.json (one measured phase each, concurrency 1)
and prints, per SET: tok/s per arm with its run-to-run spread, acceptance from vLLM's own
counters, and the speedup against the base arm of the same size.

SET A and SET B ARE NEVER POOLED WITH EACH OTHER.  They are different prompt collections with
different length and domain mixes -- set A carries structured-JSON function/tool-calling that
scores far above free prose.  A mean across both describes neither.  Each is pooled only within
itself, token-weighted (sum tokens / sum wall), which is what "prefill in the denominator"
means at the collection level.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

SET_A = ["tr_funccall", "tr_instruct", "tr_multiturn", "tr_toolcall"]
SET_B = ["tds_alpaca", "tds_holdout", "tds_wikirag"]
ARMS = ["base", "chain_v2", "chain_tr", "chain_v2_trsl"]


def load(root: str, size: str, arm: str, tag: str = "") -> dict:
    """{set: [phase, ...]} across repeats.

    `tag` selects a run directory suffix, e.g. tag="_nateos" reads `4b_chain_tr_nateos`. The
    forced-256 and natural-EOS conditions live in separate directories and are NEVER pooled:
    forcing 256 tokens drives the model past its natural stop into out-of-distribution text
    that is measurably harder to draft.
    """
    out: dict[str, list] = {}
    for f in sorted(glob.glob(f"{root}/{size}_{arm}{tag}/*_r*.json")):
        name = os.path.basename(f)[:-5]
        s, _, _rep = name.rpartition("_r")
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for p in j["phases"]:
            out.setdefault(s, []).append(p)
    return out


def agg(phases: list) -> dict:
    tps = [p["tps"] for p in phases]
    tok = sum(p["tokens"] for p in phases)
    sec = sum(p["secs"] for p in phases)
    acc = [p["accept"] for p in phases if p.get("accept")]
    nreq = sum(p["completed"] for p in phases)
    nerr = sum(p["nerrors"] for p in phases)
    # DECODE-ONLY, prefill excluded.  Speculation cannot speed up prefill, and these Turkish
    # prompts are 580-820 tokens against a 256-token generation, so prefill is a far bigger
    # share of the wall clock than in the English benchmark.  Leaving it in the denominator
    # (which the headline tok/s does, deliberately -- it is what a deployment sees) drags every
    # ratio toward 1.0 for reasons that have nothing to do with the drafter.  Reporting both
    # separates "the drafter did not help" from "there was less decode to help with".
    dec_t = sum(p["completed"] * (p["lat_mean"] - p["ttft_mean"])
                for p in phases if p.get("lat_mean") and p.get("ttft_mean"))
    dec_n = tok - nreq          # the first token of each request is emitted by prefill
    return {
        "tps_pooled": tok / sec if sec else 0.0,          # token-weighted across repeats
        "tps_decode": dec_n / dec_t if dec_t else 0.0,
        "tps_min": min(tps) if tps else 0.0,
        "tps_max": max(tps) if tps else 0.0,
        "spread_pct": (max(tps) - min(tps)) / min(tps) * 100 if tps and min(tps) else 0.0,
        "accept": sum(acc) / len(acc) if acc else None,
        "drafted": next((p["drafted"] for p in phases if p.get("drafted")), None),
        "reps": len(phases), "requests": nreq, "errors": nerr,
        "tokens": tok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/shadeform/chained-flow/logs/bench_tr")
    ap.add_argument("--sizes", default="4b,9b")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="", help="run-directory suffix, e.g. _nateos")
    args = ap.parse_args()

    report: dict = {}
    for size in args.sizes.split(","):
        data = {a: load(args.root, size, a, args.tag) for a in ARMS}
        data = {a: d for a, d in data.items() if d}
        if "base" not in data:
            print(f"\n### {size.upper()}: no base arm yet, skipping")
            continue
        report[size] = {}
        for label, sets in (("SET A -- chained-flow Turkish holdouts", SET_A),
                            ("SET B -- turkishdspark benchmark", SET_B)):
            cond = ("natural EOS, max 256 tok" if args.tag == "_nateos"
                    else "256 tok forced, ignore_eos")
            print(f"\n### {size.upper()}  {label}   (concurrency 1, greedy, {cond})")
            hdr = f"| {'set':13s} |"
            for a in data:
                hdr += f" {a} tok/s | {a} acc | {a} x |" if a != "base" else " base tok/s |"
            print(hdr)
            print("|" + "---|" * (1 + sum(3 if a != "base" else 1 for a in data)))

            pool: dict[str, dict[str, float]] = {a: {"tok": 0.0, "sec": 0.0} for a in data}
            for s in sets:
                if s not in data["base"]:
                    continue
                b = agg(data["base"][s])
                row = f"| {s:13s} | {b['tps_pooled']:7.1f} |"
                for a in data:
                    for p in data[a].get(s, []):
                        pool[a]["tok"] += p["tokens"]
                        pool[a]["sec"] += p["secs"]
                    if a == "base":
                        continue
                    if s not in data[a]:
                        row += " - | - | - |"
                        continue
                    v = agg(data[a][s])
                    dec = (f"{v['tps_decode'] / b['tps_decode']:.2f}x"
                           if b["tps_decode"] else "-")
                    row += (f" {v['tps_pooled']:7.1f} | {v['accept']:.3f} | "
                            f"**{v['tps_pooled'] / b['tps_pooled']:.3f}x** ({dec} dec) |")
                print(row)
                report[size].setdefault(s, {a: agg(data[a][s]) for a in data if s in data[a]})

            bt = pool["base"]["tok"] / pool["base"]["sec"] if pool["base"]["sec"] else 0
            row = f"| **POOLED** | **{bt:7.1f}** |"
            for a in data:
                if a == "base":
                    continue
                if not pool[a]["sec"]:
                    row += " - | - | - |"
                    continue
                t = pool[a]["tok"] / pool[a]["sec"]
                accs = [agg(data[a][s])["accept"] for s in sets if s in data[a]]
                accs = [x for x in accs if x]
                row += (f" **{t:7.1f}** | **{sum(accs)/len(accs):.3f}** | "
                        f"**{t / bt:.3f}x** |") if bt else " - | - | - |"
            print(row)

        print(f"\n{size.upper()} spread / integrity (all sets, all repeats):")
        for a in data:
            sp = [agg(v)["spread_pct"] for v in data[a].values()]
            er = sum(agg(v)["errors"] for v in data[a].values())
            rq = sum(agg(v)["requests"] for v in data[a].values())
            tk = sum(agg(v)["tokens"] for v in data[a].values())
            print(f"  {a:14s} repeats spread {min(sp):.2f}-{max(sp):.2f}%  "
                  f"requests {rq} errors {er}  tokens/req {tk/max(rq,1):.1f}")

    if args.json_out:
        json.dump(report, open(args.json_out, "w"), indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
