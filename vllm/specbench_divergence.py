#!/usr/bin/env python3
"""Where do the arms' greedy outputs diverge, and by how much?

Speculative decoding is only lossless up to the arithmetic the verifier actually does.
At fp16 a 1-ULP difference in the target logits flips a near-tie, and from that token on
the two arms decode different text.  This reports the FIRST differing character position
per prompt so a handful of late tie-flips is not mistaken for a broken verifier.

Throughput comparability is unaffected in the fixed-length config: every request emits
exactly output_tokens_count tokens whether or not the text matches, so both arms do the
same amount of work.
"""
import argparse
import json
import pathlib


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    b = d["benchmarks"][0]
    out = {}
    for r in b["requests"]["successful"]:
        body = json.loads(r["request_args"])["body"]
        msg = body["messages"][0]["content"][0]["text"]
        out[msg] = r.get("output") or ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/shadeform/chained-flow/logs/specbench")
    ap.add_argument("--size", required=True)
    ap.add_argument("--cfg", default="fixed256_synchronous_r1")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    base_f = root / f"{args.size}_base" / f"gl_{args.cfg}.json"
    if not base_f.exists():
        print(f"missing {base_f}")
        return
    base = load(base_f)

    for arm in ("chain", "tree"):
        f = root / f"{args.size}_{arm}" / f"gl_{args.cfg}.json"
        if not f.exists():
            continue
        other = load(f)
        common = sorted(set(base) & set(other))
        diffs = []
        for p in common:
            a, b = base[p], other[p]
            if a == b:
                continue
            i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                     min(len(a), len(b)))
            diffs.append((i, len(a), p[:60].replace("\n", " ")))
        print(f"\n{args.size} {arm} vs base [{args.cfg}]: "
              f"{len(common)-len(diffs)}/{len(common)} byte-identical")
        for i, n, p in sorted(diffs):
            print(f"   diverge at char {i:5d} of {n:5d} ({100*i/max(n,1):5.1f}% in) | {p}")


if __name__ == "__main__":
    main()
