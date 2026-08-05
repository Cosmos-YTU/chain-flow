"""Compare two `test_plugin_native.py` result files token-for-token, per domain.

Losslessness is a claim about TOKEN IDS, so that is what this reads -- the `outs` field the
bench already dumps -- rather than tok/s, which cannot distinguish "same text, faster" from
"different text".  Prints one row per domain plus the pooled throughput both files reported,
so the speed number and the identity check can never be quoted from different runs.

  cnd_compare.py <ref.json> <cand.json> [label]
"""
import json
import sys


def load(p):
    d = json.load(open(p))
    return {int(s["poff"]): s for s in d["sets"]}


def main():
    ref_p, cand_p = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    ref, cand = load(ref_p), load(cand_p)
    doms = sorted(set(ref) & set(cand))
    print(f"=== {label or (ref_p + ' vs ' + cand_p)} ===")
    print(f"{'dom':>3} {'ident':>6} {'first_diff':>10} {'n_ref':>6} {'n_cand':>6} "
          f"{'tps_ref':>8} {'tps_cand':>9} {'acc_cand':>8}")
    n_ident = 0
    for d in doms:
        a = ref[d]["outs"][0]
        b = cand[d]["outs"][0]
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), None)
        if i is None and len(a) != len(b):
            i = min(len(a), len(b))
        ok = i is None
        n_ident += ok
        acc = cand[d].get("accept")
        print(f"{d:>3} {'YES' if ok else 'no':>6} {('-' if ok else str(i)):>10} "
              f"{len(a):>6} {len(b):>6} {ref[d]['tps']:>8.1f} {cand[d]['tps']:>9.1f} "
              f"{(f'{acc:.3f}' if acc else '-'):>8}")
    rt = sum(s["tokens"] for s in ref.values()) / sum(s["secs"] for s in ref.values())
    ct = sum(s["tokens"] for s in cand.values()) / sum(s["secs"] for s in cand.values())
    print(f"IDENTICAL {n_ident}/{len(doms)} domains | pooled tok/s ref {rt:.1f} cand {ct:.1f} "
          f"({ct / rt:.3f}x)")
    return 0 if n_ident == len(doms) else 1


if __name__ == "__main__":
    sys.exit(main())
