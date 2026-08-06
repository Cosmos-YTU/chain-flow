"""Roll up the CF_TREE_CONV_NARROW default-ON verification matrix into two tables.

  cnd_report.py            -- everything present under logs/cnd/
  cnd_report.py <ns>       -- only the runs tagged with namespace <ns>, with <ns> stripped
                              from the arm names.  `cnd_matrix.sh ... <ns><variant>` is how a
                              session keeps its matrix from being mixed with an older one:
                              `pf` selects `basepf` / `basepfaon` / `treepf` / `treepfwide`
                              and reports them as base / base_aon / tree / tree_wide.

TABLE 1 is LOSSLESSNESS: every tree repeat against the BASE arm at the same size, token for
token, 7 domains.  The reference is always base, never another spec run, because two spec runs
agreeing proves only that they are the same wrong thing.

TABLE 2 is BATCH-1 THROUGHPUT, pooled over the 7 domains.  It deliberately reports the pooled
figure rather than a per-domain mean: domain 0 is HumanEval at 67 tokens, which swung 233 -> 277
tok/s between IDENTICAL repeats, and any statistic that weights it equally with a 256-token
domain inherits that swing.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

ROOT = "/home/shadeform/chained-flow/logs/cnd"


def load(path):
    d = json.load(open(path))
    return {int(s["poff"]): s for s in d["sets"]}


def pooled(sets):
    return sum(s["tokens"] for s in sets.values()) / sum(s["secs"] for s in sets.values())


def ident(a, b):
    """(identical domains, total, first differing domain)."""
    doms = sorted(set(a) & set(b))
    n, first = 0, None
    for d in doms:
        x, y = a[d]["outs"][0], b[d]["outs"][0]
        if x == y:
            n += 1
        elif first is None:
            first = (d, next((k for k in range(min(len(x), len(y))) if x[k] != y[k]), min(len(x), len(y))))
    return n, len(doms), first


def main():
    # Namespace: only runs whose `extra` tag starts with it, and the rest of the tag is the
    # arm variant.  Without one the reference `base` arm of one session can silently become
    # the losslessness reference for another session's tree.
    ns = sys.argv[1] if len(sys.argv) > 1 else ""
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "res_cnd*.json"))):
        m = re.match(r"res_cnd(.*)_(4b|9b|27b)_(base|tree|chain)_r(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        extra, size, arm, rep = m.group(1), m.group(2), m.group(3), int(m.group(4))
        if not extra.startswith(ns):
            continue
        variant = extra[len(ns):]
        runs[(size, arm + ("_" + variant if variant else ""), rep)] = load(p)

    sizes = [s for s in ("4b", "9b", "27b") if any(k[0] == s for k in runs)]

    print("=" * 78)
    print("TABLE 1 -- LOSSLESSNESS: token-identical to BASE, 7 domains x 256 tok, greedy")
    print("=" * 78)
    print(f"{'size':>5} {'arm':>14} {'rep':>4} {'vs base':>10} {'first diff':>18}")
    for size in sizes:
        ref = runs.get((size, "base", 1))
        if ref is None:
            continue
        for (s, arm, rep), cand in sorted(runs.items()):
            if s != size or arm == "base":
                continue
            n, tot, first = ident(ref, cand)
            fd = "-" if first is None else f"dom {first[0]} @ tok {first[1]}"
            print(f"{size:>5} {arm:>14} {rep:>4} {f'{n}/{tot}':>10} {fd:>18}")
        # base-vs-base null, and tree run-to-run
        for (s, arm, rep), cand in sorted(runs.items()):
            if s != size or (arm, rep) == ("base", 1) or not arm.startswith("base"):
                continue
            n, tot, first = ident(ref, cand)
            fd = "-" if first is None else f"dom {first[0]} @ tok {first[1]}"
            print(f"{size:>5} {arm + ' (null)':>14} {rep:>4} {f'{n}/{tot}':>10} {fd:>18}")
        treps = sorted(r for (s, a, r) in runs if s == size and a == "tree")
        for r in treps[1:]:
            n, tot, first = ident(runs[(size, "tree", treps[0])], runs[(size, "tree", r)])
            fd = "-" if first is None else f"dom {first[0]} @ tok {first[1]}"
            print(f"{size:>5} {'tree r%d vs r%d' % (r, treps[0]):>14} {'':>4} {f'{n}/{tot}':>10} {fd:>18}")
    print()

    print("=" * 78)
    print("TABLE 2 -- BATCH-1 THROUGHPUT, pooled over 7 domains (tok/s)")
    print("=" * 78)
    print(f"{'size':>5} {'arm':>12} " + " ".join(f"{'r%d' % r:>8} " for r in (1, 2, 3))
          + f"{'mean':>8} {'vs base':>9} {'vs base_aon':>12}")
    for size in sizes:
        arms = sorted({a for (s, a, _) in runs if s == size})
        base_mean = aon_mean = None
        vals = {}
        for a in arms:
            v = [pooled(runs[(size, a, r)]) for r in (1, 2, 3) if (size, a, r) in runs]
            vals[a] = v
            if not v:
                continue
            if a == "base":
                base_mean = sum(v) / len(v)
            if a == "base_aon":
                aon_mean = sum(v) / len(v)
        for a in arms:
            v = vals[a]
            if not v:
                continue                       # an arm run at one size and not another
            m = sum(v) / len(v)
            cells = " ".join(f"{v[i]:>8.1f} " if i < len(v) else f"{'-':>8} " for i in range(3))
            r1 = f"{m / base_mean:.3f}x" if base_mean else "-"
            r2 = f"{m / aon_mean:.3f}x" if aon_mean else "-"
            print(f"{size:>5} {a:>12} {cells}{m:>8.1f} {r1:>9} {r2:>12}")
    print()
    print("base      = async scheduling OFF (like-for-like: both arms run the same engine)")
    print("base_aon  = async scheduling ON  (deployment baseline: vLLM's own default)")


if __name__ == "__main__":
    main()
