"""Summarise /tmp/cf_native_*.json: t/s, accept, speedup, and LOSSLESSNESS vs the base arm.

  cf_report.py <size> [tagsuffix] [--subset a,b,c]

`--subset` selects sets by their CF_POFF value, so one run that sweeps
CF_POFF=0,1,2,3,4,5,6,64,128 can be scored BOTH as the 7-domain sweep (0..6) and as the
old-style easy-offset triple (0,64,128) without re-running anything.
"""
import json, sys, os

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
subset = None
for a in sys.argv[1:]:
    if a.startswith("--subset"):
        subset = {int(x) for x in a.split("=", 1)[1].split(",")}
size = argv[0] if argv else "4b"
suffix = argv[1] if len(argv) > 1 else ""


def load(tag):
    p = f"/tmp/cf_native_{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def sets_of(r):
    return [s for s in r["sets"] if subset is None or s["poff"] in subset]


base = load(f"base_{size}_base{suffix}")
rows = []
for arm in ("base", "chain", "tree"):
    r = load(f"{'base' if arm == 'base' else 'spec'}_{size}_{arm}{suffix}")
    if r is None:
        continue
    ss = sets_of(r)
    tot_tok = sum(s["tokens"] for s in ss)
    tot_s = sum(s["secs"] for s in ss)
    tps = tot_tok / tot_s
    accs = [s["accept"] for s in ss if s["accept"]]
    acc = sum(accs) / len(accs) if accs else 1.0
    ok = tokok = tokn = seqn = 0
    if base is not None:
        for bs, rs in zip(sets_of(base), ss):
            for bo, ro in zip(bs["outs"], rs["outs"]):
                seqn += 1
                if bo == ro:
                    ok += 1
                m = 0
                for a, b in zip(bo, ro):
                    if a != b:
                        break
                    m += 1
                tokok += m
                tokn += len(bo)
    per = " ".join(f"{s['tps']:.1f}" for s in ss)
    rows.append((arm, tps, acc, tps / (rows[0][1] if rows else tps), ok, seqn, tokok, tokn, per,
                 tot_s / (tot_tok / max(acc, 1e-9)) * 1000))
lbl = "7-domain sweep (poff 0..6)" if subset is None or subset == set(range(7)) \
    else f"poff subset {sorted(subset)}"
print(f"\n===== {size} batch1, {lbl}{' tag=' + suffix if suffix else ''} =====")
print(f"{'arm':6} {'t/s':>7} {'accept':>7} {'speedup':>8} {'step_ms':>8}  lossless")
for a, tps, acc, sp, ok, sn, tk, tn, per, stepms in rows:
    print(f"{a:6} {tps:7.1f} {acc:7.2f} {sp:8.2f}x {stepms:8.1f}  {ok}/{sn} seq {tk}/{tn} tok   [{per}]")
