"""Turn logs/eval_tr27b.log into the results JSON that scripts/push_tr27b.py consumes.

Exists so that no accept number is ever retyped between measuring it and publishing it. The push
script refuses to build a card without this file, so the model card can only ever carry figures
that came out of an actual eval run.

diff_plugin_vs_harness.py prints one row per domain:

    {domain:<18} {n:>6} {harness_tree:>12.2f} {harness_chain:>14.2f} {plugin_chain:>13.2f} {delta:>+7.2f}   PLUGIN-TREE {x:.2f}

`plugin_chain` is the published "plugin-arm accept" (the 27B English reference is 2.44) and is what
we report; PLUGIN-TREE is the shipping tree path and is carried alongside when present.

  python scripts/tr27b_results_json.py --log logs/eval_tr27b.log --out out/flow/tr27b_eval.json
"""
from __future__ import annotations

import argparse
import json
import re

# "############ 6. TR / tr / trSL   (TURKISH AFTER ...)" -> "TR:tr:trSL"
# The HEAD is a third dimension, not decoration: arms 6/7/8 differ only in it, and that A/B is how
# the -0.70 cost of shipping the stock English shortlist becomes visible.
SECTION = re.compile(r"^#+\s*\d\.\s*(EN|TR)\s*/\s*(v2|tr)\s*/\s*(enSL|trSL|full)")
ROW = re.compile(
    r"^(?P<dom>\S+)\s+(?P<n>\d+)\s+(?P<tree>[\d.]+)\s+(?P<hchain>[\d.]+)\s+"
    r"(?P<pchain>[\d.]+)\s+(?P<delta>[+-][\d.]+)(?:\s+PLUGIN-TREE\s+(?P<ptree>[\d.]+))?"
)
ENGLISH_REFERENCE = 2.44  # published 27B plugin-arm accept, offline


def parse(path: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    cur = None
    for line in open(path, errors="ignore"):
        m = SECTION.match(line.strip())
        if m:
            lang, ckpt, head = m.groups()
            cur = f"{lang}:{ckpt}:{head}"
            out.setdefault(cur, {})
            continue
        if cur is None:
            continue
        m = ROW.match(line.rstrip())
        if not m or m.group("dom") in {"MEAN", "domain"}:
            continue
        g = m.groupdict()
        out[cur][g["dom"]] = {
            "plugin_chain": float(g["pchain"]),
            "harness_tree": float(g["tree"]),
            "plugin_tree": float(g["ptree"]) if g["ptree"] else None,
        }
    return out


def merge(before: dict, after: dict) -> dict:
    doms = list(before) + [d for d in after if d not in before]
    return {d: {"before": (before.get(d) or {}).get("plugin_chain"),
                "after": (after.get(d) or {}).get("plugin_chain")} for d in doms}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/eval_tr27b.log")
    ap.add_argument("--out", default="out/flow/tr27b_eval.json")
    ap.add_argument("--parent-sha256", default="31baf9ad6a417b15eba6c896b367277b0696c4933036242900cfa1a6dbc5b072")
    # 12.95M is the MEASURED token count of the merged flow cache (28,658 rows), not the
    # 13.2M projected from prompt lengths before collection finished.
    ap.add_argument("--train-tokens-m", default="12.95")
    args = ap.parse_args()

    s = parse(args.log)

    # Sanity gate: arm 1 is EN/v2/enSL and must reproduce the published 2.44.
    gate = s.get("EN:v2:enSL", {})
    if gate:
        mean = sum(v["plugin_chain"] for v in gate.values()) / len(gate)
        delta = mean - ENGLISH_REFERENCE
        flag = "OK" if abs(delta) <= 0.15 else "*** OFF ***"
        print(f"SANITY GATE  EN/v2 mean plugin accept = {mean:.2f} "
              f"(published {ENGLISH_REFERENCE}, delta {delta:+.2f})  {flag}")
        if abs(delta) > 0.15:
            print("  The offline harness does not reproduce the published English accept. Every")
            print("  number below is suspect -- diagnose before publishing anything.")

    # Three-head A/B on the TRAINED checkpoint. On the parent these are all equal, which is exactly
    # why the hazard has to be measured here and not there.
    def hmean(key):
        d = s.get(key, {})
        return round(sum(v["plugin_chain"] for v in d.values()) / len(d), 3) if d else None

    heads = {"full": hmean("TR:tr:full"), "turkish_sl": hmean("TR:tr:trSL"),
             "packaged_english_sl": hmean("TR:tr:enSL")}
    if heads["turkish_sl"] is not None and heads["packaged_english_sl"] is not None:
        cost = heads["packaged_english_sl"] - heads["turkish_sl"]
        print(f"\nSHORTLIST A/B (trained ckpt, Turkish): full={heads['full']} "
              f"turkish_sl={heads['turkish_sl']} packaged_english_sl={heads['packaged_english_sl']}")
        print(f"  cost of shipping the stock English list: {cost:+.2f}")
        if heads["full"] is not None and heads["turkish_sl"] is not None:
            print(f"  Turkish list clips vs full head:          "
                  f"{heads['turkish_sl'] - heads['full']:+.2f}")

    res = {
        # Label the estimator on the data itself, so a number can never be lifted out of this file
        # and set beside a served figure without the mismatch being visible.
        "estimator": "offline, per-token-POSITION (diff_plugin_vs_harness.py plugin arm, K=8)",
        "not_comparable_to": "served/in-engine accept, which averages over draft STEPS (~-0.40 on 4B TR)",
        "generation_condition": "Qwen3.5-27B, natural sampling, 256-token cap, NO ignore_eos; "
                                "cap-truncated rows: 47.5% TR / 67.8% EN",
        "parent_sha256": args.parent_sha256,
        "train_tokens_m": args.train_tokens_m,
        "turkish": merge(s.get("TR:v2:trSL", {}), s.get("TR:tr:trSL", {})),
        "english": merge(s.get("EN:v2:trSL", {}), s.get("EN:tr:trSL", {})),
        "head_ab": heads,
        "shortlist_coverage": {
            "shipped_tr": "65.40", "new_rows": "77,939",
            "new_tr": "99.61", "new_en": "100.00", "new_headx": "3.19",
        },
        "raw_sections": s,
    }
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out}")
    for k in ("turkish", "english"):
        for dom, v in res[k].items():
            b, a = v["before"], v["after"]
            d = f"{a - b:+.2f}" if b is not None and a is not None else "—"
            print(f"  {k:8s} {dom:<18} before={b} after={a} delta={d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
