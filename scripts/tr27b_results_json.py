"""Turn logs/eval_tr27b.log into the results JSON that scripts/push_tr27b.py consumes.

Exists so that no accept number is ever retyped between measuring it and publishing it. The push
script refuses to build a card without this file, so the model card can only ever carry figures
that came out of an actual eval run.

diff_plugin_vs_harness.py prints one row per domain:

    {domain:<18} {n:>6} {harness_tree:>12.2f} {harness_chain:>14.2f} {plugin_chain:>13.2f} {delta:>+7.2f}   PLUGIN-TREE {x:.2f}

`plugin_chain` is the "plugin-arm accept" we report; PLUGIN-TREE is the shipping tree path,
carried alongside when present. See ENGLISH_REFERENCE for why the 2.44 smoke-test constant is
unanchored and must not be cited as validation.

  python scripts/tr27b_results_json.py --log logs/eval_tr27b.log --out out/flow/tr27b_eval.json
"""
from __future__ import annotations

import argparse
import json
import re

# "############ 6. TR / tr / trSL   (TURKISH AFTER ...)" -> "TR:tr:trSL"
# The HEAD is a third dimension, not decoration: arms 6/7/8 differ only in it, and that A/B is how
# the -0.70 cost of shipping the stock English shortlist becomes visible.
#   lang: EN | TR (in-distribution Turkish holdout) | TRO (out-of-distribution Turkish)
#   ckpt: v2 | tr300 | tr600 | tr900 | tr1500  -- finalists are carried, never one
#   head: enSL | trSL | full
SECTION = re.compile(r"^#+\s*\d+\.\s*(EN|TRO|TR)\s*/\s*(v2|tr300|tr600|tr900|tr1500|tr1800|tr2100)\s*/\s*(enSL|trSL|full)")
ROW = re.compile(
    r"^(?P<dom>\S+)\s+(?P<n>\d+)\s+(?P<tree>[\d.]+)\s+(?P<hchain>[\d.]+)\s+"
    r"(?P<pchain>[\d.]+)\s+(?P<delta>[+-][\d.]+)(?:\s+PLUGIN-TREE\s+(?P<ptree>[\d.]+))?"
)
# ---------------------------------------------------------------------------------------------
# SMOKE TEST, NOT A VALIDATION GATE. Read this before leaning on it.
#
# What it catches: a wrong checkpoint, a wrong shortlist, a wrong --states glob, a dead plugin arm.
# Gross errors, i.e. the things that would make every number below it meaningless. That is worth
# having and is why it stays.
#
# What it does NOT establish: agreement to any defined precision. Two reasons.
#
#  1. The +-0.15 tolerance is a JUDGEMENT CALL, not a measured variance. Nothing derived it. The
#     only calibration on hand is the baseline moving 2.57 -> 2.54 between per_domain 400 and 200,
#     so the observed -0.06 is ~2x the one shift actually measured -- consistent with sampling
#     noise, not demonstrated to be it.
#  2. THE REFERENCE IS UNANCHORED. 2.44 came from the task brief as prose -- "the 27B English
#     reference is plugin-arm accept 2.44 offline" -- with no domain set, window count, K, chain-vs-
#     tree, or shortlist attached. Searched for the condition and it is not recoverable on this box:
#     the value appears in no doc, log, results file or memory note (the memory note's "plugin arm
#     2.36->2.24" is a flow_steps comparison; BENCHMARKING.md's "2.444" is a latency table cell).
#     Worse, the domain set CHANGES the answer -- 5 bench domains give 2.38, 6 (adding held-out
#     gsm8k) give 2.57 -- and 5 was chosen because it reproduces the reference. That is picking the
#     instrument to match the answer, which partly assumes the conclusion.
#
# So: do not cite this on a model card or in a writeup as evidence of accuracy. The published
# numbers stand on the matrix, which is properly conditioned. This is an internal guard only.
ENGLISH_REFERENCE = 2.44
GATE_TOLERANCE = 0.15  # self-chosen; see above
# English erosion from Turkish-only fine-tuning is NOT uniform: 4B and 9B both lost accept in
# free-form (writing/summarization/qa) while technical barely moved, which is what a technical-only
# English replay cache predicts. A mean hides that, so free-form is tracked separately.
FREEFORM = {"writing", "summarization", "qa"}


def parse(path: str) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    seen_rows: set[str] = set()
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
        seen_rows.add(cur)
        g = m.groupdict()
        out[cur][g["dom"]] = {
            "plugin_chain": float(g["pchain"]),
            "harness_tree": float(g["tree"]),
            "plugin_tree": float(g["ptree"]) if g["ptree"] else None,
        }
    # A section header that appeared but produced NO parsed rows means the table format drifted
    # out from under ROW -- not that the arm was never run. Those look identical downstream (both
    # render as "-"), so say which it is.
    empty = [k for k in out if k not in seen_rows]
    if empty:
        print(f"WARNING: {len(empty)} section(s) matched a header but parsed ZERO rows: {empty}")
        print("  ROW no longer matches the eval table format -- these are NOT absent arms.")
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
    # Which finalist the card is built from. Set it from the FINALIST COMPARISON table,
    # not by default -- the whole point of carrying two is that the choice is evidence-led.
    ap.add_argument("--ship", default="tr600",
                    choices=["tr300", "tr600", "tr900", "tr1500", "tr1800", "tr2100"])
    args = ap.parse_args()

    s = parse(args.log)

    # SMOKE TEST (see ENGLISH_REFERENCE): gross-error guard, not a precision claim.
    gate = s.get("EN:v2:enSL", {})
    if gate:
        mean = sum(v["plugin_chain"] for v in gate.values()) / len(gate)
        delta = mean - ENGLISH_REFERENCE
        flag = "ok" if abs(delta) <= GATE_TOLERANCE else "*** OFF ***"
        print(f"SMOKE TEST (not validation)  EN/v2 mean plugin accept = {mean:.2f} vs "
              f"unanchored reference {ENGLISH_REFERENCE}, delta {delta:+.2f}  [{flag}]")
        print("  reference condition is NOT recoverable; tolerance is self-chosen. "
              "Do not cite as validation.")
        if abs(delta) > GATE_TOLERANCE:
            print("  The offline harness does not reproduce the published English accept. Every")
            print("  number below is suspect -- diagnose before publishing anything.")

    def hmean(key):
        d = s.get(key, {})
        return round(sum(v["plugin_chain"] for v in d.values()) / len(d), 3) if d else None

    # ---- finalist comparison: which checkpoint ships -------------------------------------------
    base = {ax: hmean(f"{ax}:v2:trSL") for ax in ("TR", "TRO", "EN")}
    print("\n" + "=" * 86)
    print("FINALIST COMPARISON (per_domain 400; TRO = out-of-distribution Turkish)")
    print(f"{'ckpt':<8} {'TR':>6} {'dTR':>7} {'TRO':>6} {'dTRO':>7} {'EN':>6} {'dEN':>7} "
          f"{'EN free':>8} {'dfree':>7}")
    print("-" * 86)
    fin = {}
    for ck in ("v2", "tr300", "tr600", "tr900", "tr1500", "tr1800", "tr2100"):
        row = {ax: hmean(f"{ax}:{ck}:trSL") for ax in ("TR", "TRO", "EN")}
        en = s.get(f"EN:{ck}:trSL", {})
        # `/ max(count, 1)` here used to turn "no free-form domain matched" into 0.00 rather than
        # None -- a silent WRONG number that reads as "no free-form erosion" on the one axis this
        # comparison exists to protect. Rename a domain and the guard-rail reports all-clear.
        ff = [v["plugin_chain"] for k, v in en.items() if k in FREEFORM]
        if en and not ff:
            raise SystemExit(
                f"FREEFORM matched no English domain for {ck}. Domains present: {sorted(en)}.\n"
                f"FREEFORM is {sorted(FREEFORM)}. A renamed domain would silently be counted as\n"
                f"'technical' and free-form erosion would read as zero. Update FREEFORM.")
        row["EN_free"] = round(sum(ff) / len(ff), 3) if ff else None
        fin[ck] = row
        d = lambda a, b: "     —" if a is None or b is None else f"{a - b:+6.2f}"
        f = lambda x: "     —" if x is None else f"{x:6.2f}"
        bf = fin.get("v2", {}).get("EN_free")
        print(f"{ck:<8} {f(row['TR'])} {d(row['TR'], base['TR'])} {f(row['TRO'])} "
              f"{d(row['TRO'], base['TRO'])} {f(row['EN'])} {d(row['EN'], base['EN'])} "
              f"{f(row['EN_free'])} {d(row['EN_free'], bf)}")
    print("\nDecide on TRO and EN free-form: TR is an in-distribution holdout drawn from the same")
    print("corpus as training, so it flatters later checkpoints; TRO and EN are independent of it.")

    # ---- three-head A/B on the shipped checkpoint ----------------------------------------------
    ship = args.ship  # which finalist the card is built from
    heads = {"full": hmean(f"TR:{ship}:full"), "turkish_sl": hmean(f"TR:{ship}:trSL"),
             "packaged_english_sl": hmean(f"TR:{ship}:enSL")}
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
        "shipped_checkpoint": ship,
        "turkish": merge(s.get("TR:v2:trSL", {}), s.get(f"TR:{ship}:trSL", {})),
        "turkish_ood": merge(s.get("TRO:v2:trSL", {}), s.get(f"TRO:{ship}:trSL", {})),
        "english": merge(s.get("EN:v2:trSL", {}), s.get(f"EN:{ship}:trSL", {})),
        "finalists": fin,
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
