"""Turn the two eval-driver logs (BEFORE = v2, AFTER = 4B-tr) into out/flow/4btr_results.json.

The push script reads only this json, so the model card cannot drift from what was measured.

Both logs are produced by logs/eval_4btr.sh, which runs scripts/diff_plugin_vs_harness.py once per
language arm and prints a table whose per-domain line is

    <domain> <n> <harness TREE> <harness chain> <plugin chain> <delta>   PLUGIN-TREE <x>

`PLUGIN-TREE` is the number the card quotes: it is the SHIPPING path (lagged context + branching,
the tree vLLM actually serves), where `harness TREE` is an upper bound the engine cannot reach.
Both are recorded so the card can say which is which.

  python scripts/parse_4btr_eval.py --before logs/eval_4btr_before.log \
                                    --after  logs/eval_4btr_after.log
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ARM = re.compile(r"^=+ (\S+) / (en|tr)\s+ckd=(\S+) windows=(\d+)")
ROW = re.compile(
    r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([+-][\d.]+)\s+PLUGIN-TREE\s+([\d.]+)")


def parse(path: Path) -> dict:
    """-> {"en": {domain: {...}}, "tr": {...}, "windows": {"en": n, "tr": n}}"""
    out: dict = {"en": {}, "tr": {}, "windows": {}, "ckd": {}}
    arm = None
    for line in path.read_text(errors="ignore").splitlines():
        m = ARM.match(line.strip())
        if m:
            arm = m.group(2)
            out["windows"][arm] = int(m.group(4))
            out["ckd"][arm] = m.group(3)
            continue
        if arm is None:
            continue
        m = ROW.match(line)
        if m:
            out[arm][m.group(1)] = {
                "n": int(m.group(2)),
                "harness_tree": float(m.group(3)),
                "harness_chain": float(m.group(4)),
                "plugin_chain": float(m.group(5)),
                "plugin_tree": float(m.group(7)),
            }
    return out


def merge(before: dict, after: dict, arm: str, key: str) -> dict:
    """{domain: {"before": x, "after": y}} over the domains present in BOTH arms."""
    doms = [d for d in after[arm] if d in before[arm]]
    missing = sorted(set(before[arm]) ^ set(after[arm]))
    if missing:
        print(f"WARN {arm}: domains in only one arm, dropped: {missing}")
    if not doms:
        raise SystemExit(f"no shared {arm} domains between the two logs -- did an arm fail?")
    return {d: {"before": before[arm][d][key], "after": after[arm][d][key]} for d in doms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="logs/eval_4btr_before.log")
    ap.add_argument("--after", default="logs/eval_4btr_after.log")
    ap.add_argument("--out", default="out/flow/4btr_results.json")
    ap.add_argument("--key", default="plugin_tree",
                    choices=["plugin_tree", "harness_tree", "plugin_chain", "harness_chain"])
    ap.add_argument("--cache", default="data/flow_cache/stage1_4btr_mix_k4")
    ap.add_argument("--config", default="train_configs/recovered/joint_4btr.yaml")
    args = ap.parse_args()

    b, a = parse(Path(args.before)), parse(Path(args.after))
    for name, d in (("before", b), ("after", a)):
        if not d["en"] and not d["tr"]:
            raise SystemExit(f"{name} log has no parseable rows: check it ran with --plugin_tree")

    meta = json.loads((Path(args.cache) / "metadata.json").read_text())
    cfg = {}
    for line in Path(args.config).read_text().splitlines():
        if ":" in line and not line.strip().startswith("#"):
            k, _, v = line.partition(":")
            cfg[k.strip()] = v.strip()

    r = {
        "metric": args.key,
        "turkish": merge(b, a, "tr", args.key),
        "english": merge(b, a, "en", args.key),
        "tr_windows": a["windows"].get("tr"),
        "en_windows": a["windows"].get("en"),
        "raw": {"before": b, "after": a},
        # from out/flow/SHORTLIST_REBUILD_CLAIM.txt, measured on held-out Turkish by the 27B agent
        "shortlist": {"n": 77939, "reduction": 3.19},
        "cov_tr_before": 65.40,
        "cov_tr_after": 99.61,
        "cov_en": 100.00,
        "data": {
            "tr_rows": meta.get("turkish_rows"),
            "tr_tokens": meta.get("turkish_tokens"),
            "en_rows": meta.get("english_rows"),
            "en_tokens": meta.get("english_tokens"),
            "en_frac": 100 * meta.get("english_tokens", 0) / max(meta.get("total_tokens", 1), 1),
            "tokens": meta.get("total_tokens", 0) / 1e6,
            "epochs": cfg.get("num_train_epochs"),
            "lr": cfg.get("learning_rate"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, indent=2))
    print(f"wrote {args.out}")
    for arm in ("turkish", "english"):
        rows = r[arm]
        mb = sum(v["before"] for v in rows.values()) / len(rows)
        ma = sum(v["after"] for v in rows.values()) / len(rows)
        print(f"  {arm:8s} mean {mb:.2f} -> {ma:.2f}  ({ma - mb:+.2f})")


if __name__ == "__main__":
    main()
