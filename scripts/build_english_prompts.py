"""Materialise the ENGLISH drafter prompts into the layout `english-flow-drafter-prompts` ships.

Unlike the Turkish corpus -- which was extracted from a local corpus and therefore had to be
published or the run was unreproducible -- the English prompts come straight from public HF
datasets. Publishing them anyway buys the same thing the Turkish dataset bought: a run pinned to
exactly the rows we used, rather than to ten upstream datasets that can be re-versioned,
re-ordered, or taken down under us.

THE PROMPTS ARE RENDERED WITH THE REPO'S OWN FORMATTERS (`select_formatter`), reading the real
collect configs for the dataset, split, range and format. So this does not re-derive what we
trained on -- it replays it. Change a formatter and this output changes with it, which is the
property we want.

  python scripts/build_english_prompts.py --out bench_data_en
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# v1 = the original mix; v2 = the sources ADDED in v2. The v2 training mix is v1 + v2, not v2
# alone -- see the dataset README. Kept split this way so the two are not duplicated on the Hub.
MIXES = {"v1": "collect_configs/stage1", "v2": "collect_configs/stage1_q3527bx"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_data_en")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--holdout", type=int, default=100,
                    help="rows held out per source, taken from the END of the range so the "
                         "train rows are byte-identical to what was collected")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from chained_flow.training.collect_teacher import select_formatter

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    for mix, cfg_dir in MIXES.items():
        out_dir = os.path.join(args.out, mix)
        os.makedirs(out_dir, exist_ok=True)
        for cfg_path in sorted(
            os.path.join(cfg_dir, f) for f in os.listdir(cfg_dir) if f.endswith(".yaml")
        ):
            cfg = yaml.safe_load(open(cfg_path))
            name = cfg.get("source") or os.path.basename(cfg_path)[:-5]
            fmt = select_formatter(cfg.get("format_name", "qwen_chat_qa"))
            take = cfg.get("stream_take") or cfg.get("dataset_end")
            if not take:
                print(f"WARN {name}: no dataset_end/stream_take in {cfg_path}; skipping")
                continue
            start = int(cfg.get("dataset_start") or 0)
            try:
                ds = load_dataset(
                    cfg["dataset_name"], cfg.get("dataset_config"),
                    split=cfg["split"], streaming=bool(cfg.get("streaming")),
                )
            except Exception as e:
                print(f"WARN {name}: {type(e).__name__}: {str(e)[:110]}")
                continue

            rows, seen, i = [], set(), 0
            for ex in ds:
                if i >= int(take):
                    break
                if i < start:
                    i += 1
                    continue
                i += 1
                try:
                    p = fmt(ex, tok)
                except Exception:
                    continue
                if not p or p in seen:
                    continue
                seen.add(p)
                rows.append((p, len(tok(p, add_special_tokens=False)["input_ids"])))

            if not rows:
                print(f"WARN {name}: 0 rows rendered")
                continue
            # Holdout from the TAIL: the collected run consumed rows front-to-back, so slicing
            # the head would change which prompts the published `train` file claims we used.
            cut = max(0, len(rows) - args.holdout)
            for split, part in (("train", rows[:cut]), ("holdout", rows[cut:])):
                p = os.path.join(out_dir, f"{name}.{split}.jsonl")
                with open(p, "w") as fh:
                    for text, k in part:
                        fh.write(json.dumps({"prompt": text, "prompt_tokens": k},
                                            ensure_ascii=False) + "\n")
                if part:
                    ks = [k for _, k in part]
                    print(f"{p:56s} rows={len(part):5d} mean_tok={sum(ks)/len(ks):7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
