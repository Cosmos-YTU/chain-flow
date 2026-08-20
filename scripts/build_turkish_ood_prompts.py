"""Build an OUT-OF-DISTRIBUTION Turkish eval set from turkishdspark's own benchmarks.

Why this exists: `bench_data_tr/*.holdout.jsonl` is a properly disjoint split (verified: zero exact
prompt overlap with training), but it is drawn from the SAME corpus, by the same builder, from the
same source mix as the training data. So it is an in-distribution holdout, and some share of any
late-checkpoint Turkish gain on it is corpus-specific fit rather than transferable Turkish ability.
That asymmetry flatters later checkpoints on the axis being bought while the English sets -- which
have no relationship to the Turkish training data -- report the axis being paid with honestly.

These sets are independent of the training mix:
  * turkish_alpaca_100  -- Alpaca-style Turkish instructions, a source not in the training mix
  * wikirag_tr_100      -- Turkish Wikipedia RAG, a different DOMAIN entirely

turkish_holdout_100 is deliberately EXCLUDED: it is turkishdspark's own held-out split of the same
tool-calling families that feed our training mix, so it is no more independent than what we have.

Prompts are chat-templated with enable_thinking=False, matching collection and every other eval.
"""
from __future__ import annotations

import argparse
import json
import os

SRC = "/root/turkishdspark/data/benchmark"
SETS = ["turkish_alpaca_100", "wikirag_tr_100"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench_data_tr_ood")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--max-prompt", type=int, default=1536)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)

    for name in SETS:
        kept, skipped = [], 0
        for line in open(os.path.join(SRC, f"{name}.jsonl")):
            d = json.loads(line)
            msgs = [m for m in d["messages"] if m["role"] != "assistant"]
            if not msgs:
                skipped += 1
                continue
            kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            if d.get("tools"):
                kw["tools"] = d["tools"]
            try:
                text = tok.apply_chat_template(msgs, **kw)
            except TypeError:
                kw.pop("enable_thinking")
                text = tok.apply_chat_template(msgs, **kw)
            n = len(tok.encode(text, add_special_tokens=False))
            if n > args.max_prompt or n < 16:
                skipped += 1
                continue
            kept.append({"prompt": text, "prompt_tokens": n})
        outp = os.path.join(args.out, f"{name}.jsonl")
        with open(outp, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mean = sum(r["prompt_tokens"] for r in kept) / max(len(kept), 1)
        print(f"{outp:44s} rows={len(kept):4d} skipped={skipped:3d} mean_prompt_tok={mean:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
