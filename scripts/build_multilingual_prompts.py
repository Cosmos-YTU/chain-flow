"""Extract chat-templated PROMPTS for German and French teacher-state collection.

Same contract as ``build_turkish_prompts.py``: we keep only the prompt (the conversation up to
and including the last human turn) and let ``collect_teacher_states.py`` generate the
continuation with the REAL target. The upstream completions are discarded -- most of these
corpora were written by other models, so their answers are off-distribution for Qwen3.5.

PROMPT LENGTH IS THE THING TO WATCH. The Turkish corpus averaged ~600 prompt tokens because its
rows carried RAG context and multi-turn history; bare Alpaca-style instructions run 20-30 tokens,
which would put the new languages in a different accept regime and make them incomparable. So we
prefer sources that carry context, keep FULL conversation history for multi-turn rows, and fold
in each row's context/input field when it has one. The realised distribution is printed per
source -- read it, do not assume it.

  python scripts/build_multilingual_prompts.py --lang de --out bench_data_de
  python scripts/build_multilingual_prompts.py --lang fr --out bench_data_fr
"""
from __future__ import annotations

import argparse
import json
import os
import random

# Per language: (source name, hf repo, kind, train rows, holdout rows, max prompt tokens).
# `kind` selects the row -> (turns) adapter in `extract`.
PLAN = {
    "de": [
        ("de-instruct",  "FreedomIntelligence/alpaca-gpt4-deutsch",   "sharegpt", 17000, 150, 1024),
        ("de-selfinst",  "CausalLM/GPT-4-Self-Instruct-German",       "instr",     6000,  80, 1024),
        ("de-rag",       "DiscoResearch/germanrag",                   "rag",       3000,  80, 1536),
        ("de-multiturn", "OpenAssistant/oasst2",                      "oasst",     3000,  80, 1536),
    ],
    "fr": [
        ("fr-instruct",  "jpacifico/French-Alpaca-dataset-Instruct-110K", "alpaca", 17000, 150, 1024),
        ("fr-multiturn", "MaziyarPanahi/french_instruct_sharegpt",        "sharegpt", 9000, 160, 1536),
        ("fr-oasst",     "OpenAssistant/oasst2",                          "oasst",   3000,  80, 1536),
    ],
}
OASST_LANG = {"de": "de", "fr": "fr"}


def _txt(m):
    if isinstance(m, dict):
        return (m.get("value") or m.get("content") or m.get("text") or "").strip()
    return str(m or "").strip()


def _role(m):
    r = (m.get("from") or m.get("role") or "").lower() if isinstance(m, dict) else ""
    return "user" if r in ("human", "user", "prompter") else "assistant"


def extract(row, kind):
    """row -> list of (role, text) ending on a USER turn, or None to skip."""
    if kind == "instr":
        q = (row.get("instruction") or "").strip()
        return [("user", q)] if q else None
    if kind == "alpaca":
        q = (row.get("instruction") or "").strip()
        ctx = (row.get("input") or "").strip()
        if not q:
            return None
        return [("user", f"{q}\n\n{ctx}" if ctx else q)]
    if kind == "rag":
        # germanrag: fold the retrieved passages in, which is what makes these prompts long.
        ctxs = row.get("contexts") or []
        q = (row.get("question") or "").strip()
        if not q:
            return None
        body = "\n\n".join(c.strip() for c in ctxs if isinstance(c, str) and c.strip())
        return [("user", f"{body}\n\nFrage: {q}" if body else q)]
    if kind == "sharegpt":
        conv = row.get("conversations") or row.get("conversation") or []
        turns = [(_role(m), _txt(m)) for m in conv if _txt(m)]
        while turns and turns[-1][0] != "user":   # must end on a user turn
            turns.pop()
        return turns or None
    return None


def oasst_threads(lang, cap):
    """OASST2 is a message FOREST. Walk each tree root->node and cut at prompter nodes, which
    yields real multi-turn history rather than isolated instructions."""
    from datasets import load_dataset

    by_id, kids = {}, {}
    for split in ("train", "validation"):
        try:
            ds = load_dataset("OpenAssistant/oasst2", split=split)
        except Exception:
            continue
        for r in ds:
            if r.get("lang") != lang:
                continue
            by_id[r["message_id"]] = r
            kids.setdefault(r.get("parent_id"), []).append(r["message_id"])
    out = []
    def walk(mid, acc):
        if len(out) >= cap:
            return
        r = by_id.get(mid)
        if r is None:
            return
        role = "user" if r.get("role") == "prompter" else "assistant"
        acc = acc + [(role, (r.get("text") or "").strip())]
        if role == "user" and len(acc) >= 1:
            out.append((r["message_id"], list(acc)))
        for k in kids.get(mid, []):
            walk(k, acc)
    for root in kids.get(None, []):
        walk(root, [])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(PLAN))
    ap.add_argument("--out", default=None)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-cap", type=int, default=400000,
                    help="max rows to stream per source before giving up on filling the quota")
    args = ap.parse_args()
    out_dir = args.out or f"bench_data_{args.lang}"

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    def render(turns):
        """Reproduce the Turkish corpus's exact surface form: no system prompt, and the
        enable_thinking=False assistant prefix baked into the text."""
        s = "".join(f"<|im_start|>{r}\n{t}<|im_end|>\n" for r, t in turns)
        return s + "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    for name, repo, kind, n_train, n_hold, max_tok in PLAN[args.lang]:
        want = n_train + n_hold
        rows, seen = [], set()
        try:
            if kind == "oasst":
                for gid, turns in oasst_threads(OASST_LANG[args.lang], want * 3):
                    if not all(t for _, t in turns):
                        continue
                    text = render(turns)
                    if text in seen:
                        continue
                    k = len(tok(text, add_special_tokens=False)["input_ids"])
                    if not (16 <= k <= max_tok):
                        continue
                    seen.add(text)
                    rows.append((gid, text, k))
                    if len(rows) >= want:
                        break
            else:
                ds = load_dataset(repo, split="train", streaming=True)
                for i, r in enumerate(ds):
                    if i >= args.scan_cap or len(rows) >= want:
                        break
                    turns = extract(r, kind)
                    if not turns or not all(t for _, t in turns):
                        continue
                    text = render(turns)
                    if text in seen:
                        continue
                    k = len(tok(text, add_special_tokens=False)["input_ids"])
                    if not (16 <= k <= max_tok):
                        continue
                    seen.add(text)
                    rows.append((r.get("id", i), text, k))
        except Exception as e:
            print(f"WARN {name}: {type(e).__name__}: {str(e)[:110]}")

        if len(rows) < want:
            print(f"WARN {name}: only {len(rows)} rows, wanted {want}")
        # HOLDOUT FIRST, so raising a train quota later leaves the holdout unchanged --
        # the same property build_turkish_prompts.py relies on for its v2 expansion.
        rng.shuffle(rows)
        hold, train = rows[:n_hold], rows[n_hold:n_hold + n_train]
        for split, part in (("train", train), ("holdout", hold)):
            p = os.path.join(out_dir, f"{name}.{split}.jsonl")
            with open(p, "w") as fh:
                for _, text, k in part:
                    fh.write(json.dumps({"prompt": text, "prompt_tokens": k},
                                        ensure_ascii=False) + "\n")
            if part:
                ks = sorted(k for _, _, k in part)
                print(f"{p:52s} rows={len(part):5d} mean_tok={sum(ks)/len(ks):7.1f} "
                      f"p50={ks[len(ks)//2]:5d} p90={ks[int(len(ks)*.9)]:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
