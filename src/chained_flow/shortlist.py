"""The drafter's shortlist head: what it is, how it is loaded and guarded, and how to build one.

WHAT IT IS
----------
The flow drafter decodes its predicted hidden states through the TARGET model's own frozen
``lm_head`` -- 248,320 rows on Qwen3.5. Drafting only ever consumes the top few candidates per
position, so ~43-47% of the draft is spent on rows the beam never looks at (measured 10.9 ms of a
94.6 ms spec step at 27B/batch 8, and ~5.6 ms/step at 4B/batch 1). A *shortlist* restricts the head
(and the markov bias head ``markov.w2``) to the rows of tokens that can actually be proposed:
62,642 rows, i.e. 4.0x less weight traffic per depth.

It is **quality-free by construction.** The shortlist only limits what the drafter may PROPOSE;
vLLM's verify still scores every drafted token against the target model, so an omitted token can
cost ACCEPT and can never cost correctness. That is also why the list is built as a UNION of
everything that occurs rather than a top-K truncation -- a 16k top-K list is known to bleed accept.

WHY IT SHIPS INSIDE THE WHEEL
-----------------------------
The list is keyed by TOKEN ID, so it is a property of the TOKENIZER, not of the model or of the
drafter checkpoint: max id 248,076 against a 248,320-row vocab, and one file serves 4B / 9B / 27B
because they share the Qwen3.5 vocabulary. Before it shipped, ``CF_SHORTLIST`` defaulted to a path
inside this repo's ``out/`` directory, which no ``pip install`` has -- so every pip user silently
ran the full head and measured **145.7 tok/s (1.04x) instead of 157.8 (1.13x)** at 4B. As int32 it
is 250 KB, which is a rounding error next to the wheel.

THE VOCAB GUARD
---------------
A shortlist is a list of integers. Point it at a model with a different vocabulary and every id
means a different token: the list is not merely suboptimal, it is nonsense, and the only symptom is
a quietly lower accept. So the payload carries the ``vocab_size`` it was built against and
``check()`` REFUSES a mismatch rather than clamping ids into range (which is what the old
``sl[sl < V]`` filter did). A refusal falls back to the full head and says so on the
``[cf-defaults]`` line -- slower, never wrong.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

# The vocabulary the packaged list was built for. Kept here (not only inside the .pt) so
# `defaults.py`, which must stay importable without torch, can name it in a message.
PACKAGED_VOCAB = 248320
PACKAGED_ROWS = 62642


def packaged_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "shortlist_qwen3_5.pt"


# ------------------------------------------------------------------ load / guard


def load(path: str):
    """``(ids int64 CPU tensor, meta dict)``.

    Two on-disk formats are accepted, and the difference matters only for the guard:

    * ``{"ids": int32 tensor, "vocab_size": int, ...}`` -- what this package ships and what
      ``build`` writes. Carries the vocabulary it was built for, so ``check`` can be exact.
    * a bare tensor of ids -- every shortlist built before that metadata existed, including
      ``out/flow/shortlist_q3527b.pt`` in this checkout. ``check`` falls back to a max-id test,
      which catches a shortlist that is too WIDE for the model but cannot catch one built for a
      different tokenizer of the same size.
    """
    import torch

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        ids = obj["ids"]
        meta = {k: v for k, v in obj.items() if k != "ids"}
    else:
        ids, meta = obj, {}
    return ids.flatten().long(), meta


def check(ids, meta: dict, vocab_size: int) -> str | None:
    """``None`` if this list may be used against a ``vocab_size``-row head, else WHY NOT."""
    built = meta.get("vocab_size")
    if built is not None:
        if int(built) != int(vocab_size):
            # Name the remedy: the most likely cause is not a wrong tokenizer at all, it is a
            # build that took `vocab_size` from the TOKENIZER (which under-reports whenever the
            # lm_head is padded -- Qwen3.5's tokenizer says 248,044 against a 248,320-row head).
            return (f"built for vocab_size={int(built)} but this model's lm_head has "
                    f"{int(vocab_size)} rows -- the ids would name different tokens. If the "
                    f"list IS for this tokenizer, rebuild it with "
                    f"`chained-flow build-shortlist --vocab-size {int(vocab_size)} ...`: the "
                    f"recorded size must be the MODEL's head, not the tokenizer's")
        return None
    mx = int(ids.max()) if ids.numel() else -1
    if mx >= int(vocab_size):
        return (f"contains token id {mx}, past this model's {int(vocab_size)}-row lm_head, and "
                f"carries no vocab_size metadata")
    return None


# ------------------------------------------------------------------ build


def _count_ids(patterns: list[str], vocab_size: int):
    import torch

    cnt = torch.zeros(vocab_size, dtype=torch.int64)
    total = 0
    for pat in patterns:
        for f in sorted(glob.glob(pat, recursive=True)):
            ids = torch.load(f, map_location="cpu").long().flatten()
            bad = int(((ids < 0) | (ids >= vocab_size)).sum())
            if bad:
                raise ValueError(f"{f}: {bad} ids outside [0, {vocab_size}) -- wrong "
                                 f"--vocab-size for this corpus?")
            cnt.index_add_(0, ids, torch.ones_like(ids))
            total += ids.numel()
            print(f"  {f:60s} {ids.numel():>12,} tok", flush=True)
    return cnt, total


def build(argv=None) -> int:
    """``chained-flow build-shortlist`` -- the command the fallback warning names.

    A shortlist is only as good as its coverage, so the sources are explicit rather than clever:
    token-id tensors (``--ids``) and/or raw text (``--jsonl``, one JSON object per line) that is
    tokenized here, unioned with every special/added token. ``--min-count 1`` (the default) keeps
    the union rather than a truncation -- see the module docstring for why that is not a knob to
    turn down casually.
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="chained-flow build-shortlist",
        description="Build a CF_SHORTLIST token-id list for the drafter's head.")
    ap.add_argument("--out", default="shortlist.pt",
                    help="output .pt (drop it next to the drafter checkpoint as shortlist.pt "
                         "and it is picked up with no env var)")
    ap.add_argument("--tokenizer", required=True,
                    help="the TARGET model's tokenizer, e.g. Qwen/Qwen3.5-27B")
    ap.add_argument("--vocab-size", type=int, default=0,
                    help="rows in the target's lm_head (default: the tokenizer's vocab_size). "
                         "This is recorded in the file and is what the runtime guard compares "
                         "against, so it must be the MODEL's head size, not the tokenizer's if "
                         "they differ.")
    ap.add_argument("--ids", action="append", default=[], metavar="GLOB",
                    help="glob of .pt files holding token-id tensors (repeatable)")
    ap.add_argument("--jsonl", action="append", default=[], metavar="GLOB",
                    help="glob of .jsonl files whose --text-field is tokenized (repeatable)")
    ap.add_argument("--text-field", default="prompt")
    ap.add_argument("--min-count", type=int, default=1,
                    help="keep tokens occurring at least this often (1 = full union)")
    ap.add_argument("--stats", action="store_true",
                    help="print the count/coverage curve and exit without writing")
    args = ap.parse_args(argv)

    if not args.ids and not args.jsonl:
        ap.error("give at least one --ids GLOB or --jsonl GLOB: a shortlist is a statement "
                 "about a corpus, and there is no useful default corpus")

    import torch
    from transformers import AutoTokenizer

    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = args.vocab_size or int(getattr(tk, "vocab_size", 0)) or len(tk)
    print(f"tokenizer {args.tokenizer}  vocab_size={vocab_size:,}", flush=True)
    if not args.vocab_size:
        # The runtime guard compares this number against the loaded model's lm_head row count,
        # and a tokenizer routinely under-reports it: Qwen3.5's tokenizer says 248,044 against a
        # 248,320-row head. Recording the tokenizer's number then gets the finished list REFUSED
        # at run time -- correct data, wrong label. Say so here, where it is still cheap to fix.
        print(f"  NOTE: --vocab-size was not given, so {vocab_size:,} (the tokenizer's) is what "
              f"gets recorded and what the runtime guard will compare against the model's "
              f"lm_head. If the target model's head is padded to a different size, pass that "
              f"number instead or the list will be refused.", flush=True)

    cnt, total = _count_ids(args.ids, vocab_size)
    print(f"corpus tokens: {total:,}  distinct: {int((cnt > 0).sum()):,}", flush=True)

    if args.stats:
        for th in (1, 2, 5, 10, 50, 100, 1000):
            keep = int((cnt >= th).sum())
            cov = float(cnt[cnt >= th].sum()) / max(total, 1)
            print(f"  min_count>={th:<5d} vocab {keep:>7,}  token coverage {100 * cov:.4f}%")
        return 0

    extra: set[int] = set()
    extra.update(int(i) for i in tk.all_special_ids)
    extra.update(int(i) for i in getattr(tk, "added_tokens_decoder", {}))
    n_special = len(extra)

    n_text_tok = 0
    for pat in args.jsonl:
        for f in sorted(glob.glob(pat, recursive=True)):
            rows = [json.loads(l) for l in open(f) if l.strip()]
            texts = [r[args.text_field] for r in rows if args.text_field in r]
            if not texts:
                raise ValueError(f"{f}: no --text-field {args.text_field!r} in {len(rows)} rows")
            before = len(extra)
            for seq in tk(texts, add_special_tokens=False)["input_ids"]:
                extra.update(int(t) for t in seq)
                n_text_tok += len(seq)
            print(f"  {f:60s} {len(texts):>5} texts, +{len(extra) - before} new ids", flush=True)

    keep = (cnt >= args.min_count).nonzero(as_tuple=True)[0]
    ids = torch.tensor(sorted(set(keep.tolist()) | {i for i in extra if 0 <= i < vocab_size}),
                       dtype=torch.int32)
    d = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(d, exist_ok=True)
    torch.save({"format": 1, "ids": ids, "vocab_size": int(vocab_size),
                "tokenizer": args.tokenizer,
                "note": f"min_count={args.min_count}; +{n_special} special/added tokens"},
               args.out)
    cov = float(cnt[ids.long()].sum()) / max(total, 1)
    print(f"\nshortlist: {ids.numel():,} / {vocab_size:,} rows "
          f"({100 * ids.numel() / vocab_size:.1f}% of the head, "
          f"{vocab_size / max(ids.numel(), 1):.2f}x less head traffic)")
    print(f"  corpus token coverage {100 * cov:.4f}%  (+{n_special} special, "
          f"{n_text_tok:,} tokenized text tokens folded in)")
    print(f"  -> {args.out}")
    print("  use it with CF_SHORTLIST=<path>, or copy it next to the drafter checkpoint as "
          "shortlist.pt")
    return 0
