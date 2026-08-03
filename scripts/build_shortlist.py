"""Build the drafter's shortlist head vocabulary (CF_SHORTLIST).

The flow drafter's lm_head is the BASE model's full 248k-row head. Drafting only ever consumes the
top few candidates per position, so the full head is pure waste (measured 10.9 ms of a 94.6 ms spec
step at 27B/batch 8). A shortlist restricts the head (and the markov bias head `markov.w2`) to the
rows of tokens that can actually be proposed.

This is LOSSLESS by construction — the shortlist only limits what the drafter may PROPOSE; vLLM's
rejection sampler still verifies every draft against the base model, so an omitted token can only
cost ACCEPT, never correctness. Coverage therefore has to be near-total to be free, which is why the
list is built as a UNION of everything that actually occurs rather than a top-K truncation
(a 16k list is known to bleed accept).

Sources (union):
  * every data/flow_cache/*/input_ids.pt  — the teacher/bench corpora the drafter was trained and
    benchmarked on (~32M tokens, 5 stage-1 domains + 6 deploy domains + gsm8k heldout + VAE mix)
  * bench_data/*.jsonl prompts, tokenized — includes the `rag` and `translation` domains that have
    no teacher_states cache, so their token support would otherwise be missing
  * all tokenizer special/added tokens

Usage:
  python scripts/build_shortlist.py --out out/flow/shortlist_q3527b.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "out/flow/shortlist_q3527b.pt"))
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--vocab-size", type=int, default=248320)
    ap.add_argument("--min-count", type=int, default=1,
                    help="keep tokens occurring at least this many times (1 = full coverage)")
    ap.add_argument("--stats", action="store_true", help="print the count/coverage curve and exit")
    args = ap.parse_args()

    cnt = torch.zeros(args.vocab_size, dtype=torch.int64)
    total = 0
    for f in sorted(glob.glob(os.path.join(REPO, "data/flow_cache/*/input_ids.pt"))):
        ids = torch.load(f, map_location="cpu").long().flatten()
        cnt.index_add_(0, ids, torch.ones_like(ids))
        total += ids.numel()
        print(f"  {os.path.basename(os.path.dirname(f)):32s} {ids.numel():>10,} tok", flush=True)
    print(f"corpus tokens: {total:,}  distinct: {int((cnt > 0).sum()):,}")

    if args.stats:
        for th in (1, 2, 5, 10, 50, 100, 1000):
            keep = int((cnt >= th).sum())
            cov = float(cnt[cnt >= th].sum()) / max(total, 1)
            print(f"  min_count>={th:<5d} vocab {keep:>7,}  token coverage {100 * cov:.4f}%")
        return

    keep = (cnt >= args.min_count).nonzero(as_tuple=True)[0]
    extra: set[int] = set()

    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(args.tokenizer)
    extra.update(int(i) for i in tk.all_special_ids)
    extra.update(int(i) for i in getattr(tk, "added_tokens_decoder", {}))
    n_special = len(extra)

    n_prompt_tok = 0
    for f in sorted(glob.glob(os.path.join(REPO, "bench_data/*.jsonl"))):
        prompts = [json.loads(l)["prompt"] for l in open(f) if l.strip()]
        enc = tk(prompts, add_special_tokens=False)["input_ids"]
        before = len(extra)
        for seq in enc:
            extra.update(int(t) for t in seq)
            n_prompt_tok += len(seq)
        print(f"  {os.path.basename(f):24s} {len(prompts):>4} prompts, +{len(extra) - before} new ids")

    ids = torch.tensor(sorted(set(keep.tolist()) | {i for i in extra if 0 <= i < args.vocab_size}),
                       dtype=torch.int64)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ids, args.out)
    cov = float(cnt[ids].sum()) / max(total, 1)
    print(f"\nshortlist: {ids.numel():,} / {args.vocab_size:,} rows "
          f"({100 * ids.numel() / args.vocab_size:.1f}% of the head, "
          f"{args.vocab_size / ids.numel():.2f}x cheaper)")
    print(f"  corpus token coverage {100 * cov:.4f}%  (+{n_special} special, "
          f"{n_prompt_tok:,} bench prompt tokens folded in)")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
