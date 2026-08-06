"""Concatenate several flow window caches into one.

Why this exists: `preprocess_flow_dataset.py` runs at ~2.5 rows/s single-process, so a 29k-row
collection is ~3.2h of CPU work that would otherwise sit SERIALLY behind ~6h of GPU collection.
Preprocessing each collection shard as it lands and concatenating the caches at the end hides
almost all of that behind the GPU time.

The cache is a flat token-major store plus per-row indexing, so the merge is a concatenation with
one offset fixup:

    hidden        [total_tokens, H]  -> cat on dim 0
    input_ids     [total_tokens]     -> cat
    row_offsets   [num_rows]         -> cat, each shard shifted by the running token count
    row_lengths   [num_rows]         -> cat unchanged
    prompt_lengths[num_rows]         -> cat unchanged
    original_row_indices             -> cat, shifted by the running SOURCE-row count (provenance
                                        only: FlowWindowCacheDataset never indexes with it)

Correctness is not assumed. --verify re-reads randomly chosen rows out of the merged tensors and
compares them elementwise against the same row read from its source shard, which is what actually
catches an off-by-one in the offset arithmetic.

  python scripts/concat_flow_caches.py --out data/flow_cache/stage1_tr27b_mix_k4 \
      --shards 'data/flow_cache/_shard_tr27b_*' --verify 64
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import torch

FILES = {
    "hidden": "hidden.pt",
    "input_ids": "input_ids.pt",
    "row_offsets": "row_offsets.pt",
    "row_lengths": "row_lengths.pt",
    "prompt_lengths": "prompt_lengths.pt",
    "original_row_indices": "original_row_indices.pt",
}
METADATA = "metadata.json"


def load_shard(d: Path) -> dict:
    out = {k: torch.load(d / v, map_location="cpu") for k, v in FILES.items()}
    with (d / METADATA).open() as f:
        out["metadata"] = json.load(f)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", required=True, help="glob of cache dirs")
    ap.add_argument("--verify", type=int, default=64, help="rows to re-read and compare")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dirs = [Path(p) for p in sorted(glob.glob(args.shards))]
    if not dirs:
        raise SystemExit(f"no cache dirs matched {args.shards!r}")

    shards = []
    tot_tokens = tot_rows = 0
    hidden_size = None
    dtype = None
    draft_length = None
    for d in dirs:
        s = load_shard(d)
        md = s["metadata"]
        if hidden_size is None:
            hidden_size, dtype, draft_length = md["hidden_size"], s["hidden"].dtype, md["draft_length"]
        # A shape or draft-length mismatch silently produces a corrupt cache, so refuse instead.
        if md["hidden_size"] != hidden_size:
            raise SystemExit(f"{d}: hidden_size {md['hidden_size']} != {hidden_size}")
        if s["hidden"].dtype != dtype:
            raise SystemExit(f"{d}: dtype {s['hidden'].dtype} != {dtype}")
        if md["draft_length"] != draft_length:
            raise SystemExit(f"{d}: draft_length {md['draft_length']} != {draft_length}")
        print(f"  {d.name:44s} rows={md['num_rows']:>7,} tokens={md['total_tokens']:>12,}")
        tot_tokens += int(s["hidden"].shape[0])
        tot_rows += int(s["row_offsets"].numel())
        shards.append(s)

    print(f"\nmerged total: rows={tot_rows:,} tokens={tot_tokens:,} "
          f"hidden={tot_tokens * hidden_size * 2 / 1e9:.1f} GB")

    hidden = torch.empty((tot_tokens, hidden_size), dtype=dtype)
    input_ids = torch.empty((tot_tokens,), dtype=shards[0]["input_ids"].dtype)
    row_offsets, row_lengths, prompt_lengths, orig_idx = [], [], [], []

    tok_cursor = 0
    row_cursor = 0
    for s in shards:
        n = int(s["hidden"].shape[0])
        hidden[tok_cursor: tok_cursor + n] = s["hidden"]
        input_ids[tok_cursor: tok_cursor + n] = s["input_ids"]
        row_offsets.append(s["row_offsets"] + tok_cursor)
        row_lengths.append(s["row_lengths"])
        prompt_lengths.append(s["prompt_lengths"])
        orig_idx.append(s["original_row_indices"] + row_cursor)
        tok_cursor += n
        row_cursor += int(s["metadata"].get("source_rows", s["row_offsets"].numel()))

    row_offsets = torch.cat(row_offsets)
    row_lengths = torch.cat(row_lengths)
    prompt_lengths = torch.cat(prompt_lengths)
    orig_idx = torch.cat(orig_idx)
    assert tok_cursor == tot_tokens, (tok_cursor, tot_tokens)
    assert row_offsets.numel() == tot_rows

    # --- verification: read rows back out of the MERGED tensors and compare to the source shard
    if args.verify:
        rng = random.Random(args.seed)
        # map merged row index -> (shard, row-within-shard)
        owner = []
        for si, s in enumerate(shards):
            owner.extend((si, r) for r in range(int(s["row_offsets"].numel())))
        picks = rng.sample(range(tot_rows), min(args.verify, tot_rows))
        bad = 0
        for gi in picks:
            si, ri = owner[gi]
            s = shards[si]
            so, sl = int(s["row_offsets"][ri]), int(s["row_lengths"][ri])
            mo, ml = int(row_offsets[gi]), int(row_lengths[gi])
            if sl != ml or not torch.equal(hidden[mo: mo + ml], s["hidden"][so: so + sl]) \
               or not torch.equal(input_ids[mo: mo + ml], s["input_ids"][so: so + sl]) \
               or int(prompt_lengths[gi]) != int(s["prompt_lengths"][ri]):
                bad += 1
                print(f"  MISMATCH merged row {gi} vs shard {si} row {ri}")
        if bad:
            raise SystemExit(f"verification FAILED on {bad}/{len(picks)} rows")
        print(f"verified {len(picks)} randomly chosen rows against their source shards: all identical")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(hidden, out / FILES["hidden"])
    torch.save(input_ids, out / FILES["input_ids"])
    torch.save(row_offsets, out / FILES["row_offsets"])
    torch.save(row_lengths, out / FILES["row_lengths"])
    torch.save(prompt_lengths, out / FILES["prompt_lengths"])
    torch.save(orig_idx, out / FILES["original_row_indices"])
    md = dict(shards[0]["metadata"])
    md.update(num_rows=tot_rows, total_tokens=tot_tokens,
              source_rows=row_cursor, dataset_path=args.shards,
              merged_from=[d.name for d in dirs])
    with (out / METADATA).open("w", encoding="utf-8") as f:
        json.dump(md, f, indent=2)
    print(f"wrote {out}  rows={tot_rows:,} tokens={tot_tokens:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
