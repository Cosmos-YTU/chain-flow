"""Build the Turkish drafter's flow cache: Turkish teacher states + an English replay slice.

The replay slice exists because the deliverable is a Turkish drafter that does NOT regress on
English. It comes out of `data/flow_cache/stage1_9bx_mix10_k4` -- the exact cache v2 was trained
on -- rather than from `teacher_states/stage1-9bx-mix`, which no longer exists on this box. That
cache was built from an already-shuffled concatenation of the 10 source domains, so a random row
sample is stratified across them without needing the `source` column the cache does not carry.

Two steps, because disk is no longer the binding constraint and reusing the shipped builder for the
Turkish half is worth more than saving one intermediate write:

  1. `build_flow_window_cache` over an in-memory concatenation of the Turkish teacher states.
     `datasets` concatenation is zero-copy over the memory-mapped arrow files, so this never
     materialises a combined teacher_states copy.
  2. concatenate that cache with a row subset of the English cache, renumbering row_offsets.

  python scripts/build_9btr_cache.py
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TURKISH = ["instructurca", "instructurca2", "multiturn", "multiturn2",
           "func_calling", "tool_calling"]
ENGLISH_CACHE = "data/flow_cache/stage1_9bx_mix10_k4"


def build_turkish(tmp_dir: str, draft_length: int) -> None:
    from datasets import concatenate_datasets, load_from_disk
    import chained_flow.training.window_dataset as W

    parts, tok = [], 0
    for name in TURKISH:
        d = load_from_disk(f"teacher_states/stage1-9btr-{name}")
        n = sum(d["num_tokens"])
        tok += n
        parts.append(d)
        print(f"  turkish stage1-9btr-{name:14s} rows={len(d):6d} tokens={n/1e6:.2f}M", flush=True)
    combined = concatenate_datasets(parts)
    print(f"  TURKISH total rows={len(combined)} tokens={tok/1e6:.2f}M", flush=True)
    W._load_teacher_source = lambda *a, **k: combined
    W.build_flow_window_cache("<in-memory turkish concatenation>", tmp_dir,
                              draft_length=draft_length, hidden_dtype="float16", overwrite=True)


def load_cache(d: Path) -> dict:
    from chained_flow.training.window_dataset import FLOW_CACHE_FILES
    out = {k: torch.load(d / v, map_location="cpu", mmap=True) for k, v in FLOW_CACHE_FILES.items()}
    out["metadata"] = json.loads((d / "metadata.json").read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/flow_cache/stage1_9btr_mix_k4")
    ap.add_argument("--tmp-dir", default="data/flow_cache/_9btr_turkish_only")
    ap.add_argument("--english-rows", type=int, default=6000)
    ap.add_argument("--draft-length", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    from chained_flow.training.window_dataset import FLOW_CACHE_FILES

    print("step 1/2: Turkish cache")
    build_turkish(args.tmp_dir, args.draft_length)
    tr = load_cache(Path(args.tmp_dir))

    print("step 2/2: append English replay slice")
    en = load_cache(Path(ENGLISH_CACHE))
    n_en_rows = int(en["row_offsets"].numel())
    rng = random.Random(args.seed)
    pick = sorted(rng.sample(range(n_en_rows), min(args.english_rows, n_en_rows)))
    en_lens = en["row_lengths"][pick]
    en_tokens = int(en_lens.sum())
    tr_tokens = int(tr["hidden"].shape[0])
    tr_rows = int(tr["row_offsets"].numel())
    total_tokens = tr_tokens + en_tokens
    total_rows = tr_rows + len(pick)
    hs = int(tr["hidden"].shape[1])
    print(f"  turkish rows={tr_rows} tokens={tr_tokens/1e6:.2f}M")
    print(f"  english rows={len(pick)} tokens={en_tokens/1e6:.2f}M "
          f"({100*en_tokens/total_tokens:.1f}% of tokens)")
    print(f"  COMBINED rows={total_rows} tokens={total_tokens/1e6:.2f}M "
          f"-> {total_tokens*hs*2/1e9:.1f} GB")

    hidden = torch.empty((total_tokens, hs), dtype=tr["hidden"].dtype)
    input_ids = torch.empty((total_tokens,), dtype=torch.long)
    row_offsets = torch.empty((total_rows,), dtype=torch.long)
    row_lengths = torch.empty((total_rows,), dtype=torch.long)
    prompt_lengths = torch.empty((total_rows,), dtype=torch.long)
    original_row_indices = torch.empty((total_rows,), dtype=torch.long)

    hidden[:tr_tokens] = tr["hidden"]
    input_ids[:tr_tokens] = tr["input_ids"]
    row_offsets[:tr_rows] = tr["row_offsets"]
    row_lengths[:tr_rows] = tr["row_lengths"]
    prompt_lengths[:tr_rows] = tr["prompt_lengths"]
    original_row_indices[:tr_rows] = tr["original_row_indices"]

    off = tr_tokens
    for j, r in enumerate(pick):
        s = int(en["row_offsets"][r])
        n = int(en["row_lengths"][r])
        hidden[off:off + n] = en["hidden"][s:s + n]
        input_ids[off:off + n] = en["input_ids"][s:s + n]
        row_offsets[tr_rows + j] = off
        row_lengths[tr_rows + j] = n
        prompt_lengths[tr_rows + j] = int(en["prompt_lengths"][r])
        # negative marks "came from the English replay cache", so provenance survives in the cache
        original_row_indices[tr_rows + j] = -(int(en["original_row_indices"][r]) + 1)
        off += n
        if (j + 1) % 1000 == 0:
            print(f"    copied {j+1}/{len(pick)} english rows", flush=True)
    assert off == total_tokens, (off, total_tokens)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, tensor in [("hidden", hidden), ("input_ids", input_ids), ("row_offsets", row_offsets),
                        ("row_lengths", row_lengths), ("prompt_lengths", prompt_lengths),
                        ("original_row_indices", original_row_indices)]:
        torch.save(tensor, out / FLOW_CACHE_FILES[key])
        print(f"  wrote {FLOW_CACHE_FILES[key]}", flush=True)
    meta = dict(tr["metadata"])
    meta.update({
        "dataset_path": "turkish stage1-9btr-* + english replay from " + ENGLISH_CACHE,
        "num_rows": total_rows, "total_tokens": total_tokens,
        "turkish_rows": tr_rows, "turkish_tokens": tr_tokens,
        "english_rows": len(pick), "english_tokens": en_tokens,
        "english_replay_seed": args.seed,
    })
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"saved={out}")

    if not args.keep_tmp:
        shutil.rmtree(args.tmp_dir)
        print(f"removed intermediate {args.tmp_dir}")


if __name__ == "__main__":
    main()
