"""Build the Turkish 4B drafter's flow cache: Turkish teacher states + an English replay slice.

Same two-step shape as `scripts/build_9btr_cache.py`, which this follows deliberately -- but a
separate file rather than an edit, because that script had not been run yet when this one was
written and three agents share this checkout. The parameters are all on the command line here, so
this one is size-agnostic; the 9B defaults are not touched.

The English replay slice exists because the deliverable is a Turkish drafter that does NOT regress
on English. Two things differ from the 9B case and both are honest limitations, not oversights:

  * The exact cache v2 trained on (`stage1_4bx_mix10_k4`, 10 domains) has been deleted from this
    box. What survives is `stage1_4b_mix5_k4` -- the 5 TECHNICAL domains (gsm8k, nemotron-math,
    nemotron-stem, alpaca-code, dolly-chat). Those five were reused byte-for-byte by the `4bx`
    "lean" pipeline, so the replay slice is a genuine SUBSET of v2's own training distribution,
    just missing the 5 free-form domains (ultrachat, opus_translation, no_robots, writingprompts,
    cnn_dailymail).
  * Consequently free-form English is replayed by NOTHING. The English bench measures exactly
    those domains (qa, summarization, writing) alongside the technical ones, so if free-form
    English drifts further than technical English, that gap is visible in the eval rather than
    hidden -- and the fix is more replay or a lower LR, not a different cache format.

That cache was built from an already-shuffled concatenation of its source domains, so a random row
sample is stratified across them without needing the `source` column the cache does not carry.

  python scripts/build_4btr_cache.py --english-rows 7500
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


def build_turkish(sources: list[str], tmp_dir: str, draft_length: int) -> None:
    """Window-cache an in-memory concatenation of the Turkish teacher-state dirs.

    `datasets` concatenation is zero-copy over the memory-mapped arrow files, so this never
    materialises a combined teacher_states copy on disk.
    """
    from datasets import concatenate_datasets, load_from_disk
    import chain_flow.training.window_dataset as W

    parts, tok = [], 0
    for path in sources:
        d = load_from_disk(path)
        n = sum(d["num_tokens"])
        tok += n
        parts.append(d)
        print(f"  turkish {path:44s} rows={len(d):6d} tokens={n/1e6:.2f}M", flush=True)
    combined = concatenate_datasets(parts)
    print(f"  TURKISH total rows={len(combined)} tokens={tok/1e6:.2f}M", flush=True)
    W._load_teacher_source = lambda *a, **k: combined
    W.build_flow_window_cache("<in-memory turkish concatenation>", tmp_dir,
                              draft_length=draft_length, hidden_dtype="float16", overwrite=True)


def load_cache(d: Path) -> dict:
    from chain_flow.training.window_dataset import FLOW_CACHE_FILES
    out = {k: torch.load(d / v, map_location="cpu", mmap=True) for k, v in FLOW_CACHE_FILES.items()}
    out["metadata"] = json.loads((d / "metadata.json").read_text())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turkish", nargs="+", default=[
        "teacher_states/stage1-4btr-instruct_a",
        "teacher_states/stage1-4btr-funccall",
        "teacher_states/stage1-4btr-multiturn",
        "teacher_states/stage1-4btr-toolcall",
    ])
    ap.add_argument("--english-cache", default="data/flow_cache/stage1_4b_mix5_k4")
    ap.add_argument("--output-dir", default="data/flow_cache/stage1_4btr_mix_k4")
    ap.add_argument("--tmp-dir", default="data/flow_cache/_4btr_turkish_only")
    ap.add_argument("--english-rows", type=int, default=7500)
    ap.add_argument("--draft-length", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    from chain_flow.training.window_dataset import FLOW_CACHE_FILES

    print("step 1/2: Turkish cache")
    build_turkish(args.turkish, args.tmp_dir, args.draft_length)
    tr = load_cache(Path(args.tmp_dir))

    print("step 2/2: append English replay slice")
    en = load_cache(Path(args.english_cache))
    if en["hidden"].shape[1] != tr["hidden"].shape[1]:
        raise SystemExit(f"hidden size mismatch: turkish {tr['hidden'].shape[1]} vs "
                         f"english {en['hidden'].shape[1]} -- wrong target model?")
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
        if (j + 1) % 2000 == 0:
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
        "dataset_path": f"turkish {'+'.join(args.turkish)} + english replay from {args.english_cache}",
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
