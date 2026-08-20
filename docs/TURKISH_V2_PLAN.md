# Turkish drafter v2 — prepared, not run

**Status: ready to fire. Nothing has been collected or trained.** Every GPU on the box was in use
when this was prepared (4 held by an unrelated `dflashvllm` server, 4 by another user).

## What v2 changes

More data, same recipe. v1 trained on 29,100 rows; v2 uses **56,699** (~12.9M → ~29.5M tokens).

| source | v1 | v2 | new |
|---|---|---|---|
| `tr-instructurca` | 24,000 | 39,264 | +15,264 |
| `tr-tool-calling-10k` | 900 | 7,290 | +6,390 |
| `tr-function-calling-20k` | 3,000 | 8,000 | +5,000 |
| `tr-multiturn` | 1,200 | 2,145 | **+945 (pool exhausted)** |

Why more data rather than more epochs: the v1 checkpoint ladder showed in-distribution Turkish
accept plateauing at ~0.7 epochs while out-of-distribution accept kept climbing to ~2.5. Both had
flattened by the end of training, so additional epochs on the same rows buy nothing.

**`tr-multiturn` is the weakest served domain (accept 1.970, 1.617x) and cannot be fixed here** —
after dedup and length filtering the whole corpus yields 2,145 usable rows. The large additions are
in tool/function calling, which are already the *strongest* domains at ~3.1 accept. If v2's gain
disappoints, this is the first place to look: we mostly added data where the model was already good.

## Why only the tail gets collected

`bench_data_tr_v2` was rebuilt with the **same seed** and larger per-source counts. The builder
fills HOLDOUT first from a seeded shuffle, so the v2 files are a strict prefix extension of v1 —
verified byte-for-byte, with **identical holdouts**, which also keeps v1-vs-v2 comparable on the
same eval set. v1's teacher states stay valid and its finished flow cache is just another shard to
concatenate.

**9B is the exception.** Its v1 collected from `data/turkish_prompts/*.jsonl` (14,746 rows,
`format_name: raw_prompt`) — a different, smaller prompt set with no prefix relationship. Unifying
it onto the same corpus costs a full 56,699-row collection. The three sizes were *not* comparable
before; this is what it costs to make them so.

## Cost

~16 GPU-hours of collection for 4B/27B (27,599 rows each) and ~33 for 9B, from v1's measured
throughput (~35 rows/min on the short instruct prompts, ~21 on the long tool/multiturn ones).
Training adds ~6-11h depending on GPU count. Peak disk is ~380 GB for 27B: concat writes a new
store while its inputs still exist, so it needs inputs + output at once.

## Running it

    .venv/bin/python scripts/gen_tr_v2_configs.py     # already run; regenerates the configs
    bash logs/run_tr_v2.sh tr27b "3 4"                # size, GPU list

The runner waits for the named GPUs to be **fully clear** — resident memory, not utilisation,
because an idle vLLM server sits at 0% while holding 88 GB — then collects, preprocesses each shard
as it lands, gates on every shard existing, concatenates onto the v1 cache, and trains.

## After training — do not skip this

Sweep the checkpoints and pick with the **both-corpora TRO ladder**
(`scripts/tr27b_results_json.py`), not by eye and not by taking the last checkpoint. Two rules the
v1 run established the hard way:

* **Never difference a 200-window sweep number against a 400-window result.** `--per_domain N`
  takes the first N windows, a prefix and not a sample. v1's apparent decline at the final
  checkpoint was a 200-window artifact that vanished at 400.
* **The eval is deterministic** (`init_mode: delta`, fixed-step Euler, no sampling), so re-running
  is bit-identical and "within jitter" is not a meaningful defence. The real question is whether a
  window subset is representative, which is why an advance must show in *both* OOD corpora.
