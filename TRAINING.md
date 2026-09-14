# Training the Turkish drafters

End-to-end from a clean clone. Assumes CUDA 13 and, for the batch shapes below, 2×B300.

## 1. Environment

```bash
git clone https://github.com/Cosmos-YTU/chain-flow && cd chain-flow
uv venv --python 3.12
uv pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` is **training only** and pins exact versions — a drafter is judged against
accept numbers measured under one environment, and a torch or transformers minor bump moves accept
by more than the effects we routinely act on. **vLLM is not in it**: training imports it nowhere,
and installing it drags in a second torch pin. Serve from a separate venv (`uv pip install
chain-flow`, see README).

## 2. Data

```bash
python scripts/fetch_tr_prompts.py
```

Pulls the prompt corpora from
[`selimaktas/turkish-flow-drafter-prompts`](https://huggingface.co/datasets/selimaktas/turkish-flow-drafter-prompts)
— v1 (29,100 rows), v2 (56,699), and the out-of-distribution eval set. Prompts only; teacher hidden
states are generated locally by running the real target over them.

## 3. Pick a preset, generate configs

```bash
python scripts/gen_tr_v2_configs.py                      # preset=instruct, warm start=continue
python scripts/gen_tr_v2_configs.py --preset varied      # the alternative
python scripts/gen_tr_v2_configs.py --init parent        # restart from the English v2 checkpoint
```

### Which preset

**Use `instruct` unless you have a reason not to.** The served per-domain numbers say why:

| domain | accept | speedup | rows available |
|---|---|---|---|
| tr_funccall | 3.136 | **2.477x** | +5,000 |
| tr_toolcall | 3.100 | **2.433x** | +6,390 |
| tr_instruct | 2.238 | 1.844x | +15,264 |
| tr_multiturn | 1.970 | **1.617x** | +945 (pool exhausted) |

Tool and function calling are already the *strongest* domains. `varied` pours 11,390 rows into
them and drops instructurca from 58% of prompt tokens to 38% — it optimises where the drafter is
already good and dilutes where it is not, for 42% more collection (19.1M vs 11.0M prompt tokens).
`instruct` takes all of instructurca and all of multiturn and holds tool/function at their v1
counts.

The "diverse data lifts the weak domains" result from the English v2 mix does **not** argue for
`varied` here: that came from adding new *kinds* of data (creative prose, summarization,
translation), not more rows of domains already represented.

**`tr_multiturn` is the weakest domain and cannot be fixed by either preset** — after dedup and
length filtering the entire corpus yields 2,145 usable rows. Fixing it needs a new source.

### Which warm start

`continue` (default) resumes the **shipped Turkish checkpoint** — carry on from where it left off.
`parent` restarts from the English v2 checkpoint, which is what v1 did; it is the only option that
keeps v1-vs-v2 a clean comparison of the *data*, since continuing confounds more data with more
epochs on the old rows.

## 4. Run

```bash
bash logs/run_tr_v2.sh tr27b "0 1"            # size, GPUs, [preset]
```

Waits for the GPUs to be **fully clear** — resident memory, not utilisation, because an idle vLLM
server sits at 0% while holding 88 GB — then restores the VAE, collects teacher states,
preprocesses each shard as it lands, gates on every shard existing, concatenates onto the v1 cache,
and trains.

Only the **tail** is collected for 4B and 27B: v2 is a strict prefix extension of v1 (same seed,
holdout fills first, verified byte-for-byte), so v1's teacher states stay valid and its finished
flow cache is just another shard. 16,209 new rows on the `instruct` preset instead of 45,309. **9B
is the exception** — its v1 came from a different prompt set with no prefix relationship, so it
collects everything.

## 5. Collection speed

Collection runs HF `model.generate()`, **not vLLM**. The generation phase is 256 sequential decode
steps per batch against a single forward for the hidden-state pass, so it dominates the wall clock
— and HF generate pads every sequence to the longest in its batch and runs without CUDA graphs or
continuous batching.

**vLLM would be materially faster for that phase and is not implemented.** It cannot do the second
phase at all — hidden states need `output_hidden_states`, which vLLM does not expose without the
runner hook — so it would be a two-stage pipeline: generate with vLLM, harvest with transformers.
It also needs its own venv because of the torch pin. Worth doing; not a thing to bolt on before a
20-hour run.

The version of that win available today is batch size — but **the two batch sizes do not scale
together**. Phase 1 holds weights + KV (~0.5 GB/sequence at 27B), so `gen=256` runs comfortably at
2.48 it/s. Phase 2 materialises every layer's hidden states *plus* the activations that produce
them; measured at 27B, `hid=128` put 266 GB in use on a 267 GB card and OOMed — about 1.66 GB per
sequence, roughly 3× what the hidden-state tensor alone accounts for. Raising both in step is the
mistake to avoid.

`gen` is a single value per size; **`hid` is derived per source** from that source's actual prompt
lengths, because phase-2 memory is per *token* and the sources differ by 2.5× — instructurca
averages 441 tokens against tool-calling's 1081. One global value either wastes half the card on
instructurca (87% of the rows) or OOMs on tool-calling. At 27B that yields `instruct 56`,
`funccall 40`, `multiturn 32`, `toolcall 24`, all landing near 190 GB of a 275 GB card. It is sized
on the mean/p95 midpoint, not the mean, since extraction pads to the longest row in each batch.
`gen` is 256 at 27B. Override either without regenerating:

```bash
CF_GEN_BS=64 CF_HID_BS=32 python scripts/gen_tr_v2_configs.py
```

These are **unverified on B300** like the training batch shapes. If collection OOMs, halve them.

### Arrow offsets

`final_hidden` is stored as `Sequence(LargeList(...))`. Arrow's plain `list` uses **int32 offsets**
into its child array, and the child here is rows × tokens × hidden halffloats — a 2000-row 27B shard
is ~8.8e9 elements against a 2^31 = 2.15e9 limit, so saving fails with *"offset overflow while
concatenating arrays"*. Only the inner level needs 64-bit offsets; the outer indexes inner lists
(~1.7M) and stays well inside int32. This is why `datasets>=4.x` is pinned — `LargeList` is required.

### Host memory

Phase 2 accumulates every row and writes once at the end of a shard, so peak **system** RAM scales
with `CF_SHARD_ROWS` (default 2000). At 27B a row is ~8.8 MB (860 tokens × 5120 × fp16), so a shard
is ~18 GB plus the Arrow conversion. Lower `CF_SHARD_ROWS` on a machine with less RAM — it costs
nothing but a few more shard boundaries.

Stopping and restarting is cheap at two levels. A shard is done when its flow cache exists, so a
restart skips it entirely. And within a shard, the generated answers are kept
(`teacher_states/_tmp_<name>`) and reused, so a phase-2 failure does **not** cost the generation —
which is the expensive half, ~24 minutes for a 4,000-row shard at 27B. That matters because phase 2
is the phase that OOMs, so tuning `hid` is cheap to iterate on. That is what makes it safe to kill a run partway and
come back with a bigger batch.

### Watching it

tqdm writes to stderr, which is redirected into the per-GPU log, so the terminal only shows the
pipeline's own lines. Every 5 minutes (`CF_HEARTBEAT_SEC`) the runner pulls the latest progress out
of each collect log. To follow one directly — note the `tr`, since tqdm separates updates with `\r`
and the file is otherwise one enormous line:

```bash
tr '\r' '\n' < logs/tr_v2_tr27b_instruct/collect_g0.log | tail -20
```

## 6. Performance knobs

| variable | default | what it does |
|---|---|---|
| `CF_COMPILE` | `1` | `torch.compile` the drafter |
| `CF_COMPILE_MODE` | `max-autotune` | several minutes of warm-up per graph, then real kernel benchmarking — worth it over a multi-hour run |
| `CF_FUSED_HEAD` | `1` | chunked lm_head reductions instead of a `[B, K, 248320]` tensor |
| `CF_FUSED_CHUNK` | `64` | rows per chunk; peak head activation is `chunk × vocab` |

The fused head is what makes the batch shapes in the generated configs fit. It is loss- and
gradient-equivalent to the reference path, proven in fp64 by `scripts/verify_fused_head.py` (every
loss and gradient to machine epsilon) and end-to-end on the real training module by
`scripts/verify_fused_head_e2e.py` (7 loss components, 112 parameter gradients). It **declines
rather than degrades**: `lambda_dist > 0` and scheduled sampling both fall back to the reference
path instead of dropping a term.

Each size's own v1 effective batch is preserved exactly — 4B 12,288 windows, 9B 24,576, 27B 6,144 —
so only the microbatch/accumulation split changes and the optimisation trajectory is untouched.
They differ per size, so there is no single rule; `gen_tr_v2_configs.py` asserts
`mb × accum × 2 == eff` to stop a hand edit silently turning a speed change into a different
experiment. **The microbatch sizes are unverified on B300** — no card was free to test. On an OOM,
halve `per_device_train_batch_size` and double `gradient_accumulation_steps`; the assert will flag
it immediately if the pair stops matching.

## 7. After training — do not skip

Sweep the checkpoints and pick with the **both-corpora TRO ladder**
(`scripts/tr27b_results_json.py`), not by eye and not by taking the last checkpoint. Two rules the
v1 run established the hard way:

* **Never difference a 200-window sweep number against a 400-window result.** `--per_domain N`
  takes the first N windows — a prefix, not a sample. v1's apparent decline at the final checkpoint
  was a 200-window artifact that vanished at 400.
* **The eval is deterministic** (`init_mode: delta`, fixed-step Euler, no sampling anywhere), so
  re-running is bit-identical and "within jitter" is not a defence. The real question is whether a
  window subset is representative, which is why an advance must show in *both* OOD corpora.
