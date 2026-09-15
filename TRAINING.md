# Training a drafter

Four stages. The expensive model is touched **once**, in stage 1 — after that the target is a
cache on disk and never loads again.

```
  1. collect    run the TARGET over prompts, save its hidden states
  2. preprocess turn a teacher-state dataset into training windows
  3. train VAE  the hidden -> latent bridge the flow runs inside
  4. train flow the drafter itself, against the cached hiddens
```

## 1. Environment

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` is **training only** and pins exact versions — a drafter is judged against
accept numbers measured under one environment, and a torch or transformers minor bump moves accept
by more than the effects we routinely act on. **vLLM is not in it**: training imports it nowhere,
and installing it drags in a second torch pin. Serve from a separate venv (`uv pip install
chain-flow`, see README).

## 2. Collect teacher states

One YAML per shard. **The config is a positional argument, not `--config`** — the collector
dispatches on `len(sys.argv) == 2 and sys.argv[1].endswith(".yaml")`, so a flag falls through to
the full argparse and is rejected instantly.

```bash
python scripts/collect_teacher_states.py collect_configs/smoke_gsm8k.yaml
```

`collect_configs/stage1/dolly_chat.yaml` is a real one. The fields that matter:

| field | what it does |
|---|---|
| `model_id` | the **target** whose hidden states you are capturing |
| `dataset_name` / `split` / `dataset_start` / `dataset_end` | the prompt rows for this shard |
| `format_name` | how a row becomes a chat-templated prompt (`pretemplated` replays one verbatim) |
| `generation_batch_size` | phase 1, generating continuations |
| `hidden_batch_size` | phase 2, capturing hiddens — `output_hidden_states=True` costs ~1.66 GB/sequence, so this is much smaller, and it is where OOMs happen |

Only prompts go in; the target generates its own continuations. Any completions shipped with an
upstream dataset are discarded — they came from other models and are off-distribution for your
target.

Shard it. A shard is a natural restart unit, and phase 1 is reusable if phase 2 OOMs.

**Size the batches from a run that finished**, not from a per-sequence estimate. A made-up
GB-per-sequence constant once produced a `generation_batch_size` ~6× the value a completed run
had used, which is an OOM hours into a shard.

## 3. Preprocess into windows

```bash
python scripts/preprocess_flow_dataset.py \
    --dataset-path teacher_states/<name> \
    --output-dir data/flow_cache/<name> \
    --draft-length 8 --hidden-dtype float16
```

`--draft-length` must match the drafter you are about to train. It is baked into the cache, so
changing it later means reprocessing — though not recollecting.

Multiple shards concatenate:

```bash
python scripts/concat_flow_caches.py --shards 'data/flow_cache/_shard_*' --out data/flow_cache/mix
```

## 4. Train the VAE

The flow runs inside a latent space, so the bridge is trained first and then frozen — or jointly
fine-tuned at a low LR, which is what the released drafters do
(`vae_learning_rate_multiplier`, 0.05).

```bash
python scripts/train_transformer_hidden_vae.py train_configs/vae/smoke_vae.yaml
```

## 5. Train the drafter

```bash
python scripts/train_tree_flow.py train_configs/tree/mix6_k4_tree_o4_cov8.yaml
```

Worth knowing about this step:

- **Warm-start with `init_from`** when adapting an existing drafter. It prints the sha256 of the
  weights it actually loaded — a config can point anywhere, and a stale default that silently
  trains from scratch is exactly the failure that print exists to prevent.
- **`CF_FUSED_HEAD=1` (default on)** keeps the three logit losses in memory at a 248k vocab by
  reducing the head in row chunks instead of materialising `[B, K, 248320]` logits three times.
- **`CF_COMPILE_MODE`**: `max-autotune` enables CUDA graphs, which recycle buffers the fused head
  saves for backward. Use `max-autotune-no-cudagraphs` if you turn it up.
- **Adaptation saturates early, and earlier as the target grows** — measured at roughly 2 epochs
  at 4B, 1.3 at 9B, 0.7 at 27B. The last checkpoint is usually *not* the best: the new domain
  flattens out while the original keeps degrading. Sweep checkpoints; do not assume the tail.

## Evaluating

`BENCHMARKING.md` (run `chain-flow docs` for its path) records every measured number for the
released drafters and the exact conditions each was taken under. Two conventions in it are easy to
conflate — pooled across domains versus a single named domain — and it says which is which.
