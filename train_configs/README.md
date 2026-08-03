# Training Configs

Training configs are grouped by model family. Keep new families in their own subdirectory.

## Smoke configs

- `smoke_mlp.yaml`: quick hidden-MLP training config.
- `vae/smoke_vae.yaml`: quick VAE training config.
- `chunked_flow/smoke_chunked_flow.yaml`: quick single-expert flow drafter config against the local smoke teacher-state dataset.

## VAE configs

VAE configs are under `vae/`. Sweep configs are under `vae/sweeps/`; see `vae/README.md` for the current grouping.

## Flow drafter configs

- `chunked_flow/smoke_chunked_flow.yaml`: local smoke run for wiring and fast failures.
- `chunked_flow/chunked_flow_k2_gsm8k_6.5k.yaml`: first real K=2/C=2 single-expert flow drafter run.
- `chunked_flow/sweeps/context_size/*.yaml`: context-size sweep with 8, 12, and 16 slots.
- `chunked_flow/sweeps/expert_dim/*.yaml`: expert-width sweep with 256, 384, and 512 dimensions.
- `chunked_flow/sweeps/ffn_multiplier/*.yaml`: FFN multiplier sweep with 8, 12, and 16.

Flow configs use the same YAML entrypoint style as the VAE trainer:

```bash
uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/chunked_flow_k2_gsm8k_6.5k.yaml
```

## Flow Dataset Cache

For repeated flow runs, preprocess teacher states once into a tensor cache:

```bash
uv run python scripts/preprocess_flow_dataset.py \
  --dataset-path sghosts/cf_gsm8k_6.5k_train \
  --dataset-split train \
  --output-dir data/flow_cache/gsm8k_6.5k_train
```

Then train against the cached config:

```bash
uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/chunked_flow_k2_gsm8k_6.5k_cached.yaml
```

The cache stores each sequence once as flat tensors and samples training windows by slicing, avoiding repeated Hugging Face row decoding.
