# VAE Training Configs

VAE configs live here so the top-level `train_configs/` directory only contains shared docs and one-off legacy configs.

## Smoke

- `smoke_vae.yaml`: quick VAE run for config and trainer wiring.

## Sweeps

- `sweeps/vae_1k_*.yaml`: smaller GSM8K teacher-state sweeps.
- `sweeps/vae_6.5k_*.yaml`: full 6.5k GSM8K teacher-state sweeps.

Run one config with:

```bash
uv run python scripts/train_vae.py train_configs/vae/sweeps/vae_6.5k_lr3e3.yaml
```

VAE training creates an internal token-level train/validation split from `validation_fraction` and evaluates every epoch.
