# Flow Sweeps

| sweep | dataset | K/C | best quality | best real speed | conclusion |
|---|---|---:|---|---|---|
| [k2_gsm8k_6.5k_sweep_001](k2_gsm8k_6.5k_sweep_001.md) | GSM8K 6.5K train / 1K test | 2/2 | expert_dim_512 | expert_dim_384 | Expert width dominates; real speed remains below baseline. |
| [k_draft_gsm8k_6.5k_sweep_002](k_draft_gsm8k_6.5k_sweep_002.md) | GSM8K 6.5K train / 1K test | 4,6,8 / same | k4_expert_dim_512 | k4_expert_dim_512 | K4 is best overall; larger K improves accepted length but not real speed. |

## Next Sweep Context

- [Trainable VAE, full-window training, and runtime break-even notes](next_sweep_trainable_vae_full_windows.md)
