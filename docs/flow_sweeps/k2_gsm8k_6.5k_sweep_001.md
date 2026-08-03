# K2 GSM8K 6.5K Flow Sweep 001

## Sweep Setup

- Purpose: first single-drafter flow sweep for `K=2`, `C=2`.
- Backbone: `Qwen/Qwen3.5-0.8B`.
- Train dataset: `data/flow_cache/gsm8k_6.5k_train`.
- Test dataset: `data/flow_cache/gsm8k_1k_test`.
- Training: 20 epochs, learning rate `0.0003`, batch size `4096`, save strategy `epoch`, report target `none`.
- Swept axes: `context_size` in `8, 12, 16`; `expert_dim` in `256, 384, 512`; `ffn_multiplier` in `8, 12, 16`.
- Eval progress checkpoints were selected from saved epoch checkpoints. This run has test progress evals only; training-data evals were final-only.

## Training Results

### Train Loss Progress

| config | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | epoch 6 | epoch 7 | epoch 8 | epoch 9 | epoch 10 | epoch 11 | epoch 12 | epoch 13 | epoch 14 | epoch 15 | epoch 16 | epoch 17 | epoch 18 | epoch 19 | epoch 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| expert_dim_512 | 2.163 | 1.320 | 1.203 | 1.142 | 1.101 | 1.072 | 1.048 | 1.030 | 1.015 | 1.002 | 0.990 | 0.980 | 0.970 | 0.962 | 0.955 | 0.948 | 0.942 | 0.936 | 0.932 | 0.928 |
| expert_dim_384 | 2.483 | 1.456 | 1.307 | 1.242 | 1.200 | 1.170 | 1.147 | 1.128 | 1.113 | 1.100 | 1.089 | 1.080 | 1.071 | 1.064 | 1.057 | 1.050 | 1.045 | 1.040 | 1.036 | 1.032 |
| expert_dim_256 | 2.965 | 2.007 | 1.754 | 1.628 | 1.558 | 1.515 | 1.482 | 1.459 | 1.440 | 1.425 | 1.412 | 1.401 | 1.391 | 1.384 | 1.377 | 1.371 | 1.366 | 1.362 | 1.358 | 1.356 |
| ffn_multiplier_16 | 3.636 | 2.818 | 2.665 | 2.591 | 2.539 | 2.497 | 2.460 | 2.431 | 2.409 | 2.393 | 2.379 | 2.369 | 2.359 | 2.352 | 2.346 | 2.341 | 2.336 | 2.333 | 2.330 | 2.328 |
| ffn_multiplier_12 | 3.676 | 2.843 | 2.682 | 2.606 | 2.551 | 2.507 | 2.470 | 2.443 | 2.422 | 2.405 | 2.392 | 2.381 | 2.371 | 2.364 | 2.357 | 2.352 | 2.347 | 2.344 | 2.341 | 2.339 |
| ffn_multiplier_8 | 3.684 | 2.886 | 2.727 | 2.644 | 2.585 | 2.536 | 2.498 | 2.471 | 2.451 | 2.435 | 2.423 | 2.412 | 2.403 | 2.397 | 2.390 | 2.385 | 2.381 | 2.378 | 2.375 | 2.373 |
| context_size_8 | 4.052 | 3.266 | 3.114 | 3.038 | 2.987 | 2.954 | 2.929 | 2.908 | 2.890 | 2.874 | 2.861 | 2.850 | 2.841 | 2.833 | 2.827 | 2.821 | 2.817 | 2.813 | 2.810 | 2.808 |
| context_size_12 | 4.209 | 3.446 | 3.293 | 3.222 | 3.176 | 3.145 | 3.122 | 3.103 | 3.088 | 3.075 | 3.064 | 3.055 | 3.046 | 3.039 | 3.032 | 3.027 | 3.022 | 3.018 | 3.015 | 3.013 |
| context_size_16 | 4.297 | 3.558 | 3.403 | 3.331 | 3.284 | 3.252 | 3.228 | 3.209 | 3.194 | 3.181 | 3.170 | 3.161 | 3.153 | 3.146 | 3.140 | 3.135 | 3.130 | 3.127 | 3.124 | 3.122 |

The train-loss cells are the mean of logged training losses within each epoch. The Trainer output only includes one cumulative `train_loss` for the full run, so epoch-level values are reconstructed from logged `loss` entries.

### Train Component Summary

| config | flow.mse | latent.mse | hidden.rel_mse | hidden.cos | logit.mse | expected accept |
|---|---:|---:|---:|---:|---:|---:|
| expert_dim_512 | 0.539 | 0.325 | 0.340 | 0.184 | - | -1.312 |
| expert_dim_384 | 0.580 | 0.363 | 0.371 | 0.202 | - | -1.258 |
| expert_dim_256 | 0.713 | 0.508 | 0.445 | 0.244 | - | -1.164 |
| ffn_multiplier_16 | 1.203 | 1.130 | 0.608 | 0.337 | - | -1.055 |
| ffn_multiplier_12 | 1.207 | 1.127 | 0.610 | 0.339 | - | -1.047 |
| ffn_multiplier_8 | 1.222 | 1.134 | 0.623 | 0.346 | - | -1.039 |
| context_size_8 | 1.399 | 1.203 | 0.718 | 0.406 | - | -0.836 |
| context_size_12 | 1.458 | 1.238 | 0.771 | 0.441 | - | -0.707 |
| context_size_16 | 1.482 | 1.258 | 0.801 | 0.461 | - | -0.644 |

Note: this sweep logged `logit.ce`; the requested component table reserves `logit.mse`, so that column is `-` for this report.

## Test Eval Results

### Eval Summary

| config | seq@2 | eval accept len | live accept len | real speed |
|---|---:|---:|---:|---:|
| expert_dim_512 | 0.668 | 1.531 | 0.776 | 0.400x |
| expert_dim_384 | 0.640 | 1.488 | 0.840 | 0.422x |
| expert_dim_256 | 0.567 | 1.384 | 0.640 | 0.374x |
| ffn_multiplier_16 | 0.523 | 1.279 | 0.396 | 0.331x |
| ffn_multiplier_12 | 0.518 | 1.272 | 0.408 | 0.330x |
| ffn_multiplier_8 | 0.506 | 1.251 | 0.454 | 0.342x |
| context_size_8 | 0.399 | 1.025 | 0.261 | 0.301x |
| context_size_12 | 0.341 | 0.890 | 0.213 | 0.293x |
| context_size_16 | 0.308 | 0.813 | 0.133 | 0.275x |


### Test Eval Progress

Cell values are eval accept length (`accept.greedy_prefix_len`).

| config | epoch 1 | epoch 6 | epoch 11 | epoch 16 |
|---|---:|---:|---:|---:|
| expert_dim_512 | 1.331 | 1.481 | 1.510 | 1.525 |
| expert_dim_384 | 1.281 | 1.442 | 1.472 | 1.483 |
| expert_dim_256 | 1.196 | 1.339 | 1.370 | 1.380 |
| ffn_multiplier_16 | 1.183 | 1.235 | 1.261 | 1.275 |
| ffn_multiplier_12 | 1.046 | 1.227 | 1.255 | 1.267 |
| ffn_multiplier_8 | 1.020 | 1.207 | 1.236 | 1.247 |
| context_size_8 | 0.769 | 0.974 | 1.008 | 1.020 |
| context_size_12 | 0.630 | 0.839 | 0.871 | 0.884 |
| context_size_16 | 0.543 | 0.754 | 0.790 | 0.807 |

## Train Eval Results

### Eval Summary

| config | seq@2 | eval accept len | live accept len | real speed |
|---|---:|---:|---:|---:|
| expert_dim_512 | 0.690 | 1.569 | 0.584 | 0.368x |
| expert_dim_384 | 0.655 | 1.515 | 0.663 | 0.376x |
| expert_dim_256 | 0.576 | 1.402 | 0.541 | 0.370x |
| ffn_multiplier_16 | 0.529 | 1.292 | 0.356 | 0.321x |
| ffn_multiplier_12 | 0.523 | 1.283 | 0.271 | 0.293x |
| ffn_multiplier_8 | 0.510 | 1.262 | 0.420 | 0.332x |
| context_size_8 | 0.402 | 1.035 | 0.191 | 0.295x |
| context_size_12 | 0.346 | 0.904 | 0.139 | 0.270x |
| context_size_16 | 0.312 | 0.827 | 0.186 | 0.284x |


No train-data eval progress table is available for this sweep because the training-data eval run did not produce checkpoint-suffixed files.

## Analysis

- `expert_dim` is the strongest sweep axis. `expert_dim_512` is best by `seq@2` and eval accept length on both test and train data.
- `expert_dim_384` is close to `expert_dim_512` and has the best real speed among the final test evals, though all real-speed values remain below baseline.
- Increasing `context_size` hurts quality in this architecture: `context_size_8` > `context_size_12` > `context_size_16`.
- `ffn_multiplier` has a modest effect. Larger FFN is slightly better in quality, but the gains are much smaller than increasing `expert_dim`.
- Eval accept length improves monotonically across evaluated checkpoints for every config, but gains flatten after the earlier checkpoints.
- Acceptance quality is promising for `K=2`, but real speed is still below `1.0x`; runtime needs to be considered alongside the next `K` sweep.

## Decision Notes

- Best quality candidate: `expert_dim_512`.
- Cost/quality candidate: `expert_dim_384`.
- Keep `context_size=8` as the baseline unless a later architecture change makes longer context useful.
- Discuss next sweep separately, likely around increasing `K` while preserving the strongest settings from this sweep.
