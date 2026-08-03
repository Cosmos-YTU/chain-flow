# K Draft Length GSM8K 6.5K Flow Sweep 002

## Sweep Setup

- Purpose: test whether increasing draft length `K` improves quality and real speed after the first `K=2` sweep.
- Backbone: `Qwen/Qwen3.5-0.8B`.
- Train dataset: `data/flow_cache/gsm8k_6.5k_train`.
- Test dataset: `data/flow_cache/gsm8k_1k_test`.
- Fixed settings: `context_size=8`, `chunk_size=draft_length`, `ffn_multiplier=4`, `num_flow_steps=1`.
- Swept axes: `draft_length` in `4, 6, 8`; `expert_dim` in `384, 512`.
- Training: 40 epochs, learning rate `0.0003`, batch size `4096`, save strategy `epoch`, report target `none`.
- Eval progress checkpoints were selected from saved epoch checkpoints. This sweep currently has test evals only.

## Training Results

### Train Loss Progress

| config | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | epoch 6 | epoch 7 | epoch 8 | epoch 9 | epoch 10 | epoch 11 | epoch 12 | epoch 13 | epoch 14 | epoch 15 | epoch 16 | epoch 17 | epoch 18 | epoch 19 | epoch 20 | epoch 21 | epoch 22 | epoch 23 | epoch 24 | epoch 25 | epoch 26 | epoch 27 | epoch 28 | epoch 29 | epoch 30 | epoch 31 | epoch 32 | epoch 33 | epoch 34 | epoch 35 | epoch 36 | epoch 37 | epoch 38 | epoch 39 | epoch 40 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k8_expert_dim_512 | 3.043 | 2.553 | 2.398 | 2.314 | 2.261 | 2.224 | 2.197 | 2.176 | 2.159 | 2.145 | 2.133 | 2.123 | 2.114 | 2.106 | 2.099 | 2.092 | 2.086 | 2.080 | 2.075 | 2.070 | 2.065 | 2.061 | 2.057 | 2.053 | 2.049 | 2.046 | 2.042 | 2.039 | 2.036 | 2.033 | 2.030 | 2.027 | 2.025 | 2.022 | 2.020 | 2.017 | 2.015 | 2.013 | 2.011 | 2.009 |
| k8_expert_dim_384 | 3.185 | 2.668 | 2.524 | 2.444 | 2.394 | 2.358 | 2.331 | 2.310 | 2.293 | 2.279 | 2.266 | 2.256 | 2.247 | 2.239 | 2.232 | 2.225 | 2.219 | 2.214 | 2.209 | 2.204 | 2.200 | 2.196 | 2.192 | 2.188 | 2.185 | 2.182 | 2.179 | 2.176 | 2.173 | 2.170 | 2.167 | 2.165 | 2.163 | 2.161 | 2.158 | 2.156 | 2.154 | 2.153 | 2.151 | 2.149 |
| k6_expert_dim_512 | 2.878 | 2.327 | 2.184 | 2.110 | 2.064 | 2.032 | 2.007 | 1.987 | 1.971 | 1.957 | 1.945 | 1.935 | 1.926 | 1.917 | 1.910 | 1.903 | 1.897 | 1.891 | 1.885 | 1.880 | 1.875 | 1.870 | 1.866 | 1.862 | 1.857 | 1.853 | 1.849 | 1.844 | 1.840 | 1.837 | 1.833 | 1.830 | 1.827 | 1.824 | 1.821 | 1.819 | 1.816 | 1.813 | 1.811 | 1.809 |
| k6_expert_dim_384 | 3.053 | 2.500 | 2.334 | 2.245 | 2.190 | 2.154 | 2.129 | 2.110 | 2.094 | 2.081 | 2.070 | 2.061 | 2.053 | 2.045 | 2.038 | 2.032 | 2.026 | 2.021 | 2.016 | 2.012 | 2.007 | 2.003 | 1.999 | 1.996 | 1.992 | 1.989 | 1.986 | 1.982 | 1.980 | 1.977 | 1.974 | 1.971 | 1.969 | 1.967 | 1.964 | 1.962 | 1.960 | 1.957 | 1.956 | 1.954 |
| k4_expert_dim_512 | 2.606 | 2.022 | 1.887 | 1.817 | 1.771 | 1.739 | 1.714 | 1.695 | 1.679 | 1.665 | 1.653 | 1.642 | 1.632 | 1.625 | 1.617 | 1.609 | 1.603 | 1.597 | 1.591 | 1.586 | 1.580 | 1.576 | 1.571 | 1.566 | 1.562 | 1.558 | 1.554 | 1.550 | 1.547 | 1.543 | 1.539 | 1.536 | 1.533 | 1.529 | 1.527 | 1.523 | 1.521 | 1.518 | 1.515 | 1.513 |
| k4_expert_dim_384 | 2.809 | 2.198 | 2.038 | 1.958 | 1.910 | 1.877 | 1.853 | 1.834 | 1.819 | 1.807 | 1.796 | 1.786 | 1.777 | 1.770 | 1.764 | 1.757 | 1.751 | 1.746 | 1.741 | 1.736 | 1.732 | 1.728 | 1.723 | 1.720 | 1.716 | 1.712 | 1.709 | 1.706 | 1.703 | 1.700 | 1.696 | 1.693 | 1.691 | 1.688 | 1.686 | 1.683 | 1.681 | 1.678 | 1.676 | 1.674 |

The train-loss cells are the mean of logged training losses within each epoch. The Trainer output only includes one cumulative `train_loss` for the full run, so epoch-level values are reconstructed from logged `loss` entries.

### Train Component Summary

| config | flow.mse | latent.mse | hidden.rel_mse | hidden.cos | logit.mse | expected accept |
|---|---:|---:|---:|---:|---:|---:|
| k8_expert_dim_512 | 0.894 | 0.653 | 0.686 | 0.397 | - | -1.516 |
| k8_expert_dim_384 | 0.963 | 0.672 | 0.706 | 0.410 | - | -1.406 |
| k6_expert_dim_512 | 0.849 | 0.594 | 0.611 | 0.348 | - | -1.633 |
| k6_expert_dim_384 | 0.918 | 0.616 | 0.633 | 0.362 | - | -1.531 |
| k4_expert_dim_512 | 0.772 | 0.524 | 0.529 | 0.295 | - | -1.508 |
| k4_expert_dim_384 | 0.835 | 0.556 | 0.560 | 0.314 | - | -1.383 |

Note: this sweep logged `logit.ce`; the requested component table reserves `logit.mse`, so that column is `-` for this report.

## Test Eval Results

### Eval Summary

This table uses the latest evaluated checkpoint for each config. `seq@K` is full draft-sequence match at that config's `draft_length`, not the old `K=2` metric.

| config | seq@K | eval accept len | live accept len | real speed |
|---|---:|---:|---:|---:|
| k4_expert_dim_512 | 0.258 | 2.067 | 0.399 | 0.320x |
| k4_expert_dim_384 | 0.229 | 1.939 | 0.255 | 0.295x |
| k6_expert_dim_512 | 0.083 | 2.259 | 0.185 | 0.265x |
| k6_expert_dim_384 | 0.071 | 2.123 | 0.349 | 0.297x |
| k8_expert_dim_512 | 0.023 | 2.290 | 0.233 | 0.270x |
| k8_expert_dim_384 | 0.019 | 2.139 | 0.256 | 0.269x |

### Test Eval Progress

Cell values are eval accept length (`accept.greedy_prefix_len`).

| config | epoch 1 | epoch 6 | epoch 11 | epoch 16 | epoch 21 | epoch 26 | epoch 31 | epoch 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k8_expert_dim_512 | 1.564 | 2.065 | 2.150 | 2.197 | 2.233 | 2.248 | 2.276 | 2.290 |
| k8_expert_dim_384 | 1.389 | 1.919 | 2.011 | 2.049 | 2.082 | 2.105 | 2.124 | 2.139 |
| k6_expert_dim_512 | 1.591 | 2.047 | 2.122 | 2.166 | 2.195 | 2.223 | 2.239 | 2.259 |
| k6_expert_dim_384 | 1.449 | 1.914 | 1.990 | 2.036 | 2.067 | 2.087 | 2.106 | 2.123 |
| k4_expert_dim_512 | 1.570 | 1.899 | 1.962 | 1.999 | 2.018 | 2.037 | 2.051 | 2.067 |
| k4_expert_dim_384 | 1.419 | 1.774 | 1.837 | 1.874 | 1.896 | 1.913 | 1.927 | 1.939 |

Checkpoint-step mapping for the progress table:

| config | epoch 1 | epoch 6 | epoch 11 | epoch 16 | epoch 21 | epoch 26 | epoch 31 | epoch 36 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k8_expert_dim_512 | 2706 | 16236 | 29766 | 43296 | 56826 | 70356 | 83886 | 97416 |
| k8_expert_dim_384 | 2706 | 16236 | 29766 | 43296 | 56826 | 70356 | 83886 | 97416 |
| k6_expert_dim_512 | 2719 | 16314 | 29909 | 43504 | 57099 | 70694 | 84289 | 97884 |
| k6_expert_dim_384 | 2719 | 16314 | 29909 | 43504 | 57099 | 70694 | 84289 | 97884 |
| k4_expert_dim_512 | 2731 | 16386 | 30041 | 43696 | 57351 | 71006 | 84661 | 98316 |
| k4_expert_dim_384 | 2731 | 16386 | 30041 | 43696 | 57351 | 71006 | 84661 | 98316 |

## Analysis

- Increasing `K` raises eval accept length, but full draft-sequence match (`seq@K`) drops sharply as K grows. This is expected: exact-match over 8 positions is much stricter than exact-match over 4 positions.
- `expert_dim_512` remains better than `expert_dim_384` at the same `K` for quality and acceptance.
- `K=8` has the highest accepted length (`2.290` for `expert_dim_512`), but it does not produce the best real speed.
- `K=4 expert_dim_512` is the best overall result in this sweep: highest `seq@K`, best real speed, and strong accepted length.
- Real speed is still below `1.0x` for all configs. Compared with the first `K=2` sweep, this sweep did not yet cross the baseline-speed threshold.
- The result suggests that simply increasing K is not enough; we need to balance accepted length with verifier/drafter overhead and full-sequence accuracy.

## Decision Notes

- Best quality candidate: `k4_expert_dim_512`.
- Best real-speed candidate: `k4_expert_dim_512`.
- Best accepted-length candidate: `k8_expert_dim_512`.
- For the next discussion, decide whether to refine around `K=4` for speed/quality, or profile `K=8` to understand why higher accepted length is not translating into real speed.
