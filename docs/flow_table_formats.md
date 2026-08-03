# Flow Table Formats

Use these table shapes when reporting flow drafter training and eval results.

## Eval Summary

Use this for the latest eval result per config.

| config | seq@K | eval accept len | live accept len | real speed |
|---|---:|---:|---:|---:|
| example_config | 0.000 | 0.000 | 0.000 | 0.000x |

- `seq@K`: `token.sequence_match` mean. For `K=2`, this is equivalent to `seq@2`.
- `eval accept len`: teacher-forced `accept.greedy_prefix_len` mean.
- `live accept len`: live-generation `speedup.mean_accept_len`. Use `-` if speedup was not measured.
- `real speed`: `speedup.real` from eval JSON. Use `-` if speedup was not measured.

## Eval Progress

Use this for checkpoint-over-training eval progress. The epoch columns must match the checkpoints selected by the eval stride/offset for that run. Each epoch cell contains eval accept length.

| config | epoch a | epoch b | epoch c | ... | epoch n |
|---|---:|---:|---:|---:|---:|
| example_config | 0.000 | 0.000 | 0.000 | ... | 0.000 |

- Cell value: `accept.greedy_prefix_len` mean for that epoch checkpoint.
- If checkpoint names are trainer steps, map them to epoch numbers from the run's saved checkpoint order.
- Do not hardcode epoch columns. For example, if eval selected every 10th epoch, use columns such as `epoch 0`, `epoch 10`, `epoch 20`, `epoch 30`.
- Use `-` when that epoch was not evaluated or the checkpoint is missing.

## Train Loss Progress

Use this for training-loss progress. Include every epoch that exists in the run.

| config | epoch 1 | epoch 2 | epoch 3 | ... | epoch n |
|---|---:|---:|---:|---:|---:|
| example_config | 0.000 | 0.000 | 0.000 | ... | 0.000 |

- Cell value: `final_train_loss` for that epoch.
- Use the epoch-level final loss, not individual logging-step losses.

## Train Component Summary

Use this for final training component metrics.

| config | flow.mse | latent.mse | hidden.rel_mse | hidden.cos | logit.mse | expected accept |
|---|---:|---:|---:|---:|---:|---:|
| example_config | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

- Use the latest/final logged value for each component.
- `expected accept` corresponds to the training expected-accept component, usually logged as `verifier.expected_accept`.
- Use `-` if a component is not logged for a given run.
