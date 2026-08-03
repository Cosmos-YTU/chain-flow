# Next Sweep Context: Trainable VAE, Full Windows, and Runtime Break-Even

## What Changed Since Frozen-VAE Flow Training

- Made the VAE optionally trainable during flow training with `train_vae: true`.
- Added separate optimizer parameter groups so VAE parameters use a smaller LR via `vae_learning_rate_multiplier`.
- Fixed `DataParallel` replica dtype/device issues by avoiding parameter-iterator lookup inside forward paths.
- Added epoch validation loss to flow training, so training logs now include `eval_loss`.
- Added plotting for train/eval loss curves.
- Added trace tooling for backbone-only, drafter-only, and drafter+verifier generation.
- Patched trace tooling to load real cached dataset windows instead of handwritten prompts.

## Overfit Findings

- Built trainable-VAE overfit configs on a fixed `1024`-window subset.
- Trainable VAE helped substantially: the model could fit sampled cached windows much better than the frozen-VAE setup.
- Static overfit success did not imply live rollout success because sampled windows were not rollout-closed.
- A traced window could be predicted exactly for the first chunk, then fail after the prefix advanced to adjacent windows that were not necessarily in the sampled subset.

## Capacity Sweep

- Swept around the best overfit setup:
  - `expert_dim`: `512`, `768`, `1024`
  - `ffn_multiplier`: `4`, `6`
- Best capacity config:
  - `context_size=8`
  - `draft_length=4`
  - `expert_dim=512`
  - `ffn_multiplier=6`
  - `num_flow_steps=1`
  - `train_vae=true`
  - `learning_rate=0.0015`
  - `vae_learning_rate_multiplier=0.1`
- Tried `learning_rate=0.003` with cosine scheduler; it was worse and was reverted.
- Larger capacity did not solve the main issue, so pure capacity is not the current bottleneck.

## Full-Window Training

- Added a full-window config using all available train windows instead of a sampled `1024`-window subset.
- Increased batch size to `6144` while keeping `learning_rate=0.0015`.
- Full-window test eval on `gsm8k_1k_test` improved quality:
  - `token.top1_match`: about `70.6%`
  - `accept.greedy_prefix_len`: about `2.58 / 4`
  - live `mean_accept_len`: about `2.1`
- This suggests the model quality is becoming promising once the training data covers more rollout-relevant prefixes.

## Runtime Profiling

- Added profiler support for direct end-to-end comparison:
  - original backbone generation
  - drafter+verifier speculative generation
- Current profile on the full-window `K=4` model:
  - original backbone alone: about `6.25s`
  - drafter+verifier: about `12.42s`
  - real speed: about `0.50x`
- Added detailed drafter+verifier timing groups.
- Main bottlenecks:
  - verifier LM forward dominates runtime
  - cache repair is also expensive
  - the drafter itself is under `1%` of generation time

## Break-Even Note

Let:

- `A`: mean accepted draft tokens per pass
- `c1`: greedy backbone cost per generated token
- `pass_cost`: speculative pass cost

Break-even requires:

```text
1 + A > pass_cost / c1
```

Using the current profile:

```text
c1 = 6.2495 / 485 = 0.01289 s/token
pass_cost = 12.4224 / 153 = 0.08119 s/pass
A_required > 5.3
```

Current observed live acceptance is about:

```text
mean_accept_len = 2.1
```

So `K=4` cannot break even with the current verifier/cache path, even with perfect `A=4`.

## Next Sweep Implication

- Do not focus only on wider flow experts.
- Either increase accepted chunk length with larger `K` while preserving accuracy, or reduce verifier/cache commit overhead.
- The next sweep should be designed around the runtime break-even constraint, not only static token accuracy.
