# Chained Flow Project Summary

This document summarizes the state of the `chained-flow` project: what was built, what was tried, which results were useful, and what direction currently looks most promising.

## Objective

The project investigates speculative drafting for `Qwen/Qwen3.5-0.8B` using learned drafters that predict multiple future tokens from the model hidden state.

The target behavior is:

1. Run the backbone model on a prompt.
2. Use a small drafter to propose `K` future tokens.
3. Verify those tokens with the original backbone.
4. Accept as many as possible.
5. Achieve real generation speedup by accepting enough draft tokens to cover verifier and cache-management overhead.

The central constraint discovered during the project is that cached, teacher-forced token accuracy is not enough. The drafter must also survive live rollout, where every accepted or rejected token changes the next context.

## Data And Execution Setup

Most training and evaluation moved to cached teacher states.

Important datasets and caches:

- Train cache: `data/flow_cache/gsm8k_6.5k_train`
- Test cache: `data/flow_cache/gsm8k_1k_test`
- Original test dataset used earlier: `sghosts/cf_gsm8k_1k_test`

The project uses `uv`, so scripts are normally run as:

```bash
uv run python <script> <args>
```

Caching became important because repeatedly materializing hidden states from the dataset made flow training look frozen or extremely slow. The data pipeline now supports loading cached flow datasets directly, and materialization paths have progress reporting.

## Metrics

The project moved away from raw unnormalized losses as the only signal and added more interpretable metrics.

### Training Metrics

Training and validation commonly log:

- `loss`: total weighted objective.
- `flow.mse`: flow/velocity prediction loss.
- `latent.mse`: latent reconstruction loss when a VAE is used.
- `hidden.rel_mse`: hidden-state reconstruction error normalized by teacher hidden-state energy.
- `hidden.cos`: cosine similarity between predicted and teacher hidden states.
- `logit.ce`: cross-entropy between predicted logits and teacher tokens.
- `verifier.expected_accept`: differentiable proxy for accepting prefix tokens.

`verifier.expected_accept` is usually optimized as a negative contribution to the loss, so more negative is better. For `K=8` and `gamma=0.8`, the maximum possible magnitude is about `4.161`.

### Evaluation Metrics

Flow evaluation reports a broader set of metrics:

- Hidden-space metrics: `hidden.rel_mse`, `hidden.rel_rmse`, `hidden.cosine_similarity`.
- Latent-space metrics when applicable: `latent.mse`.
- Logit metrics: `logit.ce`, `logit.js_div_to_teacher`.
- Token metrics: `token.top1_match`, `token.top5_contains`, `token.top10_contains`, teacher rank.
- Acceptance metrics: `accept.greedy_prefix_len`, `accept.rate@i`, `token.sequence_match`.
- Speed metrics when requested: `speedup.*`, real end-to-end speed, decode-only speed.

The key distinction is:

- Cached evaluation is teacher-forced. It checks whether predictions match cached teacher windows.
- Live profiling/evaluation rolls forward autoregressively. It checks whether the drafter remains useful once its own accepted/rejected tokens affect the future context.

This distinction became one of the main lessons of the project.

## VAE Work

The first stage used hidden-state VAEs to compress and reconstruct backbone hidden states.

### Initial Hidden VAE

The initial VAE encoded individual hidden states. It was enough to start flow modeling, but the project quickly needed better sequence modeling and better metrics.

Work completed:

- Added VAE checkpoint loading and config handling.
- Added VAE eval over checkpoints.
- Added unique output files that include checkpoint and dataset names.
- Added a VAE metrics README under `docs/vae_eval_metrics/`.
- Added support for evaluating cached flow datasets.

### Transformer Hidden VAE

A sequence-aware `TransformerHiddenVAE` was added.

Important implementation points:

- Supports input hidden sequences shaped `[B, L, D]`.
- Adds transformer depth, heads, dropout, max sequence length, and latent size controls.
- Used for sequence length `8`, matching the main `K=8` drafting experiments.

Important paths:

- `src/chained_flow/vae/transformer_hidden.py`
- `scripts/train_transformer_hidden_vae.py`
- `scripts/eval_vae.py`
- `train_configs/vae/transformer_hidden/`

### VAE Findings

The transformer VAE was substantially more useful than the original simple hidden VAE for drafting.

What worked:

- Sequence length `8` matched the first serious drafting target.
- Wider and stronger transformer VAE configs performed well.
- KL beta sweeps helped identify usable compression behavior.
- The full-dataset VAE used later was:
  - `out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full`

What did not work as expected:

- Jointly training the VAE with the flow drafter did not help in the later sweeps.
- Freezing the VAE during flow training became the better choice.

This led to a strong VAE-based baseline:

- Block-causal flow.
- Frozen transformer VAE.
- `init_mode: delta`.
- `steps=2`, `layers=4`, `dim=768`.
- Reconstruction-heavy loss.

On cached evaluation this baseline reached very strong teacher-forced metrics, including near-perfect expected acceptance.

## Original Chunked Flow Drafter

The first drafter family was a VAE-latent flow model.

Important paths:

- `src/chained_flow/drafters/chunked_flow.py`
- `src/chained_flow/training/train_chunked_flow.py`
- `scripts/train_chunked_flow.py`
- `scripts/eval_chunked_flow.py`

The initial plan was conservative:

- Start with `K=2`, `C=2`, a single drafter.
- Confirm that two-token drafting can work.
- Then gradually increase `K` to `4`, `6`, and `8`.
- Only after that consider multiple drafters or chained behavior.

### Training And Evaluation Infrastructure

The flow infrastructure was expanded substantially:

- YAML config parsing for training.
- Cached dataset loading.
- Materialization progress bars.
- Eval script for flow checkpoints.
- Eval-all-checkpoints support with checkpoint stride.
- Distinct output filenames that include checkpoint and dataset.
- Sweep scripts under `tmp/`.
- Plotting scripts for train and validation curves.
- Table formats documented under `docs/flow_table_formats.md`.

## Profiling And Speedup Work

A dedicated profiler was added:

- `scripts/profile_flow_drafter.py`

It measures:

- Backbone-only generation.
- Drafter-only components.
- Drafter plus verifier.
- VAE encode/decode time when applicable.
- Flow integration time.
- Verifier forward time.
- Cache repair time.
- Real speedup and decode-only speedup.

The most important profiling result was that the drafter itself is not the bottleneck.

One representative profile showed:

- Backbone alone: about `6.19s`.
- Drafter plus verifier: about `16.65s`.
- Real speedup: `0.372x`.
- Mean accepted/drafted: `1.320 / 7.185`.
- Drafter time: about `0.51s`, only `3.1%` of generation.
- Verifier plus nested/cache work dominated runtime.

This led to a practical lower bound: the drafter needs roughly `5` accepted tokens per step just to cover the verification and cache overhead in that setup.

The conclusion was that optimizing drafter runtime further was not the main priority. Improving live acceptance was.

## Block-Causal VAE-Based Flow

The next major architecture was a block-causal flow expert over VAE latents.

Conceptually:

```text
context_hidden [B, M, D]
  -> VAE encoder
context_latent [B, M, Z]

target_hidden [B, K, D]
  -> VAE encoder
target_latent [B, K, Z]

z_tau [B, K, Z]
  + context_latent
  + previous draft latents
  -> transformer flow expert
  -> predicted velocity
  -> integrated latent
  -> VAE decoder
  -> predicted hidden [B, K, D]
  -> lm_head
  -> draft tokens
```

The implementation introduced `FusedBlockCausalFlowExpert` with context tokens, previous draft tokens, current noisy draft tokens, time embeddings, segment embeddings, and transformer blocks.

### K And Expert Experiments

Several K/chunk layouts were tried:

- `K=2`, `C=2`
- `K=4`, `C=4`
- `K=8`, `C=8`
- `K=16`, `C=2`
- Multi-expert forms such as `K=8,C=4` and `K=8,C=2`

The idea was that multiple experts might help handle different token blocks. In practice, multiple experts did not break the live acceptance limit.

### Overfit Experiments

Overfit sweeps were used to check whether the architecture could memorize a small subset.

What was learned:

- Training on a tiny window subset can produce excellent cached metrics.
- Cached metrics can still fail in live rollout.
- Training all windows or larger subsets is necessary to understand generalization.
- Some earlier traces looked strange because the sampled prompt/window was not necessarily inside the exact overfit subset.

### Lambda Sweeps

Loss-weight sweeps were performed over reconstruction, flow, CE, and expected acceptance.

The best VAE-based result came from a reconstruction-heavy setup:

- `steps=2`
- `layers=4`
- `dim=768`
- frozen VAE
- `K=8`
- `init_mode=delta`

Same-subset cached evaluation became nearly perfect:

- `token.top1_match = 1.0`
- `token.top5_contains = 1.0`
- `accept.rate@1..8 = 1.0`
- `accept.greedy_prefix_len = 8`

But live profiling still showed poor real speedup and low live acceptance. This made rollout drift the central issue.

## Init Distribution Experiments

A major improvement came from questioning the flow initialization distribution.

Earlier flow used random noise. That is natural for diffusion-style training, but it is a poor match for speculative hidden-state drafting, where future hidden states are close to recent context hidden states.

Three init modes were implemented:

- `noise`: Gaussian noise.
- `repeat_last`: initialize all draft positions from the last context hidden/latent.
- `delta`: extrapolate from the last two context hidden/latent states.

For VAE-latent flow, the result was clear:

- `repeat_last` was much better than noise.
- `delta` was best overall.

Representative cached validation results:

| init mode | flow.mse | hidden.rel_mse | logit.ce | expected accept |
| --- | ---: | ---: | ---: | ---: |
| noise | 0.6180 | 0.4731 | 0.0044 | -4.1260 |
| repeat_last | 0.1126 | 0.2796 | 0.0016 | -4.1560 |
| delta | 0.1744 | 0.2187 | 0.0015 | -4.1560 |

`delta` became the new baseline.

## Reference Work

Reference material was reviewed under:

- `tmp/reference_papers/`
- `tmp/reference_repos/`

Important papers/repos:

- `dflash.pdf`
- `coladlm.pdf`
- `orthrus.pdf`
- `moe-diff.pdf`

Main conclusions:

- CoLaDLM resembles the block-causal latent setup: historical clean latent blocks condition the current noisy block.
- DFlash strongly motivates conditioning draft layers with target-model hidden features via KV injection.
- Orthrus motivates sharing or committing target-model cache state more directly.
- MoE-Diff is more about MoE FFNs and shared experts than the specific conditioning problem here.

This pushed the project toward hidden-KV conditioning and eventually toward removing the VAE from the drafter.

## Hidden-KV No-VAE Drafter

The latest architecture removes the VAE entirely and performs flow modeling directly in Qwen hidden space.

Architecture name:

```yaml
architecture: hidden_kv_flow
```

Important idea:

- The draft sequence is represented directly as `[B, K, 1024]`.
- Context hidden states are used as KV memory.
- Draft hidden states query the context through cross-attention.
- The model predicts hidden-state velocities directly.
- The LM head maps predicted hidden states to draft-token logits.

Tensor flow:

```text
context_hidden [B, M, 1024]
draft_state    [B, K, 1024]

for each flow block:
  draft_state -> self-attention over K draft positions
  draft_state queries context_hidden as K/V memory
  draft_state -> FFN

output velocity [B, K, 1024]
integrate to predicted_hidden [B, K, 1024]
lm_head -> logits [B, K, vocab]
```

This is closer to DFlash-style conditioning than the VAE-latent approach, but it is implemented as an external drafter rather than inside the backbone.

Important constraints:

- `expert_dim` must match hidden size, currently `1024`.
- No `vae_dir` is required.
- `init_mode: delta` still applies, now in hidden space.

Important config paths:

- `train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/`
- `train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/sweep_4096w/`
- `train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/lr_scheduler_4096w/`

## Hidden-KV Sweep Results

The first hidden-KV sweep used `4096` windows per epoch and tested:

- Layers: `2`, `4`, `6`
- FFN multiplier: `2`, `4`, `6`
- Flow steps: `1`, `2`
- Context size: `8`, `16`
- Learning rate: `3e-4`, `1e-3`

Main result:

- `layers=6`, `ffn=4`, `steps=2`, `context=8`, `lr=1e-3` was best overall in that sweep.
- More layers helped.
- `ffn=6` helped somewhat but less than more depth.
- `steps=2` beat `steps=1`.
- `context=16` did not help.
- `lr=3e-4` underfit badly.

Representative best hidden-KV result from the first sweep:

- Validation loss: about `10.08`
- `flow.mse`: about `29.95`
- `hidden.rel_mse`: about `1.566`
- `logit.ce`: about `0.074`
- `verifier.expected_accept`: about `-3.983`

This was worse than the best VAE+delta cached metrics, but it showed the architecture can get meaningful acceptance without a VAE.

## Hidden-KV LR And Scheduler Sweep

Because hidden-KV losses looked strongly affected by learning rate, a scheduler/LR sweep was run around the best hidden-KV config.

Configs tested:

- `lr=1e-3`, constant.
- `lr=3e-3`, constant.
- `lr=1e-2`, constant.
- `lr=3e-3`, cosine warmup.
- `lr=1e-2`, cosine warmup.

Main result:

- Constant LR was better than decaying schedules.
- `1e-3 constant` was best for expected acceptance.
- `3e-3 constant` was best for geometry/reconstruction, but worse for token CE and expected acceptance.
- `1e-2` was unstable or collapsed.

Representative results:

| config | hidden.rel_mse | logit.ce | expected accept | interpretation |
| --- | ---: | ---: | ---: | --- |
| `lr1e3_constant` | 0.540 | 0.150 | -3.892 | best final accept balance |
| `lr3e3_constant` | 0.380 | 0.484 | -3.518 | better geometry, worse tokens |
| `lr3e3_cosine_warmup` | worse | 0.104 | -3.491 | CE improved but accept worse |
| `lr1e2_*` | poor | poor | poor | too aggressive |

The key conclusion is that hidden-KV needs LR and loss-balancing work. The architecture is not yet beating the VAE+delta cached baseline, but it is the cleaner direction because it removes compression loss and enables direct hidden-state conditioning.

## What Went Well

Several parts of the project are now much stronger than at the start:

- Evaluation is much more reliable and interpretable.
- Cached dataset loading avoids slow repeated materialization.
- VAE metrics are documented and more understandable.
- Flow metrics include token and acceptance behavior, not just MSE.
- Eval-all-checkpoints and stride-based evaluation make sweep analysis easier.
- Profiling separates drafter cost from verifier/cache overhead.
- The project now has a realistic speedup model instead of proxy-only claims.
- The transformer VAE became a strong component.
- Freezing the VAE was identified as better than joint training for the VAE-based flow.
- `delta` initialization was a major improvement over random noise.
- Hidden-KV conditioning was implemented and tested.
- Plotting and sweep scripts make repeated experiments faster.

## What Did Not Work

Several plausible directions were tested and did not solve the main bottleneck:

- Optimizing drafter runtime alone did not matter much because verifier/cache work dominated.
- Increasing `K` without improving live acceptance did not produce speedup.
- Multiple small experts did not break the live acceptance limit.
- Simply increasing expected-acceptance loss weight could collapse training.
- Jointly training the VAE with the flow drafter was worse than freezing it.
- Cached teacher-forced success did not guarantee live speedup.
- Larger context was not automatically better.
- Cosine/linear LR decay often hurt hidden-KV because the model still needed substantial updates late in training.

## Main Lessons

The most important lessons are:

1. Real speedup depends on live accepted tokens, not cached token match.
2. The verifier and cache repair dominate runtime.
3. The drafter must accept roughly `5+` tokens per step in the measured setup before speedup becomes plausible.
4. Random diffusion noise is a poor initialization for this problem.
5. Initializing from the hidden/latent trajectory, especially `delta`, is much better.
6. A strong VAE can make cached reconstruction excellent, but it can hide rollout problems.
7. Direct hidden-KV conditioning is conceptually cleaner and closer to the reference work, but still needs optimization and loss tuning.

## Current Best Baselines

### Best Cached VAE-Based Baseline

The strongest cached baseline is:

- Block-causal VAE-latent flow.
- Frozen transformer VAE.
- `init_mode: delta`.
- `K=8`.
- `steps=2`.
- `layers=4`.
- `dim=768`.
- Reconstruction-heavy loss.

This can achieve near-perfect cached same-subset token and acceptance metrics, but live speedup remains poor.

### Best Hidden-KV Direction

The current hidden-KV baseline is:

- No VAE.
- Hidden-space flow directly over `[B, K, 1024]`.
- Context hidden states used as KV memory.
- `init_mode: delta`.
- `layers=6`.
- `ffn_multiplier=4`.
- `flow_steps=2`.
- `context_size=8`.
- Constant LR around `1e-3`.

This is the most promising architecture direction, but not yet the strongest measured model.

## Recommended Next Experiments

The next useful experiments should focus on hidden-KV optimization rather than returning to more VAE capacity.

Recommended sweeps:

1. Constant LR around the useful range:
   - `7e-4`
   - `1e-3`
   - `1.5e-3`
   - `2e-3`

2. Loss balancing:
   - Reduce `lambda_flow`.
   - Reduce or remove latent-style losses for hidden-KV.
   - Increase emphasis on `logit.ce`.
   - Keep expected acceptance important but avoid making it dominate enough to collapse training.

3. Checkpoint selection:
   - Select by validation expected acceptance and token metrics, not final total loss alone.
   - Evaluate intermediate checkpoints because final epoch is not always best.

4. Live validation:
   - Run profiler on promising checkpoints.
   - Compare cached `accept.greedy_prefix_len` with live mean accepted tokens.
   - Keep reporting real speedup.

5. Conditioning upgrades:
   - Add multi-layer hidden context from the backbone, closer to DFlash.
   - Investigate whether context KV should include more than final-layer hidden states.
   - Consider direct cache-aware conditioning inspired by Orthrus if external-drafter limits remain severe.

## Important Scripts

Training:

- `scripts/train_chunked_flow.py`
- `scripts/train_transformer_hidden_vae.py`

Evaluation:

- `scripts/eval_chunked_flow.py`
- `scripts/eval_vae.py`

Profiling:

- `scripts/profile_flow_drafter.py`

Plotting:

- `scripts/plot_flow_train_eval_loss.py`
- `scripts/plot_vae_metrics.py`

Sweep runners:

- `tmp/train_plot_hidden_kv_flow_sweep_4096w.sh`
- `tmp/train_plot_hidden_kv_flow_lr_scheduler_4096w.sh`
- `tmp/train_plot_flow_block_causal_init_mode_8192w.sh`
- `tmp/train_plot_flow_block_causal_lambda_sweep_4096w.sh`
- `tmp/train_plot_flow_block_causal_multi_expert_8192w.sh`

Docs:

- `docs/flow_table_formats.md`
- `docs/flow_sweeps/`
- `docs/vae_eval_metrics/README.md`

## Current Project State

The project has moved through three phases:

1. Build measurement, caching, VAE, and basic flow infrastructure.
2. Push VAE-latent flow until cached metrics were strong, then discover live rollout remained weak.
3. Shift toward direct hidden-KV flow without a VAE, motivated by DFlash/Orthrus-style conditioning.

The strongest practical conclusion is that the project should now optimize hidden-KV for live acceptance. The VAE path proved that teacher-forced hidden/token prediction can be learned, but it also showed that cached success is not sufficient for speculative speedup.

The next milestone should be:

- A hidden-KV checkpoint that improves live mean accepted tokens, not just cached expected acceptance.
- A profiler result showing drafter plus verifier approaching or exceeding backbone-only throughput.

