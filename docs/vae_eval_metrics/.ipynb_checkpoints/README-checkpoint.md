# VAE Eval Metrics

The VAE evaluator reports per-token metrics and then summarizes each metric with mean, std, min, max, p50, p90, p95, and p99.

## Hidden Reconstruction

`hidden.mse`: Raw mean squared error between reconstructed and real hidden states. Useful for debugging, but hard to interpret by itself because hidden-state scale changes across models and layers.

`hidden.rel_mse`: `hidden.mse` divided by the average squared size of the real hidden state. This says how large the squared error is relative to the signal being reconstructed.

`hidden.rel_rmse`: Square root of relative MSE. This is often the most interpretable hidden metric: `0.10` means the typical reconstruction error is about 10% of the typical hidden activation magnitude.

`hidden.cosine_similarity`: Directional alignment between reconstructed and real hidden states. `1.0` is perfect direction match.

`hidden.cos`: Cosine distance, equal to `1 - hidden.cosine_similarity`. Lower is better.

`hidden.norm`: Squared difference between reconstructed and real hidden-vector norms. Lower means the reconstruction preserved vector length better.

## Latent Regularization

`latent.kl`: KL term from the VAE latent distribution, after applying free bits if configured. This tracks how much latent information the model is using.

## Logit Distribution

`logit.kl`: KL divergence from the real next-token distribution to the reconstructed next-token distribution. Lower means the reconstruction preserves the LM output distribution better.

`logit.js_div`: Jensen-Shannon divergence between the real and reconstructed next-token distributions. It is symmetric and bounded, so it is usually easier to compare across runs than raw KL.

`logit.ce_delta`: Extra cross-entropy paid by reconstructed logits on the real model's top token compared with the real logits. `0` means no loss on that token; lower is better.

## Token Decision

`token.match`: Backward-compatible alias for `token.top1_match`.

`token.top1_match`: Whether reconstructed logits pick the exact same top token as real logits. This is strict.

`token.top5_match`: Whether the real top token appears in the reconstructed top 5. This is less brittle than top-1 and checks whether the right token remains a strong candidate.

`token.top10_match`: Whether the real top token appears in the reconstructed top 10. This measures whether the reconstruction preserves the plausible token set.

`token.real_top1_rank_in_recon`: Rank of the real model's top token under reconstructed logits. `1` means exact top match; lower is better.

## Token Probability

`token.recon_prob_on_real_top1`: Probability assigned by reconstructed logits to the token that the real logits ranked first. Higher is better.

`token.prob_ratio_on_real_top1`: Reconstructed probability divided by real probability for the real top token. `1.0` means the reconstruction preserved that token's probability; below `1.0` means it reduced it.
