# VAE Eval Results

Dataset: `sghosts/cf_gsm8k_1k_test`. One row per run, selecting the checkpoint with lowest mean `logit.js_div` on the test eval files.

| rank | run | ckpt | js_div↓ | ce_delta↓ | top1↑ | top5↑ | top10↑ | rank_p50↓ | rank_p90↓ | recon_p(real_top1)↑ | prob_ratio↑ | rel_rmse↓ | cos_sim↑ | latent_kl | n_ckpts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `hidden-vae-lr3e3` | 1880 | 0.0401 | 0.2077 | 0.8858 | 0.9853 | 0.9945 | 1.0000 | 2.0000 | 0.8299 | 0.9345 | 0.3437 | 0.9353 | 554.4 | 50 |
| 2 | `hidden-vae-lr2e3` | 1920 | 0.0431 | 0.2282 | 0.8810 | 0.9834 | 0.9937 | 1.0000 | 2.0000 | 0.8246 | 0.9274 | 0.3542 | 0.9312 | 556.7 | 50 |
| 3 | `hidden-vae-lr1e3` | 3960 | 0.0433 | 0.2295 | 0.8804 | 0.9830 | 0.9934 | 1.0000 | 2.0000 | 0.8248 | 0.9276 | 0.3555 | 0.9306 | 551.0 | 100 |
| 4 | `hidden-vae-lr1p5e3` | 1960 | 0.0458 | 0.2443 | 0.8764 | 0.9821 | 0.9931 | 1.0000 | 2.0000 | 0.8211 | 0.9230 | 0.3622 | 0.9280 | 555.6 | 50 |
| 5 | `hidden-vae-biglr-nonorm` | 3960 | 0.0516 | 0.2815 | 0.8672 | 0.9792 | 0.9917 | 1.0000 | 2.0000 | 0.8113 | 0.9102 | 0.3795 | 0.9209 | 550.4 | 100 |
| 6 | `hidden-vae-biglr-lownorm` | 3960 | 0.0517 | 0.2826 | 0.8669 | 0.9791 | 0.9917 | 1.0000 | 2.0000 | 0.8113 | 0.9103 | 0.3801 | 0.9207 | 550.7 | 100 |
| 7 | `hidden-vae-lr1e3-beta3e3` | 1960 | 0.0517 | 0.2795 | 0.8673 | 0.9794 | 0.9918 | 1.0000 | 2.0000 | 0.8129 | 0.9135 | 0.3702 | 0.9249 | 378.5 | 50 |
| 8 | `hidden-vae-biglr-beta3e4` | 4000 | 0.0565 | 0.3147 | 0.8593 | 0.9761 | 0.9905 | 1.0000 | 2.0000 | 0.8056 | 0.9033 | 0.3956 | 0.9141 | 732.7 | 100 |
| 9 | `hidden-vae-biglr-beta1e4` | 3920 | 0.0576 | 0.3225 | 0.8573 | 0.9756 | 0.9903 | 1.0000 | 2.0000 | 0.8037 | 0.9009 | 0.3993 | 0.9125 | 901.1 | 100 |
| 10 | `hidden-vae-lr1e3-beta1e2` | 2000 | 0.1024 | 0.6906 | 0.7945 | 0.9387 | 0.9655 | 1.0000 | 3.0000 | 0.7564 | 0.8478 | 0.5141 | 0.8694 | 126.0 | 50 |
| 11 | `hidden-vae-biglr` | 400 | 0.1885 | 1.4374 | 0.6644 | 0.8458 | 0.8981 | 1.0000 | 11.00 | 0.6246 | 0.6938 | 0.5699 | 0.8142 | 452.8 | 10 |
| 12 | `hidden-vae-bigcos` | 800 | 0.2307 | 1.8921 | 0.6011 | 0.7911 | 0.8489 | 1.0000 | 22.00 | 0.5660 | 0.6264 | 0.6153 | 0.7804 | 442.8 | 10 |
| 13 | `hidden-vae-bigbeta` | 800 | 0.2325 | 1.9134 | 0.5985 | 0.7885 | 0.8466 | 1.0000 | 23.00 | 0.5635 | 0.6235 | 0.6172 | 0.7789 | 438.9 | 10 |
| 14 | `hidden-vae-bignorm` | 400 | 0.4458 | 5.1957 | 0.2932 | 0.5528 | 0.6460 | 4.0000 | 355.0 | 0.2769 | 0.3082 | 0.8011 | 0.6209 | 288.7 | 10 |

## Analysis

- Best overall by distribution preservation is `hidden-vae-lr3e3` at checkpoint `1880`: `logit.js_div=0.0401`, `top1=0.8858`, `top5=0.9853`, and `rel_rmse=0.3437`.
- The top group is close on hidden reconstruction: the best rows sit around `rel_rmse=0.3437` to `0.3795` and cosine similarity around `0.9209` to `0.9353`.
- Token top-1 remains a strict metric: even the best models mostly land around `0.8858` top-1 agreement. Top-5/top-10 are more informative for whether the original best token stays plausible.
- Very large `rank_p90` values mean the reconstruction sometimes pushes the teacher top token far down the vocabulary, even when median rank is good. Prefer runs with both low `js_div` and low `rank_p90`.
- Weakest run by this criterion is `hidden-vae-bignorm` at its best checkpoint `400` with `js_div=0.4458` and `rel_rmse=0.8011`.

## Notes

- Lower is better for `js_div`, `ce_delta`, `rank_p50`, `rank_p90`, and `rel_rmse`.
- Higher is better for `top1`, `top5`, `top10`, `recon_p(real_top1)`, `prob_ratio`, and `cos_sim`.
- `prob_ratio` can exceed `1.0` when the reconstruction is more confident than the teacher on the teacher top token; this is not automatically better if `js_div` is worse.

## Train/Test Gap

Same selected checkpoint per run. Positive gap means test is worse than train.

| run | ckpt | train_js | test_js | js_gap | train_top1 | test_top1 | top1_gap | train_rel_rmse | test_rel_rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hidden-vae-lr3e3` | 1880 | 0.0377 | 0.0401 | 0.0025 | 0.8918 | 0.8858 | -0.0060 | 0.3410 | 0.3437 |
| `hidden-vae-lr2e3` | 1920 | 0.0406 | 0.0431 | 0.0025 | 0.8866 | 0.8810 | -0.0055 | 0.3516 | 0.3542 |
| `hidden-vae-lr1e3` | 3960 | 0.0408 | 0.0433 | 0.0025 | 0.8866 | 0.8804 | -0.0062 | 0.3529 | 0.3555 |
| `hidden-vae-lr1p5e3` | 1960 | 0.0432 | 0.0458 | 0.0026 | 0.8829 | 0.8764 | -0.0065 | 0.3597 | 0.3622 |
| `hidden-vae-biglr-nonorm` | 3960 | 0.0486 | 0.0516 | 0.0030 | 0.8740 | 0.8672 | -0.0068 | 0.3771 | 0.3795 |
| `hidden-vae-biglr-lownorm` | 3960 | 0.0488 | 0.0517 | 0.0029 | 0.8738 | 0.8669 | -0.0069 | 0.3777 | 0.3801 |
| `hidden-vae-lr1e3-beta3e3` | 1960 | 0.0489 | 0.0517 | 0.0029 | 0.8739 | 0.8673 | -0.0066 | 0.3678 | 0.3702 |
| `hidden-vae-biglr-beta3e4` | 4000 | 0.0533 | 0.0565 | 0.0032 | 0.8663 | 0.8593 | -0.0070 | 0.3931 | 0.3956 |
| `hidden-vae-biglr-beta1e4` | 3920 | 0.0545 | 0.0576 | 0.0032 | 0.8644 | 0.8573 | -0.0071 | 0.3968 | 0.3993 |
| `hidden-vae-lr1e3-beta1e2` | 2000 | 0.0962 | 0.1024 | 0.0062 | 0.8055 | 0.7945 | -0.0111 | 0.5107 | 0.5141 |
| `hidden-vae-biglr` | 400 | 0.1805 | 0.1885 | 0.0080 | 0.6789 | 0.6644 | -0.0145 | 0.5674 | 0.5699 |
| `hidden-vae-bigcos` | 800 | 0.2221 | 0.2307 | 0.0086 | 0.6175 | 0.6011 | -0.0164 | 0.6132 | 0.6153 |
| `hidden-vae-bigbeta` | 800 | 0.2239 | 0.2325 | 0.0086 | 0.6149 | 0.5985 | -0.0163 | 0.6151 | 0.6172 |
| `hidden-vae-bignorm` | 400 | 0.4463 | 0.4458 | -0.0005 | 0.2937 | 0.2932 | -0.0005 | 0.8000 | 0.8011 |

The train/test gaps are small for the strongest runs, so the ranking is not just memorization of the 1k train set. The older `big*` runs are much worse on both train and test, not primarily overfit.
