# Flow-Drafter — deployment results (RTX PRO 6000 Blackwell)

_generated 2026-07-18 05:45 UTC_

All numbers are **lossless** (spec output identical to the base model). Live tree-accept = real greedy
tree acceptance verified against the backbone; speedup = spec vs autoregressive in the same compiled-forward
framework (both cudagraphed). `vLLM-native` = fully-optimized base decode (reference).

## 1. Acceptance — RedHatAI/speculator_benchmarks + held-out gsm8k

Live tree-accept (tokens accepted per verify pass):

| domain | 4B | 9B | 27B |
|---|---|---|---|
| gsm8k held-out (in-distribution) | 5.25 | 5.70 | 5.37 |
| math_reasoning (structured) | 5.44 | 5.25 | 5.73 |
| HumanEval (code) | 4.02 | 4.27 | 3.95 |
| writing (prose) | 3.28 | 3.55 | 3.48 |
| qa (short free-form) | 2.38 | 2.69 | 2.79 |
| summarization (prose) | 2.05 | 2.30 | 2.30 |

## 2. vLLM speedup — compiled-forward + full-cudagraph verify + cudagraph draft

### 4B

| prompt | accept | AR t/s | spec t/s | speedup |
|---|---|---|---|---|
| math (step-by-step) | 2.20 | 78 | 68 | 0.87× |
| code (fibonacci) | 5.12 | 112 | 155 | 1.38× |
| short factual ("capital of France") | 1.72 | 109 | 73 | 0.67× |
| prose (ocean) | 3.00 | 103 | 94 | 0.91× |
| **mean** | 3.01 | 101 | 97 | **0.97×** |

_vLLM-native reference: 495 tok/s._


### 9B

| prompt | accept | AR t/s | spec t/s | speedup |
|---|---|---|---|---|
| math (step-by-step) | 4.30 | 37 | 65 | 1.77× |
| code (fibonacci) | 5.75 | 70 | 110 | 1.57× |
| short factual ("capital of France") | 2.43 | 69 | 66 | 0.95× |
| prose (ocean) | 3.00 | 65 | 66 | 1.02× |
| **mean** | 3.87 | 60 | 77 | **1.27×** |

_vLLM-native reference: 309 tok/s._


### 27B

| prompt | accept | AR t/s | spec t/s | speedup |
|---|---|---|---|---|
| math (step-by-step) | 3.90 | 16 | 26 | 1.64× |
| code (fibonacci) | 5.12 | 23 | 46 | 1.98× |
| short factual ("capital of France") | 2.43 | 23 | 28 | 1.21× |
| prose (ocean) | 1.88 | 22 | 22 | 1.02× |
| **mean** | 3.33 | 21 | 30 | **1.46×** |

_vLLM-native reference: 101 tok/s._


## Notes

- Accept tracks **output predictability**, not strict in/out-of-distribution: structured math/code stay high, free-form prose dips.
- Speedup wins clearly on high-accept traffic (code/math); low-accept prose is where the mean is dragged.
- Bigger base → larger win (base decode is memory-bound & scales with weight size; the latent-flow draft is base-decoupled).
