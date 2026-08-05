# Idea 3.1: Chunked Expert Final-State Drafter

## Status

Candidate to investigate.

## Core idea

Use the final-layer hidden states of the frozen Large Language Model (LLM) as the draft latent space, but split future prediction across small chunk experts.

For example, for draft length `K = 8`, use four tiny experts:

```text
expert 1 -> predicts future final states for tokens t+1, t+2
expert 2 -> predicts future final states for tokens t+3, t+4
expert 3 -> predicts future final states for tokens t+5, t+6
expert 4 -> predicts future final states for tokens t+7, t+8
```

The predicted final-layer hidden states are passed directly to the original frozen LM head to produce draft logits, and the frozen autoregressive verifier accepts or rejects the proposed tokens.

## Minimal tensor flow

Assume:

```text
B = batch size
T = prefix length
D = hidden size
K = draft length, e.g. 8
V = vocabulary size
C = chunk size, e.g. 2
E = number of experts = K / C, e.g. 4
```

Prefill:

```text
x_prompt:        [B, T]
LLM(x_prompt):   h_final [B, T, D]
```

Chunked expert drafting:

```text
context = h_final[:, -m:, :] or pooled(h_final)   # [B, m, D] or [B, D]

expert_1(context) -> z_1 [B, 2, D]
expert_2(context) -> z_2 [B, 2, D]
expert_3(context) -> z_3 [B, 2, D]
expert_4(context) -> z_4 [B, 2, D]

z_future = concat(z_1, z_2, z_3, z_4)             # [B, 8, D]
logits_draft = LMHead(z_future)                  # [B, 8, V]
draft_tokens = sample_or_argmax(logits_draft)    # [B, 8]
```

Verification:

```text
x_proposed = concat(x_prompt, draft_tokens)       # [B, T+8]
logits_verify = FrozenARVerifier(x_proposed)      # [B, T+8, V]
accept longest prefix a <= 8
commit only accepted tokens to real KV cache
```

## Why this is different from rejected old Idea 2

Old Idea 2 was rejected mainly because an internal-layer hidden-state drafter still had to pass generated states through many frozen upper layers before tokenization, making the draft branch expensive.

This variant avoids that specific issue:

- it operates at the final layer;
- the original LM head can consume the generated states directly;
- there is no proxy LM head;
- there is no upper-layer decoding path;
- the only large projection is the existing frozen LM head.

## Why chunk experts may help

A single module predicting all `K` future final states may be too hard. Chunking may help because:

- each expert predicts a short horizon, e.g. only 2 future states;
- experts can specialize by relative position in the draft block;
- small experts may be cheaper than one large K-token generator;
- the design is modular and easy to ablate.

## Important variants

### Parallel chunk experts

All experts see the same context and predict their chunks independently.

```text
expert_i(context) -> z_i
```

Pros:

- fastest;
- fully parallel;
- simple.

Cons:

- later chunks do not know whether earlier chunks are correct;
- long-horizon chunks may be weak.

### Latent autoregressive chunk experts

Later experts condition on earlier predicted latent chunks.

```text
expert_1(context) -> z_1
expert_2(context, z_1) -> z_2
expert_3(context, z_1, z_2) -> z_3
expert_4(context, z_1, z_2, z_3) -> z_4
```

Pros:

- better consistency across the draft block;
- later chunks get more context.

Cons:

- less parallel;
- slightly more latency.

### Shared expert with position embedding

Use one small expert shared across chunks, conditioned on chunk index.

```text
shared_expert(context, chunk_id) -> z_i
```

Pros:

- fewer parameters;
- cleaner scaling to different K.

Cons:

- less specialization.

## Training objective

Teacher: frozen autoregressive LLM final-layer hidden states and logits under teacher forcing.

Student: chunked final-state experts.

Possible loss:

```text
L = lambda_h * MSE(z_future, h_teacher_future_final)
  + lambda_cos * (1 - cosine(z_future, h_teacher_future_final))
  + lambda_kl * KL(p_teacher || p_draft)
  + lambda_acc * accepted_prefix_surrogate
  + lambda_stat * activation_statistics_regularization
```

Where:

```text
p_draft = softmax(LMHead(z_future))
p_teacher = softmax(teacher logits for future tokens)
```

## Minimal experiment

Use a small pretrained LLM and freeze it.

Compare:

1. baseline autoregressive decoding;
2. monolithic final-state drafter predicting `[B, K, D]`;
3. parallel chunk experts;
4. latent-autoregressive chunk experts;
5. shared expert with chunk-position embeddings.

Ablate:

- `K = 4, 8, 16`;
- chunk size `C = 1, 2, 4`;
- number of experts;
- MSE-only vs MSE+KL vs MSE+KL+acceptance surrogate;
- greedy vs sampling verification.

## Evaluation metrics

Primary:

- accepted tokens per draft;
- accepted tokens per second;
- wall-clock speedup over autoregressive decoding;
- verifier rejection rate by draft position.

Secondary:

- final hidden-state MSE / cosine similarity;
- KL divergence to teacher logits;
- top-1 draft match rate;
- memory overhead;
- expert latency.

## Main risks

- Final-layer hidden states may be difficult to generate accurately.
- Off-manifold final states may produce poor LM-head logits.
- Later experts may have low acceptance because farther future states are harder to predict.
- If the frozen LM head dominates latency for `K` positions, speedup may be limited.
- Verification still requires a full autoregressive pass over proposed tokens.

## Why it is worth investigating

This idea keeps the attractive parts of final-layer drafting while avoiding the main rejected bottlenecks:

- no shallow features;
- no proxy LM head;
- no frozen upper-layer decode path;
- no new huge vocabulary decoder;
- cache consistency is preserved through verification.

The key open question is whether chunked experts can generate final-layer states accurately enough to achieve useful verifier acceptance.
