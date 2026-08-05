# Flow Architectures for Chunked Expert Final-State Drafting

Context: this file expands `ideas-3/01-chunked-expert-final-state-drafter.md`.

The goal is to generate future final-layer hidden states of a frozen Large Language Model (LLM), then pass those states through the original frozen language-modeling head (LM head) to propose draft tokens. The frozen autoregressive LLM then verifies the proposed tokens.

## 1. Setup

Let:

$$
B = \text{batch size}, \quad
T = \text{prefix length}, \quad
D = \text{final hidden-state dimension},
$$

$$
K = \text{total draft length}, \quad
C = \text{chunk size}, \quad
E = K / C = \text{number of chunk experts},
$$

$$
m = \text{number of prefix final states used as context}.
$$

The frozen LLM processes the prefix and produces final-layer hidden states:

$$
h_{\mathrm{final}} \in \mathbb{R}^{B \times T \times D}.
$$

We use the last $m$ final states as context:

$$
h_{\mathrm{ctx}} = h_{\mathrm{final}}[:, T-m:T, :] \in \mathbb{R}^{B \times m \times D}.
$$

For example, if $K=8$ and $C=2$, then $E=4$ experts predict four chunks:

$$
\begin{aligned}
\text{Expert}_1 &: x_{t+1:t+2}, \\
\text{Expert}_2 &: x_{t+3:t+4}, \\
\text{Expert}_3 &: x_{t+5:t+6}, \\
\text{Expert}_4 &: x_{t+7:t+8}.
\end{aligned}
$$

But the experts do not directly predict token IDs. They predict final-layer hidden-state chunks:

$$
z_i \in \mathbb{R}^{B \times C \times D}.
$$

For the connected/sequential variant:

$$
\begin{aligned}
z_1 &= \text{Expert}_1(h_{\mathrm{ctx}}), \\
z_2 &= \text{Expert}_2(h_{\mathrm{ctx}}, z_1), \\
z_3 &= \text{Expert}_3(h_{\mathrm{ctx}}, z_1, z_2), \\
z_4 &= \text{Expert}_4(h_{\mathrm{ctx}}, z_1, z_2, z_3).
\end{aligned}
$$

The full drafted future hidden-state block is:

$$
z_{\mathrm{future}} = \operatorname{concat}(z_1, z_2, z_3, z_4)
\in \mathbb{R}^{B \times K \times D}.
$$

The frozen LM head maps these states to draft logits:

$$
\ell_{\mathrm{draft}} = \operatorname{LMHead}_{\mathrm{frozen}}(z_{\mathrm{future}})
\in \mathbb{R}^{B \times K \times V},
$$

where $V$ is the vocabulary size.

Draft tokens are sampled or greedily selected:

$$
\hat{x}_{t+1:t+K} \sim p_{\mathrm{draft}}(\cdot \mid z_{\mathrm{future}}).
$$

The verifier is the frozen autoregressive LLM. It receives:

$$
x_{\mathrm{proposed}} = [x_{1:T}, \hat{x}_{t+1:t+K}],
$$

computes verifier logits, accepts the longest valid prefix, and commits only accepted tokens to the real key-value (KV) cache.

## 2. Conditional flow expert formulation

Each chunk expert is a conditional flow model over future final-layer hidden states.

For chunk $i$, define:

$$
z_i^\tau \in \mathbb{R}^{B \times C \times D}
$$

as the noisy or interpolated latent chunk at flow time $\tau \in [0,1]$.

The conditioning variables are:

$$
h_{\mathrm{ctx}} \in \mathbb{R}^{B \times m \times D},
$$

$$
z_{<i} = \operatorname{concat}(z_1, \ldots, z_{i-1})
\in \mathbb{R}^{B \times (i-1)C \times D},
$$

and a chunk index embedding $e_i$.

The expert learns a velocity field:

$$
v_i^\tau
= f_{\theta_i}(z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i)
\in \mathbb{R}^{B \times C \times D}.
$$

A simple Euler update is:

$$
z_i^{\tau + \Delta \tau}
= z_i^\tau + \Delta \tau \cdot f_{\theta_i}(z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

After $S$ integration steps, the final predicted chunk is:

$$
z_i = z_i^1 \in \mathbb{R}^{B \times C \times D}.
$$

## 3. Rectified-flow training target

For each training example, obtain the teacher final hidden-state chunk from the frozen LLM:

$$
z_i^{\star} \in \mathbb{R}^{B \times C \times D}.
$$

Sample base noise:

$$
z_i^0 \sim \mathcal{N}(0, I).
$$

Sample flow time:

$$
\tau \sim \mathcal{U}(0,1).
$$

Interpolate between noise and the teacher hidden chunk:

$$
z_i^\tau = (1 - \tau) z_i^0 + \tau z_i^{\star}.
$$

The rectified-flow target velocity is:

$$
v_i^{\star} = z_i^{\star} - z_i^0.
$$

Train the expert with:

$$
\mathcal{L}_{\mathrm{flow}}
= \left\| f_{\theta_i}(z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i)
- (z_i^{\star} - z_i^0) \right\|_2^2.
$$

A broader training loss can combine flow matching, hidden-state reconstruction, logit matching, and activation-statistics regularization:

$$
\mathcal{L}
= \lambda_{\mathrm{flow}} \mathcal{L}_{\mathrm{flow}}
+ \lambda_h \|z_i - z_i^{\star}\|_2^2
+ \lambda_{\cos} \left(1 - \cos(z_i, z_i^{\star})\right)
+ \lambda_{\mathrm{KL}} \operatorname{KL}(p_{\mathrm{teacher}} \| p_{\mathrm{draft}})
+ \lambda_{\mathrm{stat}} \mathcal{L}_{\mathrm{stat}}.
$$

where:

$$
p_{\mathrm{draft}}
= \operatorname{softmax}(\operatorname{LMHead}_{\mathrm{frozen}}(z_i)),
$$

and $p_{\mathrm{teacher}}$ is the frozen LLM's teacher-forced token distribution for the corresponding future positions.

## 4. Architecture 1: Tiny Transformer flow expert

Use a small Transformer over noisy future slots, previous chunks, and prefix context.

Input sequence:

$$
S_i = [z_i^\tau; z_{<i}; h_{\mathrm{ctx}}]
\in \mathbb{R}^{B \times (C + (i-1)C + m) \times D}.
$$

Add embeddings for flow time, chunk index, and within-chunk slot index:

$$
\tilde{S}_i = S_i + e_\tau + e_i + e_{\mathrm{slot}}.
$$

A lightweight Transformer computes:

$$
H_i = \operatorname{TinyTransformer}_{\theta_i}(\tilde{S}_i).
$$

The velocity is read only from the first $C$ positions, corresponding to the noisy future slots:

$$
v_i^\tau = W_{\mathrm{out}} H_i[:, 1:C, :]
\in \mathbb{R}^{B \times C \times D}.
$$

Pros:

- expressive;
- lets future slots, previous chunks, and context interact directly;
- close to LLM hidden-state geometry;
- natural for chunk size $C > 1$.

Cons:

- more expensive than cross-attention or MLP variants;
- may reduce speedup if too deep.

## 5. Architecture 2: Cross-attention flow expert

This is the cleanest default candidate.

Use noisy future states as queries:

$$
Q_i = z_i^\tau \in \mathbb{R}^{B \times C \times D}.
$$

Use prefix context and previous generated chunks as keys/values:

$$
K_i = V_i = [h_{\mathrm{ctx}}; z_{<i}]
\in \mathbb{R}^{B \times (m + (i-1)C) \times D}.
$$

After projection to a smaller expert dimension $d_e$:

$$
q_i = W_Q(Q_i + e_\tau + e_i + e_{\mathrm{slot}}),
$$

$$
k_i = W_K(K_i), \qquad v_i = W_V(V_i).
$$

Then:

$$
U_i = \operatorname{SelfAttn}(q_i),
$$

$$
A_i = \operatorname{CrossAttn}(U_i, k_i, v_i),
$$

$$
R_i = \operatorname{FFN}(A_i),
$$

$$
\hat{v}_i^\tau = W_O R_i
\in \mathbb{R}^{B \times C \times D}.
$$

Here $\hat{v}_i^\tau$ is the predicted flow velocity.

Pros:

- explicitly separates generation from conditioning;
- cheaper than full self-attention over all tokens;
- good for small chunks such as $C=2$;
- later experts naturally condition on earlier generated chunks.

Cons:

- conditioning is mediated through cross-attention only;
- may need careful normalization to match final-state distribution.

## 6. Architecture 3: MLP-Mixer / token-mixer flow expert

Use a very cheap mixer instead of attention.

Compress context and previous chunks:

$$
c_i = \operatorname{pool}(h_{\mathrm{ctx}}, z_{<i}) \in \mathbb{R}^{B \times D}.
$$

Condition noisy slots with this summary:

$$
U_i^0 = z_i^\tau + W_c c_i + e_\tau + e_i + e_{\mathrm{slot}}
\in \mathbb{R}^{B \times C \times D}.
$$

Then apply alternating channel and token mixing:

$$
U_i^{\ell+1}
= U_i^\ell
+ \operatorname{ChannelMLP}(\operatorname{Norm}(U_i^\ell)),
$$

$$
U_i^{\ell+2}
= U_i^{\ell+1}
+ \operatorname{TokenMLP}(\operatorname{Norm}(U_i^{\ell+1})).
$$

Output:

$$
\hat{v}_i^\tau = W_O U_i^L
\in \mathbb{R}^{B \times C \times D}.
$$

Pros:

- fast;
- simple;
- useful for speed-focused ablations.

Cons:

- weaker conditioning than attention;
- context compression may discard important prefix information;
- may struggle with long-range dependencies.

## 7. Architecture 4: Low-rank flow expert

Constrain the velocity to a low-rank subspace:

$$
\hat{v}_i^\tau = A_i \; g_{\theta_i}(B_i z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i),
$$

where:

$$
B_i \in \mathbb{R}^{r \times D}, \qquad
A_i \in \mathbb{R}^{D \times r}, \qquad
r \ll D.
$$

For example, if $D=4096$, choose $r=256$ or $r=512$.

Pros:

- cheap;
- reduces parameter count;
- may keep generated states closer to plausible final-state directions;
- strong efficiency story.

Cons:

- may be too restrictive;
- final-layer hidden-state trajectories may require high-rank movement.

## 8. Architecture 5: Residual flow around a coarse initializer

Instead of starting from pure noise, initialize near the hidden-state manifold:

$$
z_i^0 = c_i(h_{\mathrm{ctx}}, z_{<i}) + \sigma \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, I),
$$

where $c_i$ is a learned coarse initializer.

Then flow only refines:

$$
z_i^{s+1}
= z_i^s + \Delta \tau_s \cdot f_{\theta_i}(z_i^s, \tau_s, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

Pros:

- fewer flow steps needed;
- better suited for inference acceleration;
- flow models uncertainty/refinement rather than full generation from noise.

Cons:

- the coarse initializer adds another component;
- if the initializer is poor, the flow may inherit its bias.

Important note: the coarse initializer is not required to be treated as the main method. It can simply be part of the flow parameterization.

## 9. Architecture 6: One-step rectified-flow expert

Train with rectified flow, sample with one Euler step.

Training interpolation:

$$
z_i^\tau = (1 - \tau)z_i^0 + \tau z_i^{\star}.
$$

Target velocity:

$$
v_i^{\star} = z_i^{\star} - z_i^0.
$$

One-step inference:

$$
z_i = z_i^0 + f_{\theta_i}(z_i^0, 0, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

Two-step inference:

$$
z_i^{1/2}
= z_i^0 + \frac{1}{2} f_{\theta_i}(z_i^0, 0, h_{\mathrm{ctx}}, z_{<i}, e_i),
$$

$$
z_i^1
= z_i^{1/2} + \frac{1}{2} f_{\theta_i}(z_i^{1/2}, 1/2, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

Pros:

- strongest speed story;
- avoids many diffusion denoising steps;
- aligned with draft acceleration;
- simpler than score-based diffusion.

Cons:

- one-step quality may be insufficient;
- may need distillation or consistency-style training.

## 10. Architecture 7: Consistency-model expert

Train a model that maps noisy states directly to clean final hidden states:

$$
F_{\theta_i}(z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i)
\approx z_i^{\star}.
$$

One-step inference:

$$
z_i = F_{\theta_i}(z_i^0, 0, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

or, for a high-noise endpoint $\tau_{\max}$:

$$
z_i = F_{\theta_i}(z_i^{\tau_{\max}}, \tau_{\max}, h_{\mathrm{ctx}}, z_{<i}, e_i).
$$

Pros:

- one-step or few-step generation;
- diffusion-like but optimized for fast inference;
- good fit for acceleration.

Cons:

- training may be less stable;
- may require teacher distillation from a stronger multi-step flow/diffusion model.

## 11. Architecture 8: Shared flow backbone with stage-specific heads

Instead of fully separate experts, share most parameters:

$$
U_i = f_{\mathrm{shared}}(z_i^\tau, \tau, h_{\mathrm{ctx}}, z_{<i}, e_i),
$$

$$
\hat{v}_i^\tau = \operatorname{Head}_i(U_i).
$$

Pros:

- shares general final-state dynamics;
- keeps horizon-specific specialization via heads or adapters;
- fewer parameters than fully separate experts.

Cons:

- less specialization than fully separate experts;
- shared backbone may become a bottleneck.

Possible variants:

- shared backbone with expert-specific output heads;
- shared backbone with expert-specific low-rank adaptation (LoRA) adapters;
- shared backbone with expert-specific time/chunk embeddings;
- shared backbone with expert-specific normalization layers.

## 12. Architecture 9: Sequential flow experts with latent memory

Each expert outputs both future hidden states and an auxiliary memory:

$$
(z_i, m_i) = \text{Expert}_i(h_{\mathrm{ctx}}, z_{<i}, m_{i-1}),
$$

where:

$$
z_i \in \mathbb{R}^{B \times C \times D},
\qquad
m_i \in \mathbb{R}^{B \times R \times D}.
$$

The next expert conditions on this memory:

$$
z_{i+1} = \text{Expert}_{i+1}(h_{\mathrm{ctx}}, z_{\leq i}, m_i).
$$

Pros:

- lets experts communicate more than token hidden states;
- memory can carry uncertainty, plan, or global trajectory information;
- may improve long-horizon coherence.

Cons:

- more complex;
- harder to analyze;
- memory could become an ungrounded latent unless regularized.

## 13. Recommended main candidate

Most promising architecture for the current idea:

$$
\textbf{Sequential rectified-flow cross-attention experts.}
$$

For each chunk $i$:

$$
z_i^\tau \in \mathbb{R}^{B \times C \times D},
\qquad
h_{\mathrm{ctx}} \in \mathbb{R}^{B \times m \times D},
\qquad
z_{<i} \in \mathbb{R}^{B \times (i-1)C \times D}.
$$

Compute:

$$
q_i = W_Q(z_i^\tau + e_\tau + e_i + e_{\mathrm{slot}}),
$$

$$
(k_i, v_i) = (W_K [h_{\mathrm{ctx}}; z_{<i}], W_V [h_{\mathrm{ctx}}; z_{<i}]),
$$

$$
U_i = \operatorname{SelfAttn}(q_i),
$$

$$
A_i = \operatorname{CrossAttn}(U_i, k_i, v_i),
$$

$$
R_i = \operatorname{FFN}(A_i),
$$

$$
\hat{v}_i^\tau = W_O R_i.
$$

Euler update:

$$
z_i^{\tau + \Delta \tau} = z_i^\tau + \Delta \tau \cdot \hat{v}_i^\tau.
$$

After $S=1$ or $S=2$ steps:

$$
\ell_i = \operatorname{LMHead}_{\mathrm{frozen}}(z_i)
\in \mathbb{R}^{B \times C \times V}.
$$

## 14. Main hypothesis

Stage-conditioned flow experts can generate short chunks of future final-layer hidden states that stay close enough to the frozen LLM's final hidden-state manifold to achieve high verifier acceptance, while costing less than generating the same tokens through full autoregressive decoding.

A useful speed condition is:

$$
\mathbb{E}[A] \cdot C_{\mathrm{AR\ step}}
>
C_{\mathrm{draft}} + C_{\mathrm{verify}} + C_{\mathrm{commit}},
$$

where $\mathbb{E}[A]$ is the expected number of accepted draft tokens.

## 15. Key ablations

Compare:

1. parallel flow experts;
2. connected/sequential flow experts;
3. fully separate experts;
4. shared flow backbone with stage-specific heads;
5. shared recurrent flow expert with chunk-position embeddings;
6. one-step vs two-step rectified flow;
7. pure-noise initialization vs residual/coarse initialization;
8. chunk size $C \in \{1,2,4\}$;
9. draft length $K \in \{4,8,16\}$.

Primary metrics:

$$
\text{accepted tokens per draft}, \quad
\text{accepted tokens per second}, \quad
\text{wall-clock speedup}, \quad
\text{verifier rejection rate by position}.
$$

Secondary metrics:

$$
\text{hidden-state MSE/cosine similarity}, \quad
\operatorname{KL}(p_{\mathrm{teacher}} \| p_{\mathrm{draft}}), \quad
\text{top-1 draft match}, \quad
\text{expert latency}, \quad
\text{memory overhead}.
$$
