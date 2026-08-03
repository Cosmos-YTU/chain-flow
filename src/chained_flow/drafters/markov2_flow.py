"""Markov-v2 hidden-space flow drafter (V7).

V6 (markov_flow) added a rank-r order-1 token->logit bias on top of the parallel flow and was the
first hidden-flow drafter to cross 1x live (gsm8k greedy 2.93, stem acc@4 0.50->0.74) — but it
REGRESSED on code (acc@4 0.52->0.40). Diagnosis: a single global bigram table averages token->token
transitions across 6 very different domains, so it applies prose-y corrections in code context.

V7 fixes that with a stronger but still-cheap Markov head, keeping the flow matcher intact:
  1. ORDER-2: condition the bias on the last TWO decoded tokens (trigram), not one.
  2. HIDDEN-GATED: gate the bias by the predicted hidden state ĥ_i, so the correction is
     context-aware (bias = g(ĥ_i) ⊙ W2(W1[prev2])) instead of a pure token bigram. This lets it
     suppress the correction where it doesn't apply (e.g. code).

Still flow matching: velocity field + Euler integration unchanged; the Markov head is a decode-time
logit correction. Reuses HiddenKVFlowExpert as the velocity net. Drop-in propose() -> DraftResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.drafters.chunked_flow import HiddenKVFlowExpert
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class Markov2FlowConfig:
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4
    expert_dim: int = 1024
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_flow_steps: int = 2
    init_mode: str = "delta"
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    noise_scale: float = 1.0
    markov_rank: int = 512        # raised from 256 (more capacity for multi-domain transitions)
    markov_order: int = 2         # condition bias on the last `order` decoded tokens
    markov_hidden_gate: bool = True  # gate the bias by the predicted hidden state
    architecture: str = "markov2_flow"

    def __post_init__(self) -> None:
        if self.draft_length % self.chunk_size != 0:
            raise ValueError("draft_length must be divisible by chunk_size")
        if self.markov_rank < 1 or self.markov_order < 1:
            raise ValueError("markov_rank and markov_order must be >= 1")
        if self.init_mode not in {"noise", "repeat_last", "delta"}:
            raise ValueError("init_mode must be 'noise', 'repeat_last', or 'delta'")
        if self.architecture != "markov2_flow":
            raise ValueError("Markov2FlowConfig.architecture must be 'markov2_flow'")


class MarkovHead2(nn.Module):
    """Order-`order`, optionally hidden-gated, low-rank token->logit bias.

    bias = W2( gate(ĥ) ⊙ sum_j W1_j[token_{i-j}] )      (j = 1..order)
    - W1_j: per-lag embedding (vocab -> rank) for the j-th previous token
    - gate: Linear(hidden -> rank) -> sigmoid, makes the correction context-aware (off if no hidden)
    - W2: rank -> vocab bias, zero-init so the head starts as a no-op
    """

    def __init__(self, vocab_size: int, hidden_size: int, rank: int, order: int, hidden_gate: bool):
        super().__init__()
        self.order = order
        self.hidden_gate = hidden_gate
        self.w1 = nn.ModuleList([nn.Embedding(vocab_size, rank) for _ in range(order)])
        for emb in self.w1:
            nn.init.normal_(emb.weight, std=0.02)
        self.w2 = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.w2.weight)
        if hidden_gate:
            self.gate = nn.Linear(hidden_size, rank)
            nn.init.zeros_(self.gate.bias)  # gate ~ sigmoid(0)=0.5 at init (mild, symmetric)

    def bias(self, prev_tokens: list[torch.Tensor], hidden: torch.Tensor | None = None) -> torch.Tensor:
        """prev_tokens: list of length up to `order`, most-recent first; each [B] (or [B,1]).
        hidden: [B, D] or [B,1,D] gating features (the position's predicted hidden). Returns [B, V]."""
        r = None
        for j in range(self.order):
            if j < len(prev_tokens) and prev_tokens[j] is not None:
                e = self.w1[j](prev_tokens[j])
                r = e if r is None else r + e
        if r is None:
            # no previous tokens available -> no bias
            return self.w2.weight.new_zeros((1, self.w2.weight.shape[0]))
        if self.hidden_gate and hidden is not None:
            h = hidden.reshape(hidden.shape[0], -1)[:, : self.gate.in_features].to(r.dtype)
            r = r * torch.sigmoid(self.gate(h))
        return self.w2(r)

    def block_bias(self, future_tokens: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training bias for all K positions at once.
        future_tokens [B,K] (real tokens), hidden [B,K,D]. Position i conditions on real tokens
        i-1..i-order; positions with no valid history get zero bias. Returns [B,K,V]."""
        b, k = future_tokens.shape
        v = self.w2.weight.shape[0]
        out = future_tokens.new_zeros((b, k, v), dtype=self.w2.weight.dtype)
        for i in range(1, k):  # position 0 has no previous draft token
            r = None
            for j in range(self.order):
                src = i - 1 - j
                if src >= 0:
                    e = self.w1[j](future_tokens[:, src])
                    r = e if r is None else r + e
            if r is None:
                continue
            if self.hidden_gate:
                h = hidden[:, i, :].to(r.dtype)
                r = r * torch.sigmoid(self.gate(h))
            out[:, i, :] = self.w2(r)
        return out


class Markov2FlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: Markov2FlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("markov2_flow requires expert_dim to equal the target LM hidden size")
        self.num_chunks = config.draft_length // config.chunk_size

        def make_expert():
            return HiddenKVFlowExpert(
                hidden_size=self.hidden_size, context_size=config.context_size,
                draft_length=config.draft_length, chunk_size=config.chunk_size,
                expert_dim=config.expert_dim, num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier, num_layers=config.num_drafter_layers,
                dropout=config.drafter_dropout,
            )
        self.expert = make_expert()
        self.extra_experts = nn.ModuleList([make_expert() for _ in range(self.num_chunks - 1)])
        self.markov = MarkovHead2(frozen_lm.model.config.vocab_size, self.hidden_size,
                                  config.markov_rank, config.markov_order, config.markov_hidden_gate)
        self._dtype = next(self.expert.parameters()).dtype

    # ---- flow machinery (identical to V6) ------------------------------

    def _chunk_experts(self):
        return [self.expert, *self.extra_experts]

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad = hidden[:, :1, :].expand(-1, self.config.context_size - hidden.shape[1], -1)
        return torch.cat([pad, hidden], dim=1)

    def init_latents(self, context: torch.Tensor) -> torch.Tensor:
        context = context.to(dtype=self._dtype)
        b, _, d = context.shape
        k = self.config.draft_length
        if self.config.init_mode == "noise":
            return torch.randn(b, k, d, device=context.device, dtype=self._dtype) * self.config.noise_scale
        last = context[:, -1:, :]
        if self.config.init_mode == "repeat_last":
            return last.expand(b, k, d).clone()
        delta = (context[:, -1:, :] - context[:, -2:-1, :]) if context.shape[1] > 1 else torch.zeros_like(last)
        steps = torch.arange(1, k + 1, device=context.device, dtype=self._dtype)
        return last + steps.view(1, k, 1) * delta

    def flow_velocity(self, z_tau, tau, context, previous=None):
        vels = []
        for ci, expert in enumerate(self._chunk_experts()):
            start = ci * self.config.chunk_size
            end = start + self.config.chunk_size
            prev = (previous if previous is not None else z_tau)[:, :start, :]
            if previous is not None and self.config.detach_previous_chunks:
                prev = prev.detach()
            vels.append(expert(context_hidden=context, previous_hidden=prev,
                               current_h_tau=z_tau[:, start:end, :], tau=tau, chunk_start=start))
        return torch.cat(vels, dim=1)

    def integrate(self, context, z0):
        z = z0
        dt = 1.0 / self.config.num_flow_steps
        for s in range(self.config.num_flow_steps):
            tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=self._dtype)
            z = z + dt * self.flow_velocity(z, tau, context)
        return z

    def predict_hidden(self, context):
        return self.integrate(context.to(self._dtype), self.init_latents(context))

    def lm_head(self, hidden):
        return self.frozen_lm.lm_head(hidden)

    # ---- training entry -------------------------------------------------

    def forward_teacher(self, context, target_hidden, future_tokens):
        context = context.to(self._dtype)
        z0 = self.init_latents(context)
        b = z0.shape[0]
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=self._dtype)
        z_tau = (1.0 - tau) * z0 + tau * target_hidden.to(self._dtype)
        v_star = target_hidden.to(self._dtype) - z0
        v_pred = self.flow_velocity(z_tau, tau, context, previous=target_hidden.to(self._dtype))
        pred_hidden = self.integrate(context, z0)
        base_logits = self.lm_head(pred_hidden)
        markov_logits = base_logits + self.markov.block_bias(future_tokens, pred_hidden)
        return {"v_pred": v_pred, "v_star": v_star, "pred_hidden": pred_hidden,
                "base_logits": base_logits, "markov_logits": markov_logits}

    # ---- inference entry ------------------------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)
        with timed_section(timings, "drafter_markov2_flow", self.frozen_lm.device):
            context = self._context(state).to(self._dtype)
            pred_hidden = self.predict_hidden(context)
            base_logits = self.frozen_lm.lm_head(pred_hidden)
            tokens = []
            hist: list[torch.Tensor] = []  # decoded tokens, most-recent appended last
            for i in range(draft_len):
                logit_i = base_logits[:, i, :]
                if hist:
                    prev = [hist[-1 - j] for j in range(self.config.markov_order) if len(hist) > j]
                    logit_i = logit_i + self.markov.bias(prev, pred_hidden[:, i, :])
                tok_i = logit_i.argmax(dim=-1)
                tokens.append(tok_i)
                hist.append(tok_i)
            draft_tokens = torch.stack(tokens, dim=1)
        return DraftResult(tokens=draft_tokens, hidden_states=pred_hidden[:, :draft_len, :],
                           logits=base_logits[:, :draft_len, :], timings=timings)
