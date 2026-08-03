"""Markov-head hidden-space flow drafter (V6).

Diagnosis that motivates this: hidden-space flow (stage2/4) hits acc@1 0.96 but deep positions
decay (acc@4 ~0.45) because the PARALLEL flow predicts all K hiddens from context only — it never
sees the actual earlier tokens. A controlled CE-reweight experiment confirmed deep positions are
conditioning-limited, not emphasis-limited. EAGLE fixes this with full causal token feedback (K
sequential drafter passes). This is the cheap middle ground, from DeepSeek DSpark:

  1. Run the parallel flow ONCE -> K predicted hiddens -> lm_head -> block logits  (unchanged, fast).
  2. Add a rank-r MARKOV head: bias_i = W2(W1[token_{i-1}]), a low-rank token->logit correction
     applied autoregressively over the block at decode time (embedding lookup + rank-r matmul per
     position; NO extra flow passes). This injects the missing "what was the previous token" signal.

Training: teacher-force the markov head with the real previous tokens (so block logits + markov
bias are supervised against future_tokens), plus the usual flow-matching + hidden losses.
Inference: integrate the flow, then decode the block left-to-right applying the markov bias from
each just-decoded token — cheap, and it gives a slice of EAGLE's feedback without its K forwards.

Reuses HiddenKVFlowExpert (chunked_flow) as the velocity net. Drop-in propose() -> DraftResult.
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
class MarkovFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4
    expert_dim: int = 1024  # must equal LM hidden size
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_flow_steps: int = 2
    init_mode: str = "delta"
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    noise_scale: float = 1.0
    markov_rank: int = 256
    architecture: str = "markov_flow"

    def __post_init__(self) -> None:
        if self.draft_length % self.chunk_size != 0:
            raise ValueError("draft_length must be divisible by chunk_size")
        if self.markov_rank < 1:
            raise ValueError("markov_rank must be >= 1")
        if self.init_mode not in {"noise", "repeat_last", "delta"}:
            raise ValueError("init_mode must be 'noise', 'repeat_last', or 'delta'")
        if self.architecture != "markov_flow":
            raise ValueError("MarkovFlowConfig.architecture must be 'markov_flow'")


class MarkovHead(nn.Module):
    """Low-rank token->logit-bias (DSpark vanilla): bias = W2(W1[prev_token]).  rank r << vocab."""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.w1 = nn.Embedding(vocab_size, rank)   # prev token -> rank
        self.w2 = nn.Linear(rank, vocab_size, bias=False)  # rank -> vocab bias
        nn.init.zeros_(self.w2.weight)  # start as no-op so it only learns a correction
        nn.init.normal_(self.w1.weight, std=0.02)

    def bias(self, prev_token: torch.Tensor) -> torch.Tensor:
        # prev_token [B] or [B, K] -> [B, vocab] or [B, K, vocab]
        return self.w2(self.w1(prev_token))


class MarkovFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: MarkovFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("markov_flow requires expert_dim to equal the target LM hidden size")
        self.num_chunks = config.draft_length // config.chunk_size

        # velocity net(s): reuse the proven hidden-KV flow expert, one per chunk (like stage4)
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

        vocab = frozen_lm.model.config.vocab_size
        self.markov = MarkovHead(vocab, config.markov_rank)
        self._dtype = next(self.expert.parameters()).dtype

    # ---- flow machinery (mirrors hidden_kv path) ------------------------

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

    def flow_velocity(self, z_tau: torch.Tensor, tau, context: torch.Tensor, previous=None) -> torch.Tensor:
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

    def integrate(self, context: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        z = z0
        dt = 1.0 / self.config.num_flow_steps
        for s in range(self.config.num_flow_steps):
            tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=self._dtype)
            z = z + dt * self.flow_velocity(z, tau, context)
        return z

    def predict_hidden(self, context: torch.Tensor) -> torch.Tensor:
        return self.integrate(context.to(self._dtype), self.init_latents(context))

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.frozen_lm.lm_head(hidden)

    # ---- training entry -------------------------------------------------

    def forward_teacher(self, context: torch.Tensor, target_hidden: torch.Tensor, future_tokens: torch.Tensor):
        """Returns v_pred, v_star (flow matching) and block_logits WITH teacher-forced markov bias.
        markov bias at position i uses the REAL previous token (future_tokens[i-1]); position 0 uses
        the last context token's prediction proxy = no bias (handled as zero)."""
        context = context.to(self._dtype)
        z0 = self.init_latents(context)
        b = z0.shape[0]
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=self._dtype)
        z_tau = (1.0 - tau) * z0 + tau * target_hidden.to(self._dtype)
        v_star = target_hidden.to(self._dtype) - z0
        v_pred = self.flow_velocity(z_tau, tau, context, previous=target_hidden.to(self._dtype))
        pred_hidden = self.integrate(context, z0)
        base_logits = self.lm_head(pred_hidden)  # [B, K, V]
        # teacher-forced markov: prev token for position i is future_tokens[i-1]; position 0 -> no bias
        k = future_tokens.shape[1]
        prev = torch.zeros_like(future_tokens)
        prev[:, 1:] = future_tokens[:, :-1]
        bias = self.markov.bias(prev)              # [B, K, V]
        bias[:, 0, :] = 0.0                         # position 0 has no real previous draft token
        markov_logits = base_logits + bias
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
        with timed_section(timings, "drafter_markov_flow", self.frozen_lm.device):
            context = self._context(state).to(self._dtype)
            pred_hidden = self.predict_hidden(context)            # one parallel flow pass
            base_logits = self.frozen_lm.lm_head(pred_hidden)     # [B, K, V]
            # autoregressive markov correction over the block (cheap: no flow re-runs)
            tokens = []
            prev_tok = None
            for i in range(draft_len):
                logit_i = base_logits[:, i, :]
                if prev_tok is not None:
                    logit_i = logit_i + self.markov.bias(prev_tok)  # bias from previously decoded token
                tok_i = logit_i.argmax(dim=-1)
                tokens.append(tok_i)
                prev_tok = tok_i
            draft_tokens = torch.stack(tokens, dim=1)
        return DraftResult(tokens=draft_tokens, hidden_states=pred_hidden[:, :draft_len, :],
                           logits=base_logits[:, :draft_len, :], timings=timings)
