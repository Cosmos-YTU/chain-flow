"""Token-conditioned bidirectional flow refiner (V3).

Motivation. Two findings shaped this design:
  - The chunked/hidden-KV flow drafter predicts K hiddens from context only and caps at
    acc@4 ~0.45 (no inter-position feedback).
  - EAGLE (causal token feedback) breaks that ceiling (acc@4 0.52-0.80, first >1x live), but
    it is strictly left-to-right: a wrong token at position i cascades and can never be revised.
  - A controlled 2x2 ablation showed num_flow_steps 2 vs 8 changes NOTHING when each step
    re-runs the same velocity net on the same conditioning. Iteration only helps if each step
    injects new information.

This drafter makes flow iteration meaningful by re-conditioning every refinement step on the
CURRENT decoded tokens, and refines all K positions JOINTLY (bidirectional self-attention) so
any position can be revised after the block reveals the others -- the thing EAGLE structurally
cannot do.

Refinement loop (N steps), block z in hidden space [B, K, D]:
    z <- delta-init (or noise)
    for s in range(N):
        tokens  = argmax(lm_head(z))           # decode current block
        tok_emb = embed(tokens)                # token feedback (the NEW info each step)
        v       = velocity_net(z, tau_s, context, tok_emb)   # bidirectional over K
        z       = z + dt * v                   # Euler step
    return argmax(lm_head(z))

Training: rectified flow-matching velocity target (z_target - z0) at random tau, PLUS a
decode-weighted CE/accept loss on the integrated output, with self-conditioning (feed the
model's own decoded tokens a fraction of the time so inference-time token drift is in-distribution).

Drop-in: propose(state, max_tokens) -> DraftResult. forward_teacher(...) is the training entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.drafters.chunked_flow import HiddenKVFlowBlock
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class RefineFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    expert_dim: int = 1024  # must equal the LM hidden size
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_refine_steps: int = 4
    init_mode: str = "delta"  # delta | noise
    noise_scale: float = 1.0
    self_cond_prob: float = 0.5  # train-time prob of feeding the model's OWN decoded tokens
    drafter_dropout: float = 0.0
    architecture: str = "refine_flow"

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.num_refine_steps < 1:
            raise ValueError("num_refine_steps must be >= 1")
        if self.init_mode not in {"delta", "noise"}:
            raise ValueError("init_mode must be 'delta' or 'noise'")
        if self.architecture != "refine_flow":
            raise ValueError("RefineFlowConfig.architecture must be 'refine_flow'")


class RefineFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: RefineFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("refine_flow requires expert_dim to equal the target LM hidden size")
        if self.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        embed = frozen_lm.model.get_input_embeddings()
        self.register_buffer("token_embedding_weight", embed.weight.detach().clone(), persistent=False)

        # Fuse [current_hidden ; emb(current_token)] -> model width (token feedback per step).
        self.input_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.position_embedding = nn.Embedding(config.draft_length, self.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                HiddenKVFlowBlock(
                    hidden_size=self.hidden_size,
                    num_heads=config.num_heads,
                    ffn_multiplier=config.ffn_multiplier,
                    dropout=config.drafter_dropout,
                )
                for _ in range(config.num_drafter_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self._cached_dtype = self.input_proj.weight.dtype

    # ---- helpers --------------------------------------------------------

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        weight = self.token_embedding_weight
        return nn.functional.embedding(token_ids.to(weight.device), weight).to(dtype=self._cached_dtype)

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.frozen_lm.lm_head(hidden)

    def init_block(self, context_hidden: torch.Tensor) -> torch.Tensor:
        """Init the K-position block [B, K, D] from the context trajectory."""
        context_hidden = context_hidden.to(dtype=self._cached_dtype)
        b, _, d = context_hidden.shape
        k = self.config.draft_length
        if self.config.init_mode == "noise":
            return torch.randn(b, k, d, device=context_hidden.device, dtype=self._cached_dtype) * self.config.noise_scale
        last = context_hidden[:, -1:, :]
        if context_hidden.shape[1] > 1:
            delta = context_hidden[:, -1:, :] - context_hidden[:, -2:-1, :]
        else:
            delta = torch.zeros_like(last)
        steps = torch.arange(1, k + 1, device=context_hidden.device, dtype=self._cached_dtype)
        return last + steps.view(1, k, 1) * delta

    def velocity(
        self,
        z: torch.Tensor,
        tau: float | torch.Tensor,
        context_hidden: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict velocity for the whole block, re-conditioned on current decoded tokens.

        z [B, K, D]; tau scalar or [B] / [B,1] / [B,1,1]; context_hidden [B, m, D];
        tokens [B, K] (current decode).
        Bidirectional: HiddenKVFlowBlock self-attends across all K positions (no causal mask).
        """
        b, k, _ = z.shape
        tok_emb = self._embed_tokens(tokens)  # [B, K, D]
        x = self.input_proj(torch.cat([z, tok_emb], dim=-1))
        pos = torch.arange(k, device=z.device)
        x = x + self.position_embedding(pos).unsqueeze(0)
        if isinstance(tau, torch.Tensor):
            tau_t = tau.reshape(b, 1).to(device=z.device, dtype=self._cached_dtype)
        else:
            tau_t = torch.full((b, 1), float(tau), device=z.device, dtype=self._cached_dtype)
        x = x + self.time_mlp(tau_t).unsqueeze(1)
        for block in self.blocks:
            x = block(x, context_hidden, attn_mask=None)  # bidirectional self-attn over K
        return self.out_proj(self.out_norm(x))

    # ---- training entry -------------------------------------------------

    def forward_teacher(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns dict with v_pred, v_star (flow-matching), and integrated pred_hidden
        (for decode-weighted CE/accept). Self-conditioning: with prob self_cond_prob the token
        feedback uses the model's OWN current decode (no-grad), else the real future tokens.
        """
        context_hidden = context_hidden.to(dtype=self._cached_dtype)
        target_hidden = target_hidden.to(dtype=self._cached_dtype)
        b, k, d = target_hidden.shape

        # --- flow-matching velocity term (random tau interpolation) ---
        z0 = self.init_block(context_hidden)
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=z0.dtype, generator=generator)
        z_tau = (1.0 - tau) * z0 + tau * target_hidden
        v_star = target_hidden - z0
        # conditioning tokens for the velocity-matching pass: real future tokens (teacher).
        v_pred = self.velocity(z_tau, tau.reshape(b), context_hidden, future_tokens)

        # --- integrated refinement (decode-weighted) with self-conditioning ---
        z = z0
        dt = 1.0 / self.config.num_refine_steps
        use_self_cond = (
            self.training
            and self.config.self_cond_prob > 0.0
            and float(torch.rand((), device=z.device, generator=generator)) < self.config.self_cond_prob
        )
        for s in range(self.config.num_refine_steps):
            if use_self_cond:
                with torch.no_grad():
                    cond_tokens = self.lm_head(z).argmax(dim=-1)
            else:
                cond_tokens = future_tokens
            v = self.velocity(z, s * dt, context_hidden, cond_tokens)
            z = z + dt * v
        pred_hidden = z

        return {"v_pred": v_pred, "v_star": v_star, "pred_hidden": pred_hidden}

    # ---- inference entry ------------------------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_refine_flow", self.frozen_lm.device):
            context_hidden = self._context(state).to(dtype=self._cached_dtype)
            z = self.init_block(context_hidden)
            dt = 1.0 / self.config.num_refine_steps
            tokens = self.lm_head(z).argmax(dim=-1)
            for s in range(self.config.num_refine_steps):
                v = self.velocity(z, s * dt, context_hidden, tokens)
                z = z + dt * v
                tokens = self.lm_head(z).argmax(dim=-1)
            logits = self.lm_head(z)
            draft_tokens = logits.argmax(dim=-1)[:, :draft_len]
        return DraftResult(
            tokens=draft_tokens,
            hidden_states=z[:, :draft_len, :],
            logits=logits[:, :draft_len, :],
            timings=timings,
        )
