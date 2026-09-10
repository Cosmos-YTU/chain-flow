"""Embedding-space flow matcher (V5).

Thesis. V1-V3 did continuous flow matching but in raw HIDDEN-state space, and plateaued
(~2.8 accept). The hidden manifold is anisotropic with huge outlier dims — a poor target for
a velocity field. DFlash showed token-EMBEDDING space works for drafting, but via DISCRETE
(masked) diffusion. This drafter is the missing combination: **continuous flow matching whose
target is clean token embeddings.** Embeddings on this backbone are tied to lm_head and have
small, uniform norm (~0.63), so they are a far better-behaved flow target than hidden states.

Pipeline:
  target  e* = embed(future_tokens)            [B, K, D]   (clean token embeddings)
  init    z0 = noise (or delta from context)
  flow    z_tau = (1-tau) z0 + tau e* ;  train v_pred -> (e* - z0)   (rectified flow)
  infer   integrate Euler N steps from z0 -> predicted embedding ẑ
  decode  logits = lm_head(ẑ)                  (tied embeddings ⇒ lm_head == nearest-embedding)

Conditioning: context features injected as KV (single-layer 1024-d or multi-layer 4096-d via
context_feature_dim). Bidirectional self-attention over the K block positions.
Reference for KV-injection block + multi-layer idea: DFlash (tmp/reference_repos/dflash).
Drop-in: propose() -> DraftResult; forward_teacher() is the training entry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chain_flow.drafters.base import DraftResult
from chain_flow.frozen_lm import FrozenLMWrapper, LMState
from chain_flow.timing import TimingStats, timed_section


@dataclass
class EmbedFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    expert_dim: int = 1024          # must equal the LM hidden / embedding size
    context_feature_dim: int = 1024  # 1024 single-layer cache, 4096 multi-layer
    num_heads: int = 8
    ffn_multiplier: int = 4
    num_drafter_layers: int = 8
    num_flow_steps: int = 4
    init_mode: str = "noise"        # noise | delta_embed  (delta_embed extrapolates last context-embed proxy)
    noise_scale: float = 0.6        # ~ embedding norm; keeps z0 on the embedding scale
    drafter_dropout: float = 0.0
    architecture: str = "embed_flow"

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.num_flow_steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        if self.init_mode not in {"noise", "delta_embed"}:
            raise ValueError("init_mode must be 'noise' or 'delta_embed'")
        if self.architecture != "embed_flow":
            raise ValueError("EmbedFlowConfig.architecture must be 'embed_flow'")


class _KVInjectBlock(nn.Module):
    """Bidirectional self-attention over the K block + context injected as extra KV, then FFN."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_multiplier: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_ctx = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        ffn_dim = hidden_size * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor, context_kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(x)
        kv = torch.cat([self.norm_ctx(context_kv), q], dim=1)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class EmbedFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: EmbedFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("embed_flow requires expert_dim to equal the target LM hidden size")
        if self.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        embed = frozen_lm.model.get_input_embeddings()
        self.register_buffer("token_embedding_weight", embed.weight.detach().clone(), persistent=False)

        self.context_proj = nn.Linear(config.context_feature_dim, self.hidden_size)
        self.position_embedding = nn.Embedding(config.draft_length, self.hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.in_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.blocks = nn.ModuleList(
            [_KVInjectBlock(self.hidden_size, config.num_heads, config.ffn_multiplier, config.drafter_dropout)
             for _ in range(config.num_drafter_layers)]
        )
        self.out_norm = nn.LayerNorm(self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self._cached_dtype = self.context_proj.weight.dtype

    # ---- helpers --------------------------------------------------------

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(token_ids.to(self.token_embedding_weight.device), self.token_embedding_weight).to(
            dtype=self._cached_dtype
        )

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def lm_head(self, embed: torch.Tensor) -> torch.Tensor:
        # tied embeddings ⇒ frozen lm_head decodes an embedding to token logits (nearest-embedding).
        return self.frozen_lm.lm_head(embed)

    def init_z0(self, context_kv: torch.Tensor, batch: int, *, generator=None) -> torch.Tensor:
        k, d = self.config.draft_length, self.hidden_size
        dev = context_kv.device
        if self.config.init_mode == "noise":
            return torch.randn(batch, k, d, device=dev, dtype=self._cached_dtype, generator=generator) * self.config.noise_scale
        # delta_embed: extrapolate from the projected context tail (kept on embedding scale)
        last = context_kv[:, -1:, :]
        if context_kv.shape[1] > 1:
            delta = context_kv[:, -1:, :] - context_kv[:, -2:-1, :]
        else:
            delta = torch.zeros_like(last)
        steps = torch.arange(1, k + 1, device=dev, dtype=self._cached_dtype)
        return last + steps.view(1, k, 1) * delta

    def velocity(self, z: torch.Tensor, tau, context_kv: torch.Tensor) -> torch.Tensor:
        b, k, _ = z.shape
        x = self.in_proj(z)
        pos = torch.arange(k, device=z.device)
        x = x + self.position_embedding(pos).unsqueeze(0)
        if isinstance(tau, torch.Tensor):
            tau_t = tau.reshape(b, 1).to(device=z.device, dtype=self._cached_dtype)
        else:
            tau_t = torch.full((b, 1), float(tau), device=z.device, dtype=self._cached_dtype)
        x = x + self.time_mlp(tau_t).unsqueeze(1)
        for block in self.blocks:
            x = block(x, context_kv)
        return self.out_proj(self.out_norm(x))

    def _context_kv(self, context_feature: torch.Tensor) -> torch.Tensor:
        return self.context_proj(context_feature.to(dtype=self._cached_dtype))

    def integrate(self, context_kv: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        z = z0
        dt = 1.0 / self.config.num_flow_steps
        for s in range(self.config.num_flow_steps):
            z = z + dt * self.velocity(z, s * dt, context_kv)
        return z

    # ---- training entry (rectified flow matching to clean embeddings) ----

    def forward_teacher(
        self,
        context_feature: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Returns v_pred, v_star (flow-matching to e_target) and integrated pred_embed (for decode loss).
        context_feature [B, m, Fdim]; future_tokens [B, K] = the K draft targets."""
        context_kv = self._context_kv(context_feature)
        b, k = future_tokens.shape
        e_target = self._embed_tokens(future_tokens)                 # [B,K,D] clean target embeddings
        z0 = self.init_z0(context_kv, b, generator=generator)
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=z0.dtype, generator=generator)
        z_tau = (1.0 - tau) * z0 + tau * e_target
        v_star = e_target - z0
        v_pred = self.velocity(z_tau, tau.reshape(b), context_kv)
        pred_embed = self.integrate(context_kv, z0)                  # full integration for decode-weighted loss
        return {"v_pred": v_pred, "v_star": v_star, "pred_embed": pred_embed, "e_target": e_target}

    # ---- inference entry ------------------------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)
        with timed_section(timings, "drafter_embed_flow", self.frozen_lm.device):
            context_kv = self._context_kv(self._context(state))
            b = state.input_ids.shape[0]
            z0 = self.init_z0(context_kv, b)
            pred_embed = self.integrate(context_kv, z0)
            logits = self.frozen_lm.lm_head(pred_embed)
            tokens = logits.argmax(dim=-1)[:, :draft_len]
        return DraftResult(
            tokens=tokens,
            hidden_states=pred_embed[:, :draft_len, :],
            logits=logits[:, :draft_len, :],
            timings=timings,
        )
