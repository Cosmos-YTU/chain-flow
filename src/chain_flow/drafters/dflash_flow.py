"""DFlash-style block-diffusion drafter (V4).

Departure from V1-V3: instead of diffusing over raw hidden states, this drafts in
TOKEN-EMBEDDING space with an absorbing-mask (discrete diffusion). The draft block starts
as [anchor_token, MASK, MASK, ...] embeddings; a bidirectional transformer denoises the
masked positions to real tokens in a single parallel pass, conditioned on the target's
context hidden states injected as extra KV (DFlash KV-injection).

Why this might beat hidden-space flow (the V1-V3 family, which plateaued ~2.8 accept):
  - token-embedding space is lower-variance / better-behaved for a generative model than
    raw 1024-d hidden states (whose outlier dims wreck flow targets);
  - the absorbing-mask objective is the proven recipe (DFlash, production speedups);
  - conditioning can use MULTIPLE target layers (concat), exposed via the cache's feature dim.

Conditioning (`context_feature`): the frozen backbone's context hidden states. With the
legacy single-layer cache this is [B, m, 1024]; with the multi-layer cache it is
[B, m, n_layers*1024] and `context_proj` maps it to model width. Either works unchanged.

Reference: tmp/reference_repos/dflash/dflash/model.py (Qwen3DFlashAttention KV-injection,
DFlashDraftModel.forward, dflash_generate). Drop-in: propose() -> DraftResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chain_flow.drafters.base import DraftResult
from chain_flow.frozen_lm import FrozenLMWrapper, LMState
from chain_flow.timing import TimingStats, timed_section


@dataclass
class DFlashFlowConfig:
    context_size: int = 8
    draft_length: int = 4          # block size (incl. the real anchor token at position 0)
    expert_dim: int = 1024         # must equal the LM hidden size (token-embedding dim)
    context_feature_dim: int = 1024  # per-token conditioning width in the cache (1024 single-layer, 4096 multi-layer)
    num_heads: int = 8
    ffn_multiplier: int = 4
    num_drafter_layers: int = 8
    drafter_dropout: float = 0.0
    mask_min: float = 0.3          # training: min fraction of block positions masked
    mask_max: float = 1.0          # training: max fraction masked (1.0 = all-masked, the inference case)
    architecture: str = "dflash_flow"

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.architecture != "dflash_flow":
            raise ValueError("DFlashFlowConfig.architecture must be 'dflash_flow'")


class _KVInjectBlock(nn.Module):
    """Bidirectional self-attention over the draft block, with target context injected as extra KV
    (DFlash-style: keys/values = [context_kv ; block_kv], no causal mask), + FFN."""

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
        kv = torch.cat([self.norm_ctx(context_kv), q], dim=1)  # inject context as extra KV
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


class DFlashFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: DFlashFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("dflash_flow requires expert_dim to equal the target LM hidden size")
        if self.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        embed = frozen_lm.model.get_input_embeddings()
        self.register_buffer("token_embedding_weight", embed.weight.detach().clone(), persistent=False)
        self.vocab_size = self.token_embedding_weight.shape[0]
        # Learned MASK embedding (absorbing state) in token-embedding space.
        self.mask_embedding = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.mask_embedding, std=0.02)

        # Project the (possibly multi-layer) context feature to model width, then inject as KV.
        self.context_proj = nn.Linear(config.context_feature_dim, self.hidden_size)
        self.position_embedding = nn.Embedding(config.draft_length, self.hidden_size)
        self.blocks = nn.ModuleList(
            [
                _KVInjectBlock(self.hidden_size, config.num_heads, config.ffn_multiplier, config.drafter_dropout)
                for _ in range(config.num_drafter_layers)
            ]
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
        """Final-layer context hidden from the live state (single-layer; inference path)."""
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.frozen_lm.lm_head(hidden)

    def denoise(self, block_emb: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        """One parallel denoise pass. block_emb [B, K, D] (mask/token embeddings),
        context_feature [B, m, Fdim] -> predicted hidden [B, K, D] (decode via lm_head)."""
        b, k, _ = block_emb.shape
        context_kv = self.context_proj(context_feature.to(dtype=self._cached_dtype))
        pos = torch.arange(k, device=block_emb.device)
        x = block_emb + self.position_embedding(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x, context_kv)
        return self.out_proj(self.out_norm(x))

    def _build_block_emb(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """tokens [B,K] (real future tokens, ignored where masked); mask [B,K] bool (True=masked).
        Returns block embeddings [B,K,D] = MASK emb at masked positions, real token emb elsewhere.
        The local left-context (h_t) enters via the injected context KV, not an in-block anchor token."""
        tok_emb = self._embed_tokens(tokens)
        mask_emb = self.mask_embedding.to(dtype=self._cached_dtype).view(1, 1, -1)
        return torch.where(mask.unsqueeze(-1), mask_emb, tok_emb)

    # ---- training entry (random-mask denoising) ------------------------

    def forward_teacher(
        self,
        context_feature: torch.Tensor,
        future_tokens: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """context_feature [B, m, Fdim]; future_tokens [B, K] = the K draft targets (input_ids[t+1:t+K+1]).
        Absorbing-mask diffusion: mask a random subset of the K positions, denoise to real tokens,
        supervise lm_head(pred) against future_tokens (weight masked positions). Returns pred_hidden + mask.
        """
        b, k = future_tokens.shape
        device = future_tokens.device
        frac = torch.empty(b, 1, device=device).uniform_(self.config.mask_min, self.config.mask_max, generator=generator)
        mask = torch.rand(b, k, device=device, generator=generator) < frac
        mask[:, 0] |= (~mask.any(dim=1))  # ensure at least one masked position per example
        block_emb = self._build_block_emb(future_tokens, mask)
        pred_hidden = self.denoise(block_emb, context_feature)
        return {"pred_hidden": pred_hidden, "mask": mask}

    # ---- inference entry ------------------------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_dflash_flow", self.frozen_lm.device):
            context_feature = self._context(state).to(dtype=self._cached_dtype)
            b = state.input_ids.shape[0]
            k = self.config.draft_length
            # fully-masked block (the inference case): every draft position is [MASK].
            mask = torch.ones(b, k, dtype=torch.bool, device=self.frozen_lm.device)
            block_tokens = torch.zeros(b, k, dtype=torch.long, device=self.frozen_lm.device)
            block_emb = self._build_block_emb(block_tokens, mask)
            pred_hidden = self.denoise(block_emb, context_feature)
            logits = self.frozen_lm.lm_head(pred_hidden)
            tokens = logits.argmax(dim=-1)[:, :draft_len]
        return DraftResult(
            tokens=tokens,
            hidden_states=pred_hidden[:, :draft_len, :],
            logits=logits[:, :draft_len, :],
            timings=timings,
        )
