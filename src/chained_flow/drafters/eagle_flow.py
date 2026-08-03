"""EAGLE-style token-feedback drafter (V2).

The chunked-flow drafter predicts all K future hidden states in parallel, conditioned only
on context — it never sees the token it just drafted. That caps deep-position acceptance
(acc@4 ~0.45 regardless of capacity/data). This drafter instead autoregresses over
(feature, token): each step consumes the previous hidden state and the embedding of the
previous token, predicts the next hidden, decodes a token, and feeds both forward.

Indexing (a window is anchor hidden h_t plus future tokens t+1..t+K):
  - draft token 0 = argmax(lm_head(h_t))           # FREE: from the REAL anchor hidden
  - h1_hat = step(h_t,    emb(token0)) -> token 1   # token0 == token at t+1
  - h2_hat = step(h1_hat, emb(token1)) -> token 2
  - ...                                             # K-1 predicted hiddens
Each step cross-attends to the frozen context hidden states as KV memory (reuses
``HiddenKVFlowBlock`` from chunked_flow).

Training (option B): self-feed the model's OWN predicted hiddens forward (drift-robust,
matches inference), while teacher-forcing the TOKEN embedding with the real previous token
(keeps the hidden chain differentiable; the drafted token equals the real token exactly when
the position would be accepted). Costs K-1 sequential drafter passes per step — cheap, the
drafter is a small fraction of runtime.

Drop-in: ``propose(state, max_tokens) -> DraftResult`` matches the BaseDrafter protocol, so
generation / verifier / profiler use it unchanged. ``forward_teacher`` is the training entry.
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
class EagleFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    expert_dim: int = 1024  # must equal the LM hidden size
    num_heads: int = 8
    ffn_multiplier: int = 4
    num_drafter_layers: int = 6
    drafter_dropout: float = 0.0
    architecture: str = "eagle_flow"

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.architecture != "eagle_flow":
            raise ValueError("EagleFlowConfig.architecture must be 'eagle_flow'")


class EagleDrafter(nn.Module):
    """Autoregressive hidden-state drafter with token feedback over a frozen backbone."""

    def __init__(self, frozen_lm: FrozenLMWrapper, config: EagleFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("eagle_flow requires expert_dim to equal the target LM hidden size")
        if self.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        # Frozen input-embedding table for token feedback (tied weights; not trained, not saved).
        embed = frozen_lm.model.get_input_embeddings()
        self.register_buffer("token_embedding_weight", embed.weight.detach().clone(), persistent=False)

        # Fuse [prev_hidden ; emb(prev_token)] -> model width.
        self.input_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.position_embedding = nn.Embedding(config.draft_length, self.hidden_size)
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

    def _step(
        self,
        prev_hidden: torch.Tensor,
        prev_token: torch.Tensor,
        context_hidden: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """One AR step. prev_hidden [B,1,D], prev_token [B,1] -> predicted hidden [B,1,D] at `position`."""
        emb = self._embed_tokens(prev_token)  # [B,1,D]
        x = self.input_proj(torch.cat([prev_hidden, emb], dim=-1))
        pos = torch.full((1,), position, device=x.device, dtype=torch.long)
        x = x + self.position_embedding(pos).unsqueeze(0)
        for block in self.blocks:
            x = block(x, context_hidden, attn_mask=None)  # single query position
        return self.out_proj(self.out_norm(x))

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.frozen_lm.lm_head(hidden)

    # ---- training entry (teacher-forced tokens, self-fed hiddens) -------

    def forward_teacher(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the K draft hiddens (option B), aligned to the cache window.

        context_hidden [B, context_size, D] : frozen context KV memory.
        target_hidden  [B, K, D]            : real hiddens h_t..h_{t+K-1}; target_hidden[:,0]=h_t.
        future_tokens  [B, K]               : real tokens t+1..t+K.
        Returns pred_hidden [B, K, D]: position 0 is the REAL anchor h_t (free); positions
        1..K-1 are autoregressively self-fed predictions. lm_head(pred[:,i]) targets future_tokens[:,i].
        """
        context_hidden = context_hidden.to(dtype=self._cached_dtype)
        anchor = target_hidden[:, 0:1, :].to(dtype=self._cached_dtype)  # h_t (real, detached input)
        k = self.config.draft_length
        preds = [anchor]
        prev_hidden = anchor
        for i in range(1, k):
            # to predict h_{t+i}: feed (prev_hidden, emb(token at t+i = future_tokens[i-1]))
            pred = self._step(prev_hidden, future_tokens[:, i - 1 : i], context_hidden, i)
            preds.append(pred)
            prev_hidden = pred  # self-feed predicted hidden
        return torch.cat(preds, dim=1)  # [B, K, D]

    # ---- inference entry ------------------------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_eagle_flow", self.frozen_lm.device):
            context_hidden = self._context(state).to(dtype=self._cached_dtype)
            anchor = state.final_hidden[:, -1:, :].to(dtype=self._cached_dtype)  # h_t (real)
            hiddens, tokens, logits_list = [], [], []
            # position 0: free token from the REAL anchor hidden
            logits0 = self.frozen_lm.lm_head(anchor)
            token0 = logits0.argmax(dim=-1)
            hiddens.append(anchor)
            tokens.append(token0)
            logits_list.append(logits0)
            prev_hidden, prev_token = anchor, token0
            for i in range(1, draft_len):
                pred = self._step(prev_hidden, prev_token, context_hidden, i)
                logits = self.frozen_lm.lm_head(pred)
                token = logits.argmax(dim=-1)
                hiddens.append(pred)
                tokens.append(token)
                logits_list.append(logits)
                prev_hidden, prev_token = pred, token
        return DraftResult(
            tokens=torch.cat(tokens, dim=1),
            hidden_states=torch.cat(hiddens, dim=1),
            logits=torch.cat(logits_list, dim=1),
            timings=timings,
        )
