from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class HiddenMLPConfig:
    context_size: int = 4
    draft_length: int = 4
    hidden_multiplier: int = 2
    vae_dir: str | None = None


class HiddenMLPDrafter(nn.Module):
    def __init__(
        self,
        frozen_lm: FrozenLMWrapper,
        config: HiddenMLPConfig | None = None,
    ):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config or HiddenMLPConfig()
        hidden_size = frozen_lm.model.config.hidden_size
        self.vae = None
        self.hidden_size = hidden_size
        self.latent_size = hidden_size
        if self.config.vae_dir is not None:
            from chained_flow.vae import load_hidden_vae_from_dir

            self.vae = load_hidden_vae_from_dir(self.config.vae_dir, device=frozen_lm.device, freeze=True)
            self.latent_size = self.vae.config.latent_size
        input_size = self.latent_size * self.config.context_size
        output_size = self.latent_size * self.config.draft_length
        mlp_hidden = self.latent_size * self.config.hidden_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, output_size),
        )

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    @property
    def uses_vae(self) -> bool:
        return self.vae is not None

    def encode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("encode_hidden requires a configured VAE")
        original_shape = hidden.shape[:-1]
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        with torch.no_grad():
            latent = self.vae.encode(flat_hidden).mu
        return latent.reshape(*original_shape, -1)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("decode_latent requires a configured VAE")
        original_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, latent.shape[-1])
        decoded = self.vae.decode(flat_latent)
        return decoded.reshape(*original_shape, -1)

    def predict_latent_from_context(self, context_hidden: torch.Tensor, max_tokens: int | None = None) -> torch.Tensor:
        if self.vae is None:
            raise RuntimeError("predict_latent_from_context requires a configured VAE")
        return self._predict_from_features(self.encode_hidden(context_hidden), max_tokens=max_tokens)

    def _predict_from_features(self, context_features: torch.Tensor, max_tokens: int | None = None) -> torch.Tensor:
        if context_features.ndim != 3:
            raise ValueError("context_features must have shape [B, m, D]")
        if context_features.shape[1] != self.config.context_size:
            raise ValueError(
                f"context_features must have context size {self.config.context_size}, "
                f"got {context_features.shape[1]}"
            )
        draft_len = self.config.draft_length if max_tokens is None else min(max_tokens, self.config.draft_length)
        batch = context_features.shape[0]
        ctx = context_features.reshape(batch, -1)
        return self.net(ctx).reshape(
            batch,
            self.config.draft_length,
            -1,
        )[:, :draft_len, :]

    def predict_from_context(self, context_hidden: torch.Tensor, max_tokens: int | None = None) -> torch.Tensor:
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B, m, D]")
        if context_hidden.shape[1] != self.config.context_size:
            raise ValueError(
                f"context_hidden must have context size {self.config.context_size}, "
                f"got {context_hidden.shape[1]}"
            )
        if self.vae is None:
            return self._predict_from_features(context_hidden, max_tokens=max_tokens)
        return self.decode_latent(self.predict_latent_from_context(context_hidden, max_tokens=max_tokens))

    def predict_hidden(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        return self.predict_from_context(self._context(state), max_tokens=max_tokens)

    def forward(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        return self.predict_hidden(state, max_tokens=max_tokens)

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_hidden_mlp", self.frozen_lm.device):
            future_latent = None
            if self.vae is None:
                future_hidden = self.predict_hidden(state, max_tokens=draft_len)
            else:
                future_latent = self.predict_latent_from_context(self._context(state), max_tokens=draft_len)
                future_hidden = self.decode_latent(future_latent)
            logits = self.frozen_lm.lm_head(future_hidden)
            tokens = logits.argmax(dim=-1)
        return DraftResult(
            tokens=tokens,
            hidden_states=future_hidden,
            latent_states=future_latent,
            logits=logits,
            timings=timings,
        )
