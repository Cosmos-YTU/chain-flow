from __future__ import annotations

import torch
from torch import nn

from chain_flow.vae.base import HiddenVAE, HiddenVAEConfig, LatentDistribution


class TransformerHiddenVAE(HiddenVAE):
    def __init__(self, config: HiddenVAEConfig):
        super().__init__()
        self.config = config
        num_heads = config.num_heads if config.intermediate_size % config.num_heads == 0 else 1
        self.encoder_in = nn.Linear(config.hidden_size, config.intermediate_size)
        self.encoder_pos = nn.Embedding(config.max_sequence_length, config.intermediate_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.intermediate_size,
            nhead=num_heads,
            dim_feedforward=config.intermediate_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.encoder_norm = nn.LayerNorm(config.intermediate_size)
        self.mu = nn.Linear(config.intermediate_size, config.latent_size)
        self.logvar = nn.Linear(config.intermediate_size, config.latent_size)

        self.decoder_in = nn.Linear(config.latent_size, config.intermediate_size)
        self.decoder_pos = nn.Embedding(config.max_sequence_length, config.intermediate_size)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.intermediate_size,
            nhead=num_heads,
            dim_feedforward=config.intermediate_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.num_layers)
        self.decoder_norm = nn.LayerNorm(config.intermediate_size)
        self.decoder_out = nn.Linear(config.intermediate_size, config.hidden_size)

    def _position_ids(self, length: int, device: torch.device) -> torch.Tensor:
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"sequence length {length} exceeds max_sequence_length={self.config.max_sequence_length}"
            )
        return torch.arange(length, device=device)

    def encode_sequence(self, hidden: torch.Tensor) -> LatentDistribution:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, D]")
        pos = self._position_ids(hidden.shape[1], hidden.device)
        x = self.encoder_in(hidden)
        x = x + self.encoder_pos(pos).unsqueeze(0)
        x = self.encoder_norm(self.encoder(x))
        return LatentDistribution(mu=self.mu(x), logvar=self.logvar(x))

    def decode_sequence(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("z must have shape [B, L, Z]")
        pos = self._position_ids(z.shape[1], z.device)
        x = self.decoder_in(z)
        x = x + self.decoder_pos(pos).unsqueeze(0)
        x = self.decoder_norm(self.decoder(x))
        return self.decoder_out(x)

    def encode(self, hidden: torch.Tensor) -> LatentDistribution:
        if hidden.ndim != 2:
            raise ValueError("hidden must have shape [N, D]")
        dist = self.encode_sequence(hidden.unsqueeze(1))
        return LatentDistribution(mu=dist.mu[:, 0, :], logvar=dist.logvar[:, 0, :])

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("z must have shape [N, Z]")
        return self.decode_sequence(z.unsqueeze(1))[:, 0, :]
