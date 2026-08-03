from __future__ import annotations

import torch
from torch import nn

from chained_flow.vae.base import HiddenVAE, HiddenVAEConfig, LatentDistribution


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ResidualMLPHiddenVAE(HiddenVAE):
    def __init__(self, config: HiddenVAEConfig):
        super().__init__()
        self.config = config
        self.encoder_in = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
        )
        self.encoder_block = ResidualBlock(config.intermediate_size)
        self.mu = nn.Linear(config.intermediate_size, config.latent_size)
        self.logvar = nn.Linear(config.intermediate_size, config.latent_size)
        self.decoder_in = nn.Sequential(
            nn.LayerNorm(config.latent_size),
            nn.Linear(config.latent_size, config.intermediate_size),
            nn.GELU(),
        )
        self.decoder_block = ResidualBlock(config.intermediate_size)
        self.decoder_out = nn.Linear(config.intermediate_size, config.hidden_size)

    def encode(self, hidden: torch.Tensor) -> LatentDistribution:
        encoded = self.encoder_block(self.encoder_in(hidden))
        return LatentDistribution(mu=self.mu(encoded), logvar=self.logvar(encoded))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.decoder_block(self.decoder_in(z))
        return self.decoder_out(decoded)
