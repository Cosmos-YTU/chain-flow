from __future__ import annotations

import torch
from torch import nn

from chained_flow.vae.base import HiddenVAE, HiddenVAEConfig, LatentDistribution


class LowRankHiddenVAE(HiddenVAE):
    def __init__(self, config: HiddenVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
        )
        self.mu = nn.Linear(config.intermediate_size, config.latent_size)
        self.logvar = nn.Linear(config.intermediate_size, config.latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )

    def encode(self, hidden: torch.Tensor) -> LatentDistribution:
        encoded = self.encoder(hidden)
        return LatentDistribution(mu=self.mu(encoded), logvar=self.logvar(encoded))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
