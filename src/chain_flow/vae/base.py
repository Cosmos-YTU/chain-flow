from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class HiddenVAEConfig:
    hidden_size: int = 1024
    latent_size: int = 256
    intermediate_size: int = 512
    num_layers: int = 2
    num_heads: int = 4
    max_sequence_length: int = 128
    dropout: float = 0.0


@dataclass
class LatentDistribution:
    mu: torch.Tensor
    logvar: torch.Tensor


@dataclass
class HiddenVAEOutput:
    recon_hidden: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class HiddenVAE(nn.Module):
    config: HiddenVAEConfig

    def encode(self, hidden: torch.Tensor) -> LatentDistribution:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode_sequence(self, hidden: torch.Tensor) -> LatentDistribution:
        if hidden.ndim != 3:
            raise ValueError("hidden must have shape [B, L, D]")
        original_shape = hidden.shape[:-1]
        dist = self.encode(hidden.reshape(-1, hidden.shape[-1]))
        return LatentDistribution(
            mu=dist.mu.reshape(*original_shape, -1),
            logvar=dist.logvar.reshape(*original_shape, -1),
        )

    def decode_sequence(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError("z must have shape [B, L, Z]")
        original_shape = z.shape[:-1]
        hidden = self.decode(z.reshape(-1, z.shape[-1]))
        return hidden.reshape(*original_shape, -1)

    def reparameterize(self, dist: LatentDistribution) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * dist.logvar)
            return dist.mu + torch.randn_like(std) * std
        return dist.mu

    def forward(self, hidden: torch.Tensor) -> HiddenVAEOutput:
        if hidden.ndim == 3:
            dist = self.encode_sequence(hidden)
            z = self.reparameterize(dist)
            recon = self.decode_sequence(z)
        else:
            dist = self.encode(hidden)
            z = self.reparameterize(dist)
            recon = self.decode(z)
        return HiddenVAEOutput(
            recon_hidden=recon,
            z=z,
            mu=dist.mu,
            logvar=dist.logvar,
        )
