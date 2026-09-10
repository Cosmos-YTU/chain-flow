from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class HiddenVAELossConfig:
    lambda_mse: float = 1.0
    lambda_cos: float = 0.2
    lambda_norm: float = 0.05
    beta: float = 1e-4
    free_bits: float = 0.0


@dataclass
class HiddenVAELossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted_components: dict[str, torch.Tensor] = field(default_factory=dict)


def hidden_mse_loss(recon_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon_hidden, target_hidden)


def hidden_cosine_loss(recon_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    recon_norm = recon_hidden.norm(dim=-1)
    target_norm = target_hidden.norm(dim=-1)
    both_zero = (recon_norm == 0) & (target_norm == 0)
    cosine = F.cosine_similarity(recon_hidden, target_hidden, dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return (1.0 - cosine).mean()


def hidden_norm_loss(recon_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon_hidden.norm(dim=-1), target_hidden.norm(dim=-1))


def latent_kl_loss(mu: torch.Tensor, logvar: torch.Tensor, *, free_bits: float = 0.0) -> torch.Tensor:
    kl_per_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        kl_per_dim = kl_per_dim.clamp_min(free_bits)
    return kl_per_dim.sum(dim=-1).mean()


def compute_hidden_vae_loss(
    recon_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    config: HiddenVAELossConfig | None = None,
) -> HiddenVAELossOutput:
    config = config or HiddenVAELossConfig()
    if recon_hidden.shape != target_hidden.shape:
        raise ValueError(
            f"recon_hidden and target_hidden must have identical shape, got "
            f"{tuple(recon_hidden.shape)} and {tuple(target_hidden.shape)}"
        )

    components = {
        "hidden.mse": hidden_mse_loss(recon_hidden, target_hidden),
        "hidden.cos": hidden_cosine_loss(recon_hidden, target_hidden),
        "hidden.norm": hidden_norm_loss(recon_hidden, target_hidden),
        "latent.kl": latent_kl_loss(mu, logvar, free_bits=config.free_bits),
    }
    weights = {
        "hidden.mse": config.lambda_mse,
        "hidden.cos": config.lambda_cos,
        "hidden.norm": config.lambda_norm,
        "latent.kl": config.beta,
    }

    total = recon_hidden.new_zeros(())
    weighted: dict[str, torch.Tensor] = {}
    for name, value in components.items():
        weighted_value = value * weights[name]
        weighted[name] = weighted_value
        total = total + weighted_value
    return HiddenVAELossOutput(total=total, components=components, weighted_components=weighted)
