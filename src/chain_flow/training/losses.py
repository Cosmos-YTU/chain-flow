from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


LMHead = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DrafterLossConfig:
    lambda_mse: float = 1.0
    lambda_cos: float = 0.2
    lambda_norm: float = 0.05
    lambda_latent_mse: float = 0.0
    lambda_latent_cos: float = 0.0
    lambda_ce: float = 0.2
    lambda_expected_accept: float = 0.1
    position_gamma: float = 0.8
    eps: float = 1e-8


@dataclass
class DrafterLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)
    weighted_components: dict[str, torch.Tensor] = field(default_factory=dict)

    def scalar_components(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.components.items()}


def _validate_shapes(
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    future_tokens: torch.Tensor | None,
) -> None:
    if pred_hidden.shape != target_hidden.shape:
        raise ValueError(
            f"pred_hidden and target_hidden must have identical shape, got "
            f"{tuple(pred_hidden.shape)} and {tuple(target_hidden.shape)}"
        )
    if pred_hidden.ndim != 3:
        raise ValueError("pred_hidden and target_hidden must have shape [B, K, D]")
    if future_tokens is not None and future_tokens.shape != pred_hidden.shape[:2]:
        raise ValueError(
            f"future_tokens must have shape [B, K], got {tuple(future_tokens.shape)} "
            f"for hidden shape {tuple(pred_hidden.shape)}"
        )


def _validate_latent_shapes(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> None:
    if pred_latent.shape != target_latent.shape:
        raise ValueError(
            f"pred_latent and target_latent must have identical shape, got "
            f"{tuple(pred_latent.shape)} and {tuple(target_latent.shape)}"
        )
    if pred_latent.ndim != 3:
        raise ValueError("pred_latent and target_latent must have shape [B, K, Z]")


def _position_weights(length: int, gamma: float, device: torch.device) -> torch.Tensor:
    if not 0.0 < gamma <= 1.0:
        raise ValueError("position_gamma must be in (0, 1]")
    return gamma ** torch.arange(length, device=device, dtype=torch.float32)


def hidden_mse_loss(pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_hidden, target_hidden)


def hidden_cosine_loss(pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    pred = pred_hidden.reshape(-1, pred_hidden.shape[-1])
    target = target_hidden.reshape(-1, target_hidden.shape[-1])
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    both_zero = (pred_norm == 0) & (target_norm == 0)
    cosine = F.cosine_similarity(pred, target, dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return (1.0 - cosine).mean()


def hidden_norm_loss(pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
    pred_norm = pred_hidden.norm(dim=-1)
    target_norm = target_hidden.norm(dim=-1)
    return F.mse_loss(pred_norm, target_norm)


def latent_mse_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_latent, target_latent)


def latent_cosine_loss(pred_latent: torch.Tensor, target_latent: torch.Tensor) -> torch.Tensor:
    pred = pred_latent.reshape(-1, pred_latent.shape[-1])
    target = target_latent.reshape(-1, target_latent.shape[-1])
    pred_norm = pred.norm(dim=-1)
    target_norm = target.norm(dim=-1)
    both_zero = (pred_norm == 0) & (target_norm == 0)
    cosine = F.cosine_similarity(pred, target, dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    return (1.0 - cosine).mean()


def position_weighted_ce_loss(
    logits: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    batch, length, vocab = logits.shape
    ce = F.cross_entropy(
        logits.reshape(batch * length, vocab),
        future_tokens.reshape(batch * length),
        reduction="none",
    ).reshape(batch, length)
    weights = _position_weights(length, gamma, logits.device).to(ce.dtype)
    return (ce * weights.unsqueeze(0)).sum() / (weights.sum() * batch)


def expected_acceptance_loss(
    logits: torch.Tensor,
    future_tokens: torch.Tensor,
    *,
    gamma: float,
    eps: float,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    token_probs = probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
    prefix_probs = torch.cumprod(token_probs.clamp_min(eps), dim=1)
    weights = _position_weights(logits.shape[1], gamma, logits.device).to(prefix_probs.dtype)
    expected_accept = (prefix_probs * weights.unsqueeze(0)).sum(dim=1) / weights.sum()
    return -expected_accept.mean()


def compute_drafter_loss(
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    *,
    pred_latent: torch.Tensor | None = None,
    target_latent: torch.Tensor | None = None,
    future_tokens: torch.Tensor | None = None,
    lm_head: LMHead | None = None,
    config: DrafterLossConfig | None = None,
) -> DrafterLossOutput:
    config = config or DrafterLossConfig()
    _validate_shapes(pred_hidden, target_hidden, future_tokens)

    components: dict[str, torch.Tensor] = {}
    weighted: dict[str, torch.Tensor] = {}

    components["hidden.mse"] = hidden_mse_loss(pred_hidden, target_hidden)
    components["hidden.cos"] = hidden_cosine_loss(pred_hidden, target_hidden)
    components["hidden.norm"] = hidden_norm_loss(pred_hidden, target_hidden)

    weights = {
        "hidden.mse": config.lambda_mse,
        "hidden.cos": config.lambda_cos,
        "hidden.norm": config.lambda_norm,
    }

    needs_latents = config.lambda_latent_mse != 0.0 or config.lambda_latent_cos != 0.0
    if needs_latents:
        if pred_latent is None or target_latent is None:
            raise ValueError("pred_latent and target_latent are required when latent losses are enabled")
        _validate_latent_shapes(pred_latent, target_latent)
        components["latent.mse"] = latent_mse_loss(pred_latent, target_latent)
        components["latent.cos"] = latent_cosine_loss(pred_latent, target_latent)
        weights.update(
            {
                "latent.mse": config.lambda_latent_mse,
                "latent.cos": config.lambda_latent_cos,
            }
        )

    needs_logits = (
        config.lambda_ce != 0.0
        or config.lambda_expected_accept != 0.0
    )
    if needs_logits:
        if lm_head is None:
            raise ValueError("lm_head is required when logit/token or verifier losses are enabled")
        if future_tokens is None:
            raise ValueError("future_tokens is required when logit/token or verifier losses are enabled")

        pred_logits = lm_head(pred_hidden)

        components["logit.ce"] = position_weighted_ce_loss(
            pred_logits,
            future_tokens,
            gamma=config.position_gamma,
        )
        components["verifier.expected_accept"] = expected_acceptance_loss(
            pred_logits,
            future_tokens,
            gamma=config.position_gamma,
            eps=config.eps,
        )
        weights.update(
            {
                "logit.ce": config.lambda_ce,
                "verifier.expected_accept": config.lambda_expected_accept,
            }
        )

    total = pred_hidden.new_zeros(())
    for name, value in components.items():
        weighted_value = value * weights.get(name, 0.0)
        weighted[name] = weighted_value
        total = total + weighted_value

    return DrafterLossOutput(total=total, components=components, weighted_components=weighted)
