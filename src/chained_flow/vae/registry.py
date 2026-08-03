from __future__ import annotations

from chained_flow.vae.base import HiddenVAE, HiddenVAEConfig
from chained_flow.vae.low_rank import LowRankHiddenVAE
from chained_flow.vae.mlp import MLPHiddenVAE
from chained_flow.vae.residual_mlp import ResidualMLPHiddenVAE
from chained_flow.vae.transformer_hidden import TransformerHiddenVAE


VAE_REGISTRY: dict[str, type[HiddenVAE]] = {
    "mlp": MLPHiddenVAE,
    "residual_mlp": ResidualMLPHiddenVAE,
    "low_rank": LowRankHiddenVAE,
    "transformer_hidden": TransformerHiddenVAE,
}


def build_hidden_vae(vae_type: str, config: HiddenVAEConfig) -> HiddenVAE:
    try:
        vae_cls = VAE_REGISTRY[vae_type]
    except KeyError as exc:
        choices = ", ".join(sorted(VAE_REGISTRY))
        raise ValueError(f"unknown vae_type={vae_type!r}; expected one of: {choices}") from exc
    return vae_cls(config)
