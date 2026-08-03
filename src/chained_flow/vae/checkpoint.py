from __future__ import annotations

import json
from pathlib import Path

import torch

from chained_flow.vae.base import HiddenVAE, HiddenVAEConfig
from chained_flow.vae.registry import build_hidden_vae


def _load_state_dict(checkpoint_dir: Path) -> dict[str, torch.Tensor]:
    safetensors_path = checkpoint_dir / "model.safetensors"
    bin_path = checkpoint_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path), device="cpu")
    if bin_path.exists():
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"missing VAE weights in {checkpoint_dir}")


def _strip_vae_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("vae.") for key in state_dict):
        return {key.removeprefix("vae."): value for key, value in state_dict.items()}
    return state_dict


def _find_config_dir(checkpoint_dir: Path) -> Path:
    for candidate in (checkpoint_dir, checkpoint_dir.parent):
        if (candidate / "chained_flow_vae_config.json").exists():
            return candidate
    raise FileNotFoundError(
        f"missing VAE config in {checkpoint_dir} or {checkpoint_dir.parent}"
    )


def load_hidden_vae_from_dir(
    checkpoint_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    freeze: bool = True,
) -> HiddenVAE:
    checkpoint_dir = Path(checkpoint_dir)
    config_dir = _find_config_dir(checkpoint_dir)
    config_path = config_dir / "chained_flow_vae_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        saved_config = json.load(f)

    model_args = saved_config["model_args"]
    vae = build_hidden_vae(
        model_args["vae_type"],
        HiddenVAEConfig(
            hidden_size=int(model_args["hidden_size"]),
            latent_size=int(model_args["latent_size"]),
            intermediate_size=int(model_args["intermediate_size"]),
            num_layers=int(model_args.get("num_layers", 2)),
            num_heads=int(model_args.get("num_heads", 4)),
            max_sequence_length=int(model_args.get("max_sequence_length", 128)),
            dropout=float(model_args.get("dropout", 0.0)),
        ),
    )
    vae.load_state_dict(_strip_vae_prefix(_load_state_dict(checkpoint_dir)))
    if device is not None:
        vae = vae.to(device)
    vae.eval()
    if freeze:
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
    return vae
