import json

import pytest
import torch

from chain_flow.vae import HiddenVAEConfig, VAE_REGISTRY, build_hidden_vae, load_hidden_vae_from_dir


@pytest.mark.parametrize("vae_type", sorted(VAE_REGISTRY))
def test_hidden_vae_architectures_round_trip_shape(vae_type):
    model = build_hidden_vae(
        vae_type,
        HiddenVAEConfig(hidden_size=8, latent_size=3, intermediate_size=5),
    )
    hidden = torch.randn(4, 8)
    output = model(hidden)

    assert output.recon_hidden.shape == hidden.shape
    assert output.mu.shape == (4, 3)
    assert output.logvar.shape == (4, 3)
    assert output.z.shape == (4, 3)


def test_unknown_hidden_vae_type_raises():
    with pytest.raises(ValueError, match="unknown vae_type"):
        build_hidden_vae("missing", HiddenVAEConfig(hidden_size=8, latent_size=3, intermediate_size=5))


def test_load_hidden_vae_from_trainer_checkpoint_uses_parent_config(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-1"
    checkpoint_dir.mkdir(parents=True)
    config = {
        "model_args": {
            "vae_type": "mlp",
            "hidden_size": 8,
            "latent_size": 3,
            "intermediate_size": 5,
        }
    }
    (run_dir / "chain_flow_vae_config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = build_hidden_vae("mlp", HiddenVAEConfig(hidden_size=8, latent_size=3, intermediate_size=5))
    torch.save({f"vae.{key}": value for key, value in vae.state_dict().items()}, checkpoint_dir / "pytorch_model.bin")

    loaded = load_hidden_vae_from_dir(checkpoint_dir)

    assert loaded.config.hidden_size == 8
    assert loaded.config.latent_size == 3
    assert all(not parameter.requires_grad for parameter in loaded.parameters())



def test_transformer_hidden_vae_sequence_methods_round_trip_shape():
    model = build_hidden_vae(
        "transformer_hidden",
        HiddenVAEConfig(
            hidden_size=8,
            latent_size=3,
            intermediate_size=8,
            num_layers=1,
            num_heads=2,
            max_sequence_length=6,
        ),
    )
    hidden = torch.randn(2, 5, 8)

    output = model(hidden)
    dist = model.encode_sequence(hidden)
    recon = model.decode_sequence(dist.mu)

    assert output.recon_hidden.shape == hidden.shape
    assert output.mu.shape == (2, 5, 3)
    assert output.logvar.shape == (2, 5, 3)
    assert output.z.shape == (2, 5, 3)
    assert recon.shape == hidden.shape
