import json

import torch

from chained_flow.drafters.hidden_mlp import HiddenMLPConfig
from chained_flow.training.losses import DrafterLossConfig
from chained_flow.training.trainer_module import HiddenMLPTrainingModule
from chained_flow.vae import HiddenVAEConfig, build_hidden_vae


def write_vae_checkpoint(path, *, hidden_size=8, latent_size=3, intermediate_size=5):
    path.mkdir()
    config = {
        "model_args": {
            "vae_type": "mlp",
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "intermediate_size": intermediate_size,
            "device": None,
        },
        "loss_args": {},
        "data_args": {},
    }
    (path / "chained_flow_vae_config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = build_hidden_vae(
        "mlp",
        HiddenVAEConfig(hidden_size=hidden_size, latent_size=latent_size, intermediate_size=intermediate_size),
    )
    torch.save({f"vae.{key}": value for key, value in vae.state_dict().items()}, path / "pytorch_model.bin")


def test_trainer_module_forward_returns_custom_loss(fake_wrapper):
    module = HiddenMLPTrainingModule(
        fake_wrapper,
        HiddenMLPConfig(context_size=2, draft_length=2),
        DrafterLossConfig(),
    )
    context_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    target_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    future_tokens = torch.tensor([[1, 2], [2, 3], [3, 4]])

    output = module(
        context_hidden=context_hidden,
        target_hidden=target_hidden,
        future_tokens=future_tokens,
    )

    assert output["loss"].ndim == 0
    assert output["pred_hidden"].shape == target_hidden.shape
    assert "loss_component/hidden.mse" in output
    output["loss"].backward()
    assert any(param.grad is not None for param in module.drafter.parameters())


def test_lm_head_buffers_are_not_persistent(fake_wrapper):
    module = HiddenMLPTrainingModule(
        fake_wrapper,
        HiddenMLPConfig(context_size=2, draft_length=2),
        DrafterLossConfig(),
    )
    state_keys = set(module.state_dict().keys())
    assert "lm_head_weight" not in state_keys
    assert all(not key.startswith("lm_head") for key in state_keys)


def test_trainer_module_with_vae_freezes_vae_and_logs_latent_loss(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = HiddenMLPTrainingModule(
        fake_wrapper,
        HiddenMLPConfig(context_size=2, draft_length=2, vae_dir=str(vae_dir)),
        DrafterLossConfig(lambda_latent_mse=1.0, lambda_latent_cos=0.5),
    )
    context_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    target_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    future_tokens = torch.tensor([[1, 2], [2, 3], [3, 4]])

    output = module(
        context_hidden=context_hidden,
        target_hidden=target_hidden,
        future_tokens=future_tokens,
    )
    output["loss"].backward()

    assert output["pred_latent"].shape == (3, 2, 3)
    assert "loss_component/latent.mse" in output
    assert any(parameter.grad is not None for parameter in module.drafter.net.parameters())
    assert all(parameter.grad is None for parameter in module.drafter.vae.parameters())
