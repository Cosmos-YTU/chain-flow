import json

import torch

from chained_flow.drafters.hidden_mlp import HiddenMLPConfig, HiddenMLPDrafter
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


def test_lm_head_matches_model_logits(fake_wrapper):
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)
    projected = fake_wrapper.lm_head(state.final_hidden)
    assert torch.equal(projected.argmax(dim=-1), state.logits.argmax(dim=-1))


def test_latest_hidden_predicts_same_next_token(fake_wrapper):
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)
    latest_logits = fake_wrapper.lm_head(fake_wrapper.latest_hidden(state).unsqueeze(1))[:, -1, :]
    assert latest_logits.argmax(dim=-1).tolist() == state.logits[:, -1, :].argmax(dim=-1).tolist()


def test_hidden_mlp_predict_from_context_shape(fake_wrapper):
    drafter = HiddenMLPDrafter(fake_wrapper, HiddenMLPConfig(context_size=3, draft_length=2))
    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)
    pred = drafter.predict_from_context(context)
    assert pred.shape == (4, 2, fake_wrapper.model.config.hidden_size)


def test_hidden_mlp_with_vae_predicts_decoded_hidden_and_freezes_vae(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    drafter = HiddenMLPDrafter(
        fake_wrapper,
        HiddenMLPConfig(context_size=3, draft_length=2, vae_dir=str(vae_dir)),
    )
    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)

    pred_hidden = drafter.predict_from_context(context)
    pred_latent = drafter.predict_latent_from_context(context)

    assert pred_hidden.shape == (4, 2, fake_wrapper.model.config.hidden_size)
    assert pred_latent.shape == (4, 2, 3)
    assert all(not parameter.requires_grad for parameter in drafter.vae.parameters())
