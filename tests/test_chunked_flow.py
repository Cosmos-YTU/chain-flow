import json

import pytest
import torch

from chain_flow.drafters.chunked_flow import CrossAttentionFlowExpert, SingleExpertFlowConfig, SingleExpertFlowDrafter
from chain_flow.training.train_chunked_flow import FlowLossArguments, SingleExpertFlowTrainingModule, _parameter_counts
from chain_flow.vae import HiddenVAEConfig, build_hidden_vae


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
    (path / "chain_flow_vae_config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = build_hidden_vae(
        "mlp",
        HiddenVAEConfig(hidden_size=hidden_size, latent_size=latent_size, intermediate_size=intermediate_size),
    )
    torch.save({f"vae.{key}": value for key, value in vae.state_dict().items()}, path / "pytorch_model.bin")


def flow_config(vae_dir, *, context_size=3, draft_length=2, num_flow_steps=1):
    return SingleExpertFlowConfig(
        context_size=context_size,
        draft_length=draft_length,
        chunk_size=draft_length,
        vae_dir=str(vae_dir),
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
        num_flow_steps=num_flow_steps,
    )


def test_cross_attention_flow_expert_returns_velocity_shape():
    expert = CrossAttentionFlowExpert(
        latent_size=3,
        chunk_size=2,
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
    )
    z_tau = torch.randn(4, 2, 3)
    tau = torch.rand(4)
    context = torch.randn(4, 3, 3)

    velocity = expert(z_tau, tau, context)

    assert velocity.shape == (4, 2, 3)


def test_single_expert_flow_config_accepts_variable_draft_length(tmp_path):
    config = SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=4, chunk_size=4)

    assert config.draft_length == 4
    assert config.chunk_size == 4


def test_single_expert_flow_config_rejects_invalid_shapes(tmp_path):
    with pytest.raises(ValueError, match="draft_length must be >= 1"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=0, chunk_size=0)
    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=4, chunk_size=0)
    with pytest.raises(ValueError, match="draft_length must be divisible by chunk_size"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=5, chunk_size=2)
    with pytest.raises(ValueError, match="init_mode must be"):
        SingleExpertFlowConfig(vae_dir=str(tmp_path), draft_length=4, chunk_size=4, init_mode="bad")


def test_single_expert_flow_drafter_initializes_from_context(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae-init"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    context = torch.randn(2, 3, fake_wrapper.model.config.hidden_size)

    repeat_drafter = SingleExpertFlowDrafter(fake_wrapper, flow_config(vae_dir, context_size=3, draft_length=4))
    repeat_drafter.config.init_mode = "repeat_last"
    z_ctx = repeat_drafter.encode_hidden(context)
    z0 = repeat_drafter.init_latents(z_ctx)
    assert z0.shape == (2, 4, 3)
    assert torch.allclose(z0, z_ctx[:, -1:, :].expand_as(z0))

    delta_drafter = SingleExpertFlowDrafter(fake_wrapper, flow_config(vae_dir, context_size=3, draft_length=4))
    delta_drafter.config.init_mode = "delta"
    z_ctx = delta_drafter.encode_hidden(context)
    z0 = delta_drafter.init_latents(z_ctx)
    delta = z_ctx[:, -1:, :] - z_ctx[:, -2:-1, :]
    expected = z_ctx[:, -1:, :] + torch.arange(1, 5, dtype=z_ctx.dtype).view(1, 4, 1) * delta
    assert torch.allclose(z0, expected)


def test_single_expert_flow_drafter_predicts_and_proposes(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    drafter = SingleExpertFlowDrafter(fake_wrapper, flow_config(vae_dir))
    input_ids = torch.tensor([[1, 2, 3]])
    state, _ = fake_wrapper.prefill(input_ids)

    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)
    latents = drafter.predict_latent_from_context(context)
    hidden = drafter.decode_latent(latents)
    proposal = drafter.propose(state, 2)

    assert latents.shape == (4, 2, 3)
    assert hidden.shape == (4, 2, fake_wrapper.model.config.hidden_size)
    assert proposal.tokens.shape == (1, 2)
    assert proposal.hidden_states.shape == (1, 2, fake_wrapper.model.config.hidden_size)
    assert proposal.latent_states.shape == (1, 2, 3)
    assert proposal.logits.shape == (1, 2, fake_wrapper.model.config.hidden_size)
    assert all(not parameter.requires_grad for parameter in drafter.vae.parameters())



def test_single_expert_flow_drafter_supports_multiple_chunk_experts(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae-multichunk"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    config = SingleExpertFlowConfig(
        context_size=3,
        draft_length=16,
        chunk_size=2,
        vae_dir=str(vae_dir),
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
    )
    drafter = SingleExpertFlowDrafter(fake_wrapper, config)

    context = torch.randn(4, 3, fake_wrapper.model.config.hidden_size)
    latents = drafter.predict_latent_from_context(context)
    hidden = drafter.decode_latent(latents)

    assert drafter.num_chunks == 8
    assert len(drafter.extra_experts) == 7
    assert latents.shape == (4, 16, 3)
    assert hidden.shape == (4, 16, fake_wrapper.model.config.hidden_size)


def test_single_expert_flow_drafter_loads_trainer_checkpoint_with_parent_config(fake_wrapper, tmp_path):
    run_dir = tmp_path / "vae-run"
    checkpoint_dir = run_dir / "checkpoint-10"
    write_vae_checkpoint(run_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    checkpoint_dir.mkdir()
    (run_dir / "pytorch_model.bin").replace(checkpoint_dir / "pytorch_model.bin")

    drafter = SingleExpertFlowDrafter(fake_wrapper, flow_config(checkpoint_dir))

    assert drafter.vae.config.hidden_size == fake_wrapper.model.config.hidden_size
    assert all(not parameter.requires_grad for parameter in drafter.vae.parameters())


def test_single_expert_flow_training_module_returns_finite_loss_and_gradients(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2, draft_length=4),
        FlowLossArguments(),
    )
    context_hidden = torch.randn(3, 2, fake_wrapper.model.config.hidden_size)
    target_hidden = torch.randn(3, 4, fake_wrapper.model.config.hidden_size)
    future_tokens = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])

    output = module(
        context_hidden=context_hidden,
        target_hidden=target_hidden,
        future_tokens=future_tokens,
    )
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert output["loss"].ndim == 0
    assert output["pred_latent"].shape == (3, 4, 3)
    assert output["pred_hidden"].shape == target_hidden.shape
    for name in [
        "loss_component/flow.mse",
        "loss_component/latent.mse",
        "loss_component/hidden.rel_mse",
        "loss_component/hidden.cos",
        "loss_component/logit.ce",
        "loss_component/verifier.expected_accept",
    ]:
        assert name in output
    assert any(parameter.grad is not None for parameter in module.drafter.expert.parameters())
    assert all(parameter.grad is None for parameter in module.drafter.vae.parameters())


def test_lm_head_buffers_are_not_persistent(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2),
        FlowLossArguments(),
    )

    state_keys = set(module.state_dict().keys())

    assert "lm_head_weight" not in state_keys
    assert all(not key.startswith("lm_head") for key in state_keys)
    assert all(".vae." not in key for key in state_keys)


def test_flow_training_lm_head_accepts_float_hidden_with_bfloat16_head(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2),
        FlowLossArguments(),
    )
    module.lm_head_weight = module.lm_head_weight.to(torch.bfloat16)
    hidden = torch.randn(2, 2, fake_wrapper.model.config.hidden_size, dtype=torch.float32)

    logits = module.lm_head(hidden)

    assert logits.dtype == torch.bfloat16
    assert logits.shape == (2, 2, fake_wrapper.model.config.hidden_size)

def test_parameter_counts_reports_total_and_trainable(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        flow_config(vae_dir, context_size=2),
        FlowLossArguments(),
    )

    total, trainable = _parameter_counts(module)

    assert total > 0
    assert trainable > 0
    assert trainable < total



def test_block_causal_flow_drafter_predicts_k8_single_block(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae-block-causal"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    config = SingleExpertFlowConfig(
        context_size=3,
        draft_length=8,
        chunk_size=8,
        vae_dir=str(vae_dir),
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
        architecture="block_causal_experts",
        num_drafter_layers=1,
    )
    drafter = SingleExpertFlowDrafter(fake_wrapper, config)

    context = torch.randn(2, 3, fake_wrapper.model.config.hidden_size)
    latents = drafter.predict_latent_from_context(context)
    hidden = drafter.decode_latent(latents)

    assert drafter.num_chunks == 1
    assert latents.shape == (2, 8, 3)
    assert hidden.shape == (2, 8, fake_wrapper.model.config.hidden_size)


def test_block_causal_flow_training_module_uses_new_architecture(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae-block-causal-train"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        SingleExpertFlowConfig(
            context_size=2,
            draft_length=8,
            chunk_size=8,
            vae_dir=str(vae_dir),
            expert_dim=8,
            num_heads=2,
            ffn_multiplier=2,
            architecture="block_causal_experts",
            num_drafter_layers=1,
        ),
        FlowLossArguments(),
    )
    output = module(
        context_hidden=torch.randn(2, 2, fake_wrapper.model.config.hidden_size),
        target_hidden=torch.randn(2, 8, fake_wrapper.model.config.hidden_size),
        future_tokens=torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0], [2, 3, 4, 5, 6, 7, 0, 1]]),
    )

    assert torch.isfinite(output["loss"])
    assert output["pred_latent"].shape == (2, 8, 3)
    assert output["pred_hidden"].shape == (2, 8, fake_wrapper.model.config.hidden_size)



def test_hidden_kv_flow_drafter_uses_hidden_states_without_vae(fake_wrapper):
    hidden_size = fake_wrapper.model.config.hidden_size
    config = SingleExpertFlowConfig(
        context_size=3,
        draft_length=4,
        chunk_size=4,
        vae_dir=None,
        expert_dim=hidden_size,
        num_heads=2,
        ffn_multiplier=2,
        num_flow_steps=1,
        init_mode="delta",
        architecture="hidden_kv_flow",
        num_drafter_layers=1,
    )
    drafter = SingleExpertFlowDrafter(fake_wrapper, config)
    context = torch.randn(2, 3, hidden_size)

    encoded = drafter.encode_hidden(context)
    pred = drafter.predict_latent_from_context(context)
    decoded = drafter.decode_latent(pred)

    assert drafter.vae is None
    assert drafter.latent_size == hidden_size
    assert torch.allclose(encoded, context.to(dtype=encoded.dtype))
    assert pred.shape == (2, 4, hidden_size)
    assert decoded.shape == pred.shape


def test_hidden_kv_flow_rejects_non_native_width(fake_wrapper):
    with pytest.raises(ValueError, match="expert_dim to equal"):
        SingleExpertFlowDrafter(
            fake_wrapper,
            SingleExpertFlowConfig(
                context_size=3,
                draft_length=4,
                chunk_size=4,
                vae_dir=None,
                expert_dim=6,
                num_heads=2,
                ffn_multiplier=2,
                architecture="hidden_kv_flow",
            ),
        )


def test_hidden_kv_flow_training_module_returns_finite_loss(fake_wrapper):
    hidden_size = fake_wrapper.model.config.hidden_size
    module = SingleExpertFlowTrainingModule(
        fake_wrapper,
        SingleExpertFlowConfig(
            context_size=2,
            draft_length=4,
            chunk_size=4,
            vae_dir=None,
            expert_dim=hidden_size,
            num_heads=2,
            ffn_multiplier=2,
            num_flow_steps=1,
            init_mode="delta",
            architecture="hidden_kv_flow",
            num_drafter_layers=1,
        ),
        FlowLossArguments(),
    )
    output = module(
        context_hidden=torch.randn(2, 2, hidden_size),
        target_hidden=torch.randn(2, 4, hidden_size),
        future_tokens=torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]]),
    )
    output["loss"].backward()

    assert torch.isfinite(output["loss"])
    assert output["pred_latent"].shape == (2, 4, hidden_size)
    assert output["pred_hidden"].shape == (2, 4, hidden_size)
    assert any(parameter.grad is not None for parameter in module.drafter.expert.parameters())
