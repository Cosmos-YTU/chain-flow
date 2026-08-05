import torch

from chained_flow.drafters.dflash_flow import DFlashFlowDrafter, DFlashFlowConfig
from chained_flow.training.train_chunked_flow import FlowLossArguments
from chained_flow.training.train_dflash_flow import DFlashModelArguments, DFlashTrainingModule, dflash_config_from_args


def _config(K=4, cfdim=8):
    # fake model: hidden_size = vocab = 8
    return DFlashFlowConfig(
        context_size=3, draft_length=K, expert_dim=8, context_feature_dim=cfdim,
        num_heads=2, ffn_multiplier=2, num_drafter_layers=2,
    )


def _state(wrapper, ids):
    state, _ = wrapper.prefill(torch.tensor([ids]))
    return state


def test_propose_shapes(fake_wrapper):
    drafter = DFlashFlowDrafter(fake_wrapper, _config(4))
    r = drafter.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)
    assert r.logits.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    drafter = DFlashFlowDrafter(fake_wrapper, _config(4))
    state = _state(fake_wrapper, [1, 2, 3])
    assert drafter.propose(state, 2).tokens.shape == (1, 2)
    assert drafter.propose(state, 0).tokens.shape == (1, 0)


def test_forward_teacher_masks_and_shape(fake_wrapper):
    drafter = DFlashFlowDrafter(fake_wrapper, _config(4))
    B, K = 5, 4
    out = drafter.forward_teacher(torch.randn(B, 3, 8), torch.randint(0, 8, (B, K)))
    assert out["pred_hidden"].shape == (B, K, 8)
    assert out["mask"].shape == (B, K)
    assert out["mask"].any(dim=1).all()  # every example has >=1 masked position


def test_mask_embedding_used_for_masked_positions(fake_wrapper):
    drafter = DFlashFlowDrafter(fake_wrapper, _config(4))
    tokens = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, False, True]])
    block = drafter._build_block_emb(tokens, mask)
    mask_emb = drafter.mask_embedding.detach()
    # masked positions equal the mask embedding; unmasked equal token embeddings
    assert torch.allclose(block[0, 1], mask_emb, atol=1e-5)
    assert torch.allclose(block[0, 3], mask_emb, atol=1e-5)
    assert not torch.allclose(block[0, 0], mask_emb, atol=1e-3)


def test_training_module_backward_reaches_drafter(fake_wrapper):
    model_args = DFlashModelArguments(
        context_size=3, draft_length=4, expert_dim=8, context_feature_dim=8,
        num_heads=2, ffn_multiplier=2, num_drafter_layers=2,
    )
    module = DFlashTrainingModule(fake_wrapper, dflash_config_from_args(model_args), FlowLossArguments())
    module.train()
    out = module(
        context_hidden=torch.randn(4, 3, 8),
        target_hidden=torch.randn(4, 4, 8),  # ignored by dflash
        future_tokens=torch.randint(0, 8, (4, 4)),
    )
    assert out["loss"].requires_grad
    out["loss"].backward()
    grads = [p.grad for n, p in module.named_parameters() if p.requires_grad and "drafter" in n]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    # mask embedding should receive gradient (it's used and trainable)
    assert module.drafter.mask_embedding.grad is not None


def test_multilayer_context_feature_dim(fake_wrapper):
    # context_feature_dim != hidden_size (multi-layer cache) must project correctly
    drafter = DFlashFlowDrafter(fake_wrapper, _config(4, cfdim=24))  # 3 layers * 8
    out = drafter.forward_teacher(torch.randn(2, 3, 24), torch.randint(0, 8, (2, 4)))
    assert out["pred_hidden"].shape == (2, 4, 8)
