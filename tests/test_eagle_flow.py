import torch

from chained_flow.drafters.eagle_flow import EagleDrafter, EagleFlowConfig
from chained_flow.frozen_lm import LMState
from chained_flow.training.train_chunked_flow import FlowLossArguments
from chained_flow.training.train_eagle_flow import EagleTrainingModule, eagle_config_from_args, EagleModelArguments


def _config(K=4):
    # fake model: hidden_size == vocab == 8, must be divisible by num_heads
    return EagleFlowConfig(context_size=3, draft_length=K, expert_dim=8, num_heads=2, ffn_multiplier=2, num_drafter_layers=2)


def _state(wrapper, ids):
    state, _ = wrapper.prefill(torch.tensor([ids]))
    return state


def test_propose_returns_k_tokens_and_free_first_token(fake_wrapper):
    K = 4
    drafter = EagleDrafter(fake_wrapper, _config(K))
    state = _state(fake_wrapper, [1, 2, 3])
    result = drafter.propose(state, K)
    assert result.tokens.shape == (1, K)
    assert result.hidden_states.shape == (1, K, 8)
    # token 0 comes from the REAL anchor hidden: fake hidden = one_hot(id%8), shift lm_head -> (id+1)%8
    anchor_id = int(state.input_ids[0, -1].item())
    assert int(result.tokens[0, 0].item()) == (anchor_id + 1) % 8


def test_propose_respects_max_tokens(fake_wrapper):
    drafter = EagleDrafter(fake_wrapper, _config(4))
    state = _state(fake_wrapper, [1, 2, 3])
    assert drafter.propose(state, 2).tokens.shape == (1, 2)
    assert drafter.propose(state, 0).tokens.shape == (1, 0)


def test_forward_teacher_shapes_and_anchor_is_real(fake_wrapper):
    K = 4
    drafter = EagleDrafter(fake_wrapper, _config(K))
    B, D = 2, 8
    context_hidden = torch.randn(B, 3, D)
    target_hidden = torch.randn(B, K, D)
    future_tokens = torch.randint(0, 8, (B, K))
    pred = drafter.forward_teacher(context_hidden, target_hidden, future_tokens)
    assert pred.shape == (B, K, D)
    # position 0 must be the real anchor (target_hidden[:,0]) passed through unchanged
    assert torch.allclose(pred[:, 0], target_hidden[:, 0].to(pred.dtype), atol=1e-5)
    # later positions must differ from the anchor (the network produced them)
    assert not torch.allclose(pred[:, 1], target_hidden[:, 0].to(pred.dtype), atol=1e-3)


def test_training_module_loss_backward_reaches_drafter(fake_wrapper):
    K = 4
    model_args = EagleModelArguments(context_size=3, draft_length=K, expert_dim=8, num_heads=2, ffn_multiplier=2, num_drafter_layers=2)
    module = EagleTrainingModule(fake_wrapper, eagle_config_from_args(model_args), FlowLossArguments())
    B, D = 3, 8
    out = module(
        context_hidden=torch.randn(B, 3, D),
        target_hidden=torch.randn(B, K, D),
        future_tokens=torch.randint(0, 8, (B, K)),
    )
    assert out["loss"].requires_grad
    out["loss"].backward()
    grads = [p.grad for n, p in module.named_parameters() if p.requires_grad and "drafter" in n]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_self_feed_chain_is_differentiable_through_positions(fake_wrapper):
    # gradient at the deepest predicted position should flow back into the input_proj (chain intact)
    drafter = EagleDrafter(fake_wrapper, _config(4))
    context_hidden = torch.randn(1, 3, 8)
    target_hidden = torch.randn(1, 4, 8)
    future_tokens = torch.randint(0, 8, (1, 4))
    pred = drafter.forward_teacher(context_hidden, target_hidden, future_tokens)
    pred[:, -1].pow(2).sum().backward()
    assert drafter.input_proj.weight.grad is not None
    assert drafter.input_proj.weight.grad.abs().sum() > 0
