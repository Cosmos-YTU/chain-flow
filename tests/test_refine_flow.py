import torch

from chain_flow.drafters.refine_flow import RefineFlowDrafter, RefineFlowConfig
from chain_flow.training.train_chunked_flow import FlowLossArguments
from chain_flow.training.train_refine_flow import RefineModelArguments, RefineTrainingModule, refine_config_from_args


def _config(K=4, steps=4, init="delta", self_cond=0.0):
    return RefineFlowConfig(
        context_size=3, draft_length=K, expert_dim=8, num_heads=2, ffn_multiplier=2,
        num_drafter_layers=2, num_refine_steps=steps, init_mode=init, self_cond_prob=self_cond,
    )


def _state(wrapper, ids):
    state, _ = wrapper.prefill(torch.tensor([ids]))
    return state


def test_propose_shapes(fake_wrapper):
    drafter = RefineFlowDrafter(fake_wrapper, _config(4))
    state = _state(fake_wrapper, [1, 2, 3])
    r = drafter.propose(state, 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)
    assert r.logits.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    drafter = RefineFlowDrafter(fake_wrapper, _config(4))
    state = _state(fake_wrapper, [1, 2, 3])
    assert drafter.propose(state, 2).tokens.shape == (1, 2)
    assert drafter.propose(state, 0).tokens.shape == (1, 0)


def test_more_steps_changes_output(fake_wrapper):
    # with token re-conditioning, more refine steps should change the integrated block
    state = _state(fake_wrapper, [1, 2, 3])
    torch.manual_seed(0)
    d2 = RefineFlowDrafter(fake_wrapper, _config(4, steps=1))
    torch.manual_seed(0)
    d8 = RefineFlowDrafter(fake_wrapper, _config(4, steps=8))
    # same init weights via same seed; differ only in step count -> hidden trajectories differ
    h1 = d2.propose(state, 4).hidden_states
    h8 = d8.propose(state, 4).hidden_states
    assert not torch.allclose(h1, h8, atol=1e-4)


def test_forward_teacher_flow_and_integrated(fake_wrapper):
    drafter = RefineFlowDrafter(fake_wrapper, _config(4))
    B, K, D = 2, 4, 8
    out = drafter.forward_teacher(torch.randn(B, 3, D), torch.randn(B, K, D), torch.randint(0, 8, (B, K)))
    assert out["v_pred"].shape == (B, K, D)
    assert out["v_star"].shape == (B, K, D)
    assert out["pred_hidden"].shape == (B, K, D)


def test_training_module_backward_reaches_drafter(fake_wrapper):
    model_args = RefineModelArguments(
        context_size=3, draft_length=4, expert_dim=8, num_heads=2, ffn_multiplier=2,
        num_drafter_layers=2, num_refine_steps=4, self_cond_prob=0.0,
    )
    module = RefineTrainingModule(fake_wrapper, refine_config_from_args(model_args), FlowLossArguments())
    module.train()
    out = module(
        context_hidden=torch.randn(3, 3, 8),
        target_hidden=torch.randn(3, 4, 8),
        future_tokens=torch.randint(0, 8, (3, 4)),
    )
    assert out["loss"].requires_grad
    out["loss"].backward()
    grads = [p.grad for n, p in module.named_parameters() if p.requires_grad and "drafter" in n]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_self_conditioning_runs(fake_wrapper):
    model_args = RefineModelArguments(
        context_size=3, draft_length=4, expert_dim=8, num_heads=2, ffn_multiplier=2,
        num_drafter_layers=2, num_refine_steps=4, self_cond_prob=1.0,  # always self-condition
    )
    module = RefineTrainingModule(fake_wrapper, refine_config_from_args(model_args), FlowLossArguments())
    module.train()
    out = module(
        context_hidden=torch.randn(2, 3, 8),
        target_hidden=torch.randn(2, 4, 8),
        future_tokens=torch.randint(0, 8, (2, 4)),
    )
    out["loss"].backward()  # self-cond uses no_grad decode internally; loss must still backprop
    assert module.drafter.input_proj.weight.grad is not None
