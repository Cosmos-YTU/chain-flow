import torch

from chain_flow.drafters.embed_flow import EmbedFlowDrafter, EmbedFlowConfig
from chain_flow.training.train_chunked_flow import FlowLossArguments
from chain_flow.training.train_embed_flow import EmbedModelArguments, EmbedTrainingModule, embed_config_from_args


def _config(K=4, steps=4, cfdim=8, init="noise"):
    return EmbedFlowConfig(
        context_size=3, draft_length=K, expert_dim=8, context_feature_dim=cfdim,
        num_heads=2, ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=steps, init_mode=init,
    )


def _state(wrapper, ids):
    state, _ = wrapper.prefill(torch.tensor([ids]))
    return state


def test_propose_shapes(fake_wrapper):
    d = EmbedFlowDrafter(fake_wrapper, _config(4))
    r = d.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)
    assert r.logits.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    d = EmbedFlowDrafter(fake_wrapper, _config(4))
    s = _state(fake_wrapper, [1, 2, 3])
    assert d.propose(s, 2).tokens.shape == (1, 2)
    assert d.propose(s, 0).tokens.shape == (1, 0)


def test_target_is_embedding_not_hidden(fake_wrapper):
    # the flow target must be embed(future_tokens), distinct from any hidden state
    d = EmbedFlowDrafter(fake_wrapper, _config(4))
    fut = torch.randint(0, 8, (2, 4))
    out = d.forward_teacher(torch.randn(2, 3, 8), fut)
    expected = d._embed_tokens(fut)
    assert torch.allclose(out["e_target"], expected, atol=1e-5)
    assert out["v_pred"].shape == out["v_star"].shape == (2, 4, 8)
    assert out["pred_embed"].shape == (2, 4, 8)


def test_more_flow_steps_changes_output(fake_wrapper):
    s = _state(fake_wrapper, [1, 2, 3])
    torch.manual_seed(0); d1 = EmbedFlowDrafter(fake_wrapper, _config(4, steps=1))
    torch.manual_seed(0); d8 = EmbedFlowDrafter(fake_wrapper, _config(4, steps=8))
    torch.manual_seed(1); h1 = d1.propose(s, 4).hidden_states
    torch.manual_seed(1); h8 = d8.propose(s, 4).hidden_states
    assert not torch.allclose(h1, h8, atol=1e-4)


def test_training_backward_reaches_drafter(fake_wrapper):
    ma = EmbedModelArguments(context_size=3, draft_length=4, expert_dim=8, context_feature_dim=8,
                             num_heads=2, ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=4)
    m = EmbedTrainingModule(fake_wrapper, embed_config_from_args(ma), FlowLossArguments())
    m.train()
    out = m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
            future_tokens=torch.randint(0, 8, (4, 4)))
    assert out["loss"].requires_grad
    out["loss"].backward()
    grads = [p.grad for n, p in m.named_parameters() if p.requires_grad and "drafter" in n]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_multilayer_context_dim(fake_wrapper):
    d = EmbedFlowDrafter(fake_wrapper, _config(4, cfdim=24))  # 3 layers * 8
    out = d.forward_teacher(torch.randn(2, 3, 24), torch.randint(0, 8, (2, 4)))
    assert out["pred_embed"].shape == (2, 4, 8)
