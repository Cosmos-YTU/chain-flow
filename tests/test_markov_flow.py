import torch

from chain_flow.drafters.markov_flow import MarkovFlowDrafter, MarkovFlowConfig
from chain_flow.training.train_chunked_flow import FlowLossArguments
from chain_flow.training.train_markov_flow import MarkovModelArguments, MarkovTrainingModule, markov_config_from_args


def _cfg(K=4, C=4):
    return MarkovFlowConfig(context_size=3, draft_length=K, chunk_size=C, expert_dim=8,
                            num_heads=2, ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2, markov_rank=4)


def _state(w, ids):
    s, _ = w.prefill(torch.tensor([ids])); return s


def test_propose_shapes(fake_wrapper):
    d = MarkovFlowDrafter(fake_wrapper, _cfg(4))
    r = d.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    d = MarkovFlowDrafter(fake_wrapper, _cfg(4))
    s = _state(fake_wrapper, [1, 2, 3])
    assert d.propose(s, 2).tokens.shape == (1, 2)
    assert d.propose(s, 0).tokens.shape == (1, 0)


def test_markov_head_starts_as_noop(fake_wrapper):
    # w2 zero-init => bias is 0 at init, so markov correction starts as a no-op (only learns delta)
    d = MarkovFlowDrafter(fake_wrapper, _cfg(4))
    bias = d.markov.bias(torch.tensor([1, 2, 3]))
    assert torch.allclose(bias, torch.zeros_like(bias))


def test_markov_bias_shapes(fake_wrapper):
    d = MarkovFlowDrafter(fake_wrapper, _cfg(4))
    assert d.markov.bias(torch.tensor([1, 2])).shape == (2, 8)          # [B] -> [B,V]
    assert d.markov.bias(torch.tensor([[1, 2, 3, 4]])).shape == (1, 4, 8)  # [B,K] -> [B,K,V]


def test_forward_teacher_outputs(fake_wrapper):
    d = MarkovFlowDrafter(fake_wrapper, _cfg(4))
    out = d.forward_teacher(torch.randn(2, 3, 8), torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4)))
    for key in ("v_pred", "v_star", "pred_hidden", "base_logits", "markov_logits"):
        assert key in out
    assert out["markov_logits"].shape == (2, 4, 8)
    # position 0 markov bias is zeroed -> base and markov logits equal there
    assert torch.allclose(out["base_logits"][:, 0], out["markov_logits"][:, 0], atol=1e-5)


def test_training_backward_reaches_flow_and_markov(fake_wrapper):
    ma = MarkovModelArguments(context_size=3, draft_length=4, chunk_size=4, expert_dim=8,
                              num_heads=2, ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2, markov_rank=4)
    m = MarkovTrainingModule(fake_wrapper, markov_config_from_args(ma), FlowLossArguments())
    m.train()
    out = m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
            future_tokens=torch.randint(0, 8, (4, 4)))
    out["loss"].backward()
    # flow expert must receive gradient
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "expert" in n)
    # markov head: w2 is zero-init so w1's grad is 0 on the FIRST step (flows through w2=0); w2 itself
    # gets gradient immediately, and after one step w1 trains too. Verify w2 learns + chain works post-step.
    assert m.drafter.markov.w2.weight.grad is not None and m.drafter.markov.w2.weight.grad.abs().sum() > 0
    # take a step so w2 != 0, then confirm w1 now receives gradient (chicken-and-egg resolved)
    torch.optim.SGD(m.parameters(), lr=1.0).step()
    m.zero_grad()
    m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
      future_tokens=torch.randint(0, 8, (4, 4)))["loss"].backward()
    assert m.drafter.markov.w1.weight.grad is not None and m.drafter.markov.w1.weight.grad.abs().sum() > 0
