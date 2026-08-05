import torch

from chained_flow.drafters.markov2_flow import Markov2FlowDrafter, Markov2FlowConfig
from chained_flow.training.train_chunked_flow import FlowLossArguments
from chained_flow.training.train_markov2_flow import Markov2ModelArguments, Markov2TrainingModule, markov2_config_from_args


def _cfg(K=4, order=2, gate=True):
    return Markov2FlowConfig(context_size=3, draft_length=K, chunk_size=K, expert_dim=8, num_heads=2,
                             ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2,
                             markov_rank=8, markov_order=order, markov_hidden_gate=gate)


def _state(w, ids):
    s, _ = w.prefill(torch.tensor([ids])); return s


def test_propose_shapes(fake_wrapper):
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4))
    r = d.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4))
    s = _state(fake_wrapper, [1, 2, 3])
    assert d.propose(s, 2).tokens.shape == (1, 2)
    assert d.propose(s, 0).tokens.shape == (1, 0)


def test_head_starts_as_noop(fake_wrapper):
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4))
    bias = d.markov.bias([torch.tensor([1]), torch.tensor([2])], torch.randn(1, 8))
    assert torch.allclose(bias, torch.zeros_like(bias))  # w2 zero-init


def test_order2_uses_two_tokens(fake_wrapper):
    # after a step (so w2 != 0), order-2 bias should depend on BOTH previous tokens
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4, order=2, gate=False))
    torch.nn.init.normal_(d.markov.w2.weight, std=0.1)
    h = torch.randn(1, 8)
    b_12 = d.markov.bias([torch.tensor([1]), torch.tensor([2])], h)
    b_13 = d.markov.bias([torch.tensor([1]), torch.tensor([3])], h)  # change the 2nd-prev token
    assert not torch.allclose(b_12, b_13)  # depends on token i-2 too (true order-2)


def test_hidden_gate_changes_bias(fake_wrapper):
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4, gate=True))
    torch.nn.init.normal_(d.markov.w2.weight, std=0.1)
    torch.nn.init.normal_(d.markov.gate.weight, std=0.5)
    prev = [torch.tensor([1]), torch.tensor([2])]
    b1 = d.markov.bias(prev, torch.randn(1, 8))
    b2 = d.markov.bias(prev, torch.randn(1, 8))
    assert not torch.allclose(b1, b2)  # gating makes the bias hidden-dependent


def test_block_bias_matches_shapes(fake_wrapper):
    d = Markov2FlowDrafter(fake_wrapper, _cfg(4))
    bb = d.markov.block_bias(torch.randint(0, 8, (3, 4)), torch.randn(3, 4, 8))
    assert bb.shape == (3, 4, 8)
    assert torch.allclose(bb[:, 0], torch.zeros_like(bb[:, 0]))  # position 0 has no history


def test_training_backward_reaches_flow_and_markov(fake_wrapper):
    ma = Markov2ModelArguments(context_size=3, draft_length=4, chunk_size=4, expert_dim=8, num_heads=2,
                               ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2,
                               markov_rank=8, markov_order=2, markov_hidden_gate=True)
    m = Markov2TrainingModule(fake_wrapper, markov2_config_from_args(ma), FlowLossArguments())
    m.train()
    out = m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
            future_tokens=torch.randint(0, 8, (4, 4)))
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "expert" in n)
    assert m.drafter.markov.w2.weight.grad is not None and m.drafter.markov.w2.weight.grad.abs().sum() > 0
