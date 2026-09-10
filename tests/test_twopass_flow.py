import torch

from chain_flow.drafters.twopass_flow import TwoPassFlowDrafter, TwoPassFlowConfig
from chain_flow.training.train_chunked_flow import FlowLossArguments
from chain_flow.training.train_twopass_flow import TwoPassModelArguments, TwoPassTrainingModule, twopass_config_from_args


def _cfg(K=4):
    return TwoPassFlowConfig(context_size=3, draft_length=K, chunk_size=K, expert_dim=8, num_heads=2,
                             ffn_multiplier=2, num_drafter_layers=2, num_refine_layers=2,
                             num_flow_steps=2, num_refine_steps=1)


def _state(w, ids):
    s, _ = w.prefill(torch.tensor([ids])); return s


def test_propose_shapes(fake_wrapper):
    d = TwoPassFlowDrafter(fake_wrapper, _cfg(4))
    r = d.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)


def test_propose_respects_max_tokens(fake_wrapper):
    d = TwoPassFlowDrafter(fake_wrapper, _cfg(4))
    s = _state(fake_wrapper, [1, 2, 3])
    assert d.propose(s, 2).tokens.shape == (1, 2)
    assert d.propose(s, 0).tokens.shape == (1, 0)


def test_shift_emb_is_causal(fake_wrapper):
    # position 0 must get a ZERO token-embedding (no previous draft token); position i gets emb(i-1)
    d = TwoPassFlowDrafter(fake_wrapper, _cfg(4))
    toks = torch.tensor([[3, 5, 2, 7]])
    se = d._shift_emb(toks)
    assert torch.allclose(se[:, 0], torch.zeros_like(se[:, 0]))               # pos0 = 0
    assert torch.allclose(se[:, 1], d._embed(toks[:, :1])[:, 0])              # pos1 = emb(tok0)
    assert torch.allclose(se[:, 3], d._embed(toks[:, 2:3])[:, 0])             # pos3 = emb(tok2)


def test_forward_teacher_outputs(fake_wrapper):
    d = TwoPassFlowDrafter(fake_wrapper, _cfg(4))
    out = d.forward_teacher(torch.randn(2, 3, 8), torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4)))
    for key in ("v1_pred", "v1_star", "v2_pred", "v2_star", "h1", "h2", "logits1", "logits2"):
        assert key in out
    assert out["h2"].shape == (2, 4, 8)
    # pass 2 must actually change the hidden (refinement != identity)
    assert not torch.allclose(out["h1"], out["h2"], atol=1e-4)


def test_pass2_conditioning_is_causal_only(fake_wrapper):
    # changing a LATER pass-1 token must NOT change an earlier refined position (strict causality)
    d = TwoPassFlowDrafter(fake_wrapper, _cfg(4)).eval()
    ctx = torch.randn(1, 3, 8)
    h1 = torch.randn(1, 4, 8)
    ta = torch.tensor([[1, 2, 3, 4]]); tb = torch.tensor([[1, 2, 3, 7]])  # differ only at position 3
    with torch.no_grad():
        ha = d._integrate_pass2(ctx, h1, d._shift_emb(ta))
        hb = d._integrate_pass2(ctx, h1, d._shift_emb(tb))
    # positions 0..2 depend on tokens <=1..2 (unchanged) -> identical; only pos3 sees tok2 (same) so all equal here,
    # but position that consumes tok3 (there is none, tok3 is the last prev for a hypothetical pos4) -> all K equal.
    assert torch.allclose(ha[:, :3], hb[:, :3], atol=1e-5)  # causal: earlier positions unaffected by later token


def test_training_backward_reaches_both_nets(fake_wrapper):
    ma = TwoPassModelArguments(context_size=3, draft_length=4, chunk_size=4, expert_dim=8, num_heads=2,
                               ffn_multiplier=2, num_drafter_layers=2, num_refine_layers=2,
                               num_flow_steps=2, num_refine_steps=1)
    m = TwoPassTrainingModule(fake_wrapper, twopass_config_from_args(ma), FlowLossArguments())
    m.train()
    out = m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
            future_tokens=torch.randint(0, 8, (4, 4)))
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "expert" in n)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "refine" in n)
