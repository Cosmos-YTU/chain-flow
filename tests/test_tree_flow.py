import torch

from chain_flow.drafters.tree_flow import DraftTree, TreeFlowConfig, TreeFlowDrafter
from chain_flow.training.train_chunked_flow import FlowLossArguments
from chain_flow.training.train_tree_flow import TreeModelArguments, TreeFlowTrainingModule, tree_config_from_args


def _cfg(K=4, order=4):
    return TreeFlowConfig(context_size=3, draft_length=K, chunk_size=K, expert_dim=8, num_heads=2,
                          ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2,
                          path_order=order, path_ffn_multiplier=2, cov_b=3, tree_top_b=2, tree_max_nodes=8)


def _state(w, ids):
    s, _ = w.prefill(torch.tensor([ids])); return s


def test_propose_shapes(fake_wrapper):
    d = TreeFlowDrafter(fake_wrapper, _cfg(4))
    r = d.propose(_state(fake_wrapper, [1, 2, 3]), 4)
    assert r.tokens.shape == (1, 4)
    assert r.hidden_states.shape == (1, 4, 8)
    assert int(r.tokens.min()) >= 0 and int(r.tokens.max()) < 8


def test_propose_respects_max_tokens(fake_wrapper):
    d = TreeFlowDrafter(fake_wrapper, _cfg(4))
    s = _state(fake_wrapper, [1, 2, 3])
    assert d.propose(s, 2).tokens.shape == (1, 2)
    assert d.propose(s, 0).tokens.shape == (1, 0)


def test_parent_stack_is_causal(fake_wrapper):
    # parent_ids[:, i, j] must be token at position i-(j+1); mask 0 where that position < 0.
    d = TreeFlowDrafter(fake_wrapper, _cfg(4, order=4))
    toks = torch.tensor([[3, 5, 2, 7]])
    ids, mask = d._parent_stack(toks)
    assert torch.allclose(mask[0, 0], torch.zeros(4))                 # position 0 has no parents
    assert int(ids[0, 1, 0]) == 3 and float(mask[0, 1, 0]) == 1.0     # pos1 parent-1 = tok0
    assert float(mask[0, 1, 1]) == 0.0                                # pos1 has only one parent
    assert [int(x) for x in ids[0, 3, :3]] == [2, 5, 3]               # pos3 parents = tok2,tok1,tok0
    assert float(mask[0, 3, 3]) == 0.0                                # pos3 parent-4 does not exist


def test_path_head_is_noop_at_init(fake_wrapper):
    # zero-init output -> residual is exactly zero at init, so V9 == V6's marginal hidden initially.
    d = TreeFlowDrafter(fake_wrapper, _cfg(4)).eval()
    res = d._path_residual(torch.tensor([[3, 5, 2, 7]]))
    assert torch.allclose(res, torch.zeros_like(res))


def test_path_residual_causality_when_active(fake_wrapper):
    # after making the path head non-identity, a LATER token must not change an EARLIER residual.
    d = TreeFlowDrafter(fake_wrapper, _cfg(4, order=4)).eval()
    with torch.no_grad():
        d.path_head.mlp[-1].weight.normal_(std=0.1)
        d.path_head.mlp[-1].bias.normal_(std=0.1)
    ta = torch.tensor([[1, 2, 3, 4]]); tb = torch.tensor([[1, 2, 3, 6]])  # differ only at position 3
    ra = d._path_residual(ta); rb = d._path_residual(tb)
    assert torch.allclose(ra[:, :3], rb[:, :3], atol=1e-5)             # positions 0..2 unaffected by tok3


def test_forward_teacher_outputs(fake_wrapper):
    d = TreeFlowDrafter(fake_wrapper, _cfg(4))
    out = d.forward_teacher(torch.randn(2, 3, 8), torch.randn(2, 4, 8), torch.randint(0, 8, (2, 4)))
    for key in ("v_pred", "v_star", "pred_hidden", "cond_hidden", "base_logits", "cond_logits"):
        assert key in out
    assert out["cond_logits"].shape == (2, 4, 8)
    # at init the path/markov heads are no-ops, so conditioned == marginal (a superset-of-V6 guarantee)
    assert torch.allclose(out["cond_hidden"], out["pred_hidden"])
    assert torch.allclose(out["cond_logits"], out["base_logits"])


def test_coverage_loss_drops_when_true_token_promoted(fake_wrapper):
    m = TreeFlowTrainingModule(fake_wrapper, tree_config_from_args(TreeModelArguments(
        context_size=3, draft_length=4, chunk_size=4, expert_dim=8, num_heads=2, ffn_multiplier=2,
        num_drafter_layers=2, num_flow_steps=2, cov_b=3)), FlowLossArguments())
    logits = torch.zeros(2, 4, 8)
    tok = torch.randint(0, 8, (2, 4))
    high = m._coverage(logits, tok)                          # true token buried among ties
    boosted = logits.scatter(-1, tok.unsqueeze(-1), 50.0)
    low = m._coverage(boosted, tok)                          # true token now clearly top-b
    assert float(low) < float(high)
    assert float(low) == 0.0


def test_training_backward_reaches_flow_path_and_markov(fake_wrapper):
    ma = TreeModelArguments(context_size=3, draft_length=4, chunk_size=4, expert_dim=8, num_heads=2,
                            ffn_multiplier=2, num_drafter_layers=2, num_flow_steps=2, cov_b=3, lambda_cov=0.3)
    m = TreeFlowTrainingModule(fake_wrapper, tree_config_from_args(ma), FlowLossArguments())
    m.train()
    out = m(context_hidden=torch.randn(4, 3, 8), target_hidden=torch.randn(4, 4, 8),
            future_tokens=torch.randint(0, 8, (4, 4)))
    out["loss"].backward()

    def grad_hits(substr):
        return any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in m.named_parameters() if substr in n)

    assert grad_hits("expert")               # the flow velocity net
    assert grad_hits("path_head")            # zero-init output layer still receives gradient
    assert grad_hits("markov")               # markov head (via w2)


def test_build_tree_structure(fake_wrapper):
    d = TreeFlowDrafter(fake_wrapper, _cfg(4)).eval()
    tree = d.build_tree(_state(fake_wrapper, [1, 2, 3]), top_b=2, max_nodes=16)
    assert isinstance(tree, DraftTree)
    assert tree.num_nodes() > 0
    # depth-0 roots count == top_b (vocab 8 >= 2)
    assert sum(1 for dep in tree.depths if dep == 0) == 2
    # every non-root node's parent has strictly smaller depth; tokens in-vocab; cum logprob <= parent
    for i, (par, dep) in enumerate(zip(tree.parents, tree.depths)):
        assert 0 <= tree.tokens[i] < 8
        if par == -1:
            assert dep == 0
        else:
            assert tree.depths[par] == dep - 1
            assert tree.cum_logprob[i] <= tree.cum_logprob[par] + 1e-4   # added a log-prob (<= 0)
    # every root-to-leaf path is contiguous in depth and no longer than draft_length
    for path in tree.root_to_leaf_paths():
        assert [tree.depths[n] for n in path] == list(range(len(path)))
        assert len(path) <= 4
