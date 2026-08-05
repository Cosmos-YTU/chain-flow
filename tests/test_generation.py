import pytest
import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.ar import ARDrafter
from chained_flow.drafters.base import DraftResult
from chained_flow.generation import generate_with_drafter


def test_generation_with_ar_drafter_matches_shift_pattern(fake_wrapper):
    result = generate_with_drafter(
        ChainedFlowContext(fake_wrapper),
        ARDrafter(fake_wrapper),
        torch.tensor([[1, 2]]),
        max_new_tokens=4,
        draft_len=2,
        eos_token_id=None,
    )
    assert result.generated_ids.tolist() == [[1, 2, 3, 4, 5, 6]]
    assert [step.accepted_len for step in result.step_stats] == [2]


class WrongDrafter:
    def propose(self, state, max_tokens):
        return DraftResult(tokens=torch.full((1, max_tokens), 0, dtype=torch.long))


def test_generation_with_wrong_drafter_falls_back_to_verifier(fake_wrapper):
    result = generate_with_drafter(
        ChainedFlowContext(fake_wrapper),
        WrongDrafter(),
        torch.tensor([[1, 2]]),
        max_new_tokens=3,
        draft_len=2,
        eos_token_id=None,
    )
    assert result.generated_ids.tolist() == [[1, 2, 3, 4, 5]]
    assert [step.accepted_len for step in result.step_stats] == [0, 0]


class PartialDrafter:
    """Proposes the correct next token then wrong tokens, so exactly one token accepts."""

    def propose(self, state, max_tokens):
        nxt = state.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if max_tokens <= 1:
            return DraftResult(tokens=nxt[:, :max_tokens])
        wrong = torch.zeros((1, max_tokens - 1), dtype=torch.long)
        return DraftResult(tokens=torch.cat([nxt, wrong], dim=1)[:, :max_tokens])


def _make_drafter(kind, wrapper):
    if kind == "ar":
        return ARDrafter(wrapper)
    if kind == "wrong":
        return WrongDrafter()
    if kind == "partial":
        return PartialDrafter()
    raise ValueError(kind)


@pytest.mark.parametrize("drafter_kind", ["ar", "wrong", "partial"])
@pytest.mark.parametrize("eos_token_id", [None, 7])
@pytest.mark.parametrize("draft_len", [1, 2, 3, 4])
@pytest.mark.parametrize("max_new_tokens", [1, 2, 3, 5, 8, 12])
@pytest.mark.parametrize("prompt", [[1], [1, 2], [3, 1, 2]])
def test_fold_anchor_matches_unfolded(
    fake_wrapper, drafter_kind, eos_token_id, draft_len, max_new_tokens, prompt
):
    context = ChainedFlowContext(fake_wrapper)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        draft_len=draft_len,
        eos_token_id=eos_token_id,
    )
    baseline = generate_with_drafter(
        context, _make_drafter(drafter_kind, fake_wrapper), torch.tensor([prompt]), fold_anchor=False, **kwargs
    )
    folded = generate_with_drafter(
        context, _make_drafter(drafter_kind, fake_wrapper), torch.tensor([prompt]), fold_anchor=True, **kwargs
    )
    assert folded.generated_ids.tolist() == baseline.generated_ids.tolist()
