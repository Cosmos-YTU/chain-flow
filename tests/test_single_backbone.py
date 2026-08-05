import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.ar import ARDrafter
from chained_flow.frozen_lm import FrozenLMWrapper
from chained_flow.timing import TimingStats
from chained_flow.verifier import SpeculativeVerifier


def test_context_loads_backbone_once(monkeypatch, fake_wrapper):
    calls = []

    def fake_from_pretrained(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_wrapper, TimingStats()

    monkeypatch.setattr(FrozenLMWrapper, "from_pretrained", fake_from_pretrained)
    context = ChainedFlowContext.from_pretrained("fake")

    assert len(calls) == 1
    assert context.frozen_lm is fake_wrapper


def test_components_share_wrapper_identity(fake_wrapper):
    drafter = ARDrafter(fake_wrapper)
    verifier = SpeculativeVerifier(fake_wrapper)
    assert drafter.frozen_lm is fake_wrapper
    assert verifier.frozen_lm is fake_wrapper


def test_frozen_lm_head_casts_hidden_to_head_dtype(fake_wrapper):
    fake_wrapper.model.lm_head.weight.data = fake_wrapper.model.lm_head.weight.data.to(torch.bfloat16)
    hidden = torch.randn(2, 3, fake_wrapper.model.config.hidden_size, dtype=torch.float32)

    logits = fake_wrapper.lm_head(hidden)

    assert logits.dtype == torch.bfloat16
    assert logits.shape == (2, 3, fake_wrapper.model.config.hidden_size)

