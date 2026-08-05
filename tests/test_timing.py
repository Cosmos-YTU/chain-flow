from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.ar import ARDrafter
from chained_flow.generation import generate_with_drafter


def test_generation_timing_fields(fake_wrapper):
    context = ChainedFlowContext(fake_wrapper)
    result = generate_with_drafter(
        context,
        ARDrafter(fake_wrapper),
        "a b",
        max_new_tokens=2,
        draft_len=1,
        eos_token_id=None,
    )
    assert result.timings.get("total_generation") >= 0.0
    assert result.timings.get("prefill") >= 0.0
    assert result.timings.get("tokens_per_second") >= 0.0
    assert "drafter.drafter_ar" in result.timings.sections
    assert "verifier.verifier_forward" in result.timings.sections
