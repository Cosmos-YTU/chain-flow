import torch

from chained_flow.verifier import SpeculativeVerifier


def test_verifier_accepts_matching_prefix_and_crops_cache(fake_wrapper):
    state, _ = fake_wrapper.prefill(torch.tensor([[1, 2]]))
    anchor = state.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    state, _ = fake_wrapper.forward_with_cache(anchor, state)

    result = SpeculativeVerifier(fake_wrapper).verify(state, torch.tensor([[4, 9]]))

    assert result.acceptance.accepted_len == 1
    assert result.acceptance.next_token.tolist() == [[5]]
    assert result.state.input_ids.tolist() == [[1, 2, 3, 4]]
    assert result.state.past_key_values.length == result.state.position
