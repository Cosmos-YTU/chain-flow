import torch

from chained_flow.acceptance import greedy_acceptance


def test_greedy_acceptance_full_match():
    result = greedy_acceptance(torch.tensor([[1, 2]]), torch.tensor([[1, 2, 3]]))
    assert result.accepted_len == 2
    assert result.accepted_tokens.tolist() == [[1, 2]]
    assert result.next_token.tolist() == [[3]]


def test_greedy_acceptance_partial_match():
    result = greedy_acceptance(torch.tensor([[1, 9, 3]]), torch.tensor([[1, 2, 3, 4]]))
    assert result.accepted_len == 1
    assert result.accepted_tokens.tolist() == [[1]]
    assert result.next_token.tolist() == [[2]]


def test_greedy_acceptance_zero_match():
    result = greedy_acceptance(torch.tensor([[9, 2]]), torch.tensor([[1, 2, 3]]))
    assert result.accepted_len == 0
    assert result.accepted_tokens.tolist() == [[]]
    assert result.next_token.tolist() == [[1]]


def test_greedy_acceptance_single_token():
    result = greedy_acceptance(torch.tensor([[1]]), torch.tensor([[1, 2]]))
    assert result.accepted_len == 1
    assert result.next_token.tolist() == [[2]]
