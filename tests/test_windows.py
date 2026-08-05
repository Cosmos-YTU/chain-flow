import torch

from chained_flow.data.windows import build_token_windows, teacher_hidden_span


def test_build_token_windows():
    windows = build_token_windows(torch.tensor([1, 2, 3, 4, 5]), prefix_length=2, future_length=2)
    assert len(windows) == 2
    assert windows[0].prefix_ids.tolist() == [[1, 2]]
    assert windows[0].future_ids.tolist() == [[3, 4]]
    assert windows[1].prefix_ids.tolist() == [[2, 3]]
    assert windows[1].future_ids.tolist() == [[4, 5]]


def test_teacher_hidden_span_alignment():
    span = teacher_hidden_span(prefix_length=4, future_length=3)
    assert span.start == 3
    assert span.stop == 6
