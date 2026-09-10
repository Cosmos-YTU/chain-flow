from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenWindow:
    prefix_ids: torch.Tensor
    future_ids: torch.Tensor
    start: int


def build_token_windows(
    token_ids: torch.Tensor,
    *,
    prefix_length: int,
    future_length: int,
    stride: int = 1,
) -> list[TokenWindow]:
    if token_ids.ndim == 2:
        if token_ids.shape[0] != 1:
            raise ValueError("windowing v1 expects a single token sequence")
        token_ids = token_ids[0]
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [N] or [1, N]")
    if prefix_length <= 0 or future_length <= 0 or stride <= 0:
        raise ValueError("prefix_length, future_length, and stride must be positive")

    windows: list[TokenWindow] = []
    total = prefix_length + future_length
    for start in range(0, token_ids.shape[0] - total + 1, stride):
        prefix = token_ids[start : start + prefix_length].unsqueeze(0)
        future = token_ids[start + prefix_length : start + total].unsqueeze(0)
        windows.append(TokenWindow(prefix_ids=prefix, future_ids=future, start=start))
    return windows


def teacher_hidden_span(prefix_length: int, future_length: int) -> slice:
    if prefix_length <= 0 or future_length <= 0:
        raise ValueError("prefix_length and future_length must be positive")
    return slice(prefix_length - 1, prefix_length + future_length - 1)
