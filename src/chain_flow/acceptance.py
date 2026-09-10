from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AcceptanceResult:
    accepted_len: int
    accepted_tokens: torch.Tensor
    next_token: torch.Tensor
    matches: torch.Tensor


def greedy_acceptance(draft_tokens: torch.Tensor, verifier_tokens: torch.Tensor) -> AcceptanceResult:
    if draft_tokens.ndim != 2 or verifier_tokens.ndim != 2:
        raise ValueError("draft_tokens and verifier_tokens must both have shape [B, K] or [B, K+1]")
    if draft_tokens.shape[0] != verifier_tokens.shape[0]:
        raise ValueError("batch sizes must match")
    if draft_tokens.shape[0] != 1:
        raise ValueError("generation v1 supports batch size 1")
    if verifier_tokens.shape[1] < draft_tokens.shape[1] + 1:
        raise ValueError("verifier_tokens must include one fallback token after the draft span")

    verifier_draft_span = verifier_tokens[:, : draft_tokens.shape[1]]
    matches = draft_tokens == verifier_draft_span
    if matches.shape[1] == 0:
        accepted_len = 0
    else:
        accepted_len = int(matches.cumprod(dim=1).sum(dim=1)[0].item())
    accepted_tokens = draft_tokens[:, :accepted_len]
    next_token = verifier_tokens[:, accepted_len : accepted_len + 1]
    return AcceptanceResult(
        accepted_len=accepted_len,
        accepted_tokens=accepted_tokens,
        next_token=next_token,
        matches=matches,
    )
