from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch

from chained_flow.frozen_lm import LMState
from chained_flow.timing import TimingStats


@dataclass
class DraftResult:
    tokens: torch.Tensor
    hidden_states: torch.Tensor | None = None
    latent_states: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    timings: TimingStats = field(default_factory=TimingStats)


class BaseDrafter(Protocol):
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        ...
