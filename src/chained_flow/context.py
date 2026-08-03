from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.timing import TimingStats


@dataclass
class ChainedFlowContext:
    frozen_lm: FrozenLMWrapper
    timings: TimingStats = field(default_factory=TimingStats)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        **kwargs: Any,
    ) -> "ChainedFlowContext":
        frozen_lm, timings = FrozenLMWrapper.from_pretrained(model_id, **kwargs)
        return cls(frozen_lm=frozen_lm, timings=timings)
