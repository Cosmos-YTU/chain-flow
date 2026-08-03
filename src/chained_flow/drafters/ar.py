from __future__ import annotations

import torch

from chained_flow.drafters.base import DraftResult
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


class ARDrafter:
    def __init__(self, frozen_lm: FrozenLMWrapper):
        self.frozen_lm = frozen_lm

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        if max_tokens <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        tokens: list[torch.Tensor] = []
        local_ids = state.input_ids
        with timed_section(timings, "drafter_ar", self.frozen_lm.device):
            for _ in range(max_tokens):
                outputs = self.frozen_lm._forward(local_ids, use_cache=False)
                token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tokens.append(token)
                local_ids = torch.cat([local_ids, token], dim=1)
        return DraftResult(tokens=torch.cat(tokens, dim=1), timings=timings)
