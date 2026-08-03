from __future__ import annotations

from dataclasses import dataclass, field
import copy

import torch

from chained_flow.acceptance import AcceptanceResult, greedy_acceptance
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class VerifyResult:
    state: LMState
    acceptance: AcceptanceResult
    timings: TimingStats = field(default_factory=TimingStats)


class SpeculativeVerifier:
    def __init__(self, frozen_lm: FrozenLMWrapper):
        self.frozen_lm = frozen_lm

    @torch.inference_mode()
    def verify(
        self,
        state: LMState,
        draft_tokens: torch.Tensor,
        *,
        max_accept_len: int | None = None,
        commit_next_token: bool = False,
    ) -> VerifyResult:
        timings = TimingStats()
        device = self.frozen_lm.device
        draft_tokens = draft_tokens.to(device)

        verify_state = LMState(
            input_ids=state.input_ids,
            past_key_values=copy.deepcopy(state.past_key_values),
            final_hidden=state.final_hidden,
            logits=state.logits,
            position=state.position,
        )
        with timed_section(timings, "verifier_forward", device):
            verified_state, forward_timings = self.frozen_lm.forward_with_cache(draft_tokens, verify_state, use_cache=True)
        timings.merge(forward_timings)

        with timed_section(timings, "acceptance", device):
            draft_position_logits = verified_state.logits[
                :,
                state.position : state.position + draft_tokens.shape[1],
                :,
            ]
            verifier_tokens = torch.cat(
                [
                    state.logits[:, -1:, :].argmax(dim=-1),
                    draft_position_logits.argmax(dim=-1),
                ],
                dim=1,
            )
            acceptance = greedy_acceptance(draft_tokens, verifier_tokens)
            if max_accept_len is not None and acceptance.accepted_len > max_accept_len:
                if max_accept_len < 0:
                    raise ValueError("max_accept_len must be non-negative")
                accepted_len = min(max_accept_len, draft_tokens.shape[1])
                acceptance = AcceptanceResult(
                    accepted_len=accepted_len,
                    accepted_tokens=draft_tokens[:, :accepted_len],
                    next_token=verifier_tokens[:, accepted_len : accepted_len + 1],
                    matches=acceptance.matches,
                )

        with timed_section(timings, "cache_repair", device):
            committed_count = acceptance.accepted_len
            full_accept = committed_count == draft_tokens.shape[1]
            if not commit_next_token:
                if full_accept:
                    repaired_state = verified_state
                elif committed_count == 0:
                    repaired_state = state
                else:
                    committed_tokens = draft_tokens[:, :committed_count]
                    repaired_state, repair_timings = self.frozen_lm.forward_with_cache(
                        committed_tokens,
                        state,
                        use_cache=True,
                    )
                    timings.merge(repair_timings)
            else:
                # Fold the verifier's fallback/next token into the same commit forward so the
                # caller does not need a separate per-step anchor forward. On full acceptance we
                # reuse the cache the verify pass already built and only append the fallback,
                # which keeps token-for-token parity with the unfolded path; otherwise we
                # recompute the accepted prefix plus the fallback in a single forward.
                next_token = acceptance.next_token.to(device)
                if full_accept:
                    repaired_state, repair_timings = self.frozen_lm.forward_with_cache(
                        next_token,
                        verified_state,
                        use_cache=True,
                    )
                else:
                    committed_tokens = torch.cat(
                        [draft_tokens[:, :committed_count], next_token], dim=1
                    )
                    repaired_state, repair_timings = self.frozen_lm.forward_with_cache(
                        committed_tokens,
                        state,
                        use_cache=True,
                    )
                timings.merge(repair_timings)

        return VerifyResult(state=repaired_state, acceptance=acceptance, timings=timings)
