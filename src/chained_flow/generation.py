from __future__ import annotations

from dataclasses import dataclass, field

import torch

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.base import BaseDrafter
from chained_flow.frozen_lm import LMState
from chained_flow.timing import TimingStats, timed_section
from chained_flow.verifier import SpeculativeVerifier


@dataclass
class GenerationStepStats:
    draft_len: int
    accepted_len: int
    generated_count: int
    timings: TimingStats = field(default_factory=TimingStats)


@dataclass
class GenerationResult:
    input_ids: torch.Tensor
    generated_ids: torch.Tensor
    step_stats: list[GenerationStepStats]
    timings: TimingStats

    @property
    def generated_token_count(self) -> int:
        return self.generated_ids.shape[1] - self.input_ids.shape[1]


def _append_state_token_ids(state: LMState, token_ids: torch.Tensor) -> LMState:
    return LMState(
        input_ids=torch.cat([state.input_ids, token_ids], dim=1),
        past_key_values=state.past_key_values,
        final_hidden=state.final_hidden,
        logits=state.logits,
        position=state.position + token_ids.shape[1],
    )


@torch.inference_mode()
def generate_with_drafter(
    context: ChainedFlowContext,
    drafter: BaseDrafter,
    prompt: str | torch.Tensor,
    *,
    max_new_tokens: int,
    draft_len: int,
    eos_token_id: int | None = None,
    force_zero_accept: bool = False,
    fold_anchor: bool = True,
) -> GenerationResult:
    frozen_lm = context.frozen_lm
    timings = TimingStats()
    step_stats: list[GenerationStepStats] = []
    eos_token_id = frozen_lm.eos_token_id if eos_token_id is None else eos_token_id

    if fold_anchor:
        return _generate_with_drafter_folded(
            context,
            drafter,
            prompt,
            max_new_tokens=max_new_tokens,
            draft_len=draft_len,
            eos_token_id=eos_token_id,
            force_zero_accept=force_zero_accept,
        )

    with timed_section(timings, "total_generation", frozen_lm.device):
        input_ids = frozen_lm.tokenize(prompt) if isinstance(prompt, str) else prompt.to(frozen_lm.device)
        state, prefill_timings = frozen_lm.prefill(input_ids)
        timings.merge(prefill_timings)

        verifier = SpeculativeVerifier(frozen_lm)
        generated = 0

        while generated < max_new_tokens:
            anchor_token, next_timings = frozen_lm.next_token(state)
            timings.merge(next_timings)
            state, anchor_timings = frozen_lm.forward_with_cache(anchor_token, state, use_cache=True)
            timings.merge(anchor_timings)
            generated += 1

            if eos_token_id is not None and int(anchor_token.item()) == int(eos_token_id):
                break

            remaining_after_anchor = max_new_tokens - generated
            if remaining_after_anchor <= 0 or draft_len <= 0:
                break

            proposal = drafter.propose(state, min(draft_len, remaining_after_anchor))
            timings.merge(proposal.timings, "drafter")
            verify_result = verifier.verify(
                state,
                proposal.tokens,
                max_accept_len=0 if force_zero_accept else None,
            )
            timings.merge(verify_result.timings, "verifier")
            state = verify_result.state

            accepted = verify_result.acceptance.accepted_len
            emitted = accepted
            generated += emitted
            step_stats.append(
                GenerationStepStats(
                    draft_len=proposal.tokens.shape[1],
                    accepted_len=accepted,
                    generated_count=emitted,
                    timings=verify_result.timings,
                )
            )

            if eos_token_id is not None:
                new_tokens = state.input_ids[:, -emitted:]
                if emitted > 0 and (new_tokens == eos_token_id).any():
                    break

        total = timings.get("total_generation")
        if total > 0:
            timings.add("tokens_per_second", generated / total)

    return GenerationResult(
        input_ids=input_ids,
        generated_ids=state.input_ids,
        step_stats=step_stats,
        timings=timings,
    )


@torch.inference_mode()
def _generate_with_drafter_folded(
    context: ChainedFlowContext,
    drafter: BaseDrafter,
    prompt: str | torch.Tensor,
    *,
    max_new_tokens: int,
    draft_len: int,
    eos_token_id: int | None,
    force_zero_accept: bool,
) -> GenerationResult:
    """Anchor-folded speculative loop.

    Identical token output to :func:`generate_with_drafter` but one fewer backbone forward
    per partial-acceptance step: the per-step "anchor" token is taken from the verify pass's
    fallback (committed together with the accepted prefix via ``commit_next_token``) instead
    of a separate forward. The first token is still bootstrapped explicitly.

    ``final_len`` tracks how many emitted tokens the unfolded loop would keep, so we can crop
    the (eagerly committed) fallback at end-of-sequence / max-token boundaries and stay
    token-for-token identical.
    """
    frozen_lm = context.frozen_lm
    timings = TimingStats()
    step_stats: list[GenerationStepStats] = []

    with timed_section(timings, "total_generation", frozen_lm.device):
        input_ids = frozen_lm.tokenize(prompt) if isinstance(prompt, str) else prompt.to(frozen_lm.device)
        state, prefill_timings = frozen_lm.prefill(input_ids)
        timings.merge(prefill_timings)
        prompt_len = input_ids.shape[1]
        verifier = SpeculativeVerifier(frozen_lm)

        final_len = 0
        if max_new_tokens >= 1:
            # Bootstrap the first token (the unfolded loop's first anchor).
            anchor_token, next_timings = frozen_lm.next_token(state)
            timings.merge(next_timings)
            state, anchor_timings = frozen_lm.forward_with_cache(anchor_token, state, use_cache=True)
            timings.merge(anchor_timings)
            final_len = 1
            anchor_is_eos = eos_token_id is not None and int(anchor_token.item()) == int(eos_token_id)

            while not anchor_is_eos and final_len < max_new_tokens and draft_len > 0:
                g_before = final_len  # emitted tokens before this round (includes current anchor)
                proposal = drafter.propose(state, draft_len)
                timings.merge(proposal.timings, "drafter")
                if proposal.tokens.shape[1] == 0:
                    break

                verify_result = verifier.verify(
                    state,
                    proposal.tokens,
                    max_accept_len=0 if force_zero_accept else None,
                    commit_next_token=True,
                )
                timings.merge(verify_result.timings, "verifier")
                state = verify_result.state
                accepted = verify_result.acceptance.accepted_len
                step_stats.append(
                    GenerationStepStats(
                        draft_len=proposal.tokens.shape[1],
                        accepted_len=accepted,
                        generated_count=accepted + 1,
                        timings=verify_result.timings,
                    )
                )

                # Mirror the unfolded loop's truncation: it caps accepted tokens to the room
                # left after the anchor, stops on eos inside the accepted block, and only emits
                # the fallback (next anchor) if there is still room and no eos was hit.
                remaining = max_new_tokens - g_before
                keep = min(accepted, remaining)
                acc_start = prompt_len + g_before
                if eos_token_id is not None and keep > 0:
                    acc_kept = state.input_ids[:, acc_start : acc_start + keep]
                    if (acc_kept == eos_token_id).any():
                        final_len = g_before + keep
                        break
                if keep < accepted:
                    # Ran out of budget inside the accepted block; no fallback emitted.
                    final_len = g_before + keep
                    break
                new_len = g_before + accepted
                if new_len >= max_new_tokens:
                    final_len = new_len
                    break
                # Emit the fallback token, which becomes the next round's anchor.
                fallback_id = int(state.input_ids[0, acc_start + accepted].item())
                final_len = new_len + 1
                anchor_is_eos = eos_token_id is not None and fallback_id == eos_token_id

        total = timings.get("total_generation")
        if total > 0:
            timings.add("tokens_per_second", final_len / total)

    generated_ids = state.input_ids[:, : prompt_len + final_len]
    return GenerationResult(
        input_ids=input_ids,
        generated_ids=generated_ids,
        step_stats=step_stats,
        timings=timings,
    )
