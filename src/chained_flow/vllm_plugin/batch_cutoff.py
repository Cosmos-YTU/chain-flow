"""Turn speculation OFF above a decode batch, so a loaded server is never worse than no
speculation at all.

THE FAILURE, MEASURED
---------------------
`vllm serve`, 4B, chain arm, `serve_ladder.sh` against the no-speculation baseline on the same
GPU (`logs/bench_serve/4b_{base_lad6,chain_lad7}`, 2026-08-05):

    conc      1       4       8      16      32      64
    base  139.0   498.8   952.9  1764.4  2968.3  4518.6   tok/s
    chain 165.2   498.6   855.7  1297.9  1743.7  1993.9
          1.19x   1.00x   0.90x   0.74x   0.59x   0.44x

Enabling speculation more than halves a loaded 4B server.  Three costs compound:

  * THE TARGET pays for every rejected draft.  With `num_speculative_tokens=K` the scheduler
    hands the target `K+1` query positions per request per step, so a batch of 64 verifies 384
    positions instead of 64.  At batch 1 that is free -- decode is memory-bound and the extra
    positions ride along in the same weight read.  At batch 64 it is not: the batch is already
    wide enough to be compute-bound, so 6x the positions is ~6x the target FLOPs, and a measured
    accept of 1.86 out of a possible 6 cannot pay for it.  Acceptance does not move across the
    whole ladder (1.854-1.863), so none of this is a drafting-quality problem.
  * THE DRAFTER's own fast paths are batch-1-gated BY CONSTRUCTION (`chunked_flow.py`'s fused
    CUDA block runner takes `x.shape[0] == 1` only), so exactly where the target's bill goes up,
    the draft's does too.
  * THE DRAFT CUDAGRAPH USED TO STOP AT BUCKET 32, and above it every distinct batch size was
    its own `torch.compile` (`CF_COMPILE` uses `dynamic=False`, and past the last bucket
    `_ctx_gpu` falls back to `bucket = B`).  The uncut ladder above logged draft batches of 34,
    37, 38, 42, 45, 46, 50, 54, 55, 59, 62, 63 and 64 -- a fresh max-autotune codegen each,
    ~70-80 s of engine stall apiece, inside live traffic.  (This is the shape of the ~850
    autotune blocks seen on a first `guidellm --rate 64` pass, whose 0.048x figure is the same
    failure with the compile storm on top of it.)

WHICH OF THOSE IS PHYSICS AND WHICH IS DEBT -- THE PROFILE, NOT THE INTUITION.  This docstring
used to say the first two "are not bugs to be fixed -- they are what speculative decoding IS".
The profile at B=64 on 4B chain falsifies that (commit 81d0ecc, docs/BENCHMARKING.md): a 65.7 ms
step with the GPU 96% busy is 35.2 ms of target forward over 384 positions and **24.2 ms of
DRAFTER**, and an A/B that skips only the draft prices the drafter at 26.0 ms/step against 18.5
ms of verify inflation.  Of the 0.56x deficit at B=64, verify width is 0.21 (38%) and THE DRAFTER
IS 0.35 (62%) -- the larger half, and the half that was engineering debt: 1901 of 2000 steps ran
the draft with no cudagraph at all and the fused batch-1 kernel fired on 2.0% of steps.  The
missing rungs are fixed in `FlowDrafterProposer._bucket_ladder` (the ladder now reaches
`max_num_seqs`), which also deletes the compile storm outright rather than by avoidance.

So the accurate statement is narrower: only the VERIFY WIDTH is intrinsic, and even it is
smaller than the `accept/(K+1)` intuition says -- `F.linear` at this model's real shapes gives
138-162 TFLOP/s at M=64 against a 330-380 asymptote, so B=64 is only half way to compute-bound
and the measured forward is 2.75x for 6x the rows, not 6x.  That is what makes a K-SCHEDULE
(below) worth having: M=128, i.e. K=1, costs only ~1.33x the M=64 forward.

WHAT THIS DOES
--------------
`CF_SPEC_MAX_BATCH=N`: when a scheduler step contains more than N requests, the NEXT step is
scheduled with no speculative tokens at all, and the drafter does not run.  Below N nothing
changes -- not one branch, not one flag -- so the batch-1 headline number is untouched.

TWO HALVES, AND BOTH ARE NEEDED
-------------------------------
1. THE SCHEDULER (this module).  This is the half that matters: it is what stops the TARGET
   from verifying drafts.  Under async scheduling the drafter cannot influence it at all --
   `AsyncScheduler._update_after_schedule` writes `request.spec_token_ids = [-1] * K` from
   `scheduler_output.num_spec_tokens_to_schedule` before any draft exists, and vLLM never reads
   our returned token ids back on that path (`_copy_draft_token_ids_to_cpu` returns early under
   `use_async_scheduling`).  So the cutoff has to be expressed where the number is decided.
2. THE PROPOSER (`FlowDrafterProposer.propose`).  Skips the flow net when the batch is over the
   cutoff, which reclaims the draft's own cost.  On its own it would not help much: the target
   would still be verifying a step's worth of zeros.

WHY NOT vLLM's OWN `num_speculative_tokens_per_batch_size`
----------------------------------------------------------
0.25.1 does have a native batch-size -> K schedule (`SpeculativeConfig.
num_speculative_tokens_per_batch_size` -> `Scheduler.dynamic_sd_lookup`), K=0 is a legal entry,
and it lands in exactly the same place this patch does.  It was rejected for one measured
reason: setting it makes `VllmConfig._maybe_override_dynamic_sd_cudagraph_mode` downgrade
`cudagraph_mode` from FULL_AND_PIECEWISE to PIECEWISE **for the whole engine**, including batch
1.  That trades a high-concurrency fix for a regression on the metric this project is judged on.
Fixing the scheduler decision alone leaves full cudagraphs captured and dispatched for every
step that still speculates.

(The above-cutoff steps are PIECEWISE either way: a step whose requests have no draft tokens has
query length 1, and `uniform_decode_query_len` is `1 + K` in a spec-configured engine, so it does
not match a captured full-decode graph.  That is a property of turning speculation off, not of
how it is turned off.)

CORRECTNESS ACROSS THE BOUNDARY
-------------------------------
THE TWO HALVES ARE IN LOCKSTEP BY CONSTRUCTION, not by luck.  Both read the batch of the SAME
step X: `_update_after_schedule` runs inside `schedule()` for step X and decides how many spec
slots step X+1 gets, and `propose()` runs during step X and produces the drafts that fill exactly
those slots.  So "the scheduler allocated slots the drafter did not fill" would require the two
to read different batches for the same step.

They can, by exactly one thing, and it points the safe way: the proposer counts the rows that
actually emitted a token (`rows`), the scheduler counts every request it scheduled, and the
former is a subset of the latter (a discarded request emits nothing).  So the proposer is only
ever MORE willing to draft than the scheduler is to schedule -- the dangerous direction is
unreachable.

And if it were reachable, both disagreements are still safe:

  * scheduler says "no spec tokens", proposer drafted anyway -> the drafts are never scheduled,
    never scattered into `input_ids` (`_prepare_input_ids` returns early on an empty
    `spec_flattened_indices`), and simply discarded.  Wasted work, correct output.
  * scheduler scheduled spec tokens, proposer skipped -> the proposer returns a FULL-WIDTH tensor
    of zeros, not a short or absent one.  Token id 0 is verified and rejected like any other bad
    draft.  Wasted work, correct output.  This is the same shape vLLM's own drafter-skip path
    uses (`gpu_model_runner.py`, `not input_fits_in_drafter` -> `torch.zeros(...).expand(nreq,
    num_spec_tokens)`), and it is why the skip must never return a NARROW draft: a short row
    leaves the tail of the persistent draft buffer holding the PREVIOUS step's tokens, which
    vLLM scatters into `input_ids` without looking -- silent corruption, see
    `greedy_guard.py`'s docstring.

A request in flight when the cutoff engages is therefore never corrupted; it decodes without
speculation for as long as the server is loaded, and resumes speculating when it is not.

DEFAULT ON, BUT ONLY WHERE THE THRESHOLD WAS MEASURED
-----------------------------------------------------
The flag shipped OFF while `_AUTO` had three entries and everything else was a guess.  It now has
six -- every size x arm this project publishes a number for -- and leaving it off means the
default deployment of a laddered combination is the 2.3x-slower one.  So `CF_SPEC_MAX_BATCH`
unset now behaves as `auto`, WITH ONE DIFFERENCE THAT IS THE WHOLE POINT: a combination that is
not in `_AUTO` resolves to 0 and speculation stays on at every batch, with the reason printed.

That asymmetry is deliberate.  `auto` asked for explicitly still takes the derived branch, because
someone who typed it asked for a best guess.  The DEFAULT must not: a derived N that is too low
silently costs speedup and nothing in the throughput number says so -- exactly what the two
27B-tree points (1.85x at concurrency 1, 1.55x at 2, against a derived N of 2) demonstrated
before that combination was laddered.  A wrong guess nobody opted into is worse than no cutoff.

`CF_SPEC_MAX_BATCH=0` (or `off`) turns it off; a number sets N directly; `auto` is the old
guess-allowed behaviour.

A K-SCHEDULE INSTEAD OF A K=0 CLIFF -- `CF_SPEC_K_SCHEDULE` (DEFAULT OFF)
------------------------------------------------------------------------
The cutoff is a cliff: below N the arm speculates with the full K, above it with none.  The
roofline says there is something in between.  `F.linear` at this model's real shapes runs at
138-162 TFLOP/s at M=64 against an M=2688 asymptote of 330-380, so a batch of 64 is only half
way to compute-bound, and M=128 -- which is exactly `B=64` at `K=1` -- costs only ~1.33x the
M=64 forward that the no-speculation baseline pays.  A K=1 arm therefore has a chance of
staying ABOVE parity at batches where K=5 provably cannot, and a cliff throws that away.

`CF_SPEC_K_SCHEDULE="4:full,16:1"` reads: at a decode batch of 4 or below schedule the engine's
full K, at 16 or below schedule K=1, above 16 schedule none.  Rungs are `<max decode batch>:<K>`,
`full` means the engine's `num_speculative_tokens`, rungs must ascend in batch and descend in K,
and there is always an implicit `K=0` above the last one.

IT LANDS IN THE SAME PLACE THE CLIFF DOES, and that is why it costs nothing new to be sure of.
`_update_after_schedule` already writes `scheduler_output.num_spec_tokens_to_schedule`; writing
`1` there instead of `0` is the same field, one step earlier in the same decision.  vLLM then
builds `request.spec_token_ids = [-1] * 1`, schedules two query positions per request, and
`_prepare_input_ids` scatters `range(start, start + draft_len)` out of the draft tensor with
`start = prev_index * self.prev_num_spec_tokens` -- i.e. it takes the FIRST `draft_len` COLUMNS
of each row of whatever width tensor the proposer returned.  For a chain that is precisely the
first `draft_len` links, in order, which is what "K=1" has to mean.  `prev_num_spec_tokens` is
set from the returned tensor's own width on every step (`_copy_draft_token_ids_to_cpu`, above
its async early-return), so the narrowing needs NO change in the proposer and cannot desynchronise
from it.  The full-width draft is still computed; the K-schedule buys VERIFY width, not draft
cost, and the profile says the two are 38% / 62% of the deficit respectively.

WHY THIS AND NOT vLLM's `num_speculative_tokens_per_batch_size`: unchanged, see below -- setting
it downgrades `cudagraph_mode` engine-wide including batch 1.  The scheduler patch expresses the
same schedule without touching the config, which is the whole reason it exists.

DEFAULT OFF, AND IT STAYS OFF UNTIL A LADDER SAYS OTHERWISE.  `_AUTO_K1` is the measured
`(hidden_size, K+1) -> highest decode batch at which K=1 is still at or above parity` table, and
it is EMPTY: a K=1 rung that is set too high is a slowdown nobody opted into, which is the same
asymmetry `_AUTO` is governed by.  Until an entry exists, the schedule is only what
`CF_SPEC_K_SCHEDULE` says, and the resolution log prints the ladder together with whether each
rung was MEASURED or DERIVED.
"""
from __future__ import annotations

import os

#: Set once the scheduler patch is actually in place.
INSTALLED = False
REASON = "not attempted"

_FLAG = "CF_SPEC_MAX_BATCH"
#: MEASUREMENT ONLY.  `CF_SPEC_CUTOFF_HALF=drafter` installs the proposer skip but NOT the
#: scheduler patch, so the target still verifies a full step of (zeroed) drafts while the drafter
#: does nothing.  The difference between that arm and the full cutoff is the VERIFY half of the
#: bill, and the difference between it and the uncut chain arm is the DRAFT half -- which is the
#: only way to say where the 21x actually goes rather than assert it.  `=scheduler` is the
#: mirror.  Default `both`, and nothing but a benchmark should ever set it.
_HALF = "CF_SPEC_CUTOFF_HALF"

#: `CF_SPEC_MAX_BATCH=auto` -> the measured crossing point for this target AND THIS ARM.
#:
#: THE THRESHOLD IS NOT A CONSTANT AND GUESSING IT IS EXPENSIVE IN BOTH DIRECTIONS.  Measured
#: 2026-08-05 against the no-speculation baseline, `serve_ladder.sh`, one RTX PRO 6000:
#:
#:      conc          1      2      4      8     16     32     64
#:   4B  chain    1.19x         1.00x  0.90x  0.74x  0.59x  0.44x   -> crosses at 4
#:   4B  tree     1.35x         0.85x  0.59x  0.35x                 -> crosses below 4
#:   27B chain    1.62x         1.40x  1.33x  1.07x  0.80x  0.50x   -> crosses between 16 and 32
#:
#: Two independent axes, and BOTH have been measured to matter:
#:
#:   * MODEL SIZE.  Setting 4 on a 27B server throws away 1.33x at concurrency 8 -- measured, not
#:     predicted: the 27B `CF_SPEC_MAX_BATCH=4` arm reads 190.4 tok/s against the uncut 253.5.
#:   * VERIFY WIDTH.  The scheduler hands the target `K+1` query positions per request per step,
#:     so the batch at which it stops being memory-bound scales inversely with `K+1`.  A chain
#:     is 6 positions and the 8x5 tree is 42 -- SEVEN TIMES wider -- and the 4B tree is
#:     correspondingly already under water at concurrency 4 where the 4B chain is exactly at
#:     parity.  A table keyed on size alone would put the tree threshold seven times too high.
#:
#: So the key is `(hidden_size, K+1)`.  Entries are measurements; anything else is derived by the
#: width rule below and SAYS SO.
#: KEYED ON `draft_width + 1`, which is what the proposer can report about itself: 6 for a chain
#: (K=5 drafts plus the bonus) and 41 for the 8x5 tree (40 tree nodes plus the bonus).  That is
#: one less than the 42 query positions the SCHEDULER hands the target for a tree, because the
#: tree's `num_speculative_tokens` is itself `keep*depth + 1`.  The exact convention matters only
#: in that the key must match what `_build()` computes -- getting it wrong is silent, and did
#: happen: a table keyed on 42 never matched, and the 4B tree quietly took the DERIVED branch.
_AUTO = {
    # (hidden_size, draft_width + 1): (N, provenance)
    (2560, 6): (4, "measured: 4B chain"),
    (5120, 6): (16, "measured: 27B chain"),
    # 9B chain, `9b_chain_thr` vs `9b_base_thr`, 2026-08-05, one RTX PRO 6000:
    #   conc     1      2      4      8     16     32     64
    #   base  83.7  157.7  315.8  632.4 1123.5 1983.8 3127.6  tok/s
    #   chain 107.3 188.5  336.1  578.4  908.8 1231.3 1463.2
    #        1.28x  1.20x  1.06x  0.91x  0.81x  0.62x  0.47x
    # Crossing between 4 and 8, so N=4 -- the last laddered batch still above parity, which is
    # the same rule the other four entries were chosen by.  The width rule DERIVED 8 for this
    # combination, i.e. it would have kept speculating at a measured 0.91x.  Acceptance is flat
    # at 1.88-1.92 across the whole ladder, so none of the decline is drafting quality.  This
    # engine reaches a decode batch of 64, so every level here is a real batch and not a queue.
    (4096, 6): (4, "measured: 9B chain (1.28x at batch 1, 1.06x at 4, 0.91x at 8)"),
    # 4B tree, laddered at 1/2/3/4: 187.5 / 268.3 / 349.9 / 426.2 tok/s against a base of
    # 139.0 / 256.8 / ~378 (interpolated) / 498.8 -> 1.35x / 1.04x / 0.93x / 0.85x.  N=2 keeps
    # the point that is still above parity and drops the first one that is not.
    (2560, 41): (2, "measured: 4B tree (1.35x at batch 1, 1.04x at 2, 0.93x at 3)"),
    # 9B tree, `9b_tree_lad` + `9b_tree_mid` vs `9b_base_thr`, 2026-08-05:
    #   conc      1      2      3*     4      6*     8     16
    #   base   83.7  157.7  236.8  315.8  474.1  632.4 1123.5  tok/s
    #   tree  123.7  191.3  258.9  307.2  383.8  419.8  464.0
    #        1.48x  1.21x  1.09x  0.97x  0.81x  0.66x  0.41x
    # N=3: batch 3 is the last one above parity and batch 4 is the first one below.  Both
    # neighbours were measured rather than assumed -- concurrency 3 and 6 were run separately
    # (`9b_tree_mid`) precisely because 1.21x at 2 and 0.97x at 4 do not say where between them
    # the crossing sits, and the answer moved N from the 2 the 4B tree's rule would suggest.
    # * base at 3 and 6 is interpolated, and it is a safe interpolation here rather than a
    #   flagged weakness: the base arm's PER-REQUEST throughput is flat at 78.8-79.1 tok/s from
    #   concurrency 2 to 8, so base(3) and base(6) are 3x and 6x that to within a fraction of a
    #   percent.  The tree points at 3 and 6 are measured.
    (4096, 41): (3, "measured: 9B tree (1.48x at batch 1, 1.21x at 2, 1.09x at 3, 0.97x at 4)"),
    # 27B tree, `27b_tree_lad` vs `27b_base_lad`, laddered at nominal concurrency 1/2/4/8:
    # 48.7 / 79.1 / 107.8 / 107.2 tok/s against 26.3 / 51.0* / 100.2 / 190.8 -> 1.85x / 1.55x /
    # 1.08x / 0.56x.  The width rule DERIVED 2; the measurement says 3, and that extra batch is
    # worth the 1.08x the derived value would have thrown away.
    #
    # N IS 3 AND NOT 4, AND THE DIFFERENCE IS THE WHOLE REASON THIS TABLE SAYS "DECODE BATCH"
    # RATHER THAN "CONCURRENCY".  `CF_BATCH_AUDIT` on this run: the decode-batch histogram is
    # {1: 4251, 2: 2636, 3: 6613} -- B NEVER REACHES 4, at any offered load, because a 27B tree
    # engine gets 27,185 KV tokens and cannot admit a fourth request.  So nominal concurrency 4
    # ran at a decode batch of 3, and batch 4 was never measured; recording 4 would be recording
    # a number this ladder did not produce.  Batches 1, 2 and 3 are all above parity, so on this
    # engine the cutoff correctly never fires.
    #
    # The 0.56x at nominal 8 is NOT a decode-batch crossing and must not be read as one: the B
    # histogram there is the same {1,2,3} and TTFT is 11.6 s.  That is the admission queue, i.e.
    # the `--speculative-config` KV reservation (docs/BENCHMARKING.md), and no decode-batch
    # threshold can address it -- cutting speculation at a batch of 3 would only remove the 2.50
    # acceptance that is currently carrying those three requests.
    #
    # * base at concurrency 2 is from the earlier `27b_base` run; `27b_base_lad` has no c=2.
    (5120, 41): (3, "measured: 27B tree (1.85x at batch 1, 1.55x at 2, 1.08x at 3; batch >= 4 "
                    "unreachable -- a 27k-token KV caps this engine's decode batch at 3)"),
}

#: The width rule, for a `(size, K+1)` that was never laddered: the crossing sits at a roughly
#: constant TOTAL verify width `B x (K+1)` for a given model size -- 4B chain crosses at 4 x 6 =
#: 24, 27B chain at 16 x 6 = 96 -- because that width is where the target forward stops being
#: memory-bound.  Interpolating on it is a guess, so it is floored at 1 (never worse than "batch
#: 1 only") and always reported as derived, and since the default refuses to use it, reaching
#: this table now takes an explicit `CF_SPEC_MAX_BATCH=auto` on an unladdered target.
#:
#: THE RULE'S OWN TRACK RECORD, NOW THAT ALL SIX POINTS ARE MEASURED, IS MIXED -- which is the
#: argument for laddering rather than trusting it.  It gets the 4B tree right (24/41 -> 1, and 2
#: measured) and is close on the 27B tree (96/41 -> 2, and 3 measured), but the 9B row was pure
#: interpolation and it was WRONG IN THE EXPENSIVE DIRECTION: 48/6 -> 8 for the 9B chain, which
#: measured 0.91x at a batch of 8.  The 9B rung is left at its interpolated 48 rather than
#: back-fitted to the measurement, because a table entry that has been tuned to the points it is
#: judged on stops being evidence about the points it has not seen.
_WIDTH_AT_CROSSING = ((3072, 24, "4B"), (4608, 48, "9B, itself interpolated"), (1 << 30, 96, "27B"))


def auto_for(hidden_size: int, k_plus_1: int = 6,
             measured_only: bool = False) -> tuple[int, str]:
    """`(N, why)` for a target of this hidden size running an arm of this verify width.

    `measured_only` is what makes the DEFAULT safe to turn on: it refuses to guess, returning
    `(0, why)` -- no cutoff at all -- for a combination nobody laddered.  See `resolve()`.
    """
    hit = _AUTO.get((int(hidden_size), int(k_plus_1)))
    if hit:
        return hit
    if measured_only:
        return 0, (f"NOT MEASURED for hidden_size={hidden_size} at verify width {k_plus_1}, and "
                   f"the default only engages a MEASURED threshold -- speculation stays ON at "
                   f"every batch. A derived N that is too low silently costs speedup, so it is "
                   f"not something to opt into by accident. Set {_FLAG}=auto for the derived "
                   f"guess, or {_FLAG}=<n> for one you measured yourself")
    for hi, width, label in _WIDTH_AT_CROSSING:
        if hidden_size <= hi:
            n = max(1, width // max(int(k_plus_1), 1))
            return n, (f"DERIVED, not measured for this combination: verify width {k_plus_1} "
                       f"into the {label} crossing width {width}")
    return 0, "no rule"


#: Resolved by `FlowDrafterProposer._build()` when the flag is `auto`.  It lives here, not in the
#: proposer, because the SCHEDULER half reads it -- and the scheduler and the proposer are the
#: same process (the engine core), so a module global is the whole mechanism.
_RESOLVED: int = 0


# ======================================================================================
# THE K-SCHEDULE.  See the module docstring.
# ======================================================================================
_KFLAG = "CF_SPEC_K_SCHEDULE"

#: `full`, as a rung's K.  Not a magic number in the ladder: the engine's own
#: `num_spec_tokens_to_schedule` is what "full" resolves to, and only the SCHEDULER knows it.
FULL = -1

#: MEASURED highest decode batch at which a K=1 arm is still at or above the no-speculation
#: baseline, keyed exactly like `_AUTO`.  DELIBERATELY EMPTY: an entry here turns a K=1 rung ON
#: by default for that target, and a rung set too high is a silent slowdown -- the same asymmetry
#: that keeps `_AUTO` measured-only.  Add an entry only with a ladder behind it, with the tok/s
#: in the provenance string the way `_AUTO`'s entries carry theirs.
_AUTO_K1: dict[tuple[int, int], tuple[int, str]] = {}

#: The ladder actually in force: ascending `(max decode batch, K)` rungs, implicit K=0 above the
#: last.  Empty means "no K-schedule" -- `should_cut` then falls back to the plain `_AUTO` cliff,
#: which is the shipping behaviour.
_RESOLVED_LADDER: tuple[tuple[int, int], ...] = ()
_LADDER_WHY: str = "no K-schedule"


def parse_schedule(spec: str) -> tuple[tuple[int, int], ...]:
    """`"4:full,16:1"` -> `((4, FULL), (16, 1))`.

    Validates rather than tolerates.  A schedule that ascends in K, or repeats a batch, or names
    a K of 0 in the middle, is a typo whose only symptom would be a throughput number -- so it
    raises here, in the process that read the flag, instead of quietly becoming a different
    schedule than the one that was typed.
    """
    out: list[tuple[int, int]] = []
    for part in spec.replace(",", " ").split():
        n_s, _, k_s = part.partition(":")
        if not _:
            raise ValueError(f"{_KFLAG}: rung {part!r} is not '<max decode batch>:<K>'")
        n = int(n_s)
        k = FULL if k_s.strip().lower() in ("full", "max", "k") else int(k_s)
        if n < 1:
            raise ValueError(f"{_KFLAG}: rung {part!r} has a max decode batch below 1; use "
                             f"{_FLAG}=0 to turn speculation off entirely")
        if k == 0:
            raise ValueError(f"{_KFLAG}: rung {part!r} names K=0, which is what the IMPLICIT "
                             f"rung above the last one already means -- drop it")
        if k < FULL:
            raise ValueError(f"{_KFLAG}: rung {part!r} has a negative K")
        if out and n <= out[-1][0]:
            raise ValueError(f"{_KFLAG}: rung {part!r} does not ascend in decode batch")
        if out and out[-1][1] != FULL and (k == FULL or k > out[-1][1]):
            raise ValueError(f"{_KFLAG}: rung {part!r} raises K as the batch grows, which is "
                             f"backwards -- a wider batch can afford LESS speculation, not more")
        out.append((n, k))
    return tuple(out)


def auto_ladder(hidden_size: int, k_plus_1: int,
                measured_only: bool = True) -> tuple[tuple[tuple[int, int], ...], str]:
    """The default K-schedule for this target and arm, and where each rung came from.

    Returns `((), why)` when there is nothing measured, which is the shipping answer today: the
    plain `_AUTO` cliff then stands unchanged.
    """
    n1, why1 = auto_for(hidden_size, k_plus_1, measured_only=measured_only)
    hit = _AUTO_K1.get((int(hidden_size), int(k_plus_1)))
    if not hit:
        return (), (f"no K=1 rung is MEASURED for hidden_size={hidden_size} at verify width "
                    f"{k_plus_1}, so the schedule stays the {_FLAG} cliff (N={n1}). Set "
                    f"{_KFLAG} to try one.")
    n2, why2 = hit
    if not n1:
        return ((n2, 1),), f"K=1 up to {n2} ({why2}); no measured full-K rung ({why1})"
    if n2 <= n1:
        return ((n1, FULL),), (f"K=1 rung {n2} ({why2}) is not above the full-K rung {n1} "
                               f"({why1}), so it would never fire -- ignored")
    return ((n1, FULL), (n2, 1)), f"full K up to {n1} ({why1}); K=1 up to {n2} ({why2})"


def ladder() -> tuple[tuple[int, int], ...]:
    """The K-schedule in force.  `CF_SPEC_K_SCHEDULE` beats the resolved default.

    Read from the environment every call for the same reason `_mode()` is: this module is
    imported independently by the API-server and engine-core processes and must be the same fact
    in both.
    """
    spec = (os.environ.get(_KFLAG, "") or "").strip()
    if spec and spec.lower() not in _OFF_WORDS:
        # An EXPLICIT schedule supersedes `CF_SPEC_MAX_BATCH` rather than being gated by it: its
        # last rung IS the cutoff, so obeying both would mean two thresholds for one boundary.
        return parse_schedule(spec)
    if _mode() == "off":
        return ()
    return _RESOLVED_LADDER


def set_resolved_ladder(rungs, why: str) -> None:
    global _RESOLVED_LADDER, _LADDER_WHY
    _RESOLVED_LADDER = tuple((int(n), int(k)) for n, k in rungs)
    _LADDER_WHY = why


def describe_ladder() -> str:
    lad = ladder()
    if not lad:
        return f"no K-schedule ({_LADDER_WHY})"
    body = ", ".join(f"B<={n} -> K={'full' if k == FULL else k}" for n, k in lad)
    src = (f"{_KFLAG}={os.environ.get(_KFLAG, '').strip()}"
           if (os.environ.get(_KFLAG, "") or "").strip() else _LADDER_WHY)
    return f"{body}, else K=0  [{src}]"


def k_for(nreq: int, full: int) -> int:
    """How many speculative tokens a step of `nreq` requests should be scheduled.

    `full` is the engine's own `num_spec_tokens_to_schedule`, i.e. what it would have used with
    no patch at all -- so with no ladder and no cutoff this returns exactly that and the patch is
    a no-op by construction rather than by matching a constant.
    """
    lad = ladder()
    if not lad:
        return 0 if should_cut(nreq) else full
    for n, k in lad:
        if nreq <= n:
            return full if k == FULL else min(k, full)
    return 0


#: Words that mean "off".  `0` is one of them and has to stay one: before the default flipped,
#: `CF_SPEC_MAX_BATCH=0` was already the way to say "no cutoff", and anything that had it pinned
#: off must keep reading off rather than silently acquire a threshold.
_OFF_WORDS = ("0", "off", "no", "none", "false", "disable", "disabled")


def _mode() -> str:
    """`off` | `fixed` | `auto` | `default`.

    Read from the environment on every call rather than cached: this module is imported by the
    API-server process and the engine-core process independently, and the value has to be the
    same fact in both.  It is a couple of dict lookups on a path that already does far more.

    A value that parses as nothing at all reads `off`, not `default`: an engine must never
    acquire a behaviour because someone typo'd the flag that was meant to configure it.
    """
    v = (os.environ.get(_FLAG, "") or "").strip().lower()
    if not v:
        return "default"
    if v == "auto":
        return "auto"
    if v in _OFF_WORDS:
        return "off"
    try:
        return "fixed" if int(v) > 0 else "off"
    except ValueError:
        return "off"


def resolve(hidden_size: int, k_plus_1: int) -> tuple[int, str]:
    """`(N, why)` for THIS engine under the mode actually in force.

    The one place the default's extra rule lives: `auto` typed by a human may guess, the default
    may not.  `_build()` calls this and nothing else, so the two cannot drift apart.
    """
    return auto_for(hidden_size, k_plus_1, measured_only=_mode() == "default")


def set_resolved(n: int, why: str) -> None:
    global _RESOLVED
    _RESOLVED = max(int(n), 0)
    how = "unset -> default" if _mode() == "default" else f"{_FLAG}=auto"
    if _RESOLVED:
        print(f"[cf-plugin] {_FLAG} ({how}) resolved to {_RESOLVED} ({why}). Speculation is off "
              f"above a decode batch of {_RESOLVED}; at or below it nothing changes.", flush=True)
    else:
        print(f"[cf-plugin] {_FLAG} ({how}) resolved to NO CUTOFF: {why}.", flush=True)


def max_batch() -> int:
    """The configured cutoff, or 0 for "no cutoff".

    `auto` and the default read 0 until the proposer's lazy `_build()` resolves it, which is
    correct rather than merely tolerable: `_build()` runs inside the FIRST `propose()`, so the
    only steps that can precede it are the first few of the first request -- a decode batch of 1,
    which no threshold in the table would have cut anyway.
    """
    m = _mode()
    if m in ("auto", "default"):
        return _RESOLVED
    if m == "fixed":
        return int((os.environ.get(_FLAG, "") or "").strip())
    return 0


def requested() -> bool:
    """Is a cutoff IN FORCE, whatever it resolves to?

    Distinct from `max_batch() > 0` and the difference is load-bearing: under `auto` (and under
    the default) the value is 0 until `_build()` resolves it, and `install()` runs long before
    that.  Keying the scheduler patch on the resolved number would silently install nothing.

    True by default.  A non-speculative engine never calls `_build()`, so `_RESOLVED` stays 0 and
    the installed patch never cuts anything -- the scheduler wrapper is inert, not merely
    harmless.

    An explicit `CF_SPEC_K_SCHEDULE` also counts: its top rung is a cutoff, and it has to be able
    to install the patch on its own.
    """
    return _mode() != "off" or bool(ladder())


def half() -> str:
    return os.environ.get(_HALF, "both").strip().lower() or "both"


def drafter_half() -> bool:
    """Should `FlowDrafterProposer` skip the draft above the cutoff?"""
    return half() in ("both", "drafter")


def scheduler_half() -> bool:
    """Should the scheduler stop allocating spec slots above the cutoff?"""
    return half() in ("both", "scheduler")


def should_cut(nreq: int) -> bool:
    """Does a step of `nreq` requests get NO speculative tokens at all?

    N is inclusive -- a batch of exactly N still speculates.  Both halves of the cutoff call
    this (the scheduler patch below, `FlowDrafterProposer._cut` in the proposer), because the two
    disagreeing by one would be invisible in any throughput number.

    With a K-schedule in force this is the schedule's implicit top rung: above the last rung the
    step gets K=0 and the drafter skips, exactly as under the cliff.  Note a K=1 step is NOT a
    cut -- the drafter still runs, and must, because the scatter reads the first column of the
    draft it returns.
    """
    lad = ladder()
    if lad:
        return nreq > lad[-1][0]
    n = max_batch()
    return n > 0 and nreq > n


def install() -> None:
    """Patch `AsyncScheduler._update_after_schedule` to zero K for over-cutoff steps.

    Called from `async_guard.register()`, i.e. from the `vllm.general_plugins` entry point in
    `EngineArgs.__post_init__` -- before the engine config exists and long before a `Scheduler`
    is constructed, so patching the CLASS is enough and there is no instance to chase.

    Never raises: this runs in every vLLM process that merely has chained-flow installed.
    """
    global INSTALLED, REASON
    if INSTALLED:
        return
    if not requested():
        REASON = (f"{_FLAG}={(os.environ.get(_FLAG, '') or '').strip()} -- turned off explicitly; "
                  f"speculation stays on at every batch")
        return
    if not scheduler_half():
        REASON = (f"{_HALF}={half()} -- MEASUREMENT ONLY: the target will keep verifying a full "
                  f"step of drafts above the cutoff, so this arm is NOT the shipping behaviour")
        print(f"[cf-plugin] batch cutoff: scheduler half DISABLED ({REASON})", flush=True)
        return
    try:
        from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    except Exception as e:                                   # noqa: BLE001
        REASON = f"AsyncScheduler not importable ({e!r})"
        return

    orig = AsyncScheduler._update_after_schedule
    if getattr(orig, "_cf_batch_cutoff", False):
        INSTALLED = True
        return

    def _update_after_schedule(self, scheduler_output) -> None:
        # `len(num_scheduled_tokens)` is the number of requests in THIS step -- the exact
        # quantity vLLM's own dynamic-SD schedule keys on (`scheduler.py`, `dynamic_sd_lookup[
        # len(num_scheduled_tokens)]`).  Setting the field the base implementation is about to
        # read makes the placeholder list the scheduled width for every request, which is
        # bit-identical to what the native mechanism produces -- no second code path.
        #
        # The engine's own value is passed in as `full` rather than re-derived, so a step this
        # patch has no opinion about is written back unchanged and the wrapper is provably inert.
        full = scheduler_output.num_spec_tokens_to_schedule
        k = k_for(len(scheduler_output.num_scheduled_tokens), full)
        if k != full:
            scheduler_output.num_spec_tokens_to_schedule = k
        return orig(self, scheduler_output)

    _update_after_schedule._cf_batch_cutoff = True            # type: ignore[attr-defined]
    AsyncScheduler._update_after_schedule = _update_after_schedule  # type: ignore[assignment]
    INSTALLED = True
    REASON = ""
    m = _mode()
    asked = (os.environ.get(_FLAG, "") or "").strip().lower() or f"unset -> {m}"
    print(f"[cf-plugin] batch cutoff INSTALLED: {_FLAG}={asked} -- a scheduler step with more "
          f"than N requests schedules NO speculative tokens for the next step, and the drafter "
          f"does not run."
          + (f" N={max_batch()}." if m == "fixed" else
             " N is resolved from the target's size and the arm's verify width when the drafter "
             "is built, and a combination with no measured ladder resolves to NO CUTOFF."
             if m == "default" else
             " N is resolved from the target's size when the drafter is built."), flush=True)


def status() -> str:
    if not INSTALLED:
        return f"not installed ({REASON})"
    lad = ladder()
    if lad:
        return f"installed ({describe_ladder()})"
    n = max_batch()
    return f"installed (N={n})" if n else "installed (N not yet resolved: auto, drafter not built)"
