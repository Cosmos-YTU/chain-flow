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
    CUDA block runner takes `x.shape[0] == 1` only; the draft cudagraph has buckets up to 32 and
    none beyond), so exactly where the target's bill goes up, the draft's does too.
  * ABOVE BUCKET 32 EVERY DISTINCT BATCH SIZE IS ITS OWN `torch.compile`.  `CF_COMPILE` uses
    `dynamic=False`, and past the last bucket `_ctx_gpu` falls back to `bucket = B`.  The uncut
    ladder above logged draft batches of 34, 37, 38, 42, 45, 46, 50, 54, 55, 59, 62, 63 and 64 --
    a fresh max-autotune codegen each, ~70-80 s of engine stall apiece, inside live traffic.
    (This is the shape of the ~850 autotune blocks seen on a first `guidellm --rate 64` pass,
    whose 0.048x figure is the same failure with the compile storm on top of it.)

The first two are not bugs to be fixed -- they are what speculative decoding IS.  The bug is
that nothing stopped it.  The third is fixed here as a side effect, and by `CF_WARM_BUCKETS`
directly.

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

DEFAULT OFF.  Unlike the greedy guard -- which turns a crash into a 4xx and has no case where it
is worse -- this one has a real trade: it gives up speculative speedup at concurrency, and the
right N is a property of the deployment.  It ships off, with a measured N in the docs.
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
    # 4B tree, laddered at 1/2/3/4: 187.5 / 268.3 / 349.9 / 426.2 tok/s against a base of
    # 139.0 / 256.8 / ~378 (interpolated) / 498.8 -> 1.35x / 1.04x / 0.93x / 0.85x.  N=2 keeps
    # the point that is still above parity and drops the first one that is not.
    (2560, 41): (2, "measured: 4B tree (1.35x at batch 1, 1.04x at 2, 0.93x at 3)"),
}

#: The width rule, for a `(size, K+1)` that was never laddered: the crossing sits at a roughly
#: constant TOTAL verify width `B x (K+1)` for a given model size -- 4B chain crosses at 4 x 6 =
#: 24, 27B chain at 16 x 6 = 96 -- because that width is where the target forward stops being
#: memory-bound.  Interpolating on it is a guess, so it is floored at 1 (never worse than "batch
#: 1 only") and always reported as derived.
_WIDTH_AT_CROSSING = ((3072, 24, "4B"), (4608, 48, "9B, itself interpolated"), (1 << 30, 96, "27B"))


def auto_for(hidden_size: int, k_plus_1: int = 6) -> tuple[int, str]:
    """`(N, why)` for a target of this hidden size running an arm of this verify width."""
    hit = _AUTO.get((int(hidden_size), int(k_plus_1)))
    if hit:
        return hit
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


def set_resolved(n: int, why: str) -> None:
    global _RESOLVED
    _RESOLVED = max(int(n), 0)
    print(f"[cf-plugin] {_FLAG}=auto resolved to {_RESOLVED} ({why}). Speculation is off above a "
          f"decode batch of {_RESOLVED}; at or below it nothing changes.", flush=True)


def max_batch() -> int:
    """The configured cutoff, or 0 for "no cutoff".

    Read from the environment on every call rather than cached: this module is imported by the
    API-server process and the engine-core process independently, and the value has to be the
    same fact in both.  It is a couple of dict lookups on a path that already does far more.

    `auto` reads 0 until the proposer's lazy `_build()` resolves it, which is correct rather than
    merely tolerable: `_build()` runs inside the FIRST `propose()`, so the only steps that can
    precede it are the first few of the first request -- a decode batch of 1, which no threshold
    in the table would have cut anyway.
    """
    v = (os.environ.get(_FLAG, "") or "").strip().lower()
    if v == "auto":
        return _RESOLVED
    try:
        n = int(v or 0)
    except ValueError:
        return 0
    return n if n > 0 else 0


def requested() -> bool:
    """Was a cutoff ASKED FOR, whatever it resolves to?

    Distinct from `max_batch() > 0` and the difference is load-bearing: under `auto` the value is
    0 until `_build()` resolves it, and `install()` runs long before that.  Keying the scheduler
    patch on the resolved number would silently install nothing.
    """
    v = (os.environ.get(_FLAG, "") or "").strip().lower()
    return v == "auto" or max_batch() > 0


def half() -> str:
    return os.environ.get(_HALF, "both").strip().lower() or "both"


def drafter_half() -> bool:
    """Should `FlowDrafterProposer` skip the draft above the cutoff?"""
    return half() in ("both", "drafter")


def scheduler_half() -> bool:
    """Should the scheduler stop allocating spec slots above the cutoff?"""
    return half() in ("both", "scheduler")


def should_cut(nreq: int) -> bool:
    """Does a step of `nreq` requests fall ABOVE the cutoff?

    N is inclusive -- a batch of exactly N still speculates.  Both halves of the cutoff call
    this (the scheduler patch below, `FlowDrafterProposer._cut` in the proposer), because the two
    disagreeing by one would be invisible in any throughput number.
    """
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
        REASON = f"{_FLAG} unset (no cutoff; speculation stays on at every batch)"
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
        # read makes the placeholder list empty for every request, which is bit-identical to
        # what the native mechanism produces -- no second code path.
        if should_cut(len(scheduler_output.num_scheduled_tokens)):
            scheduler_output.num_spec_tokens_to_schedule = 0
        return orig(self, scheduler_output)

    _update_after_schedule._cf_batch_cutoff = True            # type: ignore[attr-defined]
    AsyncScheduler._update_after_schedule = _update_after_schedule  # type: ignore[assignment]
    INSTALLED = True
    REASON = ""
    asked = (os.environ.get(_FLAG, "") or "").strip().lower()
    print(f"[cf-plugin] batch cutoff INSTALLED: {_FLAG}={asked} -- a scheduler step with more "
          f"than N requests schedules NO speculative tokens for the next step, and the drafter "
          f"does not run."
          + (" N is resolved from the target's size when the drafter is built."
             if asked == "auto" else f" N={max_batch()}."), flush=True)


def status() -> str:
    if not INSTALLED:
        return f"not installed ({REASON})"
    n = max_batch()
    return f"installed (N={n})" if n else "installed (N not yet resolved: auto, drafter not built)"
