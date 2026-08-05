"""What `CF_SPEC_MAX_BATCH` promises, pinned where it is cheap to check.

The cutoff's whole job is to be a *no-op below N* and *complete above N*, and both halves of it
(the scheduler patch, the proposer skip in `flow_proposer`) fail SILENTLY rather than loudly: a
cutoff that fires one request early gives up speedup nobody asked to give up, and a cutoff whose
scheduler half is missing looks exactly like "the cutoff does not work" -- the drafter stops but
the target keeps verifying a step's worth of zeros.

The boundary tests need no vLLM.  The patch test does, and skips without it.
"""
import pytest

from chained_flow.vllm_plugin import batch_cutoff


@pytest.mark.parametrize("n,nreq,cut", [
    (0, 1, False), (0, 999, False),          # unset: never
    (8, 1, False), (8, 7, False),
    (8, 8, False),                           # N is INCLUSIVE: exactly N still speculates
    (8, 9, True), (8, 64, True),
    (1, 1, False), (1, 2, True),             # batch-1-only speculation is expressible
])
def test_boundary(monkeypatch, n, nreq, cut):
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", str(n))
    assert batch_cutoff.should_cut(nreq) is cut


def test_bad_value_does_not_take_the_engine_down(monkeypatch):
    """This is read in every vLLM process that merely has chained-flow installed."""
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "not-a-number")
    assert batch_cutoff.max_batch() == 0
    assert batch_cutoff.should_cut(1000) is False


@pytest.mark.parametrize("hidden,width,n,what", [
    (2560, 6, 4, "4B chain"),
    (5120, 6, 16, "27B chain"),
    (2560, 41, 2, "4B tree"),
])
def test_auto_returns_the_measured_threshold_where_one_was_measured(hidden, width, n, what):
    """The threshold is a measurement, not a constant, and both ways of being wrong cost real
    throughput: N=4 on 27B measured 190.4 tok/s at concurrency 8 against the uncut arm's 253.5,
    and N=16 on 4B leaves it under water from concurrency 8 up."""
    got, why = batch_cutoff.auto_for(hidden, width)
    assert got == n, f"{what}: {why}"
    assert "measured" in why and "DERIVED" not in why


def test_verify_width_moves_the_threshold_as_much_as_model_size_does():
    """The scheduler hands the target `K+1` positions per request, so a 42-wide tree saturates it
    seven times sooner than a 6-wide chain. A table keyed on model size alone would put the 4B
    tree threshold at 4, where the tree measured 0.85x."""
    assert batch_cutoff.auto_for(2560, 41)[0] < batch_cutoff.auto_for(2560, 6)[0]
    assert batch_cutoff.auto_for(5120, 41)[0] < batch_cutoff.auto_for(5120, 6)[0]


def test_a_combination_never_laddered_is_derived_floored_and_says_so():
    """A guess must be labelled as one, and must not be able to guess ABOVE 'batch 1 only' into
    a regime nobody measured."""
    n, why = batch_cutoff.auto_for(3584, 41)                 # no such target here: a pure guess
    assert n >= 1 and "DERIVED" in why


def test_auto_is_inert_until_the_drafter_resolves_it(monkeypatch):
    """`install()` runs from the vllm.general_plugins entry point, long before any model is
    loaded, so it CANNOT key off the resolved number -- it must key off `requested()`. Keying it
    off `max_batch()` would silently install no scheduler patch at all under `auto`."""
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "auto")
    monkeypatch.setattr(batch_cutoff, "_RESOLVED", 0)
    assert batch_cutoff.requested() is True
    assert batch_cutoff.max_batch() == 0
    assert batch_cutoff.should_cut(9999) is False        # inert, not accidentally cutting
    batch_cutoff.set_resolved(*batch_cutoff.auto_for(2560, 6))
    assert batch_cutoff.max_batch() == 4
    assert batch_cutoff.should_cut(4) is False and batch_cutoff.should_cut(5) is True


@pytest.mark.parametrize("val,mode", [
    (None, "default"), ("", "default"),
    ("auto", "auto"), ("AUTO", "auto"),
    ("0", "off"), ("off", "off"), ("none", "off"), ("-3", "off"),
    ("not-a-number", "off"),                 # a typo must not confer a behaviour
    ("8", "fixed"),
])
def test_mode(monkeypatch, val, mode):
    if val is None:
        monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    else:
        monkeypatch.setenv("CF_SPEC_MAX_BATCH", val)
    assert batch_cutoff._mode() == mode


def test_zero_still_means_off_after_the_default_flipped(monkeypatch):
    """`CF_SPEC_MAX_BATCH=0` was the way to say "no cutoff" while the flag shipped OFF. Anything
    that has it pinned must keep reading off rather than silently acquire a threshold."""
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "0")
    monkeypatch.setattr(batch_cutoff, "_RESOLVED", 16)
    assert batch_cutoff.requested() is False
    assert batch_cutoff.max_batch() == 0
    assert batch_cutoff.should_cut(9999) is False


@pytest.mark.parametrize("hidden,width", [
    (2560, 6), (2560, 41), (4096, 6), (4096, 41), (5120, 6), (5120, 41),
])
def test_the_default_engages_exactly_the_laddered_combinations(monkeypatch, hidden, width):
    """DEFAULT ON is only defensible because it cannot guess. Every combination this project
    publishes a number for is laddered and resolves to its measured N; the assertion that keeps
    the two in step is that `resolve()` under the default returns a MEASURED threshold or none."""
    monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    n, why = batch_cutoff.resolve(hidden, width)
    assert n > 0 and "measured" in why, why
    assert batch_cutoff.auto_for(hidden, width)[0] == n      # explicit `auto` agrees


def test_the_default_refuses_to_guess_but_auto_still_guesses(monkeypatch):
    """The asymmetry is the whole design: `auto` typed by a human asked for a best guess, the
    default did not. A derived N that is too low silently costs speedup -- measured: the derived
    N=2 for 27B tree was still leaving 1.55x on the table at concurrency 2."""
    unladdered = (3584, 6)                                   # no such target here
    monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    n, why = batch_cutoff.resolve(*unladdered)
    assert n == 0 and "NOT MEASURED" in why
    batch_cutoff.set_resolved(n, why)
    assert batch_cutoff.max_batch() == 0
    assert batch_cutoff.should_cut(9999) is False            # speculation stays on everywhere

    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "auto")
    n_auto, why_auto = batch_cutoff.resolve(*unladdered)
    assert n_auto >= 1 and "DERIVED" in why_auto


def test_no_measured_threshold_can_touch_batch_1(monkeypatch):
    """The headline metric this project is judged on is batch 1, and the default now engages the
    cutoff without anyone asking. Every entry must therefore be >= 1, so that `should_cut(1)` is
    False by the inclusive-N rule -- not as a matter of the numbers happening to be large, but
    checked against the table as it stands."""
    monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    for (hidden, width), (n, why) in batch_cutoff._AUTO.items():
        assert n >= 1, f"({hidden}, {width}) -> {n}: would disable speculation at batch 1"
        monkeypatch.setattr(batch_cutoff, "_RESOLVED", n)
        assert batch_cutoff.should_cut(1) is False, f"({hidden}, {width}) cuts at batch 1"


def test_off_installs_nothing(monkeypatch):
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "off")
    monkeypatch.setattr(batch_cutoff, "INSTALLED", False)
    batch_cutoff.install()
    assert not batch_cutoff.INSTALLED, batch_cutoff.status()


class _FakeOutput:
    def __init__(self, nreq, k):
        self.num_scheduled_tokens = {f"r{i}": 1 for i in range(nreq)}
        self.num_spec_tokens_to_schedule = k


def test_patch_zeros_the_field_the_base_impl_reads(monkeypatch):
    """The patch works by mutating `num_spec_tokens_to_schedule` BEFORE calling super, because
    that is the single field `AsyncScheduler._update_after_schedule` turns into the per-request
    `[-1] * K` placeholder list -- and under async scheduling those placeholders, not our
    returned draft tokens, are what the next `schedule()` allocates spec slots from.  If upstream
    stops deriving them there, the cutoff silently becomes a drafter-only skip."""
    async_sched = pytest.importorskip("vllm.v1.core.sched.async_scheduler")
    import inspect
    assert "num_spec_tokens_to_schedule" in inspect.getsource(
        async_sched.AsyncScheduler._update_after_schedule), (
        "AsyncScheduler no longer derives its spec-token placeholders from "
        "scheduler_output.num_spec_tokens_to_schedule -- the batch cutoff is now half a cutoff")

    seen = []
    monkeypatch.setattr(async_sched.AsyncScheduler, "_update_after_schedule",
                        lambda self, out: seen.append(out.num_spec_tokens_to_schedule))
    monkeypatch.setattr(batch_cutoff, "INSTALLED", False)
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "8")
    batch_cutoff.install()
    fn = async_sched.AsyncScheduler._update_after_schedule

    fn(object(), _FakeOutput(8, 5))
    fn(object(), _FakeOutput(9, 5))
    assert seen == [5, 0]

    batch_cutoff.install()                                   # idempotent
    assert async_sched.AsyncScheduler._update_after_schedule is fn


# ======================================================================================
# THE K-SCHEDULE.  A cliff to K=0 is one rung of a ladder; these pin the rest of it.
# ======================================================================================

@pytest.fixture(autouse=True)
def _no_stray_ladder(monkeypatch):
    """Every test starts with NO K-schedule.

    `_RESOLVED_LADDER` is a module global set by `_build()`, so a test that resolves one would
    otherwise change what `should_cut` means for every test after it -- and `should_cut` is what
    the cliff tests above are asserting."""
    monkeypatch.delenv("CF_SPEC_K_SCHEDULE", raising=False)
    monkeypatch.setattr(batch_cutoff, "_RESOLVED_LADDER", ())


def test_a_k_schedule_is_off_unless_asked_for(monkeypatch):
    """`_AUTO_K1` is deliberately empty: a K=1 rung set too high is a slowdown nobody opted into,
    the same asymmetry that keeps `_AUTO` measured-only. Until a ladder exists the shipping
    behaviour is the plain cliff, at every target this project publishes a number for."""
    assert batch_cutoff._AUTO_K1 == {}
    for hidden, width in batch_cutoff._AUTO:
        rungs, why = batch_cutoff.auto_ladder(hidden, width)
        assert rungs == (), why
        assert "MEASURED" in why


@pytest.mark.parametrize("spec,nreq,k", [
    ("4:full,16:1", 1, 5), ("4:full,16:1", 4, 5),
    ("4:full,16:1", 5, 1), ("4:full,16:1", 16, 1),
    ("4:full,16:1", 17, 0), ("4:full,16:1", 64, 0),
    ("64:1", 1, 1), ("64:1", 64, 1), ("64:1", 65, 0),
    ("4:full,16:2,32:1", 8, 2), ("4:full,16:2,32:1", 20, 1), ("4:full,16:2,32:1", 33, 0),
])
def test_k_schedule_rungs(monkeypatch, spec, nreq, k):
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", spec)
    assert batch_cutoff.k_for(nreq, 5) == k


def test_a_k_rung_is_not_a_cut_and_the_drafter_must_still_run(monkeypatch):
    """`should_cut` is what `FlowDrafterProposer._cut` skips the draft on. A K=1 step needs a
    draft -- vLLM scatters the FIRST COLUMN of the returned tensor -- so only the implicit rung
    above the last one may read as a cut. Getting this wrong scatters a stale draft, silently."""
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    assert [batch_cutoff.should_cut(b) for b in (1, 4, 5, 16, 17)] == \
           [False, False, False, False, True]


def test_the_patch_is_inert_on_a_step_it_has_no_opinion_about(monkeypatch):
    """The wrapper passes the engine's OWN `num_spec_tokens_to_schedule` in as `full` and writes
    it back unchanged below the first rung, so 'no cutoff configured' cannot become 'a different
    K' by the patch matching a constant that later changes."""
    monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    monkeypatch.setattr(batch_cutoff, "_RESOLVED", 0)
    for full in (0, 1, 5, 41):
        assert batch_cutoff.k_for(1, full) == full
        assert batch_cutoff.k_for(9999, full) == full        # no cutoff resolved -> untouched


def test_a_rung_can_never_exceed_the_engines_own_k(monkeypatch):
    """The draft tensor is `draft_width` columns wide and `_prepare_input_ids` reads
    `range(start, start + draft_len)` out of it. A schedule asking for more spec tokens than the
    engine was configured with would index past the row and scatter the NEXT request's draft."""
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "64:3")
    assert batch_cutoff.k_for(8, 1) == 1
    assert batch_cutoff.k_for(8, 5) == 3


@pytest.mark.parametrize("bad", [
    "4:0",                       # K=0 is the IMPLICIT rung above the last one
    "4:full,2:1",                # does not ascend in batch
    "4:1,16:full",               # raises K as the batch grows
    "4:1,16:3",                  # ditto, numerically
    "4", "4:", "0:1", "-1:full",
])
def test_a_typo_in_the_schedule_raises_rather_than_becoming_a_different_schedule(bad):
    """Its only other symptom would be a throughput number, which is how the last two wrong
    thresholds in this file were shipped."""
    with pytest.raises(ValueError):
        batch_cutoff.parse_schedule(bad)


def test_an_explicit_schedule_supersedes_the_cliff_flag(monkeypatch):
    """The schedule's last rung IS the cutoff; obeying `CF_SPEC_MAX_BATCH` as well would mean two
    thresholds for one boundary."""
    monkeypatch.setenv("CF_SPEC_MAX_BATCH", "0")
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    assert batch_cutoff.requested() is True
    assert batch_cutoff.k_for(20, 5) == 0 and batch_cutoff.k_for(10, 5) == 1


def test_the_resolution_log_states_value_and_provenance(monkeypatch, capsys):
    """`_AUTO` entries are measured and `_AUTO_K1` is empty, so the line has to say WHICH -- a
    derived threshold that reads like a measured one is how a guess gets published."""
    monkeypatch.delenv("CF_SPEC_MAX_BATCH", raising=False)
    batch_cutoff.set_resolved(*batch_cutoff.resolve(2560, 6))
    out = capsys.readouterr().out
    assert "4" in out and "measured" in out
    batch_cutoff.set_resolved_ladder(*batch_cutoff.auto_ladder(2560, 6))
    assert "MEASURED" in batch_cutoff.describe_ladder()

    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    d = batch_cutoff.describe_ladder()
    assert "B<=4 -> K=full" in d and "B<=16 -> K=1" in d and "CF_SPEC_K_SCHEDULE" in d


def test_the_scheduler_patch_carries_a_k_ladder_not_only_a_zero(monkeypatch):
    """The whole reason the ladder can exist at all: the field the cutoff already writes is the
    same field a K-schedule needs, so K=1 costs no second code path in vLLM."""
    async_sched = pytest.importorskip("vllm.v1.core.sched.async_scheduler")
    seen = []
    monkeypatch.setattr(async_sched.AsyncScheduler, "_update_after_schedule",
                        lambda self, out: seen.append(out.num_spec_tokens_to_schedule))
    monkeypatch.setattr(batch_cutoff, "INSTALLED", False)
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    batch_cutoff.install()
    fn = async_sched.AsyncScheduler._update_after_schedule
    for nreq in (4, 5, 16, 17):
        fn(object(), _FakeOutput(nreq, 5))
    assert seen == [5, 1, 1, 0]
