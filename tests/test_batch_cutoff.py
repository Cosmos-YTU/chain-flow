"""What `CF_SPEC_MAX_BATCH` promises, pinned where it is cheap to check.

The cutoff's whole job is to be a *no-op below N* and *complete above N*, and both halves of it
(the scheduler patch, the proposer skip in `flow_proposer`) fail SILENTLY rather than loudly: a
cutoff that fires one request early gives up speedup nobody asked to give up, and a cutoff whose
scheduler half is missing looks exactly like "the cutoff does not work" -- the drafter stops but
the target keeps verifying a step's worth of zeros.

The boundary tests need no vLLM.  The patch test does, and skips without it.
"""
import pytest

from chain_flow.vllm_plugin import batch_cutoff


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
    """This is read in every vLLM process that merely has chain-flow installed."""
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



# ======================================================================================
# THE SPEC-SLOT INVARIANT: a request is scheduled its FULL drafted width, or none.
#
# Both halves are checked here because both fail SILENTLY on a chain (a junk draft token is
# verified and rejected, costing only a query position) and both are WRONG TEXT on a
# CF_TREE_CONV_NARROW tree: the row misses the out-of-band tree registry, which matches on
# LENGTH, the step goes all-stale, and the GDN layer leaves the tree conv kernel for a sliding
# window this engine did not allocate.
# ======================================================================================
class _FakeRequest:
    def __init__(self, num_tokens, spec, computed, placeholders=0):
        self._n = num_tokens
        self.spec_token_ids = spec
        self.num_computed_tokens = computed
        self.num_output_placeholders = placeholders

    @property
    def num_tokens(self):
        return self._n

    @property
    def num_tokens_with_spec(self):
        return self._n + len(self.spec_token_ids)


def _guarded(monkeypatch, **overrides):
    """A FRESH stand-in `Scheduler` with the guard installed on it.

    Injected as `vllm.v1.core.sched.scheduler.Scheduler` so this runs with no vLLM present --
    these are the tests that say what the guard does to a request, and they should not be the
    ones that get skipped on a machine without a GPU.  A fresh class per call because
    `install_slot_guard` is idempotent by a marker on `__init__`, so a reused class would be
    patched once and then silently skipped.
    """
    import sys
    import types

    attrs = dict(
        num_spec_tokens=41,
        num_sampled_tokens_per_step=1,
        max_model_len=2048,
        max_num_scheduled_tokens=16384,
        scheduler_config=type("C", (), {"max_num_seqs": 64})(),
        # What stock vLLM leaves this as unless `num_speculative_tokens_per_batch_size` is set.
        __init__=lambda self, running=(): (setattr(self, "running", list(running)),
                                           setattr(self, "dynamic_sd_lookup", None))[0],
        schedule=lambda self, *a, **k: ("ORIGINAL RAN", a, k)[0],
    )
    attrs.update(overrides)
    cls = type("_FakeScheduler", (), attrs)

    mod = types.ModuleType("vllm.v1.core.sched.scheduler")
    mod.Scheduler = cls
    for name in ("vllm", "vllm.v1", "vllm.v1.core", "vllm.v1.core.sched"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.v1.core.sched.scheduler", mod)
    monkeypatch.setattr(batch_cutoff, "SLOT_GUARD_INSTALLED", False)
    monkeypatch.setattr(batch_cutoff, "_TRUNC_AVOIDED", [0])
    batch_cutoff.install_slot_guard()
    assert batch_cutoff.SLOT_GUARD_INSTALLED, batch_cutoff.SLOT_GUARD_REASON
    return cls


def test_full_width_survives_untouched(monkeypatch):
    """The guard must be a NO-OP in steady state, or it is just a slower way to not speculate."""
    spec = [-1] * 40
    reqs = [_FakeRequest(num_tokens=100, spec=spec, computed=100) for _ in range(4)]
    cls = _guarded(monkeypatch)
    assert cls(reqs).schedule(True) == "ORIGINAL RAN", "extra scheduler args pass through"
    assert all(len(r.spec_token_ids) == 40 for r in reqs)
    assert batch_cutoff._TRUNC_AVOIDED[0] == 0


def test_a_request_that_would_be_truncated_by_max_model_len_loses_its_spec(monkeypatch):
    """scheduler.py clamps `num_new_tokens` to `max_model_len - computed - 1` and then SHORTENS
    `spec_token_ids` to fit.  A short tree cannot match the registry, so it must not be
    scheduled at all -- the request decodes without speculation for its last ~K tokens."""
    spec = [-1] * 40
    ok = _FakeRequest(num_tokens=2000, spec=spec, computed=2000)      # 2040 <= 2047
    edge = _FakeRequest(num_tokens=2007, spec=spec, computed=2007)    # 2047 <= 2047, exact fit
    doomed = _FakeRequest(num_tokens=2008, spec=spec, computed=2008)  # 2048  > 2047
    _guarded(monkeypatch)([ok, edge, doomed]).schedule()
    assert len(ok.spec_token_ids) == 40
    assert len(edge.spec_token_ids) == 40, "the boundary must be inclusive, not off by one"
    assert doomed.spec_token_ids == []
    assert batch_cutoff._TRUNC_AVOIDED[0] == 1


def test_the_shared_placeholder_list_is_rebound_not_mutated(monkeypatch):
    """`AsyncScheduler` hands EVERY running request the same list object, so clearing one by
    mutation would silently unspeculate the whole batch."""
    shared = [-1] * 40
    doomed = _FakeRequest(num_tokens=2100, spec=shared, computed=2100)
    safe = _FakeRequest(num_tokens=10, spec=shared, computed=10)
    _guarded(monkeypatch)([doomed, safe]).schedule()
    assert doomed.spec_token_ids == []
    assert len(shared) == 40 and safe.spec_token_ids is shared


def test_the_token_budget_half_errs_towards_dropping_spec(monkeypatch):
    """The other clamp that truncates is `min(num_new_tokens, token_budget)`.  This walk cannot
    know which requests the real loop will skip, so it must OVER-count the budget: dropping spec
    from a request that would have kept it is a lost draft, keeping it on one the real loop
    truncates is wrong text."""
    spec = [-1] * 40
    reqs = [_FakeRequest(num_tokens=100, spec=spec, computed=100) for _ in range(8)]
    # Room for three full-width rows, not eight.
    _guarded(monkeypatch, max_num_scheduled_tokens=41 * 3)(reqs).schedule()
    kept = [r for r in reqs if r.spec_token_ids]
    assert 0 < len(kept) < 8
    assert all(len(r.spec_token_ids) == 40 for r in kept), "kept rows keep the FULL width"


def test_a_non_speculative_engine_is_untouched(monkeypatch):
    """This runs in every vLLM process that merely has chain-flow installed."""
    spec = [-1] * 40
    r = _FakeRequest(num_tokens=9999, spec=spec, computed=9999)
    cls = _guarded(monkeypatch, num_spec_tokens=0)
    s = cls([r])
    s.schedule()
    assert r.spec_token_ids is spec
    assert s.dynamic_sd_lookup is None, "no table on an engine that never speculates"


def test_pad_spec_decode_is_disabled_by_an_identity_lookup(monkeypatch):
    """HALF 1.  `pad_spec_decode` is gated on `dynamic_sd_lookup is None`, so declaring the
    schedule dynamic is how vLLM's own guard is asked to stand down -- and the table is the
    IDENTITY so the only other reader (the seed of `num_spec_tokens_to_schedule`) is unchanged
    and the single place that decides K stays `_update_after_schedule`."""
    s = _guarded(monkeypatch)()
    assert s.dynamic_sd_lookup is not None, "the padding branch would still be live"
    assert set(s.dynamic_sd_lookup) == {41}, "the table must not change any scheduled K"
    assert len(s.dynamic_sd_lookup) == 65, "1-indexed by decode batch, up to max_num_seqs"


def test_an_engine_with_its_own_schedule_keeps_it(monkeypatch):
    """Only ever ADD a table where vLLM left None: a user who really did configure
    `num_speculative_tokens_per_batch_size` must not have it overwritten."""
    cls = _guarded(
        monkeypatch,
        __init__=lambda self, running=(): (setattr(self, "running", list(running)),
                                           setattr(self, "dynamic_sd_lookup", [0, 5, 5, 1, 1]))[0],
    )
    assert cls().dynamic_sd_lookup == [0, 5, 5, 1, 1]


def test_the_guard_can_be_turned_off_for_measurement(monkeypatch):
    monkeypatch.setenv("CF_SPEC_SLOT_GUARD", "0")
    assert batch_cutoff.slot_guard_on() is False
    monkeypatch.setenv("CF_SPEC_SLOT_GUARD", "1")
    assert batch_cutoff.slot_guard_on() is True
    monkeypatch.delenv("CF_SPEC_SLOT_GUARD")
    assert batch_cutoff.slot_guard_on() is True, "default ON"


def test_a_partial_k_rung_is_refused_on_a_tree(monkeypatch):
    """A tree's shape is matched by LENGTH, so scheduling a slice of it makes every row stale --
    the same failure the two scheduler holes produce, arrived at by configuration."""
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    monkeypatch.setenv("VLLM_SPEC_TREE", "1")
    with pytest.raises(ValueError, match="TREE engine"):
        batch_cutoff.ladder()
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full")
    assert batch_cutoff.ladder() == ((4, batch_cutoff.FULL),), "a pure cliff is fine on a tree"
    monkeypatch.setenv("VLLM_SPEC_TREE", "0")
    monkeypatch.setenv("CF_SPEC_K_SCHEDULE", "4:full,16:1")
    assert batch_cutoff.ladder() == ((4, batch_cutoff.FULL), (16, 1)), "a chain is unaffected"
