"""The fork-free enabler's guard rails.

`chained_flow.vllm_plugin.async_guard` rebinds a PRIVATE vLLM symbol, so its blast radius and
its failure modes are the things worth pinning down.  Every assertion here is about a way it
could go wrong QUIETLY -- a rebind that replaces instead of extends, a rebind that leaks into
another module, a version drift that turns the relaxation into a no-op, or an engine that comes
out synchronous anyway (~10% slower at 4B, and nothing in the log says so).

Skipped when vLLM is not importable, so the rest of the suite still runs on a CPU box.
"""
from typing import get_args

import pytest

vllm = pytest.importorskip("vllm", reason="the async guard is only meaningful with vLLM installed")
cfg = pytest.importorskip("vllm.config.vllm")

from chained_flow.vllm_plugin import async_guard  # noqa: E402


@pytest.fixture
def guard(monkeypatch):
    """A fresh, opted-in plugin against a restored `NgramGPUTypes`."""
    monkeypatch.setattr(cfg, "NgramGPUTypes", cfg.NgramGPUTypes)   # restored on teardown
    monkeypatch.setattr(async_guard, "RELAXED", False)
    monkeypatch.setenv("CF_ASYNC_SPEC", "1")
    monkeypatch.delenv("CF_NO_ASYNC_PLUGIN", raising=False)
    return async_guard


def test_relax_is_additive_and_module_scoped(guard):
    guard.register()
    assert guard.RELAXED, guard.status()
    got = get_args(cfg.NgramGPUTypes)
    assert "custom_class" in got, "the whole point"
    # ngram_gpu must survive: this ADDS us to the allow-list, it does not take it over.
    assert "ngram_gpu" in got
    # ...and only `vllm.config.vllm` may see it. Anything else that imports the type keeps the
    # upstream meaning.
    from vllm.config.speculative import NgramGPUTypes as upstream

    assert get_args(upstream) == ("ngram_gpu",)


def test_opting_out_does_not_touch_vllm(guard, monkeypatch):
    monkeypatch.setenv("CF_ASYNC_SPEC", "0")
    monkeypatch.setattr("chained_flow.defaults._APPLIED", True)   # don't re-default it to 1
    guard.register()
    assert not guard.RELAXED
    assert get_args(cfg.NgramGPUTypes) == ("ngram_gpu",)


def test_unverified_vllm_version_does_not_relax_and_does_not_raise(guard, monkeypatch):
    """A future vLLM must neither be silently rebound NOR take down an unrelated engine.

    `register()` runs in EVERY vLLM process that has this package installed, so raising here
    would break a plain `vllm serve` that never asked for us. The failure is deferred to the
    per-config check instead.
    """
    monkeypatch.setattr(vllm, "__version__", "0.26.0")
    guard.register()
    assert not guard.RELAXED
    assert "0.25" in guard.REASON
    assert get_args(cfg.NgramGPUTypes) == ("ngram_gpu",)


def test_missing_symbol_does_not_relax(guard, monkeypatch):
    monkeypatch.delattr(cfg, "NgramGPUTypes")
    guard.register()
    assert not guard.RELAXED
    assert "has moved" in guard.REASON


class _Sched:
    async_scheduling = None


def _fake_config(monkeypatch, requested, model="chained_flow.vllm_plugin.flow_proposer.X"):
    """A stand-in VllmConfig whose __post_init__ leaves async_scheduling exactly as given."""
    ran = {}

    class Spec:
        method = "custom_class"

    Spec.model = model

    class Fake:
        scheduler_config = _Sched()
        speculative_config = Spec()

        def __post_init__(self):
            ran["yes"] = True

    Fake.scheduler_config.async_scheduling = requested
    monkeypatch.setattr(cfg, "VllmConfig", Fake)
    monkeypatch.setattr(cfg.VllmConfig, "_cf_async_checked", False, raising=False)
    async_guard._install_resolution_check(cfg)
    return Fake, ran


def test_engine_that_stays_synchronous_is_fatal(guard, monkeypatch):
    """AUTO in, not-True out => the run is a 10% regression wearing the shipping config's name,
    and the proposer's GPU-token-native path is expecting an async engine."""
    Fake, ran = _fake_config(monkeypatch, requested=None)
    with pytest.raises(RuntimeError, match="resolved async_scheduling"):
        Fake().__post_init__()
    assert ran, "the wrapper must still have run the original __post_init__"


def test_explicitly_synchronous_is_honoured_silently(guard, monkeypatch):
    """`async_scheduling=False` is the deliberate like-for-like baseline. It must not raise."""
    Fake, ran = _fake_config(monkeypatch, requested=False)
    Fake().__post_init__()
    assert ran


def test_somebody_elses_custom_class_proposer_is_left_alone(guard, monkeypatch):
    """`custom_class` is vLLM's generic hook. Another project's proposer must not inherit our
    fatal check just because chained-flow happens to be installed."""
    Fake, ran = _fake_config(monkeypatch, requested=None, model="someone_else.Proposer")
    Fake().__post_init__()
    assert ran
