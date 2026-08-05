"""The fork-free enabler: a ``vllm.general_plugins`` entry point that relaxes ONE guard.

WHY THIS EXISTS
---------------
``vllm/config/vllm.py`` *auto-decides* ``scheduler_config.async_scheduling`` and gives the BASE
arm a feature a ``custom_class`` speculative proposer is structurally denied::

    elif (self.speculative_config is not None
          and self.speculative_config.method not in get_args(EagleModelTypes)
          and self.speculative_config.method not in get_args(NgramGPUTypes)   # <-- :1007
          and self.speculative_config.method != "dspark"):
        logger.warning_once("Async scheduling not supported with %s-based "
                            "speculative decoding and will be disabled.", ...)
        self.scheduler_config.async_scheduling = False

Async scheduling is worth **+10.5% / +5.5% / +1.6%** at 4B / 9B / 27B (docs/BENCHMARKING.md), and
without it a fork-free chain build is barely above break-even.  Our proposer IS async-safe --
``FlowDrafterProposer`` is GPU-token-native under ``CF_ASYNC_SPEC`` -- so the guard is wrong *for
us* and only for us.

WHAT IT DOES, AND WHY IT IS A REBIND AND NOT A POST-HOC ASSIGNMENT
-----------------------------------------------------------------
``NgramGPUTypes`` is referenced at EXACTLY TWO places in ``vllm.config.vllm`` (verified on
0.25.1: lines 974 and 1007), and both are the async guard -- one for the explicit-request path
and one for the auto-decide path.  Rebinding the name *in that module only* therefore relaxes
precisely that guard and nothing else; ``vllm.config.speculative`` keeps its own definition, so
no other consumer of the type sees the change.

The tempting alternative -- letting vLLM decide and then setting
``vllm_config.scheduler_config.async_scheduling = True`` afterwards -- is WRONG.  Everything
downstream of :1040 in the same function reads that field to make further decisions:

  * ``:1047-1058`` ``disable_nccl_for_dp_synchronization``
  * ``:1060-1070`` **disable_cascade_attn** -- "not yet compatible with async speculative
    decoding".  A post-hoc write leaves cascade attention ENABLED under async spec decode.

A post-hoc write produces an engine that says async and behaves like neither.  So: relax the
INPUT to the decision, then let vLLM make it, then CHECK the answer.

LOUDNESS
--------
This codebase has been bitten twice by silent fallbacks, and a silent fall back to synchronous
scheduling here is a ~10% regression that announces itself nowhere.  So:

  * the vLLM version is pinned (``0.25.*``) and checked;
  * the symbol must EXIST in ``vllm.config.vllm`` before we touch it (a rename upstream would
    otherwise turn this into a no-op that still "succeeds");
  * both call sites are located by source inspection and counted;
  * ``VllmConfig.__post_init__`` is wrapped so that the RESOLVED value is asserted: if we
    relaxed the guard on an auto-decide config and the engine still came out synchronous, that
    is a hard ``RuntimeError``, not a log line.

Anything that fails raises out of the plugin FUNCTION (vLLM's ``load_general_plugins`` swallows
exceptions from ``entry_point.load()`` but not from calling the loaded callable, which is why
this module's import is stdlib-only and every check lives inside ``register()``).

SCOPE: ``CF_ASYNC_SPEC`` IS THE SWITCH
-------------------------------------
The guard is relaxed **only when ``CF_ASYNC_SPEC`` is on**, never blanket.  Async scheduling
makes ``valid_sampled_token_ids`` an empty list and fills ``token_ids_cpu`` with ``-1``
placeholders; a proposer that has not gone GPU-token-native reads zero rows every step and
silently drafts nothing (accept collapses to 1.0).  Tying the two together makes the mixed state
unreachable rather than merely unlikely.
"""
from __future__ import annotations

import os

#: Set once ``register()`` has actually rebound the symbol.  Read by
#: ``chained_flow.defaults.finalize_async`` to tell "stock vLLM, guard relaxed" apart from
#: "forked vLLM, guard relaxed by the fork" apart from "not relaxed at all".
RELAXED: bool = False
#: Why the guard was not relaxed, when it was not.  Reaches the startup summary line.
REASON: str = "register() not called (the vllm.general_plugins entry point never fired)"

_ENV = "CF_ASYNC_SPEC"
_DISABLE = "CF_NO_ASYNC_PLUGIN"      # escape hatch: never touch vLLM at all
_SUPPORTED = "0.25."
#: The name we rebind and the number of times it is referenced in ``vllm/config/vllm.py`` on a
#: supported vLLM.  Both references are async-guard sites; a different count means the module
#: changed underneath us and the "relaxes exactly that guard" argument no longer holds.
_SYMBOL = "NgramGPUTypes"
_EXPECTED_SITES = 2


def _truthy(v) -> bool:
    return v is not None and str(v) not in ("0", "", "false", "False", "FALSE", "no", "off", "OFF")


def _wanted() -> bool:
    """Is the async spec path requested?

    ``chained_flow.defaults`` owns the value; importing it is cheap (stdlib only, no torch) and
    idempotent, and it is what makes ``CF_ASYNC_SPEC`` default ON without the caller having to
    know the flag exists.
    """
    try:
        from chained_flow import defaults

        defaults.apply()
    except Exception:                                       # noqa: BLE001 - defaults are advisory
        pass
    return _truthy(os.environ.get(_ENV))


def register() -> None:
    """The ``vllm.general_plugins`` entry point.  Called by ``load_general_plugins()`` from
    ``EngineArgs.__post_init__``, i.e. BEFORE ``create_engine_config()`` builds the
    ``VllmConfig`` whose ``__post_init__`` runs the guard.

    IT MUST NOT RAISE HERE, and that is not squeamishness.  vLLM runs every installed general
    plugin in every engine process, so this function also executes for a plain ``vllm serve``
    that merely has chained-flow on disk and no intention of using it.  Taking that engine down
    because OUR guard could not be relaxed would be a packaging bug of the worst kind.

    So a failure is *deferred*, not swallowed: the reason is recorded, printed once, and a check
    is installed on ``VllmConfig.__post_init__`` that raises only for a config that is actually
    ours.  Loud where it matters, invisible where it does not.
    """
    global RELAXED, REASON

    if _truthy(os.environ.get(_DISABLE)):
        REASON = f"{_DISABLE} is set"
        return
    if not _wanted():
        REASON = f"{_ENV}=0 (async spec path not requested)"
        return

    import vllm
    import vllm.config.vllm as cfg

    _install_resolution_check(cfg)

    version = getattr(vllm, "__version__", "?")
    if not str(version).startswith(_SUPPORTED):
        REASON = (f"vLLM {version} is not the verified {_SUPPORTED}x. The guard relaxed here "
                  f"({_SYMBOL} in vllm.config.vllm) is a private detail upstream may move or "
                  f"rename, and a rebind that quietly stopped working would cost ~10% at 4B "
                  f"with nothing in the log to say so")
        print(f"[cf-plugin] NOT relaxing vLLM's async-scheduling guard: {REASON}. A "
              f"chained-flow speculative run on this vLLM will REFUSE to start; anything else "
              f"is unaffected.", flush=True)
        return

    if not hasattr(cfg, _SYMBOL):
        REASON = (f"vllm.config.vllm has no `{_SYMBOL}` -- the async-scheduling guard this "
                  f"plugin relaxes has moved")
        print(f"[cf-plugin] NOT relaxing vLLM's async-scheduling guard: {REASON}.", flush=True)
        return

    _check_sites(cfg)

    from typing import Literal

    # The rebind.  MODULE-SCOPED: `vllm.config.speculative.NgramGPUTypes` is untouched, so the
    # only behaviour that changes is the two `get_args(NgramGPUTypes)` membership tests in this
    # module -- both of them the async guard.  "ngram_gpu" is preserved: this ADDS us to the
    # allow-list, it does not replace it.
    cfg.NgramGPUTypes = Literal["ngram_gpu", "custom_class"]

    RELAXED = True
    REASON = ""
    print("[cf-plugin] vLLM async-scheduling guard relaxed for method='custom_class' "
          f"(vllm {version}); the engine's resolved async_scheduling will be asserted after "
          "config creation.", flush=True)


def _check_sites(cfg) -> None:
    """Count the references to the symbol in the module SOURCE.

    The safety argument for this plugin is "the name is used only by the async guard".  That is
    a property of a specific vLLM source file, so it is checked against the file rather than
    assumed.  A mismatch is a warning, not a raise: the version pin is the hard gate, and a
    harmless upstream edit (a comment mentioning the name) must not stop an engine from
    starting.  It is printed because a change here is the signal to re-read config/vllm.py.
    """
    try:
        import inspect

        src = inspect.getsource(cfg)
    except Exception as e:                                  # noqa: BLE001 - source may be absent
        print(f"[cf-plugin] could not read vllm/config/vllm.py to verify the guard sites "
              f"({e!r}); relying on the version pin alone.", flush=True)
        return
    n = src.count(f"get_args({_SYMBOL})")
    if n != _EXPECTED_SITES:
        print(f"[cf-plugin] WARNING: expected {_EXPECTED_SITES} `get_args({_SYMBOL})` sites in "
              f"vllm/config/vllm.py (both async-scheduling guards) and found {n}. The rebind "
              f"may now affect something else -- re-read that file before trusting this run.",
              flush=True)


def _install_resolution_check(cfg) -> None:
    """Check the RESOLVED ``async_scheduling`` after vLLM has decided it -- and raise.

    This is where ``register()``'s deferred failures land, and it is deliberately narrow.  It
    fires only when

      * the speculative config is OURS (``method="custom_class"`` and a chained-flow proposer
        class), so an unrelated engine sharing the machine is never affected, and
      * the caller left ``async_scheduling`` on AUTO (``None``) -- an explicit ``False`` is a
        deliberate request for the synchronous engine (the like-for-like baseline in
        ``vllm/bench_cf.sh``) and must be honoured in silence.

    Under those conditions the relaxed guard is the only thing that was stopping the auto-decide
    from returning True, so a non-True answer means either we never relaxed it (see ``REASON``)
    or something ELSE in ``config/vllm.py:992-1040`` turned it off.  Either way the run would be
    a ~10% regression wearing the shipping config's name, and the proposer's GPU-token-native
    path is expecting the async engine.  Raise.
    """
    if getattr(cfg.VllmConfig, "_cf_async_checked", False):
        return
    orig = cfg.VllmConfig.__post_init__

    def __post_init__(self, *a, **k):
        sched = getattr(self, "scheduler_config", None)
        requested = getattr(sched, "async_scheduling", None) if sched is not None else None
        out = orig(self, *a, **k)
        if _is_ours(getattr(self, "speculative_config", None)) and requested is None:
            resolved = getattr(self.scheduler_config, "async_scheduling", None)
            if resolved is not True:
                raise RuntimeError(
                    "chained-flow needs vLLM's async scheduling for this speculative config "
                    f"({_ENV}=1) and the engine resolved async_scheduling={resolved!r}.\n"
                    + (f"  The guard was never relaxed: {REASON}.\n" if not RELAXED else
                       "  The guard WAS relaxed, so something else in "
                       "vllm/config/vllm.py:992-1040 disabled it -- a pooling runner, "
                       "disable_padded_drafter_batch, an executor backend without async "
                       "support, or ROCm DeepEP DBO. The vLLM log line just above says "
                       "which.\n")
                    + "  Running anyway would silently cost ~10% at 4B and hand a GPU draft "
                      "tensor to a synchronous _prepare_inputs, which ignores it. "
                      f"Set {_ENV}=0 to run the synchronous path on purpose.")
        return out

    cfg.VllmConfig.__post_init__ = __post_init__
    cfg.VllmConfig._cf_async_checked = True


def _is_ours(spec) -> bool:
    """A `custom_class` proposer that is a chained-flow one.

    ``method == "custom_class"`` alone is not enough: it is vLLM's generic hook and somebody
    else's proposer must not inherit our fatal check.  The proposer CLASS PATH is the
    discriminator, and it is the same string the user passes as ``speculative_config["model"]``.
    """
    if spec is None or getattr(spec, "method", None) != "custom_class":
        return False
    for attr in ("model", "draft_model", "custom_class"):
        v = getattr(spec, attr, None)
        if isinstance(v, str) and "chained_flow" in v:
            return True
    return False


def status() -> str:
    """One phrase for the startup summary."""
    return "relaxed" if RELAXED else f"NOT relaxed: {REASON}"
