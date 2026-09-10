"""CF_VPROF=1: cudagraph-safe HOST wall-clock profiler for the whole vLLM spec step.

Wraps a handful of vLLM entry points with `time.perf_counter` only — no
`torch.cuda.synchronize`, no D2H — so it can stay enabled during cudagraph capture
and in a shipping-config run.  Reported time is HOST time: a function that blocks
waiting on the GPU (any D2H of a not-yet-ready tensor) shows up as a large host
segment, which is exactly the serialization we are hunting.

INCLUSIVE is reported for nested wrappers; `self` subtracts the children we know about.
"""
from __future__ import annotations

import atexit
import collections
import time

_T: dict[str, float] = collections.defaultdict(float)
_N: dict[str, int] = collections.defaultdict(int)
_STACK: list[list] = []
_installed = False


import os

# Warmup steps to DISCARD.  The first steps carry vLLM's own cudagraph capture and our
# draft capture; a cumulative mean over them decays like 1/n and hides the steady state.
_SKIP = int(os.environ.get("CF_VPROF_SKIP", "60"))
#: CF_VPROF_EVERY=N: also dump the report every N steps, not only at exit.
#:
#: `report()` is registered with `atexit`, which is enough for the offline `LLM()` harness --
#: that process ends by returning from `main()`.  Under `vllm serve` the profiled process is the
#: ENGINE CORE, which is torn down by a signal, so atexit is the one thing that does not run and
#: a serve profile would produce no output at all.  Reporting on a step count instead of a timer
#: keeps it off the clock: a step that does not cross the boundary pays one integer compare.
_EVERY = int(os.environ.get("CF_VPROF_EVERY", "0"))
_STEP = [0]
_WALL = [0.0, 0.0]   # [t of first counted step, t of last counted step]


# --- CF_VPROF_CUDA=1: DEVICE-timeline span of a wrapped call -----------------
# Host timers cannot separate "the host is dispatching" from "the host is blocked
# on the GPU".  A pair of cuda events straddling the call measures how long the
# call OCCUPIES THE DEVICE TIMELINE, which includes the bubbles a PIECEWISE
# (eager-dispatched) forward leaves between its kernels -- exactly the thing a
# FULL cudagraph removes.  Readback is DEFERRED by _LAG steps so nothing ever
# blocks the step loop.
_CUDA = os.environ.get("CF_VPROF_CUDA", "0") == "1"
#: Which wrapped calls get the event pair.  The DRAFT is in the set because at a large decode
#: batch it is the segment in question: its host time is ~4x the target forward's, and host time
#: alone cannot say whether that is the drafter occupying the device or the drafter's launch
#: stream stalled behind a target forward that has not finished.  Only a device span separates
#: those two, and they call for opposite fixes.
_CUDA_LABELS = ("1.execute_model", "1b._model_forward", "2b.draft(propose)", "2.sample_tokens")
_LAG = 64
_PEND: dict[str, list] = collections.defaultdict(list)
_GT: dict[str, float] = collections.defaultdict(float)
_GN: dict[str, int] = collections.defaultdict(int)


def _drain(label, force=False):
    q = _PEND[label]
    while q and (force or len(q) > _LAG):
        s, e = q.pop(0)
        if not e.query():
            q.insert(0, (s, e))
            if not force:
                return
            e.synchronize()
        _GT[label] += s.elapsed_time(e)
        _GN[label] += 1


def _wrap(obj, name, label):
    fn = getattr(obj, name, None)
    if fn is None:
        return

    def w(*a, **k):
        if _CUDA and label in _CUDA_LABELS:
            import torch

            _drain(label)
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            if label == "1.execute_model":
                _STEP[0] += 1
                if _STEP[0] == _SKIP:
                    _T.clear(); _N.clear(); _GT.clear(); _GN.clear()
                    _WALL[0] = time.perf_counter()
                _WALL[1] = time.perf_counter()
                if _EVERY and _STEP[0] > _SKIP and _STEP[0] % _EVERY == 0:
                    report(force_drain=False)
            t0 = time.perf_counter()
            _STACK.append([label, 0.0])
            s.record()
            try:
                return fn(*a, **k)
            finally:
                e.record()
                _PEND[label].append((s, e))
                dt = time.perf_counter() - t0
                me = _STACK.pop()
                _T[label] += dt
                _T[label + " (self)"] += dt - me[1]
                _N[label] += 1
                if _STACK:
                    _STACK[-1][1] += dt
        return _w_host(fn, label, *a, **k)

    def _w_host(fn, label, *a, **k):
        if label == "1.execute_model":
            _STEP[0] += 1
            if _STEP[0] == _SKIP:
                _T.clear()
                _N.clear()
                _WALL[0] = time.perf_counter()
            _WALL[1] = time.perf_counter()
            if _EVERY and _STEP[0] > _SKIP and _STEP[0] % _EVERY == 0:
                report(force_drain=False)
        t0 = time.perf_counter()
        _STACK.append([label, 0.0])
        try:
            if label == "1.execute_model":
                import torch
                # a countable per-step marker in the chrome trace
                with torch.profiler.record_function("CF_STEP"):
                    return fn(*a, **k)
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            me = _STACK.pop()
            _T[label] += dt
            _T[label + " (self)"] += dt - me[1]
            _N[label] += 1
            if _STACK:
                _STACK[-1][1] += dt

    setattr(obj, name, w)


def install() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    from vllm.v1.spec_decode import tree_state
    from vllm.v1.core.sched.scheduler import Scheduler

    _wrap(GPUModelRunner, "execute_model", "1.execute_model")
    _wrap(GPUModelRunner, "_prepare_inputs", "1a._prepare_inputs")
    _wrap(GPUModelRunner, "_update_states", "1a0._update_states")
    _wrap(GPUModelRunner, "_prepare_input_ids", "1a2._prepare_input_ids")
    _wrap(GPUModelRunner, "_model_forward", "1b._model_forward")
    _wrap(GPUModelRunner, "sample_tokens", "2.sample_tokens")
    _wrap(GPUModelRunner, "_sample", "2a._sample")
    _wrap(GPUModelRunner, "_bookkeeping_sync", "2c._bookkeeping_sync")
    _wrap(GPUModelRunner, "propose_draft_token_ids", "2b.draft(propose)")
    _wrap(RejectionSampler, "forward", "2a1.rejsampler.forward")
    _wrap(RejectionSampler, "parse_output", "2a2.rejsampler.parse_output(D2H)")
    _wrap(tree_state, "build_attn_plumbing", "1a1.tree.build_attn_plumbing")
    _wrap(Scheduler, "schedule", "0.sched.schedule")
    _wrap(Scheduler, "update_from_output", "3.sched.update_from_output")
    atexit.register(report)


def install_cgmode(every: int = 500) -> None:
    """`CF_DBG_CG=1`: tally which cudagraph mode each step actually dispatches.

    The same tally `vllm/test_plugin_native.py` has had, moved somewhere the SERVE path can
    reach.  It answers a question no throughput number can: whether a wide decode batch still
    replays a captured FULL graph or has fallen to PIECEWISE.  A spec engine's
    `uniform_decode_query_len` is `1 + num_spec_tokens`, so the captured decode graph is keyed
    on `B x (K+1)` tokens -- and `cudagraph_capture_sizes` is a finite list, so there is a batch
    above which no key matches and every step is eager-dispatched.  Printed every `every`
    dispatches because the engine core is killed by a signal and atexit does not run there.
    """
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

    tally: dict = collections.Counter()
    n = [0]
    orig = CudagraphDispatcher.dispatch

    def dispatch(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            mode, desc = (r if isinstance(r, tuple) else (r, None))
            tally[f"{mode} ntok={getattr(desc, 'num_tokens', None)} "
                  f"uniform={getattr(desc, 'uniform', None)}"] += 1
        except Exception as e:                               # noqa: BLE001
            tally[f"ERR {e!r}"] += 1
        n[0] += 1
        if every and n[0] % every == 0:
            print(f"[cf-cgmode] dispatched modes (n={n[0]}): "
                  f"{dict(tally.most_common(10))}", flush=True)
        return r

    CudagraphDispatcher.dispatch = dispatch
    atexit.register(lambda: print(f"\n[cf-cgmode] dispatched modes: {dict(tally.most_common(10))}",
                                  flush=True))


_SYNCS: dict = collections.Counter()


def install_sync_debug() -> None:
    """CF_SYNCDBG=1: attribute every D2H / device sync to a source line.

    `torch.cuda.set_sync_debug_mode("warn")` raises a Python warning at each
    implicit synchronization; we capture the call site (skipping torch internals)
    and tally it.  This is the definitive list of host syncs on the step path.
    """
    import warnings, traceback, atexit as _a

    def showwarning(message, category, filename, lineno, file=None, line=None):
        st = traceback.extract_stack()[:-1]
        site = "?"
        for fr in reversed(st):
            if ("/torch/" not in fr.filename and "vprof.py" not in fr.filename
                    and "warnings.py" not in fr.filename):
                site = f"{fr.filename.split('/site-packages/')[-1].split('/chain-flow/')[-1]}:{fr.lineno} {fr.name}()"
                break
        _SYNCS[site] += 1

    warnings.showwarning = showwarning
    warnings.simplefilter("always")   # the default filter dedupes per (msg, lineno) -> undercounts
    import torch

    torch.cuda.set_sync_debug_mode("warn")

    def rep():
        n = max(_N.get("1.execute_model", 0), 1)
        print(f"\n[cf-sync] implicit device syncs per step (n={n}):")
        for k, v in _SYNCS.most_common(30):
            print(f"[cf-sync]   {v / n:6.2f}/step  ({v:6d})  {k}")
        print(f"[cf-sync]   TOTAL {sum(_SYNCS.values()) / n:.2f}/step")

    _a.register(rep)


def report(force_drain: bool = True) -> None:
    """`force_drain=False` for the mid-run `CF_VPROF_EVERY` dump.

    Forcing the drain calls `Event.synchronize()` on the events still in flight, which is
    exactly the host block this profiler exists to find -- harmless at exit, and a measurement
    artefact if it happens every N steps inside a live server.  The unforced drain leaves the
    last `_LAG` events pending; against a cumulative n in the thousands that is noise.
    """
    n = max(_N.get("1.execute_model", 0), 1)
    step = (_WALL[1] - _WALL[0]) / max(n - 1, 1) * 1000 if _WALL[0] else float("nan")
    print(f"\n[cf-vprof] HOST ms per execute_model (n={n}, first {_SKIP} steps discarded) "
          "— INCLUSIVE, (self) = minus tracked children")
    print(f"[cf-vprof]   >>> WALL ms per engine step: {step:.3f} <<<")
    if _CUDA:
        for k in sorted(_PEND):
            _drain(k, force=force_drain)
        for k in sorted(_GT):
            print(f"[cf-vprof]   GPU-span {k:28s} {_GT[k] / max(_GN[k], 1):7.3f} ms"
                  f"  (n={_GN[k]})")
    for k in sorted(_T):
        if k.endswith(" (self)"):
            continue
        print(f"[cf-vprof]   {k:36s} incl {_T[k] / n * 1000:7.3f}  self {_T[k + ' (self)'] / n * 1000:7.3f}"
              f"  (calls/step {_N[k] / n:.2f})")
