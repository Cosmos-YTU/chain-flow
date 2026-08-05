"""One mode per process (CF_MODE=base|spec) — avoids cudagraph-pool corruption from two LLMs."""
import os, sys, time, json
# Only fall back to the source tree when chained-flow is NOT installed. Prepending it
# unconditionally would shadow an installed package and quietly invalidate the one thing a
# pristine-venv run is meant to prove -- that the WHEEL works, `vllm.general_plugins` entry
# point and all.
import importlib.util
if importlib.util.find_spec("chained_flow") is None:
    sys.path.insert(0, "/home/shadeform/chained-flow/src")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

if os.environ.get("CF_DBG_CG") == "1":      # tally which cudagraph mode each step dispatches
    from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
    import collections, atexit
    _tally = collections.Counter()
    _orig_dispatch = CudagraphDispatcher.dispatch
    def _dispatch(self, *a, **k):
        r = _orig_dispatch(self, *a, **k)
        try:
            mode, desc = (r if isinstance(r, tuple) else (r, None))
            _tally[f"{mode} ntok={getattr(desc,'num_tokens',None)} "
                   f"uniform={getattr(desc,'uniform',None)}"] += 1
        except Exception as e:
            _tally[f"ERR {e!r}"] += 1
        return r
    CudagraphDispatcher.dispatch = _dispatch
    atexit.register(lambda: print("\n[cf-cgmode] dispatched modes:",
                                  dict(_tally.most_common(8)), flush=True))

if os.environ.get("CF_VPROF") == "1":
    from chained_flow.vllm_plugin import vprof
    vprof.install()
    if os.environ.get("CF_SYNCDBG") == "1":
        vprof.install_sync_debug()

MODE = os.environ.get("CF_MODE", "base")
MODEL = os.environ.get("CF_MODEL", "Qwen/Qwen3.5-4B")
K = int(os.environ.get("CF_K", "4"))
GMU = float(os.environ.get("CF_GMU", "0.55"))
_BASE = ["Q: What is 15*23? A: Let's think step by step.", "def fibonacci(n):",
         "The capital of France is", "Write a short paragraph about the ocean.",
         "Summarize the causes of the French Revolution.", "Explain gradient descent simply.",
         "Translate to French: the weather is nice today.", "Who wrote Pride and Prejudice?"]
_N = int(os.environ.get("CF_BATCH", "4"))
# CF_PROMPTS=<dir of bench_data/*.jsonl> draws REAL distinct prompts round-robin across domains.
# The 8-prompt _BASE list cycled with "(vN)" suffixes gives near-duplicates, which makes both
# accept and throughput look far more stable than they are.
_pd = os.environ.get("CF_PROMPTS")
# CF_POFF may be a COMMA LIST of offsets: each is a disjoint prompt set, all measured in ONE process.
# At batch 1 the 27B model load dominates a run, so measuring the 3 required disjoint sets in separate
# processes would triple the wall clock for no statistical gain.
_OFFS = [int(x) for x in os.environ.get("CF_POFF", "0").split(",")]
if _pd:
    import glob, json as _j, itertools
    _per = []
    for f in sorted(glob.glob(os.path.join(_pd, "*.jsonl"))):
        _per.append([_j.loads(l)["prompt"] for l in open(f) if l.strip()])
    _rr = [p for grp in itertools.zip_longest(*_per) for p in grp if p]
    SETS = [[_rr[(i + _off) % len(_rr)] for i in range(_N)] for _off in _OFFS]
else:
    SETS = [[_BASE[(i + _off) % len(_BASE)] + f" (v{i//len(_BASE)})" for i in range(_N)]
            for _off in _OFFS]
PROMPTS = SETS[0]
_T = float(os.environ.get("CF_TEMP", "0"))   # CF_TEMP>0 exercises the NON-GREEDY path
sp = SamplingParams(temperature=_T, max_tokens=int(os.environ.get("CF_MAXTOK","64")),
                    seed=1234 if _T > 0 else None)

kw = dict(model=MODEL, gpu_memory_utilization=GMU, max_model_len=2048, dtype="float16",
          enforce_eager=os.environ.get("CF_EAGER","0")=="1", max_num_seqs=64)
# CF_SSM_DTYPE=bfloat16 halves the recurrent-state cache (Qwen3.5 ships fp32).
_sd = os.environ.get("CF_SSM_DTYPE")
if _sd:
    kw["mamba_ssm_cache_dtype"] = _sd
if MODE == "spec":
    kw["speculative_config"] = {"method": "custom_class",
                                "model": "chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
                                "num_speculative_tokens": K}
# CF_ASYNC_SCHED: vLLM auto-decides async_scheduling (config/vllm.py:992-1040) and gives the
# BASE arm a feature custom_class spec is FORCED to give up (:1003). Leaving it auto makes every
# base-vs-spec ratio understate the spec arm. See docs/BENCHMARKING.md.
_as = os.environ.get('CF_ASYNC_SCHED')
if _as is not None:
    kw['async_scheduling'] = _as == '1'
llm = LLM(**kw)


def _acc_counters():
    """Cumulative (tokens emitted, request-steps) from the proposer, for per-set accept."""
    try:
        from chained_flow.vllm_plugin.flow_proposer import _STASH
        p = _STASH.get("proposer")
        if p is None:
            return (0, 0)
        if getattr(p, "async_spec", False):
            # CF_ASYNC_SPEC: the CPU token lists are empty every step, so the counters are
            # accumulated on the GPU and drained here (once per prompt set, not per step).
            g = getattr(p, "_acc_gpu", None)
            return (0, 0) if g is None else tuple(int(x) for x in g.tolist())
        return (p._t.get("acc_tok", 0), p._t.get("acc_req", 0))
    except Exception as e:
        print(f"[cf-warn] accept counters unavailable: {e!r}", flush=True)
        return (0, 0)


llm.generate(PROMPTS, sp)                      # warmup
llm.generate(PROMPTS, sp)                      # warmup 2: our draft cudagraph is captured lazily

_tp = None
if os.environ.get("CF_TORCHPROF") == "1":
    # GPU-busy vs host-wall split. Cudagraph replays still emit their kernels to the
    # CUPTI stream, so kernel time inside the captured draft is accounted correctly.
    import torch
    _tp = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False, with_stack=False)
    _tp.start()

sets = []
for _off, _ps in zip(_OFFS, SETS):
    a0 = _acc_counters()
    t0 = time.time(); outs = llm.generate(_ps, sp); dt = time.time() - t0
    a1 = _acc_counters()
    n = sum(len(o.outputs[0].token_ids) for o in outs)
    acc = (a1[0] - a0[0]) / (a1[1] - a0[1]) if a1[1] > a0[1] else None
    sets.append({"poff": _off, "tps": n / dt, "tokens": n, "secs": dt, "accept": acc,
                 "outs": [list(o.outputs[0].token_ids) for o in outs]})
    print(f"\n[{MODE}] poff={_off} {n} tok / {dt:.2f}s = {n/dt:.1f} tok/s"
          + (f" | accept {acc:.3f}" if acc else ""), flush=True)
if _tp is not None:
    _tp.stop()
    _f = f"/tmp/cf_trace_{MODE}{os.environ.get('CF_TAG','')}.json"
    _tp.export_chrome_trace(_f)
    print(f"[cf-trace] wrote {_f}", flush=True)
if os.environ.get("CF_GDN_CUDA_CHECK") == "1":
    from vllm.v1.spec_decode import tree_gdn_verify as _gv
    print(f"[cf-gdn-cuda-check] mismatching elements: {_gv.check_counters()}", flush=True)
if os.environ.get("CF_GDN_FACTOR_CHECK") == "1":
    from vllm.v1.spec_decode import tree_gdn_factor as _gf
    print(f"[cf-gdn-factor-check] mismatching output elements: {_gf.check_counters()}",
          flush=True)
if os.environ.get("CF_TFA_CHECK") == "1":
    from vllm.v1.spec_decode import tree_attn_fused as _tfa
    print(f"[cf-tfa-check] mismatching output elements: {_tfa.check_counters()}", flush=True)
res = dict(sets[0], mode=MODE, sets=sets)
json.dump(res, open(f"/tmp/cf_native_{MODE}{os.environ.get('CF_TAG','')}.json", "w"))
_m = sum(s["tokens"] for s in sets) / sum(s["secs"] for s in sets)
print(f"\n[{MODE}] SETS " + " ".join(f"{s['tps']:.1f}" for s in sets) + f" | mean {_m:.1f} tok/s",
      flush=True)
