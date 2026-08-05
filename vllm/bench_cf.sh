#!/bin/bash
# Chained-Flow batch-1 7-domain sweep driver.
#   bench_cf.sh <4b|9b|27b> <base|chain|tree> [tag]
# Each CF_POFF offset is ONE prompt from a DIFFERENT domain (prompts are round-robin
# interleaved across bench_data/*.jsonl, 7 files) -> 0..6 == the full 7-domain sweep.
set -euo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
export CF_TAG=${3:-_${SIZE}_${ARM}}
export CUDA_VISIBLE_DEVICES=${CF_GPU:-5}
export CF_BATCH=1
export CF_POFF=${CF_POFF:-0,1,2,3,4,5,6}
export CF_PROMPTS=/home/shadeform/chained-flow/bench_data
export CF_MAXTOK=${CF_MAXTOK:-64}
export CF_ACCEPT=1

# ---------------------------------------------------------------------------------------
# CAPABILITY-GATED DEFAULTS.  Single source of truth: src/chained_flow/defaults.py.
#
# Almost every CF_* default is applied in-process at `import chained_flow` (which the fork does
# while constructing the custom_class proposer, EARLIER in GPUModelRunner.__init__ than it reads
# its own flags).  Exactly one cannot be on the FORK: CF_ASYNC_SPEC is read by the forked
# config/vllm.py inside VllmConfig.__post_init__, i.e. while LLM(...) is still being built,
# before anything imports us.  So the table is also exported from the shell here.  (On STOCK
# vLLM our `vllm.general_plugins` entry point reads it and applies the defaults itself, so the
# emitter is redundant there -- it also emits CF_VLLM_BUILD, which the arms below need.)
#
# The emitter never overwrites a variable that is already set, so `CF_CUDA_BLOCK=0 ./bench_cf.sh`
# still wins.  It is run as a FILE, not `-m`, so it does not import torch (~40 ms, not ~5 s), and
# it prints its fork verdict as a comment which is kept in the log.
# CF_PY selects the interpreter, i.e. WHICH vLLM. The default is the forked venv (tree +
# chain); /home/shadeform/vllm-pristine/.venv/bin/python is unmodified vLLM 0.25.1, where the
# chain arm runs through the `vllm.general_plugins` entry point instead. The defaults emitter is
# run with the SAME interpreter, or it would probe the wrong install for the fork markers.
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}

_CF_DEF=$(PYTHONPATH=/home/shadeform/chained-flow/src \
          "$CF_PY" \
          /home/shadeform/chained-flow/src/chained_flow/defaults.py --sh)
echo "$_CF_DEF" | sed 's/^/[bench_cf] /' >&2
eval "$_CF_DEF"

if [ "$SIZE" = "4b" ]; then
  export CF_MODEL=Qwen/Qwen3.5-4B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-4B-v2}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.55}
elif [ "$SIZE" = "9b" ]; then
  export CF_MODEL=Qwen/Qwen3.5-9B
  # v2 (2026-08-05): measured v1-vs-v2 here, 7 domains, batch 1, maxtok 256, 8x5 tree, 2 repeats.
  # Tree arm pooled accept 2.165 -> 2.302 and 110.1 -> 117.3 tok/s; chain arm 1.716 -> 1.819 and
  # 96.3 -> 102.3 tok/s. This line was the last size still on v1, so every published 9B number
  # before that date is a v1 number. docs/BENCHMARKING.md.
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-9B-v2}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.70}
else
  export CF_MODEL=Qwen/Qwen3.5-27B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-Qwen3.5-27B-v2}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.85}
fi

# How the SPEC arms ask for async scheduling, which differs by build and must not be guessed:
#
#  * FORK: config/vllm.py only relaxes the EXPLICIT-request branch (:971) when CF_ASYNC_SPEC=1.
#    The auto-decide branch (:1007) still forces async OFF for custom_class, so the engine flag
#    has to be passed in or the proposer gates itself back off ("engine async_scheduling is
#    OFF") and the flag does nothing.
#  * STOCK: the `vllm.general_plugins` entry point relaxes BOTH branches, so auto-decide now
#    returns True on its own. Leaving it on AUTO is deliberate -- it is the only configuration
#    in which the plugin's post-resolution assert fires, which is what turns a silent fall back
#    to synchronous (~10% at 4B) into a hard error.
#
# CF_ASYNC_SPEC=0 in the caller's environment reaches here as 0 and suppresses both.
_cf_async_engine() {
  [ "${CF_ASYNC_SPEC:-0}" = "1" ] || return 0
  [ "${CF_VLLM_BUILD:-stock}" = "fork" ] || return 0
  : ${CF_ASYNC_SCHED:=1}; export CF_ASYNC_SCHED
}

case "$ARM" in
  # LIKE-FOR-LIKE BY DEFAULT: vLLM would hand the base arm async_scheduling (which
  # custom_class spec is forced to give up), worth +10.5%/+5.5%/+1.6% at 4B/9B/27B and
  # silently understating every speedup. Set CF_ASYNC_SCHED=1 for the DEPLOYMENT number.
  # docs/BENCHMARKING.md. The chain/tree arms are already async-off; only base needs this.
  base)  export CF_MODE=base; : ${CF_ASYNC_SCHED:=0}; export CF_ASYNC_SCHED ;;
  # The chain arm is the FORK-FREE SHIPPING TARGET: only the drafter-side defaults apply, and
  # it runs on unmodified vLLM. CF_ASYNC_SPEC now applies here too -- the async CONTRACT is
  # generic and only the TREE-SHAPE hand-off needed the fork -- so the chain arm gets the
  # +10.5%/+5.5%/+1.6% that used to belong to the base arm alone. See `_cf_async_engine` below.
  # CF_COMPILE is NO LONGER set here. It was hard-coded to 1 in both spec arms while the
  # proposer defaulted it to 0, so every published number came from a torch.compile'd flow net
  # that a pip user did not get and no line in the log mentioned. It is now a capability-gated
  # default (defaults.py, gate: an inductor backend exists) and appears on the [cf-defaults]
  # line like every other flag -- which also means this script must NOT set it, or the benchmark
  # goes back to measuring a path the table cannot report on.
  chain) export CF_MODE=spec CF_CUDAGRAPH=1 CF_K=5; _cf_async_engine ;;
  # CF_TREE_FULLCG=1 makes a tree step dispatch to vLLM's FULL decode cudagraph
  # instead of PIECEWISE (bit-exact; set it to 0 to reproduce the PIECEWISE path).
  tree)  export CF_MODE=spec CF_CUDAGRAPH=1 VLLM_SPEC_TREE=1
         # shape is caller-overridable (default 4x4=16 nodes). CF_K must be nodes+1: the spare
         # mamba state column. Previously these were `export`ed unconditionally, silently
         # overriding a caller-supplied shape -- same trap as CF_DRAFTER_DIR.
         : ${CF_TREE_KEEP:=4}; : ${CF_TREE_DEPTH:=4}
         : ${CF_K:=$(( CF_TREE_KEEP * CF_TREE_DEPTH + 1 ))}
         export CF_TREE_KEEP CF_TREE_DEPTH CF_K
         _cf_async_engine ;;
  *) echo "bad arm"; exit 1 ;;
esac
exec "$CF_PY" /home/shadeform/chained-flow/vllm/test_plugin_native.py
