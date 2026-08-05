#!/bin/bash
# Serve a Chained-Flow arm over the OpenAI HTTP API so the RedHat AI
# `speculator_benchmarks` dataset can be driven by guidellm (the harness in
# neuralmagic/speculators examples/evaluate/eval-guidellm).
#
#   serve_cf.sh <4b|9b|27b> <base|chain|tree> <port> <logfile>
#
# The env table is a transcription of bench_cf.sh's arms; the only differences are
# (a) HTTP serving instead of the offline LLM() entry point and (b) async scheduling
# forced ON for EVERY arm, including base, so the comparison is like-for-like.
set -euo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
PORT=${3:?port}
LOG=${4:?logfile}

export CUDA_VISIBLE_DEVICES=${CF_GPU:-5}
export VLLM_ENABLE_V1_MULTIPROCESSING=0
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
export PYTHONPATH=/home/shadeform/chained-flow/src

# Same capability-gated default table bench_cf.sh uses (src/chained_flow/defaults.py).
_CF_DEF=$(PYTHONPATH=/home/shadeform/chained-flow/src "$CF_PY" \
          /home/shadeform/chained-flow/src/chained_flow/defaults.py --sh)
eval "$_CF_DEF"
# (The `unset CF_SHORTLIST` that used to be here is gone: `defaults.py --sh` no longer exports
#  the shortlist at all, so the packaged list resolves in-process the way a pip user gets it.
#  Fixing it in one place also fixed it for every other launcher, which did NOT have this
#  workaround -- see `_sh()`.)

if [ "$SIZE" = "4b" ]; then
  CF_MODEL=Qwen/Qwen3.5-4B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-4B-v2}
  GMU=${CF_GMU:-0.55}
elif [ "$SIZE" = "9b" ]; then
  CF_MODEL=Qwen/Qwen3.5-9B
  # v2: the v1-vs-v2 decision landed 2026-08-05 in bench_cf.sh -- v2 wins at 9B.
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-9B-v2}
  GMU=${CF_GMU:-0.70}
else
  CF_MODEL=Qwen/Qwen3.5-27B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-Qwen3.5-27B-v2}
  GMU=${CF_GMU:-0.85}
fi
export CF_DRAFTER_DIR

SPEC_ARGS=()
case "$ARM" in
  base) ;;
  chain) export CF_CUDAGRAPH=1 CF_K=${CF_K:-5}
         SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}") ;;
  tree)  export CF_CUDAGRAPH=1 VLLM_SPEC_TREE=1
         : ${CF_TREE_KEEP:=8}; : ${CF_TREE_DEPTH:=5}
         : ${CF_K:=$(( CF_TREE_KEEP * CF_TREE_DEPTH + 1 ))}
         export CF_TREE_KEEP CF_TREE_DEPTH CF_K
         SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}") ;;
  *) echo "bad arm"; exit 1 ;;
esac
# Async scheduling ON for every arm. On the fork the spec arms additionally need
# CF_ASYNC_SPEC=1 (already in the default table) or config/vllm.py refuses the request.
export CF_ASYNC_SCHED=1

echo "[serve_cf] size=$SIZE arm=$ARM port=$PORT gpu=$CUDA_VISIBLE_DEVICES model=$CF_MODEL" >&2
echo "[serve_cf] drafter=$CF_DRAFTER_DIR K=${CF_K:-n/a} keep=${CF_TREE_KEEP:-n/a} depth=${CF_TREE_DEPTH:-n/a}" >&2

exec "${CF_PY%python}vllm" serve "$CF_MODEL" \
  --seed 42 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "${CF_MAXLEN:-8192}" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "${CF_MAXSEQS:-64}" \
  --async-scheduling \
  "${SPEC_ARGS[@]}" \
  > "$LOG" 2>&1
