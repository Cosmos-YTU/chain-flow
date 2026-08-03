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
export CF_SHORTLIST=/home/shadeform/chained-flow/out/flow/shortlist_q3527b.pt

if [ "$SIZE" = "4b" ]; then
  export CF_MODEL=Qwen/Qwen3.5-4B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-4B-v2}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.55}
elif [ "$SIZE" = "9b" ]; then
  export CF_MODEL=Qwen/Qwen3.5-9B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-9B}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.70}
else
  export CF_MODEL=Qwen/Qwen3.5-27B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-Qwen3.5-27B-v2}; export CF_DRAFTER_DIR
  export CF_GMU=${CF_GMU:-0.85}
fi

case "$ARM" in
  base)  export CF_MODE=base ;;
  chain) export CF_MODE=spec CF_COMPILE=1 CF_CUDAGRAPH=1 CF_K=5 ;;
  # CF_TREE_FULLCG=1 makes a tree step dispatch to vLLM's FULL decode cudagraph
  # instead of PIECEWISE (bit-exact; set it to 0 to reproduce the PIECEWISE path).
  tree)  export CF_MODE=spec CF_COMPILE=1 CF_CUDAGRAPH=1 VLLM_SPEC_TREE=1
         # shape is caller-overridable (default 4x4=16 nodes). CF_K must be nodes+1: the spare
         # mamba state column. Previously these were `export`ed unconditionally, silently
         # overriding a caller-supplied shape -- same trap as CF_DRAFTER_DIR.
         : ${CF_TREE_KEEP:=4}; : ${CF_TREE_DEPTH:=4}
         : ${CF_K:=$(( CF_TREE_KEEP * CF_TREE_DEPTH + 1 ))}
         export CF_TREE_KEEP CF_TREE_DEPTH CF_K
         export CF_TREE_FULLCG=${CF_TREE_FULLCG:-1} ;;
  *) echo "bad arm"; exit 1 ;;
esac
exec /home/shadeform/vllm/.venv/bin/python /home/shadeform/chained-flow/vllm/test_plugin_native.py
