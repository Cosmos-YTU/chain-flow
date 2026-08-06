#!/usr/bin/env bash
# Turkish 27B teacher-state collection, data-parallel over two GPUs.
# Prompts come from bench_data_tr/*.train.jsonl (chat-templated prefixes lifted from the
# turkishdspark corpus); Qwen3.5-27B generates its own continuations, so the completions are
# on-distribution for OUR target rather than the Qwen3.6-35B that built that corpus.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

collect_one(){
  local gpu=$1; shift
  for s in "$@"; do
    echo "==== [G$gpu] SRC $s START $(date -u +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/collect_teacher_states.py \
      collect_configs/stage1_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
    echo "==== [G$gpu] SRC $s DONE rc=$? $(date -u +%H:%M:%S) df=$(df -h / | awk 'NR==2{print $4}') ===="
  done
}

collect_one 3 instruct_a multiturn toolcall > logs/collect_tr27b_g3.log 2>&1 &
P3=$!
collect_one 4 instruct_b funccall           > logs/collect_tr27b_g4.log 2>&1 &
P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "COLLECT DONE g3=$R3 g4=$R4 $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
