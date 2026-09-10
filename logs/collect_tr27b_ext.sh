#!/usr/bin/env bash
# Phase 2 of the Turkish 27B collection: the EXTENSION shards.
# The disk budget went 120 GB -> ~400 GB mid-run, so rather than restart the running job we
# append disjoint shards. bench_data_tr/*.train.jsonl grows as a stable PREFIX (verified:
# the first 8000/1000/400/300 rows are byte-identical to what the first wave is collecting,
# and the holdouts are unchanged), so these ranges do not overlap the first wave.
# Waits for the first wave to release the GPUs before starting.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

while pgrep -f "collect_teacher_states.py collect_configs/stage1_tr27b/(instruct_a|instruct_b|funccall|multiturn|toolcall)\.yaml" >/dev/null 2>&1; do sleep 60; done
echo "first wave released the GPUs at $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"

collect_one(){
  local gpu=$1; shift
  for s in "$@"; do
    echo "==== [G$gpu] SRC $s START $(date -u +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/collect_teacher_states.py \
      collect_configs/stage1_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
    echo "==== [G$gpu] SRC $s DONE rc=$? $(date -u +%H:%M:%S) df=$(df -h / | awk 'NR==2{print $4}') ===="
  done
}

# also collect the Turkish holdout eval sets (small) on whichever GPU frees first
collect_one 3 instruct_c funccall2 > logs/collect_tr27b_g3_ext.log 2>&1 &
P3=$!
collect_one 4 instruct_d multiturn2 toolcall2 > logs/collect_tr27b_g4_ext.log 2>&1 &
P4=$!
wait $P3; R3=$?
wait $P4; R4=$?
echo "COLLECT EXT DONE g3=$R3 g4=$R4 $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
