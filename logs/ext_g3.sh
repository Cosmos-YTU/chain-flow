#!/usr/bin/env bash
# GPU 3 extension shards. Waits only for GPU 3's OWN wave-1 tail (multiturn, toolcall) rather
# than for all of wave 1, so GPU 4 is free to run ahead on its own chain.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

while pgrep -f "collect_teacher_states.py collect_configs/stage1_tr27b/(instruct_a|multiturn|toolcall)\.yaml" >/dev/null 2>&1; do
  sleep 60
done
echo "GPU 3 wave-1 tail finished at $(date -u)"

for s in instruct_c funccall2; do
  echo "==== [G3] SRC $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=3 $PY scripts/collect_teacher_states.py \
    collect_configs/stage1_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== [G3] SRC $s DONE rc=$? $(date -u +%H:%M:%S) df=$(df -h / | awk 'NR==2{print $4}') ===="
done
echo "G3 EXT DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
