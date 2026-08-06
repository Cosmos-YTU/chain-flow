#!/usr/bin/env bash
# GPU 4 extension shards. Starts IMMEDIATELY: GPU 4's wave-1 list (instruct_b, funccall) is
# already done, and the original single gate for the whole of wave 1 would have left this GPU
# idle for ~1h waiting on GPU 3's multiturn+toolcall. Per-GPU gating reclaims that hour.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

for s in instruct_d multiturn2 toolcall2; do
  echo "==== [G4] SRC $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=4 $PY scripts/collect_teacher_states.py \
    collect_configs/stage1_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== [G4] SRC $s DONE rc=$? $(date -u +%H:%M:%S) df=$(df -h / | awk 'NR==2{print $4}') ===="
done
echo "G4 EXT DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
