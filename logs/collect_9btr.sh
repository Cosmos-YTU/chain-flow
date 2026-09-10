#!/bin/bash
# Turkish teacher states for Qwen/Qwen3.5-9B, GPU 2 only. Sequential over the 4 dspark corpora.
set -o pipefail
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=2 HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for s in instructurca multiturn func_calling tool_calling; do
  echo "==== SRC $s START $(date -u) df_avail=$(df -BG --output=avail / | tail -1) ===="
  $PY scripts/collect_teacher_states.py collect_configs/stage1_9btr/$s.yaml 2>&1 \
    | grep -vE "it/s\]$|examples/s\]$|^Loading weights|^Fetching"
  echo "==== SRC $s DONE rc=$? $(date -u) df_avail=$(df -BG --output=avail / | tail -1) ===="
done
echo "==== ALL TURKISH COLLECTION DONE $(date -u) ===="
touch logs/collect_9btr_DONE.flag
