#!/bin/bash
# Batch 2 (disk constraint lifted) + the Turkish eval bench. Queued behind batch 1 on GPU 2.
set -o pipefail
cd /home/shadeform/chained-flow
until [ -f logs/collect_9btr_DONE.flag ]; do sleep 30; done
export CUDA_VISIBLE_DEVICES=2 HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
for cfg in stage1_9btr/instructurca2 stage1_9btr/multiturn2 bench_9btr/alpaca bench_9btr/holdout bench_9btr/wikirag; do
  echo "==== CFG $cfg START $(date -u) df_avail=$(df -BG --output=avail / | tail -1) ===="
  $PY scripts/collect_teacher_states.py collect_configs/$cfg.yaml 2>&1 \
    | grep -vE "it/s\]$|examples/s\]$|^Loading weights|^Fetching"
  echo "==== CFG $cfg DONE rc=$? $(date -u) ===="
done
echo "==== BATCH2 + BENCH DONE $(date -u) ===="
touch logs/collect_9btr_b2_DONE.flag
