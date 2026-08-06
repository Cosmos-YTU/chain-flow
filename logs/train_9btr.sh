#!/bin/bash
# Turkish 9B drafter: warm-start fine-tune from Flow-Drafter-9B-v2. GPU 2 only, single process.
set -o pipefail
cd /home/shadeform/chained-flow
export CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "==== TRAIN 9btr START $(date -u) ===="
.venv/bin/python scripts/train_tree_flow.py train_configs/recovered/joint_9btr.yaml 2>&1 \
  | grep -viE "it/s\]$|examples/s\]$|^Loading weights|^Fetching"
echo "==== TRAIN 9btr DONE rc=$? $(date -u) ===="
touch logs/train_9btr_DONE.flag
