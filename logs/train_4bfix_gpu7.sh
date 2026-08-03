#!/bin/bash
# 4B alignment-fix retrain (prev_token_cond + scheduled sampling), SINGLE GPU 7.
set -o pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/train_tree_flow.py \
    train_configs/recovered/joint_4bfix.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
