#!/bin/bash
# P1: total-variation loss (alpha = 1 - d_TV IS the acceptance rate). DDP on GPUs 3,4,5,6.
set -o pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=3,4,5,6 .venv/bin/python -m torch.distributed.run --nproc_per_node=4 \
    --master_port=29545 scripts/train_tree_flow.py train_configs/recovered/joint_4btv.yaml 2>&1 \
    | grep --line-buffered -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
touch logs/train_4btv_DONE.flag
