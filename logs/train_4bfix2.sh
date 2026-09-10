#!/bin/bash
# 4B ablation: prev_token_cond ONLY, scheduled sampling off. DDP on GPUs 6+7.
set -o pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=6,7 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
    --master_port=29537 scripts/train_tree_flow.py train_configs/recovered/joint_4bfix2.yaml 2>&1 \
    | grep --line-buffered -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
