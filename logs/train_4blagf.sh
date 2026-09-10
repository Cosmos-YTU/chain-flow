#!/bin/bash
# 4B lag-retrain: same v1 data/VAE as joint_4b, ONLY the context construction differs
# (ends at h(t-1) + learned slot from emb(token t)). Tests whether the -0.68 accept the
# vLLM proposer loses to the one-step lag is recoverable. DDP on GPUs 5+6.
set -o pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=5,6 .venv/bin/python -m torch.distributed.run --nproc_per_node=2 \
    --master_port=29525 scripts/train_tree_flow.py train_configs/recovered/joint_4blagf.yaml 2>&1 \
    | grep -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
