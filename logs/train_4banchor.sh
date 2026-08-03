#!/bin/bash
# Token anchoring: emb(committed token) conditions every flow block; trained under the lag. GPU 3.
set -o pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=4,5,6,7 .venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29551 scripts/train_tree_flow.py \
    train_configs/recovered/joint_4banchor.yaml 2>&1 \
    | grep --line-buffered -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
touch logs/train_4banchor_DONE.flag
