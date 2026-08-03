#!/bin/bash
# P0: widen the flow's context 8 -> 32 hidden states. VAE first (its learned pos-emb caps the
# context length), then the joint drafter. DDP on GPUs 3,4,5,6.
set -o pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
echo "======== [$(date -u '+%m-%d %H:%M:%S')] 1/2 train VAE (ctx32, max_seq 64) ========"
CUDA_VISIBLE_DEVICES=3 $PY scripts/train_transformer_hidden_vae.py \
    train_configs/recovered/vae_4bctx32.yaml 2>&1 | grep --line-buffered -viE "it/s\]$|examples/s\]$"
ls out/vae/ckpts/transformer-hidden-4bctx32-2560-latent640-fp16/model.safetensors >/dev/null 2>&1 \
  || { echo "FATAL vae ckpt missing"; exit 1; }
echo "======== [$(date -u '+%m-%d %H:%M:%S')] 2/2 train drafter (context_size 32), DDP 3,4,5,6 ========"
CUDA_VISIBLE_DEVICES=3,4,5,6 $PY -m torch.distributed.run --nproc_per_node=4 \
    --master_port=29541 scripts/train_tree_flow.py train_configs/recovered/joint_4bctx32.yaml 2>&1 \
    | grep --line-buffered -viE "it/s\]$|examples/s\]$"
echo "TRAIN_RC=$? $(date -u)"
touch logs/train_4bctx32_DONE.flag
