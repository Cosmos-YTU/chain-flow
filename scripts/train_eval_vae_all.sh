#!/usr/bin/env bash
set -euo pipefail

pids=()

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py train_configs/vae/sweeps/vae_6.5k_lr4e3.yaml > tmp/out_lr4e3.log 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py train_configs/vae/sweeps/vae_6.5k_lr5e3.yaml > tmp/out_lr5e3.log 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py train_configs/vae/sweeps/vae_6.5k_lr7e3.yaml > tmp/out_lr7e3.log 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_vae.py train_configs/vae/sweeps/vae_6.5k_lr1e2.yaml > tmp/out_lr1e2.log 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if [ "$status" -ne 0 ]; then
  echo "one or more training jobs failed; skipping eval" >&2
  exit "$status"
fi

eval_pids=()

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr4e3 \
  --dataset_path sghosts/cf_gsm8k_1k_test \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr4e3_test.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr4e3 \
  --dataset_path sghosts/cf_gsm8k_1k_train \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr4e3_train.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr5e3 \
  --dataset_path sghosts/cf_gsm8k_1k_test \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr5e3_test.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr5e3 \
  --dataset_path sghosts/cf_gsm8k_1k_train \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr5e3_train.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr7e3 \
  --dataset_path sghosts/cf_gsm8k_1k_test \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr7e3_test.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr7e3 \
  --dataset_path sghosts/cf_gsm8k_1k_train \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr7e3_train.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr1e2 \
  --dataset_path sghosts/cf_gsm8k_1k_test \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr1e2_test.log 2>&1 &
eval_pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/eval_vae.py \
  --vae_dir out/vae/ckpts/hidden-vae-lr1e2 \
  --dataset_path sghosts/cf_gsm8k_1k_train \
  --dataset_split train \
  --eval_all_checkpoints \
  --checkpoint_stride 5 \
  > tmp/eval_lr1e2_train.log 2>&1 &
eval_pids+=("$!")

for pid in "${eval_pids[@]}"; do
  wait "$pid"
done
