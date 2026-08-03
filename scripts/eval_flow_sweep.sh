#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-data/flow_cache/gsm8k_1k_test}"
EVAL_DATASET_SPLIT="${EVAL_DATASET_SPLIT:-train}"
CHECKPOINT_STRIDE="${CHECKPOINT_STRIDE:-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LOG_DIR="${LOG_DIR:-tmp/flow_eval_sweep_logs}"

mkdir -p "$LOG_DIR"

names=(
  context_size_8
  context_size_12
  context_size_16
  expert_dim_256
  expert_dim_384
  expert_dim_512
  ffn_multiplier_8
  ffn_multiplier_12
  ffn_multiplier_16
)

flow_dirs=(
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-context_size-8
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-context_size-12
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-context_size-16
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-expert_dim-256
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-expert_dim-384
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-expert_dim-512
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-ffn_multiplier-8
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-ffn_multiplier-12
  out/flow/ckpts/chunked-flow-k2-gsm8k-6.5k-ffn_multiplier-16
)

pids=()

for i in "${!names[@]}"; do
  name="${names[$i]}"
  flow_dir="${flow_dirs[$i]}"
  log_path="$LOG_DIR/${name}.log"
  echo "evaluating $name: flow_dir=$flow_dir dataset=$EVAL_DATASET_PATH stride=$CHECKPOINT_STRIDE log=$log_path"
  CUDA_VISIBLE_DEVICES="$GPU_ID" uv run python scripts/eval_flow.py \
    --flow_dir "$flow_dir" \
    --dataset_path "$EVAL_DATASET_PATH" \
    --dataset_split "$EVAL_DATASET_SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --eval_all_checkpoints \
    --checkpoint_stride "$CHECKPOINT_STRIDE" \
    > "$log_path" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
