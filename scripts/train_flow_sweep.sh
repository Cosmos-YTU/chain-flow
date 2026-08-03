#!/usr/bin/env bash
set -euo pipefail

mkdir -p tmp/flow_sweep_logs

pids=()

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/context_size/context_size_8.yaml > tmp/flow_sweep_logs/context_size_8.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/context_size/context_size_12.yaml > tmp/flow_sweep_logs/context_size_12.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/context_size/context_size_16.yaml > tmp/flow_sweep_logs/context_size_16.log 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/expert_dim/expert_dim_256.yaml > tmp/flow_sweep_logs/expert_dim_256.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/expert_dim/expert_dim_384.yaml > tmp/flow_sweep_logs/expert_dim_384.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/expert_dim/expert_dim_512.yaml > tmp/flow_sweep_logs/expert_dim_512.log 2>&1 &
pids+=("$!")

CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/ffn_multiplier/ffn_multiplier_8.yaml > tmp/flow_sweep_logs/ffn_multiplier_8.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/ffn_multiplier/ffn_multiplier_12.yaml > tmp/flow_sweep_logs/ffn_multiplier_12.log 2>&1 &
pids+=("$!")
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_chunked_flow.py train_configs/chunked_flow/sweeps/ffn_multiplier/ffn_multiplier_16.yaml > tmp/flow_sweep_logs/ffn_multiplier_16.log 2>&1 &
pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
