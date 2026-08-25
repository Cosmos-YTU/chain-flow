#!/usr/bin/env bash
# Out-of-distribution Turkish eval sets, on GPU 4 while the sweep runs on GPU 3.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in turkish_alpaca_100 wikirag_tr_100; do
  echo "==== OOD $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/collect_teacher_states.py \
    collect_configs/bench_tr27b_ood/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== OOD $s DONE rc=$? $(date -u +%H:%M:%S) ===="
done
echo "OOD COLLECT DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
