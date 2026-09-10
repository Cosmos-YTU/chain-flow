#!/usr/bin/env bash
# Turkish HOLDOUT eval sets (301 rows) -- the accept gate for the Turkish drafter.
#
# Runs on GPU 4 IMMEDIATELY rather than waiting for both extension chains. GPU 4 finished its
# chain at 09:36 and would otherwise idle ~1.3h while funccall2 finishes on GPU 3; the bench is
# only ~20 min, so running it here takes it off the critical path entirely instead of appending
# it after the last training shard.
#
# Writes to logs/bench_after_ext.log because logs/train_tr27b.sh gates on the string
# "BENCH DONE" appearing in THAT file. Keep both in sync if either moves.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

echo "bench starting on GPU 4 at $(date -u) (funccall2 still collecting on GPU 3)"
for s in tr_instruct tr_funccall tr_multiturn tr_toolcall; do
  echo "==== BENCH $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=4 $PY scripts/collect_teacher_states.py \
    collect_configs/bench_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== BENCH $s DONE rc=$? $(date -u +%H:%M:%S) ===="
done
echo "BENCH DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
