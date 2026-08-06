#!/usr/bin/env bash
# Turkish HOLDOUT eval sets (301 rows) -- the accept gate for the Turkish drafter.
# Uses the same `pretemplated` handler as training, so the `<think>\n\n</think>\n\n`
# non-thinking prefix is byte-identical between collection and evaluation.
#
# Gated on BOTH per-GPU extension markers. Process-absence checks are not used: there are
# gaps between shards where no collector is running, and a presence check would fire inside
# one of them and contend for a GPU that is about to be reclaimed.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

until grep -q "G3 EXT DONE" logs/ext_g3.log 2>/dev/null && grep -q "G4 EXT DONE" logs/ext_g4.log 2>/dev/null; do
  sleep 60
done
echo "both extension chains done, collecting Turkish bench at $(date -u)"

for s in tr_instruct tr_funccall tr_multiturn tr_toolcall; do
  echo "==== BENCH $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=3 $PY scripts/collect_teacher_states.py \
    collect_configs/bench_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== BENCH $s DONE rc=$? $(date -u +%H:%M:%S) ===="
done
echo "BENCH DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
