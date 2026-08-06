#!/usr/bin/env bash
# Turkish HOLDOUT eval sets (301 rows total) -- the accept gate for the Turkish drafter.
# Same `pretemplated` handler as training, so the `<think>\n\n</think>\n\n` non-thinking
# prefix is byte-identical between collection and evaluation.
#
# Gated on the EXTENSION DRIVER's completion marker rather than on "no collector is
# running": there is an up-to-60s gap between wave 1 exiting and wave 2 claiming the
# GPUs, and a process-absence check would fire inside that gap and contend for GPU 3.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
until grep -q "COLLECT EXT DONE" logs/collect_tr27b_ext_main.log 2>/dev/null; do sleep 60; done
echo "extension done, collecting Turkish bench at $(date -u)"
for s in tr_instruct tr_funccall tr_multiturn tr_toolcall; do
  echo "==== BENCH $s START $(date -u +%H:%M:%S) ===="
  CUDA_VISIBLE_DEVICES=3 $PY scripts/collect_teacher_states.py \
    collect_configs/bench_tr27b/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$"
  echo "==== BENCH $s DONE rc=$? $(date -u +%H:%M:%S) ===="
done
echo "BENCH DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
