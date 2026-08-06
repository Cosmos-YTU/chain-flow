#!/usr/bin/env bash
# Turkish teacher states for Qwen/Qwen3.5-4B. GPU 7 ONLY -- this box has three other agents on
# GPUs 2/3/4 and 5/6, so this run is deliberately single-device and sized to fit one.
#
# Sizing (measured, not guessed): a 256-row smoke of tr-instructurca ran at 2.44 rows/s and
# 388 tokens/row on GPU 7. The 27B agent's full plan (29,100 rows / ~15M tokens) would be ~4.5h
# on one device. This CORE mix is 13,100 rows / ~7.2M tokens in ~2.5h -- still 1.5x the English
# v2 mix (4.87M tokens) that trained the parent, and better balanced across the four Turkish
# domains than spending the same hours on tr-instructurca alone. instruct_b/instruct_c exist as
# configs and can extend the mix later if the accept gate says it needs more data.
#
# Holdout first (301 rows, ~8 min): it is the accept gate for everything downstream, so it is
# cheapest to de-risk it before committing hours to the training mix.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
say(){ echo "[$(date -u +%H:%M:%S)] $* df=$(df -h / | awk 'NR==2{print $4}')"; }

say "START Turkish 4B collection on GPU 7"

for s in tr_instruct tr_funccall tr_multiturn tr_toolcall; do
  say "BENCH $s start"
  CUDA_VISIBLE_DEVICES=7 $PY scripts/collect_teacher_states.py \
      collect_configs/bench_4btr/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$|s/it\]$"
  say "BENCH $s done rc=$?"
done
say "BENCH ALL DONE"

for s in instruct_a funccall multiturn toolcall; do
  say "TRAIN $s start"
  CUDA_VISIBLE_DEVICES=7 $PY scripts/collect_teacher_states.py \
      collect_configs/stage1_4btr/$s.yaml 2>&1 | grep -vE "it/s\]$|examples/s\]$|it\]$|s/it\]$"
  say "TRAIN $s done rc=$?"
done

say "COLLECT 4BTR DONE"
touch logs/collect_4btr_DONE.flag
