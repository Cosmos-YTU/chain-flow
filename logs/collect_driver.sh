#!/bin/bash
# $1=GPU  $2=config-subdir  $3=logfile
GPU=$1; CFGDIR=$2; LOG=$3
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /home/shadeform/chain-flow
{
echo "==== COLLECT START $CFGDIR on GPU$GPU $(date -u) ===="
for s in gsm8k_fp16 nemotron_math nemotron_stem alpaca_code dolly_chat; do
  echo "==== SRC $s START $(date -u) ===="
  .venv/bin/python scripts/collect_teacher_states.py collect_configs/$CFGDIR/$s.yaml
  echo "==== SRC $s DONE rc=$? $(date -u) ===="
done
echo "==== COLLECT ALL DONE $CFGDIR $(date -u) ===="
} >> "$LOG" 2>&1
