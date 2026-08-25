#!/usr/bin/env bash
# Adds checkpoint-1800 and checkpoint-2100 to the finalist table at per_domain 400.
#
# Why these and not 300: the selection rule changed to Turkish-first, which voids 300's entire case
# (its only advantage was costing less English). The live question is whether a LATER checkpoint
# beats 600 on Turkish. The 200-window sweep has 900/1200/1500/1800 all sitting above 600 -- each
# within jitter individually, but four consecutive points above is the same shape that was correctly
# read as signal for free-form English. It cannot be settled from the sweep: 600 scores 2.64 at 200
# windows and 2.84 at 400, so 900's 2.67 is not comparable to 600's 2.84. Different instrument.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CK=out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8
EN='teacher_states/[bh]*-q3527b-*'; TRS='teacher_states/bench-tr27b-*'; OOD='teacher_states/ood-tr27b-*'
SL=out/flow/shortlist_q3527b_tr_bare.pt
run(){ echo ""; echo "############ $2"; echo "# ckpt=$3 states=$4"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/diff_plugin_vs_harness.py --ckd "$3" --model Qwen/Qwen3.5-27B \
    --states "$4" --shortlist "$SL" --per_domain 400 --plugin_tree 2>&1 | grep -vE "it/s\]$|examples/s\]$"; }
( run 3 "30. TR / tr1800 / trSL  (Turkish, ckpt-1800)"   "$CK/checkpoint-1800"  "$TRS"
  run 3 "31. TRO / tr1800 / trSL (OOD Turkish, ckpt-1800)" "$CK/checkpoint-1800" "$OOD"
  run 3 "32. EN / tr1800 / trSL  (English floor, ckpt-1800)" "$CK/checkpoint-1800" "$EN"
  echo "LATE2-A DONE" ) > logs/matrix_late2_a.log 2>&1 &
A=$!
( run 4 "33. TR / tr2100 / trSL  (Turkish, ckpt-2100)"   "$CK/checkpoint-2100" "$TRS"
  run 4 "34. TRO / tr2100 / trSL (OOD Turkish, ckpt-2100)" "$CK/checkpoint-2100" "$OOD"
  run 4 "35. EN / tr2100 / trSL  (English floor, ckpt-2100)" "$CK/checkpoint-2100" "$EN"
  echo "LATE2-B DONE" ) > logs/matrix_late2_b.log 2>&1 &
B=$!
wait $A; wait $B
cat logs/matrix_late2_a.log logs/matrix_late2_b.log >> logs/eval_tr27b.log
echo "LATE2 MATRIX DONE $(date -u)"
