#!/usr/bin/env bash
# FINAL accept matrix at per_domain 400, deciding between TWO checkpoint finalists.
#
#   usage: bash logs/matrix_tr27b.sh          (runs both GPU streams, waits for both)
#
# Why two finalists rather than one: the sweep (per_domain 200) put Turkish on a plateau from
# checkpoint-600 onward with spread +-0.02, and the 300->600 Turkish gap is only 0.06 -- about 3x
# that spread. Enough to suspect 300 sits below the plateau, not enough to bet the shipped model on
# it. 400 windows is the instrument that can separate them. If 600 holds its lead, ship 600; if they
# converge, ship 300, which costs strictly less English.
#
# Sweep numbers (200 windows) are NEVER mixed with these (400 windows). Different denominators on
# the same tool; the sweep selects, this reports.
#
# TRO = out-of-distribution Turkish (turkish_alpaca + wikirag_tr). The TR holdout is disjoint from
# training (verified, zero exact prompt overlap) but drawn from the SAME corpus and source mix, so
# late-checkpoint gains on it partly reflect corpus-specific fit. TRO has no relationship to the
# training data, so it is the honest read on whether Turkish ability transfers. Expect TRO to favour
# the earlier checkpoint more strongly than TR does.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

CK=out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8
V2=out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8
C300=$CK/checkpoint-300
C600=$CK/checkpoint-600
EN='teacher_states/[bh]*-q3527b-*'
TRS='teacher_states/bench-tr27b-*'
OOD='teacher_states/ood-tr27b-*'
SL_OLD=out/flow/shortlist_shipped_bare.pt
SL_NEW=out/flow/shortlist_q3527b_tr_bare.pt

# HF Trainer does not write the tree config into checkpoint-N/; the eval reads it from --ckd.
for c in "$C300" "$C600"; do
  [ -f "$c/chained_flow_tree_config.json" ] || cp "$CK/chained_flow_tree_config.json" "$c/"
done

run(){ # gpu label ckpt states [shortlist]
  local gpu=$1 label=$2 ckd=$3 states=$4 sl=${5:-}
  echo ""
  echo "############ $label"
  echo "# ckpt=$ckd states=$states head=${sl:-FULL}"
  local a=()
  [ -n "$sl" ] && a=(--shortlist "$sl")
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/diff_plugin_vs_harness.py \
    --ckd "$ckd" --model Qwen/Qwen3.5-27B --states "$states" "${a[@]}" \
    --per_domain 400 --plugin_tree 2>&1 | grep -vE "it/s\]$|examples/s\]$"
}

stream_a(){   # GPU 3: baselines + the 300 finalist
  run 3 "1. EN / v2 / enSL    (SANITY GATE, bench-only)" "$V2" 'teacher_states/bench-q3527b-*' "$SL_OLD"
  run 3 "2. EN / v2 / trSL    (baseline English, 6 domains)"  "$V2"   "$EN"  "$SL_NEW"
  run 3 "3. TR / v2 / trSL    (baseline Turkish)"             "$V2"   "$TRS" "$SL_NEW"
  run 3 "4. TR / v2 / enSL    (Turkish as it serves today)"   "$V2"   "$TRS" "$SL_OLD"
  run 3 "5. TRO / v2 / trSL   (baseline OOD Turkish)"         "$V2"   "$OOD" "$SL_NEW"
  run 3 "6. EN / tr300 / trSL (English after, ckpt-300)"      "$C300" "$EN"  "$SL_NEW"
  run 3 "7. TR / tr300 / trSL (Turkish after, ckpt-300)"      "$C300" "$TRS" "$SL_NEW"
  run 3 "8. TRO / tr300 / trSL (OOD Turkish, ckpt-300)"       "$C300" "$OOD" "$SL_NEW"
  run 3 "9. TR / tr300 / enSL (stock-list cost, ckpt-300)"    "$C300" "$TRS" "$SL_OLD"
  echo "STREAM A DONE $(date -u)"
}

stream_b(){   # GPU 4: the 600 finalist + the head A/B
  run 4 "10. EN / tr600 / trSL (English after, ckpt-600)"     "$C600" "$EN"  "$SL_NEW"
  run 4 "11. TR / tr600 / trSL (Turkish after, ckpt-600)"     "$C600" "$TRS" "$SL_NEW"
  run 4 "12. TRO / tr600 / trSL (OOD Turkish, ckpt-600)"      "$C600" "$OOD" "$SL_NEW"
  run 4 "13. TR / tr600 / full (does trSL clip anything?)"    "$C600" "$TRS"
  run 4 "14. TR / tr600 / enSL (stock-list cost, ckpt-600)"   "$C600" "$TRS" "$SL_OLD"
  run 4 "15. TRO / tr600 / enSL (stock-list cost, OOD)"       "$C600" "$OOD" "$SL_OLD"
  echo "STREAM B DONE $(date -u)"
}

stream_a > logs/matrix_a.log 2>&1 &
A=$!
stream_b > logs/matrix_b.log 2>&1 &
B=$!
wait $A; wait $B
cat logs/matrix_a.log logs/matrix_b.log > logs/eval_tr27b.log
echo "MATRIX DONE $(date -u)"
