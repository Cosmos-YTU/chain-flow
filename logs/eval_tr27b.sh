#!/usr/bin/env bash
# Accept matrix for the Turkish 27B drafter.
#
#   usage: bash logs/eval_tr27b.sh [checkpoint_dir] [gpu]
#
# The checkpoint is an ARGUMENT, chosen from the per-epoch curve in
# scripts/sweep_tr27b_checkpoints.py -- not assumed to be the last one. 4B converged at 2 epochs and
# 9B at ~1.3 (it stopped at checkpoint-800 because more training bought Turkish nothing and slowly
# eroded English), so the final 6-epoch checkpoint here is an upper bound, not the default answer.
#
# English states use [bh]*-q3527b-* -- bench-* AND heldout-* (6 domains incl. held-out gsm8k).
# NOT *-q3527b-*, which also sweeps in the _tmp_ staging dirs (answer datasets with no hidden
# states) and the stage1 TRAINING sets, i.e. in-distribution data in a held-out measurement.
#
# Arm 1 is a SANITY GATE, not a result: v2 + the packaged shortlist on English must reproduce the
# published plugin-arm accept of 2.44 (measured 2.38 over 5 domains -- passes). If it drifts, the
# invocation is wrong and every number below it is meaningless.
#
# Arms 6/7/8 are the THREE-HEAD A/B on the trained checkpoint and are the point of this script.
# 9B measured full 2.81 / Turkish-SL 2.81 / packaged-English-SL 2.11 -- shipping the stock English
# list cost -0.70, HALF its entire training gain. That hazard is INVISIBLE before adaptation: on the
# parent all three heads score alike, because a drafter that cannot predict Turkish is not yet being
# clipped by the shortlist. Measuring it on the parent (arms 4 vs 5, which differ by +0.02 here)
# clears it FALSELY. Arm 7 vs 6 additionally shows whether the Turkish list itself clips anything.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

V2=out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8
TR=${1:-out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8}
GPU=${2:-3}
EN='teacher_states/[bh]*-q3527b-*'
TRS='teacher_states/bench-tr27b-*'
SL_OLD=out/flow/shortlist_shipped_bare.pt      # 62,642 ids -- the packaged English list
SL_NEW=out/flow/shortlist_q3527b_tr_bare.pt    # 77,939 ids -- en+tr min50, ships with this model

echo "TR checkpoint under test: $TR"
sha256sum "$TR/model.safetensors" 2>/dev/null || echo "  (NO model.safetensors -- not a real checkpoint)"

run(){ # label ckpt states [shortlist]  -- omitting the shortlist runs the FULL head
  echo ""
  echo "############ $1"
  echo "# ckpt=$2 states=$3 head=${4:-FULL}"
  local sl=()
  [ -n "${4:-}" ] && sl=(--shortlist "$4")
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/diff_plugin_vs_harness.py \
    --ckd "$2" --model Qwen/Qwen3.5-27B --states "$3" "${sl[@]}" \
    --plugin_tree 2>&1 | grep -vE "it/s\]$|examples/s\]$"
}

# Arm 1 deliberately uses the 5-domain BENCH set, not the 6-domain $EN glob: the published 2.44 is
# a bench-only figure, and adding held-out gsm8k (3.56, the highest-accept domain, and a held-out
# split of an IN-DISTRIBUTION domain) lifts the mean to 2.57 and makes the gate read +0.13 against a
# reference that never included it. Bench-only reproduces at 2.38. The gate exists to validate the
# invocation, so it has to compare like with like; arms 2/3 use all 6 domains for real coverage.
run "1. EN / v2 / enSL    (SANITY GATE, bench-only: expect ~2.44)"      "$V2" 'teacher_states/bench-q3527b-*' "$SL_OLD"
run "2. EN / v2 / trSL    (shortlist swap on English -- must be ~free)"  "$V2" "$EN"  "$SL_NEW"
run "3. EN / tr / trSL    (ENGLISH AFTER -- regression check)"           "$TR" "$EN"  "$SL_NEW"
run "4. TR / v2 / enSL    (Turkish as it would serve TODAY)"             "$V2" "$TRS" "$SL_OLD"
run "5. TR / v2 / trSL    (TURKISH BEFORE -- warm-start baseline)"       "$V2" "$TRS" "$SL_NEW"
run "6. TR / tr / trSL    (TURKISH AFTER -- the deliverable)"            "$TR" "$TRS" "$SL_NEW"
run "7. TR / tr / full    (does the Turkish list clip anything at all?)" "$TR" "$TRS"
run "8. TR / tr / enSL    (COST of shipping the stock English list)"     "$TR" "$TRS" "$SL_OLD"
echo ""
echo "EVAL DONE $(date -u)"
