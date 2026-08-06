#!/usr/bin/env bash
# The four BASELINE arms of the accept matrix -- every one of them uses the v2 checkpoint, so
# none of them needs the Turkish training run to have finished. Run now on GPU 4, which went idle
# at 09:48 when the bench finished and would otherwise sit dark until training starts.
#
# The point is arm 1. It is a SANITY GATE: v2 + the shipped shortlist on the English bench must
# reproduce the published plugin-arm accept of 2.44. Running it now means a broken harness
# invocation surfaces with hours of slack to diagnose, instead of at the very end when every
# Turkish number would already be suspect and there would be no time left to redo them.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

V2=out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8
SL_OLD=out/flow/shortlist_shipped_bare.pt      # 62,642 ids -- what ships today
SL_NEW=out/flow/shortlist_q3527b_tr_bare.pt    # 77,939 ids -- en+tr min50

run(){
  echo ""
  echo "############ $1"
  echo "# ckpt=$2 states=$3 shortlist=$(basename $4)"
  CUDA_VISIBLE_DEVICES=4 $PY scripts/diff_plugin_vs_harness.py \
    --ckd "$2" --model Qwen/Qwen3.5-27B --states "$3" --shortlist "$4" \
    --plugin_tree 2>&1 | grep -vE "it/s\]$|examples/s\]$"
}

run "1. EN / v2 / shipped SL   (SANITY GATE: expect plugin accept ~2.44)" \
    "$V2" 'teacher_states/bench-q3527b-*' "$SL_OLD"
run "2. EN / v2 / new SL       (isolates the shortlist swap on English)" \
    "$V2" 'teacher_states/bench-q3527b-*' "$SL_NEW"
run "4. TR / v2 / shipped SL   (Turkish as it would serve TODAY)" \
    "$V2" 'teacher_states/bench-tr27b-*' "$SL_OLD"
run "5. TR / v2 / new SL       (TURKISH BEFORE -- warm-start baseline)" \
    "$V2" 'teacher_states/bench-tr27b-*' "$SL_NEW"
echo ""
echo "BASELINES DONE $(date -u)"
