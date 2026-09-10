#!/bin/bash
# Offline accept gate for the Turkish 9B drafter. GPU 2 only.
#
#   $1 = checkpoint dir (v2 snapshot for the BEFORE arm, out/flow/ckpts/... for AFTER)
#   $2 = tag used in the log header
#   $3 = windows per domain (default 600)
#   $4 = optional shortlist .pt (RAW id tensor, not the dict payload) -- pass to measure the head
#        the drafter would actually score in production, omit for the full-head ceiling
#
# Tree settings match the v2 model card (keep=8, depth=5, topb=8, K=8). Windows default to 600
# rather than the card's 200: the Turkish bench rows have ~800-token prompts and only generated
# tokens are scored, so 200 windows would come from ~1 row and measure that row, not the domain.
# Every arm uses the same value, so before/after stays comparable.
set -o pipefail
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=2 HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKD=$1; TAG=$2; NW=${3:-600}; SL=$4
PY=.venv/bin/python
SLARG=""; [ -n "$SL" ] && SLARG="--shortlist $SL"

for arm in "en:teacher_states/bench-9b-*" "tr:teacher_states/bench-9btr-*"; do
  lang=${arm%%:*}; states=${arm#*:}
  echo "======== $TAG / $lang  ckd=$CKD windows=$NW shortlist=${SL:-none} $(date -u) ========"
  $PY scripts/diff_plugin_vs_harness.py \
      --ckd "$CKD" --model Qwen/Qwen3.5-9B --states "$states" \
      --K 8 --width 4 --per_domain "$NW" --batch 64 \
      --keep 8 --depth 5 --topb 8 --plugin_tree $SLARG 2>&1 \
    | grep -vE "it/s\]$|examples/s\]$|^Loading weights|^Fetching"
done
