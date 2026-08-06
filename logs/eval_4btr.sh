#!/usr/bin/env bash
# Offline accept gate for the Turkish 4B drafter. GPU 7 ONLY.
#
#   $1 = checkpoint dir  (v2 dir for the BEFORE arm, out/flow/ckpts/tree-vae-joint-4btr-... for AFTER)
#   $2 = tag printed in the log header
#   $3 = windows per domain (default 600)
#   $4 = optional shortlist .pt -- must be the RAW id tensor
#        (out/flow/shortlist_q3527b_tr_ids.pt), NOT the dict payload that ships as shortlist.pt:
#        diff_plugin_vs_harness.py does torch.load(...).flatten() with no unwrapping.
#
# Tree settings match the v2 model card (keep=8, depth=5, topb=8, K=8). 600 windows/domain rather
# than the card's 200 because the Turkish bench rows have 600-800 token prompts and only GENERATED
# tokens are scored -- 200 windows would come from about one row and would measure that row, not
# the domain. Both arms use the same value, so before/after stays comparable.
set -uo pipefail
cd /home/shadeform/chained-flow
export CUDA_VISIBLE_DEVICES=7 HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CKD=$1; TAG=$2; NW=${3:-600}; SL=${4:-}
PY=.venv/bin/python
SLARG=""; [ -n "$SL" ] && SLARG="--shortlist $SL"

for arm in "en:teacher_states/bench-4b-*" "tr:teacher_states/bench-4btr-*"; do
  lang=${arm%%:*}; states=${arm#*:}
  echo "======== $TAG / $lang  ckd=$CKD windows=$NW shortlist=${SL:-none} $(date -u) ========"
  $PY scripts/diff_plugin_vs_harness.py \
      --ckd "$CKD" --model Qwen/Qwen3.5-4B --states "$states" \
      --K 8 --width 4 --per_domain "$NW" --batch 64 \
      --keep 8 --depth 5 --topb 8 --plugin_tree $SLARG 2>&1 \
    | grep -vE "it/s\]$|examples/s\]$|^Loading weights|^Fetching"
done
echo "======== EVAL $TAG DONE $(date -u) ========"
