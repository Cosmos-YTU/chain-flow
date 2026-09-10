#!/usr/bin/env bash
# Turkish 4B drafter: cache -> BEFORE eval -> warm-started train -> AFTER eval -> results json.
# GPU 7 ONLY. Waits for logs/collect_4btr.sh to plant its completion flag.
#
# The cache build is CPU/disk bound and the BEFORE eval is GPU bound, so they run concurrently --
# the only place in this pipeline where two things can overlap on one device.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CACHE=data/flow_cache/stage1_4btr_mix_k4
CKD=out/flow/ckpts/tree-vae-joint-4btr-640-k8-l8
V2=out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8
SL=out/flow/shortlist_q3527b_tr_ids.pt
say(){ echo "[$(date -u +%H:%M:%S)] $* df=$(df -h / | awk 'NR==2{print $4}')"; }

until [ -f logs/collect_4btr_DONE.flag ]; do sleep 60; done
say "collection flag seen"

# ---------------------------------------------------------------- 1. cache + BEFORE eval
if [ ! -f "$CACHE/hidden.pt" ]; then
  say "building flow cache (CPU) and BEFORE eval (GPU) concurrently"
  $PY scripts/build_4btr_cache.py --output-dir "$CACHE" > logs/build_4btr_cache.log 2>&1 &
  CP=$!
else
  say "cache already present, skipping build"
  CP=""
fi

if [ ! -s logs/eval_4btr_before.log ]; then
  bash logs/eval_4btr.sh "$V2" v2-BEFORE 600 "$SL" > logs/eval_4btr_before.log 2>&1
  say "BEFORE eval done rc=$?"
fi

if [ -n "$CP" ]; then
  wait $CP; RC=$?
  say "cache build rc=$RC"
  tail -6 logs/build_4btr_cache.log
  [ -f "$CACHE/hidden.pt" ] || { say "ABORT: no hidden.pt in $CACHE"; exit 1; }
fi

# ---------------------------------------------------------------- 2. train
say "training -- WATCH FOR THE 'WARM START' sha256 LINE"
CUDA_VISIBLE_DEVICES=7 $PY scripts/train_tree_flow.py train_configs/recovered/joint_4btr.yaml 2>&1 \
  | grep -viE "it/s\]$|examples/s\]$|s/it\]$" | tee logs/train_4btr.log | tail -400
[ -f "$CKD/model.safetensors" ] || { say "ABORT: training wrote no model.safetensors"; exit 1; }
say "training done"

# ---------------------------------------------------------------- 3. AFTER eval + results
bash logs/eval_4btr.sh "$CKD" 4btr-AFTER 600 "$SL" > logs/eval_4btr_after.log 2>&1
say "AFTER eval done rc=$?"

$PY scripts/parse_4btr_eval.py --before logs/eval_4btr_before.log --after logs/eval_4btr_after.log \
    --cache "$CACHE" 2>&1 | tail -10

say "4BTR PIPELINE DONE"
touch logs/train_4btr_DONE.flag
