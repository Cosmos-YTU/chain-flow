#!/usr/bin/env bash
# Runs the 6-arm accept matrix once training finishes, then renders the results JSON.
#
# Deliberately stops short of pushing. Everything upstream of this is compute and can run
# unattended; publishing to HF is public and irreversible, and the sanity gate in
# tr27b_results_json.py (English/v2 must reproduce the published 2.44) has to be READ by
# someone before a card is built on top of it.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src
PY=.venv/bin/python
CKPT=out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8

# training is done when the driver logs DONE *and* the checkpoint is on disk
until grep -q "^\[..:..:..\] DONE" logs/train_tr27b.log 2>/dev/null; do sleep 120; done
if [ ! -f "$CKPT/model.safetensors" ]; then
  echo "TRAIN FINISHED WITHOUT A CHECKPOINT at $CKPT -- not evaluating"
  exit 1
fi
echo "training done, checkpoint present; sha256:"
sha256sum "$CKPT/model.safetensors"

# Per-epoch curve FIRST. Which checkpoint to ship is an empirical question -- the 4B Turkish run
# saturated at 2 epochs and this one is budgeted for 6 -- so the sweep runs before the full matrix
# and the full matrix is NOT launched automatically against the last checkpoint.
$PY scripts/sweep_tr27b_checkpoints.py --out out/flow/tr27b_sweep.json --per-domain 200 --gpu 3 \
  2>&1 | tee logs/sweep_tr27b.log
echo
echo "SWEEP DONE -- read the curve above, then run the full 6-arm matrix on the chosen checkpoint:"
echo "    bash logs/eval_tr27b.sh <checkpoint_dir> 2>&1 | tee logs/eval_tr27b.log"
echo "    .venv/bin/python scripts/tr27b_results_json.py --log logs/eval_tr27b.log --out out/flow/tr27b_eval.json"
echo "EVAL PIPELINE DONE $(date -u)"
