#!/usr/bin/env bash
# Waits for BOTH the sweep (GPU 3) and the OOD collection (GPU 4) to release their GPUs, then runs
# the final two-finalist matrix at per_domain 400 across both cards.
set -uo pipefail
cd /home/shadeform/chain-flow
until grep -q "SWEEP DONE" logs/eval_after_train.log 2>/dev/null; do sleep 60; done
echo "sweep released GPU 3 at $(date -u)"
until grep -q "OOD COLLECT DONE" logs/collect_tr_ood.log 2>/dev/null; do sleep 30; done
echo "OOD collection released GPU 4 at $(date -u)"
for d in teacher_states/ood-tr27b-turkish_alpaca_100 teacher_states/ood-tr27b-wikirag_tr_100; do
  [ -f "$d/dataset_info.json" ] || { echo "ABORT: missing OOD set $d"; exit 1; }
done
echo "both OOD sets present; starting matrix $(date -u)"
bash logs/matrix_tr27b.sh
