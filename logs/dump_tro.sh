#!/usr/bin/env bash
# Per-window (row, accepted_len) dumps for the plugin arm on the TRO corpora, one per ladder
# candidate. Feeds the PAIRED bootstrap that derives the selection floor.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CK=out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8
SL=out/flow/shortlist_q3527b_tr_bare.pt
run(){ # gpu step
  local d=$CK/checkpoint-$2
  [ -f "$d/chained_flow_tree_config.json" ] || cp "$CK/chained_flow_tree_config.json" "$d/"
  echo "--- dump tr$2 on GPU $1 $(date -u +%H:%M:%S) ---"
  CUDA_VISIBLE_DEVICES=$1 $PY scripts/diff_plugin_vs_harness.py --ckd "$d" \
    --model Qwen/Qwen3.5-27B --states 'teacher_states/ood-tr27b-*' --shortlist "$SL" \
    --per_domain 400 --dump-windows out/flow/tro_dumps/tr$2.json 2>&1 | tail -3
}
( for s in 300 600 900 1500; do run 3 $s; done; echo "DUMP-A DONE" ) > logs/dump_tro_a.log 2>&1 &
A=$!
( for s in 1800 2100 2400 2553; do run 4 $s; done; echo "DUMP-B DONE" ) > logs/dump_tro_b.log 2>&1 &
B=$!
wait $A; wait $B
echo "TRO DUMPS DONE $(date -u)"
