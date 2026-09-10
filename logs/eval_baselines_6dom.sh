#!/usr/bin/env bash
# Re-run the ENGLISH baseline arms over the full 6-domain held-out glob ([bh]*-q3527b-*, which
# adds held-out gsm8k). The first baseline pass used bench-* only (5 domains); before/after
# comparisons must be over an identical domain set, so this replaces those two arms.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
V2=out/flow/ckpts/tree-vae-joint-q3527bx-1024-k8-l8
run(){ echo ""; echo "############ $1"; CUDA_VISIBLE_DEVICES=4 $PY scripts/diff_plugin_vs_harness.py \
    --ckd "$V2" --model Qwen/Qwen3.5-27B --states 'teacher_states/[bh]*-q3527b-*' --shortlist "$2" \
    --plugin_tree 2>&1 | grep -vE "it/s\]$|examples/s\]$"; }
run "1. EN / v2 / shipped SL   (SANITY GATE, 6 domains)" out/flow/shortlist_shipped_bare.pt
run "2. EN / v2 / new SL       (shortlist swap on English, 6 domains)" out/flow/shortlist_q3527b_tr_bare.pt
echo ""; echo "BASELINES-6DOM DONE $(date -u)"
