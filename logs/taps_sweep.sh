#!/bin/bash
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python; ED=/tmp/eval-4b-full
for DOM in math_reasoning writing; do
  echo "==== DOMAIN $DOM ===="
  for MN in 8 16 32 64; do
    echo "-- max_nodes=$MN --"
    $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.5-4B \
      --dataset_path data/flow_cache/deploy_$DOM --dtype float16 \
      --batch_size 128 --max_batches 5 --measure_tree --tree_top_b 8 --tree_max_nodes $MN 2>&1 \
      | grep -E "LIVE tree_accept" | sed "s/^/   [$DOM mn=$MN] /"
  done
done
echo "==== TAPS SWEEP DONE ===="
