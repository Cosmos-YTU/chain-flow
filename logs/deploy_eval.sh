#!/bin/bash
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
ED=/tmp/eval-4b-full   # has config + model.safetensors symlink
st(){ echo "======== [$(date -u +%H:%M:%S)] $1 ========"; }

run_domain(){  # $1=name $2=collect_config $3=teacher_dir
  local NAME=$1 CFG=$2 TD=$3
  st "COLLECT $NAME"
  $PY scripts/collect_teacher_states.py $CFG 2>&1 | grep -E "saved=|rows=" | tail -2
  [ -d "$TD" ] || { echo "SKIP $NAME (no teacher dir)"; return; }
  st "CACHE $NAME"
  $PY scripts/preprocess_flow_dataset.py --dataset-path $TD \
      --output-dir data/flow_cache/deploy_$NAME --draft-length 4 --hidden-dtype float16 --overwrite 2>&1 | grep -E "saved=|tokens="
  [ -f data/flow_cache/deploy_$NAME/hidden.pt ] || { echo "SKIP $NAME (no cache)"; return; }
  st "ACCEPT $NAME"
  $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.5-4B \
      --dataset_path data/flow_cache/deploy_$NAME --dtype float16 \
      --batch_size 128 --max_batches 6 --measure_tree 2>&1 | grep -E "LIVE tree_accept|CHAIN greedy" | sed "s/^/  [$NAME] /"
}

# held-out in-distribution
run_domain gsm8k_heldout collect_configs/stage1_4b/heldout_gsm8k_test.yaml teacher_states/heldout-4b-gsm8k-test
# OOD spread
for d in math_reasoning HumanEval qa writing summarization; do
  run_domain $d collect_configs/bench/$d.yaml teacher_states/bench-4b-$d
done
st "DEPLOY EVAL DONE"
