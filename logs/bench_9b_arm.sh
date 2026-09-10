#!/bin/bash
# Waits for the 9B drafter to finish, then on GPU 5 (which 9B frees): RedHatAI+held-out accept
# rates -> vLLM speedup -> combined 4B+9B docs/deployment_results.md
set -o pipefail
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
VPY=vllm/.venv/bin/python
CKD9=out/flow/ckpts/tree-vae-joint-9b-1024-k8-l8
ED=/tmp/eval-9b
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for 9B final model.safetensors"
while [ ! -f "$CKD9/model.safetensors" ]; do sleep 120; done
sleep 30
st "9B done — starting benchmark on GPU5"

# --- synthesize 9B eval config + eval dir ---
$PY - <<'PYEOF'
import json
from dataclasses import asdict
from transformers import HfArgumentParser, TrainingArguments
from chain_flow.training.train_tree_flow import TreeModelArguments
from chain_flow.training.train_chunked_flow import TeacherDataArguments, FlowLossArguments
p=HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
m,d,l,t=p.parse_yaml_file(yaml_file="train_configs/recovered/joint_9b.yaml")
json.dump({"model_args":asdict(m),"data_args":asdict(d),"loss_args":asdict(l)},
          open("out/flow/ckpts/tree-vae-joint-9b-1024-k8-l8/chained_flow_tree_config.json","w"), indent=2)
print("9B config synthesized")
PYEOF
mkdir -p $ED
cp $CKD9/chained_flow_tree_config.json $ED/
ln -sf /home/shadeform/chain-flow/$CKD9/model.safetensors $ED/model.safetensors

# --- accept bench: collect -> cache -> measure per domain ---
> logs/deploy_eval_9b.log
for DOM in gsm8k_heldout math_reasoning HumanEval qa writing summarization; do
  { st "COLLECT 9B $DOM"
    $PY scripts/collect_teacher_states.py collect_configs/bench_9b/$DOM.yaml 2>&1 | grep -E "saved=|rows=" | tail -1
    TD=teacher_states/bench-9b-$DOM; [ "$DOM" = gsm8k_heldout ] && TD=teacher_states/heldout-9b-gsm8k-test
    [ -d "$TD" ] || { echo "SKIP $DOM (no teacher dir)"; continue; }
    st "CACHE 9B $DOM"
    $PY scripts/preprocess_flow_dataset.py --dataset-path $TD --output-dir data/flow_cache/deploy9b_$DOM \
        --draft-length 4 --hidden-dtype float16 --overwrite 2>&1 | grep -E "saved=|tokens="
    [ -f data/flow_cache/deploy9b_$DOM/hidden.pt ] || { echo "SKIP $DOM (no cache)"; continue; }
    st "ACCEPT 9B $DOM"
    $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.5-9B \
        --dataset_path data/flow_cache/deploy9b_$DOM --dtype float16 \
        --batch_size 96 --max_batches 6 --measure_tree 2>&1 | grep -E "LIVE tree_accept" | sed "s/^/  [$DOM] /"
  } >> logs/deploy_eval_9b.log 2>&1
done

# --- vLLM speedup (compiled-forward + full-cudagraph verify + cudagraph draft) ---
st "9B vLLM SPEEDUP" >> logs/deploy_eval_9b.log
CUDA_VISIBLE_DEVICES=5 $VPY vllm/compiled_forward_cgdraft_9b.py > logs/speedup_9b.log 2>&1

# --- write combined 4B+9B results md ---
$PY vllm/write_results_md.py
st "9B BENCH DONE -> docs/deployment_results.md"
