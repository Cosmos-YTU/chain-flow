#!/bin/bash
# Waits for the 27B drafter to finish, then on GPU 6 (which 27B frees): RedHatAI+held-out accept
# rates -> best-effort vLLM speedup -> combined 4B+9B+27B docs/deployment_results.md -> push to HF.
set -o pipefail
cd /home/shadeform/chain-flow
export CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
VPY=vllm/.venv/bin/python
CKD=out/flow/ckpts/tree-vae-joint-27b-1024-k8-l8
ED=/tmp/eval-27b
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for 27B final model.safetensors"
while [ ! -f "$CKD/model.safetensors" ]; do sleep 120; done
sleep 30
st "27B done — starting benchmark on GPU6"

# --- synthesize 27B eval config + eval dir ---
$PY - <<'PYEOF'
import json
from dataclasses import asdict
from transformers import HfArgumentParser, TrainingArguments
from chain_flow.training.train_tree_flow import TreeModelArguments
from chain_flow.training.train_chunked_flow import TeacherDataArguments, FlowLossArguments
p=HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
m,d,l,t=p.parse_yaml_file(yaml_file="train_configs/recovered/joint_27b.yaml")
json.dump({"model_args":asdict(m),"data_args":asdict(d),"loss_args":asdict(l)},
          open("out/flow/ckpts/tree-vae-joint-27b-1024-k8-l8/chained_flow_tree_config.json","w"), indent=2)
print("27B config synthesized")
PYEOF
mkdir -p $ED
cp $CKD/chained_flow_tree_config.json $ED/
ln -sf /home/shadeform/chain-flow/$CKD/model.safetensors $ED/model.safetensors

# --- accept bench: collect -> cache -> measure per domain ---
> logs/deploy_eval_27b.log
for DOM in gsm8k_heldout math_reasoning HumanEval qa writing summarization; do
  { st "COLLECT 27B $DOM"
    $PY scripts/collect_teacher_states.py collect_configs/bench_27b/$DOM.yaml 2>&1 | grep -E "saved=|rows=" | tail -1
    TD=teacher_states/bench-27b-$DOM; [ "$DOM" = gsm8k_heldout ] && TD=teacher_states/heldout-27b-gsm8k-test
    [ -d "$TD" ] || { echo "SKIP $DOM (no teacher dir)"; continue; }
    st "CACHE 27B $DOM"
    $PY scripts/preprocess_flow_dataset.py --dataset-path $TD --output-dir data/flow_cache/deploy27b_$DOM \
        --draft-length 4 --hidden-dtype float16 --overwrite 2>&1 | grep -E "saved=|tokens="
    [ -f data/flow_cache/deploy27b_$DOM/hidden.pt ] || { echo "SKIP $DOM (no cache)"; continue; }
    st "ACCEPT 27B $DOM"
    $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.6-27B \
        --dataset_path data/flow_cache/deploy27b_$DOM --dtype float16 \
        --batch_size 32 --max_batches 6 --measure_tree 2>&1 | grep -E "LIVE tree_accept" | sed "s/^/  [$DOM] /"
  } >> logs/deploy_eval_27b.log 2>&1
done

# --- vLLM speedup (best-effort: 27B is a VL backbone; non-fatal if it can't load) ---
st "27B vLLM SPEEDUP (best-effort)" >> logs/deploy_eval_27b.log
CUDA_VISIBLE_DEVICES=6 $VPY vllm/compiled_forward_cgdraft_27b.py > logs/speedup_27b.log 2>&1 \
  || echo "27B vLLM speedup FAILED (non-fatal) — see logs/speedup_27b.log" >> logs/deploy_eval_27b.log

# --- write combined 4B+9B+27B results md ---
$PY vllm/write_results_md.py

# --- push 27B to HF + refresh all three cards with the combined accept table ---
st "27B PUSH TO HF + refresh cards"
$PY scripts/push_27b_and_cards.py 2>&1 | grep -E "DONE|updated|FAIL" | tail -6

st "27B BENCH DONE -> docs/deployment_results.md + HF"
touch logs/bench_27b_DONE.flag
