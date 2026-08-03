#!/bin/bash
# Waits for the Qwen3.5-27B drafter to finish (logs/train_q3527b_DONE.flag), then on GPU 5 (freed):
# RedHatAI+held-out accept -> best-effort vLLM speedup -> push selimaktas/Flow-Drafter-Qwen3.5-27B.
set -o pipefail
cd /home/shadeform/chained-flow
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
VPY=vllm/.venv/bin/python
CKD=out/flow/ckpts/tree-vae-joint-q3527b-1024-k8-l8
ED=/tmp/eval-q3527b
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for Qwen3.5-27B training (logs/train_q3527b_DONE.flag)"
while [ ! -f logs/train_q3527b_DONE.flag ]; do sleep 180; done
sleep 30
st "training done — starting q3527b benchmark on GPU5"

# synthesize eval config + eval dir
$PY - <<'PYEOF'
import json
from dataclasses import asdict
from transformers import HfArgumentParser, TrainingArguments
from chained_flow.training.train_tree_flow import TreeModelArguments
from chained_flow.training.train_chunked_flow import TeacherDataArguments, FlowLossArguments
p=HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
m,d,l,t=p.parse_yaml_file(yaml_file="train_configs/recovered/joint_q3527b.yaml")
json.dump({"model_args":asdict(m),"data_args":asdict(d),"loss_args":asdict(l)},
          open("out/flow/ckpts/tree-vae-joint-q3527b-1024-k8-l8/chained_flow_tree_config.json","w"), indent=2)
print("q3527b config synthesized")
PYEOF
mkdir -p $ED
cp $CKD/chained_flow_tree_config.json $ED/
ln -sf /home/shadeform/chained-flow/$CKD/model.safetensors $ED/model.safetensors

# accept bench: collect -> cache -> measure per domain
> logs/deploy_eval_q3527b.log
for DOM in gsm8k_heldout math_reasoning HumanEval qa writing summarization; do
  { st "COLLECT q3527b $DOM"
    $PY scripts/collect_teacher_states.py collect_configs/bench_q3527b/$DOM.yaml 2>&1 | grep -E "saved=|rows=" | tail -1
    TD=teacher_states/bench-q3527b-$DOM; [ "$DOM" = gsm8k_heldout ] && TD=teacher_states/heldout-q3527b-gsm8k-test
    [ -d "$TD" ] || { echo "SKIP $DOM (no teacher dir)"; continue; }
    st "CACHE q3527b $DOM"
    $PY scripts/preprocess_flow_dataset.py --dataset-path $TD --output-dir data/flow_cache/deployq3527b_$DOM \
        --draft-length 4 --hidden-dtype float16 --overwrite 2>&1 | grep -E "saved=|tokens="
    [ -f data/flow_cache/deployq3527b_$DOM/hidden.pt ] || { echo "SKIP $DOM (no cache)"; continue; }
    st "ACCEPT q3527b $DOM"
    $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.5-27B \
        --dataset_path data/flow_cache/deployq3527b_$DOM --dtype float16 \
        --batch_size 32 --max_batches 6 --measure_tree 2>&1 | grep -E "LIVE tree_accept" | sed "s/^/  [$DOM] /"
  } >> logs/deploy_eval_q3527b.log 2>&1
done

# best-effort vLLM speedup (non-fatal)
st "q3527b vLLM SPEEDUP (best-effort)" >> logs/deploy_eval_q3527b.log
CUDA_VISIBLE_DEVICES=5 $VPY vllm/compiled_forward_cgdraft_q3527b.py > logs/speedup_q3527b.log 2>&1 \
  || echo "q3527b vLLM speedup FAILED (non-fatal) — see logs/speedup_q3527b.log" >> logs/deploy_eval_q3527b.log

# push to HF
st "q3527b PUSH TO HF"
$PY scripts/push_q3527b.py 2>&1 | grep -E "DONE|uploading|staged" | tail -4

st "q3527b BENCH DONE"
touch logs/bench_q3527b_DONE.flag
