#!/bin/bash
# Waits for the 4bx drafter, then benches its accept on the 6 domains by REUSING the existing
# base-model caches (data/flow_cache/deploy_*) — no re-collection. Compares 4bx vs original 4B.
set -o pipefail
cd /home/shadeform/chained-flow
export CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
CKD=out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8
ED=/tmp/eval-4bx
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for 4bx drafter (logs/train_4bx_DONE.flag)"
while [ ! -f logs/train_4bx_DONE.flag ]; do sleep 120; done
sleep 20
st "4bx done — benching accept on reused caches (GPU5)"

# synthesize eval config + dir
$PY - <<'PYEOF'
import json
from dataclasses import asdict
from transformers import HfArgumentParser, TrainingArguments
from chained_flow.training.train_tree_flow import TreeModelArguments
from chained_flow.training.train_chunked_flow import TeacherDataArguments, FlowLossArguments
p=HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
m,d,l,t=p.parse_yaml_file(yaml_file="train_configs/recovered/joint_4bx.yaml")
json.dump({"model_args":asdict(m),"data_args":asdict(d),"loss_args":asdict(l)},
          open("out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8/chained_flow_tree_config.json","w"), indent=2)
print("4bx config synthesized")
PYEOF
mkdir -p $ED
cp $CKD/chained_flow_tree_config.json $ED/
ln -sf /home/shadeform/chained-flow/$CKD/model.safetensors $ED/model.safetensors

# eval each domain on the EXISTING base-model cache (domain -> cache dir)
> logs/deploy_eval_4bx.log
declare -A CACHE=( [gsm8k_heldout]=deploy_gsm8k_heldout [math_reasoning]=deploy_math_reasoning
                   [HumanEval]=deploy_HumanEval [qa]=deploy_qa [writing]=deploy_writing
                   [summarization]=deploy_summarization )
for DOM in gsm8k_heldout math_reasoning HumanEval qa writing summarization; do
  C=data/flow_cache/${CACHE[$DOM]}
  { st "ACCEPT 4bx $DOM"
    [ -f "$C/hidden.pt" ] || { echo "SKIP $DOM (no cache $C)"; continue; }
    $PY scripts/eval_tree_flow.py --flow_dir $ED --model_id Qwen/Qwen3.5-4B \
        --dataset_path $C --dtype float16 --batch_size 256 --max_batches 6 --measure_tree 2>&1 \
        | grep -E "LIVE tree_accept" | sed "s/^/  [$DOM] /"
  } >> logs/deploy_eval_4bx.log 2>&1
done

# comparison: 4bx vs original 4B + GOOD/NOTGOOD verdict (weak domains up AND mean not regressed)
st "4bx vs 4B accept comparison" | tee -a logs/deploy_eval_4bx.log
$PY - <<'PYEOF' | tee docs/accept_4bx_vs_4b.md
import re, os
def acc(p):
    d={}
    if os.path.exists(p):
        for l in open(p,errors="ignore"):
            m=re.search(r"\[(\w+)\] LIVE tree_accept=([0-9.]+)",l)
            if m: d[m.group(1)]=float(m.group(2))
    return d
old=acc("logs/deploy_eval.log"); new=acc("logs/deploy_eval_4bx.log")
order=["gsm8k_heldout","math_reasoning","HumanEval","writing","qa","summarization"]
weak=["writing","qa","summarization"]
print("# 4B accept — original v1 (5 tech sources) vs expanded v2 (10 diverse sources)\n")
print("| domain | v1 | v2 | Δ |")
print("|---|---|---|---|")
tot_o=tot_n=n=0
for d in order:
    if d in new and d in old:
        o,v=old[d],new[d]; tot_o+=o; tot_n+=v; n+=1
        print(f"| {d} | {o:.2f} | {v:.2f} | {v-o:+.2f} |")
mean_o=tot_o/n if n else 0; mean_n=tot_n/n if n else 0
wo=[old[d] for d in weak if d in old]; wn=[new[d] for d in weak if d in new]
weak_o=sum(wo)/len(wo) if wo else 0; weak_n=sum(wn)/len(wn) if wn else 0
if n: print(f"| **mean** | {mean_o:.2f} | {mean_n:.2f} | {mean_n-mean_o:+.2f} |")
print(f"\n- weak-domain mean (writing/qa/summ): v1 {weak_o:.2f} -> v2 {weak_n:.2f} ({weak_n-weak_o:+.2f})")
# GOOD if weak domains improved AND overall mean did not regress meaningfully (tol 0.10)
good = (weak_n > weak_o) and (mean_n >= mean_o - 0.10)
print(f"- verdict: {'GOOD (push v2)' if good else 'NOT GOOD (hold)'}")
open("/tmp/4bx_verdict","w").write("GOOD" if good else "NOTGOOD")
PYEOF

VERDICT=$(cat /tmp/4bx_verdict 2>/dev/null)
if [ "$VERDICT" = "GOOD" ]; then
  st "verdict GOOD -> pushing selimaktas/Flow-Drafter-4B-v2"
  $PY scripts/push_4bx_v2.py 2>&1 | grep -E "DONE|uploading|staged" | tail -3
else
  st "verdict NOT GOOD -> holding push (see docs/accept_4bx_vs_4b.md)"
fi
st "4bx BENCH DONE -> docs/accept_4bx_vs_4b.md"
touch logs/bench_4bx_DONE.flag
