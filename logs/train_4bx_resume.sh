#!/bin/bash
# RESUME 4bx: re-collect no_robots (fixed batch) then combine->caches->VAE->drafter
set -o pipefail
TAG=4bx
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }
st "RESUME: re-collect no_robots (gen48/hid16) on GPU5"
CUDA_VISIBLE_DEVICES=5 $PY scripts/collect_teacher_states.py collect_configs/stage1_4bx/no_robots.yaml 2>&1 | grep -E "saved=|rows=|dataset loaded|Error" | tail -3
[ -d teacher_states/stage1-4bx-no_robots ] || { echo "FATAL no_robots re-collect failed"; exit 1; }
st "no_robots re-collected — resuming pipeline"
export CUDA_VISIBLE_DEVICES=5

st "2/6 combine 5 reused technical + 5 new -> mix"
$PY - <<'PYEOF'
from datasets import load_from_disk, concatenate_datasets
tech=[f"teacher_states/stage1-4b-{x}" for x in ["gsm8k","nemotron-math","nemotron-stem","alpaca-code","dolly-chat"]]
new =[f"teacher_states/stage1-4bx-{x}" for x in ["ultrachat","no_robots","writing","summarization","translation"]]
parts=[]
for p in tech+new:
    d=load_from_disk(p); parts.append(d); print(f"  {p}: {len(d)} rows")
ds=concatenate_datasets(parts).shuffle(seed=0)
ds.save_to_disk("teacher_states/stage1-4bx-mix"); print("mix rows", len(ds))
PYEOF
[ $? -ne 0 ] && { echo FATAL combine; exit 1; }

st "3/6 preprocess drafter flow cache (draft-length 4, fp16)"
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-mix \
    --output-dir data/flow_cache/stage1_${TAG}_mix10_k4 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/stage1_${TAG}_mix10_k4/hidden.pt ] || { echo FATAL drafter-cache; exit 1; }

st "4/6 build 3k VAE subset + cache"
$PY -c "from datasets import load_from_disk; load_from_disk('teacher_states/stage1-$TAG-mix').select(range(3000)).save_to_disk('teacher_states/stage1-$TAG-vae3k')" || { echo FATAL vae-subset; exit 1; }
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-vae3k \
    --output-dir data/flow_cache/vae${TAG}_3k_fp16 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/vae${TAG}_3k_fp16/hidden.pt ] || { echo FATAL vae-cache; exit 1; }

st "5/6 train VAE (transformer_hidden, single GPU)"
$PY scripts/train_transformer_hidden_vae.py train_configs/recovered/vae_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
ls out/vae/ckpts/transformer-hidden-${TAG}-*/model.safetensors >/dev/null 2>&1 || { echo FATAL vae-ckpt; exit 1; }

st "6/6 train JOINT-VAE drafter — DDP on GPUs 5+6 (torchrun nproc=2)"
CUDA_VISIBLE_DEVICES=5,6 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29519 \
    scripts/train_tree_flow.py train_configs/recovered/joint_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
ls out/flow/ckpts/tree-vae-joint-${TAG}-640-k8-l8/model.safetensors >/dev/null 2>&1 \
  && st "$TAG DRAFTER DONE (model.safetensors written)" || echo "WARN: no final drafter model"

st "$TAG PIPELINE DONE"
touch logs/train_${TAG}_DONE.flag
