#!/bin/bash
# Auto-triggered training pipeline for a model tag once its collection finishes.
# $1 = tag (4b|9b)   $2 = GPU index
# Waits for "COLLECT ALL DONE stage1_<tag>" in the collection log, then runs:
#   combine 5 sources -> drafter flow cache -> 3k VAE cache -> VAE -> joint-VAE drafter
set -o pipefail
TAG=$1; GPU=$2
cd /home/shadeform/chained-flow
export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0
PY=.venv/bin/python
CLOG=logs/collect_${TAG}.log
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for collection ($TAG) to finish"
while ! grep -q "COLLECT ALL DONE stage1_$TAG" "$CLOG" 2>/dev/null; do sleep 60; done
# guard: every source dir must exist
for x in gsm8k nemotron-math nemotron-stem alpaca-code dolly-chat; do
  [ -d "teacher_states/stage1-$TAG-$x" ] || { echo "FATAL missing source stage1-$TAG-$x"; exit 1; }
done
st "collection done — starting pipeline on GPU$GPU"

st "1/5 combine 5 sources -> mix"
$PY - <<PYEOF
from datasets import load_from_disk, concatenate_datasets
s=['gsm8k','nemotron-math','nemotron-stem','alpaca-code','dolly-chat']
ds=concatenate_datasets([load_from_disk(f'teacher_states/stage1-$TAG-{x}') for x in s]).shuffle(seed=0)
ds.save_to_disk('teacher_states/stage1-$TAG-mix5'); print('mix rows', len(ds))
PYEOF
[ $? -ne 0 ] && { echo FATAL combine; exit 1; }

st "2/5 preprocess drafter flow cache (draft-length 4, fp16)"
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-mix5 \
    --output-dir data/flow_cache/stage1_${TAG}_mix5_k4 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/stage1_${TAG}_mix5_k4/hidden.pt ] || { echo FATAL drafter-cache; exit 1; }

st "3/5 build 3k VAE subset + cache"
$PY -c "from datasets import load_from_disk; load_from_disk('teacher_states/stage1-$TAG-mix5').select(range(3000)).save_to_disk('teacher_states/stage1-$TAG-vae3k')" || { echo FATAL vae-subset; exit 1; }
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-vae3k \
    --output-dir data/flow_cache/vae${TAG}_3k_fp16 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/vae${TAG}_3k_fp16/hidden.pt ] || { echo FATAL vae-cache; exit 1; }

st "4/5 train VAE (transformer_hidden)"
$PY scripts/train_transformer_hidden_vae.py train_configs/recovered/vae_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
ls out/vae/ckpts/transformer-hidden-${TAG}-*/model.safetensors >/dev/null 2>&1 || { echo FATAL vae-ckpt; exit 1; }

st "5/5 train JOINT-VAE drafter (unfrozen, accept-optimized)"
$PY scripts/train_tree_flow.py train_configs/recovered/joint_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"

st "$TAG PIPELINE DONE"
