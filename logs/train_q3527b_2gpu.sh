#!/bin/bash
# Qwen3.5-27B drafter — same recipe as the Qwen3.6-27B run, ONLY difference: 2 GPUs (5+6) for speed.
# Gated on the Qwen3.6-27B bench finishing (logs/bench_27b_DONE.flag) so both GPUs are free.
#   parallel teacher collect (GPU5 || GPU6, embarrassingly parallel) -> combine -> drafter+VAE caches
#   -> train VAE (1 GPU) -> train joint-VAE drafter (DDP across 5+6; frozen backbone => light comm)
set -o pipefail
TAG=q3527b
cd /home/shadeform/chain-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
st(){ echo "======== [$(date -u '+%m-%d %H:%M:%S')] $1 ========"; }

st "WAIT for Qwen3.6-27B bench to finish (logs/bench_27b_DONE.flag) so GPUs 5+6 are free"
while [ ! -f logs/bench_27b_DONE.flag ]; do sleep 120; done
sleep 30
st "GPUs 5+6 free — starting Qwen3.5-27B pipeline"

# ---- 1. parallel teacher collection: GPU5 gets 3 sources, GPU6 gets 2 (concurrent) ----
collect_one(){ # $1=gpu  $2..=sources
  local gpu=$1; shift
  for s in "$@"; do
    echo "==== [G$gpu] SRC $s START $(date -u +%H:%M:%S) ===="
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/collect_teacher_states.py collect_configs/stage1_$TAG/$s.yaml 2>&1 \
      | grep -E "saved=|rows=|Error" | tail -2
    echo "==== [G$gpu] SRC $s DONE rc=$? $(date -u +%H:%M:%S) ===="
  done
}
st "1/6 parallel collect (GPU5: gsm8k,nemotron_stem,dolly_chat | GPU6: nemotron_math,alpaca_code)"
collect_one 5 gsm8k_fp16 nemotron_stem dolly_chat > logs/collect_${TAG}_g5.log 2>&1 &
P5=$!
collect_one 6 nemotron_math alpaca_code > logs/collect_${TAG}_g6.log 2>&1 &
P6=$!
wait $P5; R5=$?
wait $P6; R6=$?
echo "collect rc: g5=$R5 g6=$R6"
# guard: all 5 source dirs must exist
for x in gsm8k nemotron-math nemotron-stem alpaca-code dolly-chat; do
  [ -d "teacher_states/stage1-$TAG-$x" ] || { echo "FATAL missing source stage1-$TAG-$x"; exit 1; }
done
st "collection done"

# ---- single-GPU prep steps run on GPU 5 ----
export CUDA_VISIBLE_DEVICES=5

st "2/6 combine 5 sources -> mix"
$PY - <<PYEOF
from datasets import load_from_disk, concatenate_datasets
s=['gsm8k','nemotron-math','nemotron-stem','alpaca-code','dolly-chat']
ds=concatenate_datasets([load_from_disk(f'teacher_states/stage1-$TAG-{x}') for x in s]).shuffle(seed=0)
ds.save_to_disk('teacher_states/stage1-$TAG-mix5'); print('mix rows', len(ds))
PYEOF
[ $? -ne 0 ] && { echo FATAL combine; exit 1; }

st "3/6 preprocess drafter flow cache (draft-length 4, fp16)"
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-mix5 \
    --output-dir data/flow_cache/stage1_${TAG}_mix5_k4 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/stage1_${TAG}_mix5_k4/hidden.pt ] || { echo FATAL drafter-cache; exit 1; }

st "4/6 build 3k VAE subset + cache"
$PY -c "from datasets import load_from_disk; load_from_disk('teacher_states/stage1-$TAG-mix5').select(range(3000)).save_to_disk('teacher_states/stage1-$TAG-vae3k')" || { echo FATAL vae-subset; exit 1; }
$PY scripts/preprocess_flow_dataset.py --dataset-path teacher_states/stage1-$TAG-vae3k \
    --output-dir data/flow_cache/vae${TAG}_3k_fp16 --draft-length 4 --hidden-dtype float16 --overwrite \
    2>&1 | grep -E "saved=|tokens="
[ -f data/flow_cache/vae${TAG}_3k_fp16/hidden.pt ] || { echo FATAL vae-cache; exit 1; }

st "5/6 train VAE (transformer_hidden, single GPU)"
$PY scripts/train_transformer_hidden_vae.py train_configs/recovered/vae_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
ls out/vae/ckpts/transformer-hidden-${TAG}-*/model.safetensors >/dev/null 2>&1 || { echo FATAL vae-ckpt; exit 1; }

# ---- 6. drafter training across BOTH GPUs (DDP). Frozen 27B backbone has no grads -> light all-reduce ----
st "6/6 train JOINT-VAE drafter — DDP on GPUs 5+6 (torchrun nproc=2)"
CUDA_VISIBLE_DEVICES=5,6 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29517 \
    scripts/train_tree_flow.py train_configs/recovered/joint_$TAG.yaml 2>&1 | grep -viE "it/s\]$|examples/s\]$"
ls out/flow/ckpts/tree-vae-joint-${TAG}-1024-k8-l8/model.safetensors >/dev/null 2>&1 \
  && st "$TAG DRAFTER DONE (model.safetensors written)" || echo "WARN: no final drafter model yet"

st "$TAG PIPELINE DONE"
touch logs/train_${TAG}_DONE.flag
