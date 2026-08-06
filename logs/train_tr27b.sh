#!/usr/bin/env bash
# Turkish 27B flow drafter: concat per-shard caches -> warm-started train. GPUs 3 and 4.
#
# The teacher_states -> flow_cache conversion no longer happens here. logs/preprocess_shards.sh
# converts each shard as it lands, CONCURRENTLY with the GPU collection, because that conversion
# is ~3.2h of single-process CPU work that would otherwise sit serially behind ~6h of GPU work.
# All that is left is a concatenation of the per-shard caches, which is IO-bound.
#
# The intermediate `teacher_states/stage1-tr27b-mix` merge is also gone: it was a full 136 GB
# copy of data that the per-shard caches already represent, and nothing downstream reads it.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

CACHE=data/flow_cache/stage1_tr27b_mix_k4
say(){ echo "[$(date -u +%H:%M:%S)] $*"; }
say "START df=$(df -h / | awk 'NR==2{print $4}')"

# ---------------------------------------------------------------- 1. wait for caches AND the GPUs
# Two gates, not one. PREPROCESS ALL DONE only means the CPU-side conversion finished; the Turkish
# bench collection is still holding GPU 3 at that point, and starting a 2-GPU training run on top
# of it would put two 27B models on the same card.
until grep -q "PREPROCESS ALL DONE" logs/preprocess_shards.log 2>/dev/null; do sleep 60; done
say "all shard caches built df=$(df -h / | awk 'NR==2{print $4}')"
until grep -q "BENCH DONE" logs/bench_after_ext.log 2>/dev/null; do sleep 60; done
say "bench collection finished, GPUs are free"

# ---------------------------------------------------------------- 2. concat
if [ ! -f "$CACHE/metadata.json" ]; then
  say "concatenating shard caches -> $CACHE"
  $PY scripts/concat_flow_caches.py --out "$CACHE" \
      --shards 'data/flow_cache/_shard_tr27b_*' --verify 128 || { say "ABORT: concat failed"; exit 1; }
fi
ROWS=$($PY -c "import json;print(json.load(open('$CACHE/metadata.json'))['num_rows'])") || exit 1
TOK=$($PY -c "import json;print(json.load(open('$CACHE/metadata.json'))['total_tokens'])") || exit 1
say "cache rows=$ROWS tokens=$TOK df=$(df -h / | awk 'NR==2{print $4}')"
# 29,100 collected rows minus the few dropped for being shorter than context+draft.
[ "$ROWS" -lt 25000 ] && { say "ABORT: cache has only $ROWS rows"; exit 1; }

# concat verified row-by-row against its sources -> the per-shard caches are redundant
rm -rf data/flow_cache/_shard_tr27b_*
say "dropped per-shard caches df=$(df -h / | awk 'NR==2{print $4}')"

# ---------------------------------------------------------------- 3. train
say "training (warm start from v2) -- CHECK THE 'WARM START' sha256 LINE, it must read"
say "  31baf9ad6a417b15eba6c896b367277b0696c4933036242900cfa1a6dbc5b072"
CUDA_VISIBLE_DEVICES=3,4 $PY -m torch.distributed.run --nproc_per_node=2 --master_port=29533 \
    scripts/train_tree_flow.py train_configs/recovered/joint_tr27b.yaml 2>&1 \
    | grep -viE "it/s\]$|examples/s\]$"
say "DONE rc=$? df=$(df -h / | awk 'NR==2{print $4}')"
