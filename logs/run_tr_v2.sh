#!/usr/bin/env bash
# Turkish v2 pipeline: collect the NEW rows -> preprocess each shard as it lands -> concat onto the
# EXISTING v1 cache -> warm-started train.  One size per invocation.
#
#   bash logs/run_tr_v2.sh tr27b "3 4"        # size, GPU list
#
# WHY IT CONCATS ONTO v1 RATHER THAN REBUILDING
# ---------------------------------------------
# `bench_data_tr_v2` is a strict PREFIX EXTENSION of `bench_data_tr` (same seed, holdout fills
# first, verified byte-for-byte with identical holdouts).  So v1's 29,100 rows are already correct
# and its finished cache is just another shard to concatenate.  Only the 27,599-row tail is
# collected.  9B is the exception -- its v1 came from a DIFFERENT prompt set
# (data/turkish_prompts, 14,746 rows, raw_prompt), so it has no reusable prefix and collects all
# 56,699 rows.  See scripts/gen_tr_v2_configs.py.
#
# DISK.  Concat writes a new store while its inputs still exist, so peak = inputs + output.  Each
# stage refuses to start below a floor rather than dying half-written and leaving a corrupt cache.
set -uo pipefail
cd /home/shadeform/chained-flow
export PYTHONPATH=src HF_HUB_ENABLE_HF_TRANSFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python

SIZE="${1:?usage: run_tr_v2.sh <4btr|9btr|tr27b> "<gpu list>"}"
GPUS="${2:?usage: run_tr_v2.sh <size> "<gpu list>"}"
case "$SIZE" in
  4btr)  V1_CACHE=data/flow_cache/stage1_4btr_mix_k4  ; NEED_GB=180 ;;
  9btr)  V1_CACHE=""                                  ; NEED_GB=320 ;;  # full recollect, no reuse
  tr27b) V1_CACHE=data/flow_cache/stage1_tr27b_mix_k4 ; NEED_GB=380 ;;
  *) echo "unknown size $SIZE (want 4btr|9btr|tr27b)"; exit 2 ;;
esac
CFG_DIR=collect_configs/stage1_${SIZE}_v2
CACHE=data/flow_cache/stage1_${SIZE}_v2_mix_k4
LOG=logs/tr_v2_${SIZE}
mkdir -p "$LOG"
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }
freegb(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

[ -d "$CFG_DIR" ] || { say "ABORT: no $CFG_DIR -- run scripts/gen_tr_v2_configs.py"; exit 1; }
[ -n "$V1_CACHE" ] && [ ! -f "$V1_CACHE/metadata.json" ] && { say "ABORT: v1 cache $V1_CACHE missing; it is an INPUT, not optional"; exit 1; }

say "START size=$SIZE gpus=[$GPUS] free=$(freegb)G need>=${NEED_GB}G"
[ "$(freegb)" -lt "$NEED_GB" ] && { say "ABORT: only $(freegb)G free, need ${NEED_GB}G. Free space or shard the run."; exit 1; }

# ---------------------------------------------------------------- 1. wait for the GPUs
# A card is FREE only if nothing at all is resident.  Checking utilisation is not enough: an idle
# vLLM server sits at 0% while holding 88 GB, and starting a 27B collector on top of it OOMs.
for g in $GPUS; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -lt 2000 ]; do
    say "waiting for GPU $g ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g") MiB resident)"; sleep 300
  done
done
say "GPUs [$GPUS] are clear"

# ---------------------------------------------------------------- 2. collect, round-robin by GPU
i=0; pids=""
for cfg in "$CFG_DIR"/*.yaml; do
  set -- $GPUS; shift $(( i % $# )); g=$1
  name=$(basename "$cfg" .yaml)
  ( say "[g$g] COLLECT $name START"
    CUDA_VISIBLE_DEVICES=$g $PY scripts/collect_teacher_states.py --config "$cfg" \
        >>"$LOG/collect_g$g.log" 2>&1
    rc=$?; say "[g$g] COLLECT $name DONE rc=$rc free=$(freegb)G"
    [ $rc -ne 0 ] && say "[g$g] !! $name FAILED -- pipeline will abort at the gate"
    # preprocess immediately: ~2.5 rows/s single-process, so hiding it behind the GPU work
    # is most of the saving.  A shard that failed to collect must NOT be preprocessed.
    if [ $rc -eq 0 ]; then
      $PY scripts/preprocess_flow_dataset.py --dataset-path "teacher_states/stage1-${SIZE}-v2-${name}" \
          --output-dir "data/flow_cache/_shard_${SIZE}v2_${name}" --draft-length 4 --overwrite \
          >>"$LOG/preprocess_g$g.log" 2>&1
      say "[g$g] PREPROCESS $name rc=$? free=$(freegb)G"
    fi ) &
  pids="$pids $!"; i=$((i+1))
  # keep at most one job per GPU in flight
  [ "$(jobs -rp | wc -l)" -ge "$(set -- $GPUS; echo $#)" ] && wait -n
done
wait $pids
say "collection+preprocess complete free=$(freegb)G"

# ---------------------------------------------------------------- 3. gate: every shard must exist
missing=0
for cfg in "$CFG_DIR"/*.yaml; do
  n=$(basename "$cfg" .yaml)
  [ -f "data/flow_cache/_shard_${SIZE}v2_${n}/metadata.json" ] || { say "MISSING shard cache: $n"; missing=1; }
done
[ $missing -eq 1 ] && { say "ABORT: shards missing. Fix and re-run -- finished shards are skipped by --overwrite guards."; exit 1; }

# ---------------------------------------------------------------- 4. concat (v1 cache first)
if [ ! -f "$CACHE/metadata.json" ]; then
  [ "$(freegb)" -lt 250 ] && { say "ABORT: $(freegb)G free, concat needs headroom for inputs+output"; exit 1; }
  say "concat -> $CACHE"
  $PY scripts/concat_flow_caches.py --out "$CACHE" \
      ${V1_CACHE:+--shards "$V1_CACHE"} --shards "data/flow_cache/_shard_${SIZE}v2_*" --verify 128 \
      >>"$LOG/concat.log" 2>&1 || { say "ABORT: concat failed, see $LOG/concat.log"; exit 1; }
fi
ROWS=$($PY -c "import json;print(json.load(open('$CACHE/metadata.json'))['num_rows'])") || exit 1
say "cache rows=$ROWS free=$(freegb)G"
# v1 27B kept 25k of 29.1k collected rows after the context+draft length filter (~86%).
MIN=$([ "$SIZE" = "9btr" ] && echo 45000 || echo 48000)
[ "$ROWS" -lt "$MIN" ] && { say "ABORT: cache has only $ROWS rows, expected >=$MIN"; exit 1; }

# ---------------------------------------------------------------- 5. train
NG=$(set -- $GPUS; echo $#)
say "TRAIN start ($NG GPUs)"
CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',') $PY -m torch.distributed.run --nproc_per_node=$NG \
    --master_port=29531 scripts/train_tree_flow.py train_configs/recovered/joint_${SIZE}_v2.yaml \
    2>&1 | grep -viE "it/s\]$|examples/s\]$" >>"$LOG/train.log"
say "TRAIN rc=$? free=$(freegb)G"
say "NEXT: sweep the checkpoints with scripts/sweep_tr27b_checkpoints.py and pick with the TRO"
say "      both-corpora ladder in scripts/tr27b_results_json.py -- do NOT ship the last checkpoint"
say "      untested, and do NOT difference 200-window sweep numbers against 400-window results."
