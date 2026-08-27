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

SIZE="${1:?usage: run_tr_v2.sh <4btr|9btr|tr27b> \"<gpus>\" [preset]}"
GPUS="${2:?usage: run_tr_v2.sh <size> \"<gpus>\" [preset]}"
PRESET="${3:-instruct}"

# torch.compile mode. max-autotune costs several minutes of warm-up per graph and then benchmarks
# real kernel variants; over a multi-hour run that is the right trade, which is why it is the
# default here and only `default` in the library. CF_FUSED_HEAD/CF_FUSED_CHUNK default on in the
# training module -- set CF_FUSED_HEAD=0 to fall back to the reference path.
export CF_COMPILE="${CF_COMPILE:-1}"
export CF_COMPILE_MODE="${CF_COMPILE_MODE:-max-autotune}"
export CF_FUSED_HEAD="${CF_FUSED_HEAD:-1}"
case "$SIZE" in
  4btr)  V1_CACHE=data/flow_cache/stage1_4btr_mix_k4  ;;
  9btr)  V1_CACHE=""                                  ;;  # v1 used a different prompt set: no reuse
  tr27b) V1_CACHE=data/flow_cache/stage1_tr27b_mix_k4 ;;
  *) echo "unknown size $SIZE (want 4btr|9btr|tr27b)"; exit 2 ;;
esac
CFG_DIR=collect_configs/stage1_${SIZE}_v2
TRAIN_CFG=train_configs/recovered/joint_${SIZE}_v2_${PRESET}.yaml
CACHE=$($PY -c "import yaml;print(yaml.safe_load(open('$TRAIN_CFG'))['dataset_path'])" 2>/dev/null)
LOG=logs/tr_v2_${SIZE}_${PRESET}
mkdir -p "$LOG"
rm -f "$LOG/.collect_failed"
say(){ echo "[$(date -u +%F' '%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }
freegb(){ df --output=avail -BG / | tail -1 | tr -dc '0-9'; }

[ -f "$TRAIN_CFG" ] || { say "ABORT: no $TRAIN_CFG -- run: $PY scripts/gen_tr_v2_configs.py --preset $PRESET"; exit 1; }
[ -d "$CFG_DIR" ] || { say "ABORT: no $CFG_DIR -- run: $PY scripts/gen_tr_v2_configs.py --preset $PRESET"; exit 1; }
[ -n "$CACHE" ]   || { say "ABORT: could not read dataset_path from $TRAIN_CFG"; exit 1; }

# Prompts. A fresh clone has no bench_data_tr_v2/ -- it is data, not source. Fetch it from the
# published dataset rather than requiring the upstream turkishdspark corpus to be present.
if [ ! -f bench_data_tr_v2/tr-instructurca.train.jsonl ]; then
  say "fetching prompts from selimaktas/turkish-flow-drafter-prompts"
  $PY scripts/fetch_tr_prompts.py || { say "ABORT: prompt fetch failed"; exit 1; }
fi
# v1 cache: an OPTIMISATION, not a requirement. When it exists, v2 is a strict prefix extension of
# it and only the tail needs collecting. It is 124 GB of DERIVED data and is published nowhere, so
# on a fresh machine it is simply absent -- in which case collect everything instead of aborting.
# Regenerating the configs is what actually changes the plan; without that the shards would still
# start at v1's offsets and the run would train on a cache missing its first 29,100 rows.
REUSE_ARG=""
if [ -n "$V1_CACHE" ] && [ ! -f "$V1_CACHE/metadata.json" ]; then
  say "v1 cache $V1_CACHE is ABSENT -- collecting every row instead of just the tail."
  say "  This is the normal path on a fresh clone. It costs more GPU time and more disk;"
  say "  nothing is lost, because the tail-only plan only ever SKIPPED rows that cache held."
  V1_CACHE=""
  REUSE_ARG="--no-reuse"
fi
say "regenerating configs (preset=$PRESET, init=continue${REUSE_ARG:+, full collection})"
$PY scripts/gen_tr_v2_configs.py --preset "$PRESET" $REUSE_ARG >/dev/null || {
  say "ABORT: config generation failed"; exit 1; }

# Rows this plan will actually collect, read from the configs rather than assumed -- the number
# differs by preset AND by whether the v1 cache was reusable, so a hard-coded gate is wrong in at
# least one of those cases.
PLANNED=$(CF_SIZE="$SIZE" $PY - <<'PYEOF'
import glob, os, yaml
d = f"collect_configs/stage1_{os.environ['CF_SIZE']}_v2"
print(sum(yaml.safe_load(open(f))["dataset_end"] - yaml.safe_load(open(f))["dataset_start"]
          for f in glob.glob(f"{d}/*.yaml")))
PYEOF
)
[ -n "$PLANNED" ] && [ "$PLANNED" -gt 0 ] 2>/dev/null || { say "ABORT: could not count planned rows from $CFG_DIR"; exit 1; }
# ~5.3 MB of teacher states per row at 27B, and the flow cache is about the same again. Teacher
# states are deleted per shard once preprocessed, so the peak is roughly one shard plus the cache.
NEED_GB=$(( PLANNED * 6 / 1000 + 60 ))
say "START size=$SIZE preset=$PRESET gpus=[$GPUS] free=$(freegb)G need>=${NEED_GB}G"
say "  collecting $PLANNED rows${V1_CACHE:+ (tail only; v1 cache supplies the rest)}"
say "  compile=$CF_COMPILE mode=$CF_COMPILE_MODE fused_head=$CF_FUSED_HEAD"
say "  train config: $TRAIN_CFG"
say "  cache:        $CACHE"
[ "$(freegb)" -lt "$NEED_GB" ] && { say "ABORT: only $(freegb)G free, need ${NEED_GB}G. Free space or shard the run."; exit 1; }

# ---------------------------------------------------------------- 0. restore the VAE weights
# `vae_dir` points at a LOCAL directory whose weights are byte-identical to the vae/ bundled in the
# published drafter, so the local copies were deleted to reclaim disk. They still have to be on disk
# before training starts, and finding that out at launch -- after waiting hours for GPUs -- is the
# expensive way to learn it. Restore is a no-op when the file is already there.
VDIR=$($PY -c "import yaml;print(yaml.safe_load(open('$TRAIN_CFG'))['vae_dir'])")
say "vae_dir: $VDIR"
$PY scripts/restore_vae.py --dir "$VDIR" 2>&1 | tee -a "$LOG/pipeline.log"
[ -f "$VDIR/model.safetensors" ] || { say "ABORT: could not restore the VAE at $VDIR"; exit 1; }

# ---------------------------------------------------------------- 1. wait for the GPUs
# A card is FREE only if nothing at all is resident.  Checking utilisation is not enough: an idle
# vLLM server sits at 0% while holding 88 GB, and starting a 27B collector on top of it OOMs.
for g in $GPUS; do
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g")" -lt 2000 ]; do
    say "waiting for GPU $g ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g") MiB resident)"; sleep 300
  done
done
say "GPUs [$GPUS] are clear"

# ---------------------------------------------------------------- 1b. invocation smoke test
# Every script this pipeline drives has its own CLI contract, and getting one wrong costs a full
# dispatch round to discover -- `--config` instead of a positional YAML killed 13 shards in three
# seconds each. This proves the collector ACCEPTS the exact form used below, without loading a
# model: a nonexistent config must fail on the missing FILE, not on argument parsing.
_probe=$($PY scripts/collect_teacher_states.py "$CFG_DIR/__does_not_exist__.yaml" 2>&1 | tail -5)
case "$_probe" in
  *"unrecognized arguments"*|*"invalid choice"*|*"the following arguments are required"*)
    say "ABORT: the collector rejects this invocation form, not the missing file:"
    printf '%s\n' "$_probe" | sed 's/^/      /' | tee -a "$LOG/pipeline.log"; exit 1 ;;
  *) say "collector accepts a positional YAML (probe failed on the file, as intended)" ;;
esac

# ---------------------------------------------------------------- 2. collect, round-robin by GPU
i=0; pids=""
for cfg in "$CFG_DIR"/*.yaml; do
  set -- $GPUS; shift $(( i % $# )); g=$1
  name=$(basename "$cfg" .yaml)
  ( say "[g$g] COLLECT $name START"
    # POSITIONAL, and it must be the ONLY argument: collect_teacher_states.py dispatches on
    # `len(sys.argv) == 2 and sys.argv[1].endswith(".yaml")`. A `--config` flag falls through to
    # the full argparse, which rejects it instantly -- which is what killed all 13 shards in
    # three seconds each.
    CUDA_VISIBLE_DEVICES=$g $PY scripts/collect_teacher_states.py "$cfg" \
        >>"$LOG/collect_g$g.log" 2>&1
    rc=$?; say "[g$g] COLLECT $name DONE rc=$rc free=$(freegb)G"
    if [ $rc -ne 0 ]; then
      touch "$LOG/.collect_failed"
      say "[g$g] !! $name FAILED rc=$rc -- last lines of $LOG/collect_g$g.log:"
      tail -n 15 "$LOG/collect_g$g.log" | sed 's/^/      /' | tee -a "$LOG/pipeline.log"
    fi
    # preprocess immediately: ~2.5 rows/s single-process, so hiding it behind the GPU work
    # is most of the saving.  A shard that failed to collect must NOT be preprocessed.
    if [ $rc -eq 0 ]; then
      $PY scripts/preprocess_flow_dataset.py --dataset-path "teacher_states/stage1-${SIZE}-v2-${name}" \
          --output-dir "data/flow_cache/_shard_${SIZE}v2_${name}" --draft-length 4 --overwrite \
          >>"$LOG/preprocess_g$g.log" 2>&1
      prc=$?
      say "[g$g] PREPROCESS $name rc=$prc free=$(freegb)G"
      # Teacher states are ~5.3 MB/row and the shard cache now represents them, so holding both
      # doubles peak disk for no benefit. Deleted ONLY on a clean preprocess with a real cache on
      # disk -- on failure they are kept so the shard can be reprocessed without recollecting.
      if [ $prc -eq 0 ] && [ -f "data/flow_cache/_shard_${SIZE}v2_${name}/metadata.json" ]; then
        rm -rf "teacher_states/stage1-${SIZE}-v2-${name}"
        say "[g$g] freed teacher states for $name  free=$(freegb)G"
      else
        say "[g$g] KEEPING teacher states for $name (preprocess rc=$prc) -- reprocess, do not recollect"
      fi
    fi ) &
  pids="$pids $!"; i=$((i+1))
  # keep at most one job per GPU in flight
  [ "$(jobs -rp | wc -l)" -ge "$(set -- $GPUS; echo $#)" ] && wait -n
  # Checked AFTER waiting on a slot so it sees the most recent completion. The first failure is
  # almost always systematic -- a bad invocation, a missing model, no disk -- so the remaining
  # shards fail identically and bury the one message worth reading.
  if [ -f "$LOG/.collect_failed" ]; then
    say "ABORT after the first failure -- $((i-1))/$(ls "$CFG_DIR"/*.yaml | wc -l) shards dispatched."
    say "  The tail above is the real error. Fix it and re-run; finished shards are skipped."
    wait; exit 1
  fi
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
# v1 kept ~86% of collected rows after the context+draft length filter; 75% is a floor that catches
# a lost shard without tripping on normal filtering. Derived from what this run actually planned,
# plus whatever the v1 cache contributes, so it stays correct across presets and reuse modes.
V1ROWS=0
[ -n "$V1_CACHE" ] && V1ROWS=$($PY -c "import json;print(json.load(open('$V1_CACHE/metadata.json'))['num_rows'])")
MIN=$(( (PLANNED + V1ROWS) * 75 / 100 ))
[ "$ROWS" -lt "$MIN" ] && { say "ABORT: cache has $ROWS rows, expected >=$MIN (planned $PLANNED + v1 $V1ROWS)"; exit 1; }

# ---------------------------------------------------------------- 5. train
NG=$(set -- $GPUS; echo $#)
say "TRAIN start ($NG GPUs)"
CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',') $PY -m torch.distributed.run --nproc_per_node=$NG \
    --master_port=29531 scripts/train_tree_flow.py "$TRAIN_CFG" \
    2>&1 | grep -viE "it/s\]$|examples/s\]$" >>"$LOG/train.log"
say "TRAIN rc=$? free=$(freegb)G"
say "NEXT: sweep the checkpoints with scripts/sweep_tr27b_checkpoints.py and pick with the TRO"
say "      both-corpora ladder in scripts/tr27b_results_json.py -- do NOT ship the last checkpoint"
say "      untested, and do NOT difference 200-window sweep numbers against 400-window results."
