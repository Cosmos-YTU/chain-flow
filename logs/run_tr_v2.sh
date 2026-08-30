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
NGPU=$(set -- $GPUS; echo $#)          # defined HERE: config generation needs it long before the
                                       # heartbeat section where it used to live

# torch.compile mode. max-autotune costs several minutes of warm-up per graph and then benchmarks
# real kernel variants; over a multi-hour run that is the right trade, which is why it is the
# default here and only `default` in the library. CF_FUSED_HEAD/CF_FUSED_CHUNK default on in the
# training module -- set CF_FUSED_HEAD=0 to fall back to the reference path.
export CF_COMPILE="${CF_COMPILE:-1}"
# max-autotune-no-cudagraphs, NOT max-autotune. `max-autotune` turns on CUDA graphs, which recycle
# output buffers between runs -- and the fused head is a custom autograd Function that saves tensors
# produced by the compiled forward and reads them again in the backward, by which point the graph
# has re-run and overwritten them ("accessing tensor output of CUDAGraphs that has been overwritten
# by a subsequent run"). The kernel autotuning, which is the actual win here, is unaffected; CUDA
# graphs mainly remove launch overhead, and at a 512-window microbatch that is already amortised.
export CF_COMPILE_MODE="${CF_COMPILE_MODE:-max-autotune-no-cudagraphs}"
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
rm -f "$LOG/.collect_failed" "$LOG/.collect_done"
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
# If the merged cache is already built, none of the collection machinery applies: not the v1-cache
# question, not config regeneration, not the shards. Checking this FIRST matters because the
# v1-cache branch below rewrites the configs for a full 45k-row collection, and it fires whenever
# the v1 cache is gone -- including when it is gone precisely BECAUSE it was merged into $CACHE and
# then deleted to reclaim disk. That is how a finished dataset turned into a 3-hour recollection.
CACHE_READY=0
if [ -f "$CACHE/metadata.json" ]; then
  CACHE_READY=1
  say "cache already built: $CACHE"
  $PY -c "
import json; m=json.load(open('$CACHE/metadata.json'))
print(f\"  rows={m['num_rows']:,} tokens={m['total_tokens']:,} dtype={m['hidden_dtype']}\")" \
    | tee -a "$LOG/pipeline.log"
  say "skipping collection and concat -- going straight to training"
fi

# v1 cache: an OPTIMISATION, not a requirement. When it exists, v2 is a strict prefix extension of
# it and only the tail needs collecting. It is 124 GB of DERIVED data and is published nowhere, so
# on a fresh machine it is simply absent -- in which case collect everything instead of aborting.
# Regenerating the configs is what actually changes the plan; without that the shards would still
# start at v1's offsets and the run would train on a cache missing its first 29,100 rows.
REUSE_ARG=""
if [ "$CACHE_READY" = 0 ] && [ -n "$V1_CACHE" ] && [ ! -f "$V1_CACHE/metadata.json" ]; then
  say "v1 cache $V1_CACHE is ABSENT -- collecting every row instead of just the tail."
  say "  This is the normal path on a fresh clone. It costs more GPU time and more disk;"
  say "  nothing is lost, because the tail-only plan only ever SKIPPED rows that cache held."
  V1_CACHE=""
  REUSE_ARG="--no-reuse"
fi
say "regenerating configs (preset=$PRESET, init=continue${REUSE_ARG:+, full collection})"
# --gpus sets gradient_accumulation_steps so the effective batch stays at v1's value regardless of
# how many cards this run uses. Without it a 4-GPU run would silently double the effective batch,
# which at a constant LR is a different experiment, not a speed-up.
$PY scripts/gen_tr_v2_configs.py --preset "$PRESET" --gpus "$NGPU" $REUSE_ARG >/dev/null || {
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
# SINGLE RUN PER SIZE. Ctrl-C stops the script, but collectors already running may take a while to
# die -- and starting a second run immediately puts two processes in the same
# teacher_states/stage1-<size>-v2-<shard> directory, which corrupts both with no error from either.
LOCK="$LOG/.run.lock"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  say "ABORT: run already active (pid $(cat "$LOCK")). Two collectors would write the same"
  say "  teacher_states directory and corrupt each other. Wait for it, or: kill $(cat "$LOCK")"
  exit 1
fi
[ -f "$LOCK" ] && say "clearing a stale lock (pid $(cat "$LOCK") is gone)"
echo $$ > "$LOCK"
# ONE exit trap for everything. A second `trap ... EXIT` REPLACES the first rather than adding to
# it, so the lock cleanup and the heartbeat kill cannot be registered separately.
cleanup() { [ -n "${HEARTBEAT_PID:-}" ] && kill "$HEARTBEAT_PID" 2>/dev/null; rm -f "$LOCK"; }
trap cleanup EXIT

# Orphaned collectors from a previous Ctrl-C would fight this one for the GPUs and the output dirs.
_orphans=$(pgrep -f "collect_teacher_states.py .*stage1_${SIZE}_v2" | grep -v "^$$\$" | tr '\n' ' ')
if [ -n "$_orphans" ]; then
  say "ABORT: collectors from a previous run are still alive: $_orphans"
  say "  They hold the GPUs and would write the same shard dirs. Stop them first:  kill $_orphans"
  exit 1
fi

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

# ---------------------------------------------------------------- 1c. progress heartbeat
# tqdm writes to stderr, which is redirected into the per-GPU collect log, so the terminal shows
# nothing between START and DONE -- on a 20-hour collection that is indistinguishable from a hang.
# This pulls the most recent progress line out of each log every few minutes. tqdm separates
# updates with \r rather than \n, so the log is one enormous line until it is split.
HEARTBEAT_SEC="${CF_HEARTBEAT_SEC:-300}"
(
  while true; do
    sleep "$HEARTBEAT_SEC"
    [ -f "$LOG/.collect_done" ] && exit 0
    for g in $GPUS; do
      [ -f "$LOG/collect_g$g.log" ] || continue
      # Only report if the log is still being written. A dead collector leaves its last progress
      # line behind, and repeating it every 5 minutes reads exactly like a running job.
      now=$(date +%s); mtime=$(stat -c %Y "$LOG/collect_g$g.log" 2>/dev/null || echo 0)
      if [ $((now - mtime)) -gt $((HEARTBEAT_SEC * 2)) ]; then
        say "  [g$g] no output for $((now - mtime))s -- process is not writing"
        continue
      fi
      line=$(tr '\r' '\n' < "$LOG/collect_g$g.log" | grep -aE "[0-9]+%\|" | tail -1 | cut -c1-100)
      [ -n "$line" ] && say "  [g$g] $line"
    done
  done
) & HEARTBEAT_PID=$!

if [ "$CACHE_READY" = 0 ]; then
  # ---------------------------------------------------------------- 2. collect, round-robin by GPU
  i=0; pids=""
  # GPU slots are LOCK FILES, not a job count. Two things were wrong with counting jobs:
  #   1. the GPU came from the shard INDEX (i % nGPU), so the next shard could be handed to a card
  #      that was still busy while the other sat idle;
  #   2. the count included the whole subshell -- collection AND the ~10 minutes of CPU-only
  #      preprocessing that follows it -- so a preprocessing job held a GPU slot while using no GPU.
  # A slot is now released the moment collection ends, and preprocessing continues outside it.
  rm -f "$LOG"/.gpu_*.busy
  claim_gpu() {                       # echoes the first free GPU, waiting until one is
    while :; do
      for _g in $GPUS; do
        if ( set -o noclobber; echo $$ > "$LOG/.gpu_${_g}.busy" ) 2>/dev/null; then echo "$_g"; return; fi
      done
      sleep 5
    done
  }

  for cfg in "$CFG_DIR"/*.yaml; do
    name=$(basename "$cfg" .yaml)
    # RESUME. A shard is done when its flow cache exists -- that is the artifact the concat consumes,
    # and it is only written after a clean preprocess. Skipping here is what makes it safe to stop a
    # run and restart it with different batch sizes: everything already collected is kept.
    # (`--overwrite` on the preprocess step overwrites, it does not skip; it is not a resume guard.)
    if [ -f "data/flow_cache/_shard_${SIZE}v2_${name}/metadata.json" ]; then
      say "[skip] $name already has a shard cache"
      continue
    fi
    # PHASE-1 REUSE. collect_teacher_states.py always writes the generated answers to
    # teacher_states/_tmp_<output_dir_name>, but only AFTER generation finishes -- and it never reads
    # them back unless `answer_dataset_path` says to. Generation is the expensive phase (~24 min for
    # a 4000-row shard) and phase 2 is the one that OOMs, so without this every phase-2 retry redoes
    # all of it. `state.json` is the completeness marker: save_to_disk writes it last, so its presence
    # means generation ran to the end.
    g=$(claim_gpu)
    tmpd="teacher_states/_tmp_stage1-${SIZE}-v2-${name}"
    shard_cfg="$cfg"
    if [ -f "$tmpd/state.json" ]; then
      # The tmp dataset holds ONLY this shard's rows, indexed from 0. The shard config's
      # dataset_start/end address the SOURCE prompt file, and the resume path re-applies them to the
      # answer dataset -- so leaving them in slices [2000:3000] out of a 1000-row set, yields nothing,
      # and phase 2 then builds a dataset from an empty list. Reset the range to the whole file.
      ntmp=$($PY -c "
  from datasets import load_from_disk; print(len(load_from_disk('$tmpd')))" 2>/dev/null || echo 0)
      want=$($PY -c "
  import yaml; d=yaml.safe_load(open('$cfg')); print(d['dataset_end']-d['dataset_start'])")
      if [ "$ntmp" != "$want" ]; then
        say "[g$g] tmp answers for $name have $ntmp rows, shard wants $want -- regenerating instead"
      else
        shard_cfg="$LOG/resume_${name}.yaml"
        grep -vE '^(answer_dataset_path|dataset_start|dataset_end):' "$cfg" > "$shard_cfg"
        printf 'answer_dataset_path: %s\ndataset_start: 0\ndataset_end: %s\n' "$tmpd" "$ntmp" >> "$shard_cfg"
        say "[g$g] REUSING phase-1 answers for $name ($ntmp rows) -- skipping generation"
      fi
    fi
    ( say "[g$g] COLLECT $name START"
      # POSITIONAL, and it must be the ONLY argument: collect_teacher_states.py dispatches on
      # `len(sys.argv) == 2 and sys.argv[1].endswith(".yaml")`. A `--config` flag falls through to
      # the full argparse, which rejects it instantly -- which is what killed all 13 shards in
      # three seconds each.
      CUDA_VISIBLE_DEVICES=$g $PY scripts/collect_teacher_states.py "$shard_cfg" \
          >>"$LOG/collect_g$g.log" 2>&1
      rc=$?
      if [ $rc -eq 0 ]; then
        say "[g$g] COLLECT $name DONE free=$(freegb)G"
      elif [ $rc -eq 130 ] || [ $rc -eq 2 ]; then
        # SIGINT. Not a failure of the shard -- somebody stopped the run. Saying "DONE rc=1" here
        # made an interrupted shard read exactly like one that ran and failed.
        touch "$LOG/.collect_failed"
        say "[g$g] COLLECT $name INTERRUPTED (rc=$rc) -- stopped by signal, not by an error"
      else
        touch "$LOG/.collect_failed"
        say "[g$g] COLLECT $name FAILED rc=$rc -- last lines of $LOG/collect_g$g.log:"
        tail -n 15 "$LOG/collect_g$g.log" | sed 's/^/      /' | tee -a "$LOG/pipeline.log"
      fi
      # Release the GPU HERE. Everything below is CPU-only; holding the card through it left the
      # other GPU idle for roughly a third of the run.
      rm -f "$LOG/.gpu_${g}.busy"
      # preprocess immediately: ~2.5 rows/s single-process, so hiding it behind the GPU work
      # is most of the saving.  A shard that failed to collect must NOT be preprocessed.
      if [ $rc -eq 0 ]; then
        $PY scripts/preprocess_flow_dataset.py --dataset-path "teacher_states/stage1-${SIZE}-v2-${name}" \
            --output-dir "data/flow_cache/_shard_${SIZE}v2_${name}" --draft-length 4 \
            --hidden-dtype float16 --overwrite \
            >>"$LOG/preprocess_g$g.log" 2>&1
        prc=$?
        say "[g$g] PREPROCESS $name rc=$prc free=$(freegb)G"
        # Teacher states are ~5.3 MB/row and the shard cache now represents them, so holding both
        # doubles peak disk for no benefit. Deleted ONLY on a clean preprocess with a real cache on
        # disk -- on failure they are kept so the shard can be reprocessed without recollecting.
        if [ $prc -eq 0 ] && [ -f "data/flow_cache/_shard_${SIZE}v2_${name}/metadata.json" ]; then
          rm -rf "teacher_states/stage1-${SIZE}-v2-${name}" "teacher_states/_tmp_stage1-${SIZE}-v2-${name}"
          say "[g$g] freed teacher states + phase-1 answers for $name  free=$(freegb)G"
        else
          say "[g$g] KEEPING teacher states for $name (preprocess rc=$prc) -- reprocess, do not recollect"
        fi
      fi ) &
    pids="$pids $!"; i=$((i+1))
    # No job-count cap: claim_gpu already blocks until a card frees, and it is the GPU that is
    # scarce -- CPU preprocessing may pile up a little behind it, which is fine and is the point.
    # Checked AFTER waiting on a slot so it sees the most recent completion. The first failure is
    # almost always systematic -- a bad invocation, a missing model, no disk -- so the remaining
    # shards fail identically and bury the one message worth reading.
    if [ -f "$LOG/.collect_failed" ]; then
      say "ABORT after the first failure -- $((i-1))/$(ls "$CFG_DIR"/*.yaml | wc -l) shards dispatched."
      say "  The tail above is the real error. Fix it and re-run; finished shards are skipped."
      # Kill the heartbeat FIRST. A bare `wait` includes it, and it loops until .collect_done is
      # written -- which an aborting run never does -- so the script would hang here forever,
      # printing stale progress from logs whose processes had already died.
      touch "$LOG/.collect_done"; kill "$HEARTBEAT_PID" 2>/dev/null
      for _p in $pids; do wait "$_p" 2>/dev/null; done
      exit 1
    fi
  done
  touch "$LOG/.collect_done"; kill "$HEARTBEAT_PID" 2>/dev/null
  for _p in $pids; do wait "$_p" 2>/dev/null; done; kill $HEARTBEAT_PID 2>/dev/null
  say "collection+preprocess complete free=$(freegb)G"
  say "  shard caches present: $(ls -d data/flow_cache/_shard_${SIZE}v2_* 2>/dev/null | wc -l)/$(ls "$CFG_DIR"/*.yaml | wc -l)"

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
fi

# ---------------------------------------------------------------- 5. train
# Rotate: train.log is appended, so a previous run's traceback stays at the tail and reads as if
# it belonged to this one -- which cost a round of confusion diagnosing a run that was 15 seconds
# old against an error from ten minutes earlier.
[ -f "$LOG/train.log" ] && mv "$LOG/train.log" "$LOG/train.$(date -u +%Y%m%d-%H%M%S).log"
say "TRAIN start ($NGPU GPUs)"
CUDA_VISIBLE_DEVICES=$(echo $GPUS | tr ' ' ',') $PY -m torch.distributed.run --nproc_per_node=$NGPU \
    --master_port=29531 scripts/train_tree_flow.py "$TRAIN_CFG" \
    2>&1 | grep -viE "it/s\]$|examples/s\]$" >>"$LOG/train.log"
say "TRAIN rc=$? free=$(freegb)G"
say "NEXT: sweep the checkpoints with scripts/sweep_tr27b_checkpoints.py and pick with the TRO"
say "      both-corpora ladder in scripts/tr27b_results_json.py -- do NOT ship the last checkpoint"
say "      untested, and do NOT difference 200-window sweep numbers against 400-window results."
