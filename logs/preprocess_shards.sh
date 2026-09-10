#!/usr/bin/env bash
# Preprocess each collection shard into its own flow cache AS IT LANDS, concurrently with the
# GPU collection that is still producing later shards.
#
# preprocess_flow_dataset.py runs at ~2.5 rows/s and uses ~5 of this box's 120 cores, so the
# full 29k-row collection is ~3.2h of CPU work. Run serially after collection it is 3.2h added
# to the critical path; run alongside the GPU work it is very nearly free. Shards are done ONE
# AT A TIME on purpose -- the disk is shared with the collectors (and the 9B agent's job), and
# parallel preprocessing would trade GPU throughput for CPU throughput.
#
# scripts/concat_flow_caches.py then merges the per-shard caches, with row-level verification.
set -uo pipefail
cd /home/shadeform/chain-flow
export PYTHONPATH=src
PY=.venv/bin/python

SHARDS="instruct-a instruct-b funccall multiturn toolcall instruct-c instruct-d funccall2 multiturn2 toolcall2"
mkdir -p data/flow_cache

done_all=0
while [ "$done_all" -eq 0 ]; do
  done_all=1
  for s in $SHARDS; do
    src="teacher_states/stage1-tr27b-$s"
    dst="data/flow_cache/_shard_tr27b_$s"
    # already cached?
    [ -f "$dst/metadata.json" ] && continue
    done_all=0
    # collected yet? dataset_info.json is written last by save_to_disk
    if [ -f "$src/dataset_info.json" ] || [ -f "$src/state.json" ]; then
      echo "==== PREPROCESS $s START $(date -u +%H:%M:%S) ===="
      $PY scripts/preprocess_flow_dataset.py --dataset-path "$src" --output-dir "$dst" \
          --draft-length 4 --hidden-dtype float16 --overwrite 2>&1 \
          | grep -vE "row/s\]$|examples/s\]$|it/s\]$"
      rc=$?
      echo "==== PREPROCESS $s DONE rc=$rc $(date -u +%H:%M:%S) df=$(df -h / | awk 'NR==2{print $4}') ===="
      [ "$rc" -ne 0 ] && echo "PREPROCESS $s FAILED"
    fi
  done
  [ "$done_all" -eq 0 ] && sleep 120
done
echo "PREPROCESS ALL DONE $(date -u) df=$(df -h / | awk 'NR==2{print $4}')"
