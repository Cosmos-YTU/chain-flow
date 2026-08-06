#!/bin/bash
# Run the per-domain specbench matrix across exactly TWO GPUs (0 and 1), two arms at a
# time, each arm pinned to one device.  Jobs are taken from a queue by two workers.
#
#   specbench_dom_matrix.sh "<job> <job> ..."      job = "<size>:<arm>"
set -uo pipefail
ROOT=/home/shadeform/chained-flow
JOBS=(${1:?jobs})
QUEUE=$ROOT/logs/specbench_dom/queue.txt
printf '%s\n' "${JOBS[@]}" > "$QUEUE"
LOCK=$ROOT/logs/specbench_dom/queue.lock

worker() {  # gpu port
  local gpu=$1 port=$2
  while :; do
    local job=""
    exec 9>"$LOCK"; flock 9
    job=$(head -1 "$QUEUE")
    if [ -n "$job" ]; then tail -n +2 "$QUEUE" > "$QUEUE.tmp" && mv "$QUEUE.tmp" "$QUEUE"; fi
    flock -u 9; exec 9>&-
    [ -z "$job" ] && break
    local size=${job%%:*} arm=${job##*:}
    echo "[matrix] gpu$gpu -> $size/$arm $(date +%H:%M:%S)"
    bash "$ROOT/vllm/specbench_dom_run.sh" "$size" "$arm" "$gpu" "$port" \
      > "$ROOT/logs/specbench_dom/run_${size}_${arm}.log" 2>&1
    echo "[matrix] gpu$gpu done $size/$arm rc=$? $(date +%H:%M:%S)"
  done
  echo "[matrix] gpu$gpu queue empty, worker exit"
}

worker 0 8601 &
W0=$!
worker 1 8611 &
W1=$!
wait $W0 $W1
echo "[matrix] ALL DONE $(date +%H:%M:%S)"
