#!/bin/bash
# Drive RedHatAI/speculator_benchmarks PER DOMAIN against one Chained-Flow arm.
#
#   specbench_dom_run.sh <4b|9b|27b> <base|chain|tree> <gpu> <port>
#
# One server per arm; one guidellm benchmark per DOMAIN per repeat, with an exact
# Prometheus snapshot around each so acceptance is attributable to that domain.
# Batch 1 (synchronous), greedy, fixed 256 output tokens, async scheduling on for
# every arm including base (serve_cf.sh exports CF_ASYNC_SCHED=1 unconditionally).
set -uo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
GPU=${3:?gpu}
PORT=${4:?port}

ROOT=/home/shadeform/chain-flow
OUT=$ROOT/logs/specbench_dom/${SIZE}_${ARM}
DATA=$ROOT/logs/specbench_dom/data
GL=/home/shadeform/specbench-venv/bin/guidellm
REPEATS=${SPECBENCH_REPEATS:-2}

mkdir -p "$OUT"
SERVER_LOG=$OUT/vllm_server.log

echo "[dom] === $SIZE / $ARM on GPU $GPU port $PORT ==="
CF_GPU=$GPU nohup "$ROOT/vllm/serve_cf.sh" "$SIZE" "$ARM" "$PORT" "$SERVER_LOG" \
  > "$OUT/serve.launch" 2>&1 &
LAUNCH_PID=$!

for i in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[dom] SERVER FAILED for $SIZE/$ARM" >&2
  tail -40 "$SERVER_LOG" >&2
  exit 1
fi
echo "[dom] server ready"
grep -m1 "chain-flow] drafter=" "$SERVER_LOG" | tee "$OUT/drafter_hash.txt"

# Teardown is by PID, never pkill (a broad pkill has killed unrelated jobs here).
SERVE_PIDS=$(pgrep -f "vllm serve .*--port ${PORT}" | tr '\n' ' ')
echo "[dom] serve pids: $SERVE_PIDS"

run_file() {  # datafile tag maxreq
  local df=$1 tag=$2 maxreq=$3
  echo "[dom] --- $SIZE/$ARM $tag ---"
  curl -s "http://localhost:${PORT}/metrics" | grep -E "^vllm:spec_decode" \
    > "$OUT/metrics_${tag}.before" 2>/dev/null
  echo "CFMARK_START $tag $(date +%s.%N)" >> "$OUT/marks.txt"
  GUIDELLM__PREFERRED_ROUTE=chat_completions "$GL" benchmark \
    --target "http://localhost:${PORT}/v1" \
    --data "$df" \
    --profile synchronous \
    --max-requests "$maxreq" \
    --output-path "$OUT/gl_${tag}.json" \
    --disable-progress \
    --backend-args '{"extras": {"body": {"temperature":0}}}' \
    > "$OUT/gl_${tag}.log" 2>&1
  local rc=$?
  echo "CFMARK_END $tag $(date +%s.%N)" >> "$OUT/marks.txt"
  curl -s "http://localhost:${PORT}/metrics" | grep -E "^vllm:spec_decode" \
    > "$OUT/metrics_${tag}.after" 2>/dev/null
  echo "[dom] done $tag rc=$rc"
}

# Sanity gate: the legacy pooled 70-prompt subset, so a known number is reproduced
# before any new one is read.
if [ "${SPECBENCH_GATE:-0}" = "1" ]; then
  GDATA=$ROOT/logs/specbench/data/subset_fixed256.jsonl
  run_file "$GDATA" "gate_pooled_r0" "$(wc -l < "$GDATA")"   # warm
  run_file "$GDATA" "gate_pooled_r1" "$(wc -l < "$GDATA")"
fi

# Warm compile / cudagraph capture on a mixed file so repeat 1 is not the warmup.
[ "${SPECBENCH_GATE:-0}" = "1" ] || \
  run_file "$DATA/dom_WARMUP.jsonl" "WARMUP_r0" "$(wc -l < "$DATA/dom_WARMUP.jsonl")"

DOMAINS=$(ls "$DATA"/dom_*.jsonl | grep -v WARMUP | sed 's#.*/dom_##; s#\.jsonl##')
for rep in $(seq 1 "$REPEATS"); do
  for d in $DOMAINS; do
    run_file "$DATA/dom_${d}.jsonl" "${d}_r${rep}" "$(wc -l < "$DATA/dom_${d}.jsonl")"
  done
done

echo "[dom] stopping server"
for p in $SERVE_PIDS; do kill -TERM "$p" 2>/dev/null; done
kill -TERM "$LAUNCH_PID" 2>/dev/null
for i in $(seq 1 30); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || break
  sleep 2
done
for p in $SERVE_PIDS; do kill -KILL "$p" 2>/dev/null; done
sleep 5
echo "[dom] $SIZE/$ARM COMPLETE"
