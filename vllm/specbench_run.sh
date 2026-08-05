#!/bin/bash
# Drive the RedHat AI `speculator_benchmarks` prompt set against one Chained-Flow arm.
#
#   specbench_run.sh <4b|9b|27b> <base|chain|tree> <gpu> <port>
#
# Starts ONE server for the arm, runs every (config x profile x repeat) against it,
# then stops it.  Server stays up across repeats so model load is not re-paid.
set -uo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
GPU=${3:?gpu}
PORT=${4:?port}

ROOT=/home/shadeform/chained-flow
OUT=$ROOT/logs/specbench/${SIZE}_${ARM}
DATA=$ROOT/logs/specbench/data
GL=/home/shadeform/specbench-venv/bin/guidellm
REPEATS=${SPECBENCH_REPEATS:-2}

mkdir -p "$OUT"
SERVER_LOG=$OUT/vllm_server.log
# One pass over the subset; guidellm would otherwise cycle it indefinitely.
NPROMPT=$(wc -l < "$DATA/subset_fixed256.jsonl")

echo "[specbench] === $SIZE / $ARM on GPU $GPU port $PORT ==="
CF_GPU=$GPU nohup "$ROOT/vllm/serve_cf.sh" "$SIZE" "$ARM" "$PORT" "$SERVER_LOG" \
  > "$OUT/serve.launch" 2>&1 &
LAUNCH_PID=$!

# Wait for readiness (27B cold-compiles for several minutes).
for i in $(seq 1 120); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  sleep 10
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[specbench] SERVER FAILED for $SIZE/$ARM" >&2
  tail -30 "$SERVER_LOG" >&2
  exit 1
fi
echo "[specbench] server ready"

# EngineCore is a child of the launcher; record both so teardown is by PID, never pkill.
SERVE_PIDS=$(pgrep -f "vllm serve .*--port ${PORT}" | tr '\n' ' ')
echo "[specbench] serve pids: $SERVE_PIDS"

run_one() {  # cfg profile rate repeat body [maxreq]
  local cfg=$1 profile=$2 rate=$3 rep=$4 body=$5 maxreq=${6:-}
  local tag="${cfg}_${profile}${rate:+$rate}_r${rep}"
  local rargs=(); [ -n "$rate" ] && rargs=(--rate "$rate")
  # --max-requests is MANDATORY. guidellm's default --data-samples -1 makes the request
  # loader report "inf unique requests" -- it recycles the file forever -- so without an
  # explicit stopping condition a synchronous benchmark never terminates.
  rargs+=(--max-requests "${maxreq:-$NPROMPT}")
  echo "[specbench] --- $SIZE/$ARM $tag ---"
  # Marker into the server log so acceptance can be attributed to this window, plus an
  # EXACT snapshot of vLLM's own cumulative spec counters. The delta of those counters
  # gives 1 + accepted/drafts with no window-attribution error, which is exactly the
  # acceptance_length formula in speculators/tests/e2e/run_vllm.py.
  curl -s "http://localhost:${PORT}/v1/models" >/dev/null
  curl -s "http://localhost:${PORT}/metrics" | grep -E "^vllm:spec_decode" \
    > "$OUT/metrics_${tag}.before" 2>/dev/null
  echo "CFMARK_START $tag $(date +%s.%N)" >> "$OUT/marks.txt"
  GUIDELLM__PREFERRED_ROUTE=chat_completions "$GL" benchmark \
    --target "http://localhost:${PORT}/v1" \
    --data "$DATA/subset_${cfg}.jsonl" \
    --profile "$profile" "${rargs[@]}" \
    --output-path "$OUT/gl_${tag}.json" \
    --disable-progress \
    --backend-args "{\"extras\": {\"body\": $body}}" \
    > "$OUT/gl_${tag}.log" 2>&1
  local rc=$?
  echo "CFMARK_END $tag $(date +%s.%N)" >> "$OUT/marks.txt"
  curl -s "http://localhost:${PORT}/metrics" | grep -E "^vllm:spec_decode" \
    > "$OUT/metrics_${tag}.after" 2>/dev/null
  echo "[specbench] done $tag rc=$rc"
}

# Warm the server (compile + cudagraph capture) so repeat 1 is not the warmup.
echo "[specbench] warmup"
run_one fixed256 synchronous "" 0 '{"temperature":0}'

for rep in $(seq 1 "$REPEATS"); do
  # PRIMARY: greedy, exactly 256 output tokens per request (ignore_eos), batch 1.
  run_one fixed256 synchronous "" "$rep" '{"temperature":0}'
  # PRIMARY: same but server-saturated, max_num_seqs=64.
  run_one fixed256 throughput 64 "$rep" '{"temperature":0}'
  # HARNESS-AS-SHIPPED: temp 0.6 / top_p 0.95 / top_k 20, generate to EOS.
  # Qwen3.5 is a reasoning model, so "to EOS" means a median ~1160 and p95 ~8100
  # output tokens per request; SPECBENCH_EOS_MAXREQ trims the request count at the
  # larger sizes. The subset is round-robin interleaved, so a prefix stays balanced.
  run_one eos synchronous "" "$rep" '{"temperature":0.6,"top_p":0.95,"top_k":20}' \
          "${SPECBENCH_EOS_MAXREQ:-70}"
done

echo "[specbench] stopping server"
for p in $SERVE_PIDS; do kill -TERM "$p" 2>/dev/null; done
kill -TERM "$LAUNCH_PID" 2>/dev/null
for i in $(seq 1 30); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || break
  sleep 2
done
for p in $SERVE_PIDS; do kill -KILL "$p" 2>/dev/null; done
echo "[specbench] $SIZE/$ARM COMPLETE"
