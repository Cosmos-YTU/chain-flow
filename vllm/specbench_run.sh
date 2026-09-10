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

ROOT=/home/shadeform/chain-flow
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
# BOTH profiles must be warmed. CF_COMPILE defaults on with mode=max-autotune-no-cudagraphs,
# so the flow net re-autotunes for every new batch shape: a batch-1 warmup leaves the
# concurrency-64 run paying minutes of Triton autotuning inside the measured window, which
# the base arm (no drafter to compile) never pays. That asymmetry is a fake regression.
echo "[specbench] warmup"
run_one fixed256 synchronous "" 0 '{"temperature":0}'

for rep in $(seq 1 "$REPEATS"); do
  # PRIMARY: greedy, exactly 256 output tokens per request (ignore_eos), batch 1.
  [ "${SPECBENCH_ONLY:-}" = "eos" ] || \
  run_one fixed256 synchronous "" "$rep" '{"temperature":0}'
  # NO concurrency-64 arm. It does not measure inference for the spec arms: CF_COMPILE=1
  # (mode=max-autotune-no-cudagraphs, a capability-gated default) re-autotunes the flow net
  # for EVERY new batch shape, and a throughput run walks every shape from 64 down to 1 as
  # requests retire. Measured at 4B/chain: 850 AUTOTUNE events and 9.2 tok/s generation with
  # 51 requests resident, against 3099 tok/s for base -- a compile storm, not a throughput
  # number. Warming it honestly means warming all 64 shapes. Batch 1 is the regime this
  # project's numbers describe, so that is what is reported.
  # HARNESS-AS-SHIPPED: temp 0.6 / top_p 0.95 / top_k 20, generate to EOS.
  # NOT RUNNABLE ON THE TREE ARM. flow_proposer.py:1666 raises
  #   "chain-flow tree mode requires greedy sampling (temperature=0, no logprobs,
  #    no penalties)"
  # and it raises inside propose(), i.e. inside the EngineCore step -- so the request does
  # not fail, the ENGINE dies (EngineDeadError) and every later benchmark against that
  # server fails to connect. The RedHat harness ships temperature=0.6/top_p=0.95/top_k=20,
  # so the tree arm cannot run the benchmark's default sampling at all.
  if [ "$ARM" = "tree" ]; then
    echo "[specbench] SKIP eos config on tree arm (requires greedy; non-greedy kills EngineCore)"
    continue
  fi
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
