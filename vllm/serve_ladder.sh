#!/bin/bash
# One server, a CONCURRENCY LADDER with a per-level request count, and a summary.
#
#   serve_ladder.sh <4b|9b|27b> <base|chain|tree> <gpu> <port> <tag>
#
# Why this exists next to bench_serve.sh: bench_serve.sh drives ONE `--requests` for the whole
# ladder, so a count that makes c=1 finish in a minute is barely one wave at c=64 (70 prompts /
# 64 in flight = 1.1 waves -- the measured window is then almost entirely ramp-up and drain, and
# the steady state it is supposed to report never happens).  Here each level gets its own count,
# `CF_LADDER` = "c:requests,c:requests,...", so every level measures ~4 waves.
#
# Everything else -- the env table, the engine-process [cf-defaults] check, the teardown by PID
# -- is bench_serve.sh's, on purpose: this must not become a second definition of the arms.
set -uo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
GPU=${3:?gpu}
PORT=${4:?port}
TAG=${5:?tag}

ROOT=/home/shadeform/chain-flow
OUT=$ROOT/logs/bench_serve/${SIZE}_${ARM}_${TAG}
mkdir -p "$OUT"
SERVER_LOG=$OUT/server.log

export CUDA_VISIBLE_DEVICES=$GPU
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
export PYTHONPATH=$ROOT/src

_CF_DEF=$(PYTHONPATH=$ROOT/src "$CF_PY" $ROOT/src/chain_flow/defaults.py --sh)
eval "$_CF_DEF"

if [ "$SIZE" = "4b" ]; then
  CF_MODEL=Qwen/Qwen3.5-4B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-4B-v2}
  GMU=${CF_GMU:-0.55}
elif [ "$SIZE" = "9b" ]; then
  CF_MODEL=Qwen/Qwen3.5-9B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-9B-v2}
  GMU=${CF_GMU:-0.70}
else
  CF_MODEL=Qwen/Qwen3.5-27B
  : ${CF_DRAFTER_DIR:=selimaktas/Flow-Drafter-Qwen3.5-27B-v2}
  GMU=${CF_GMU:-0.85}
fi
export CF_DRAFTER_DIR

SPEC_ARGS=()
case "$ARM" in
  base) ;;
  chain) export CF_CUDAGRAPH=1 CF_K=${CF_K:-5}
         SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chain_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}") ;;
  tree)  export CF_CUDAGRAPH=1 VLLM_SPEC_TREE=1
         : ${CF_TREE_KEEP:=8}; : ${CF_TREE_DEPTH:=5}
         : ${CF_K:=$(( CF_TREE_KEEP * CF_TREE_DEPTH + 1 ))}
         export CF_TREE_KEEP CF_TREE_DEPTH CF_K
         SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chain_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}") ;;
  *) echo "bad arm"; exit 1 ;;
esac
ASYNC_ARG=(--async-scheduling)
[ "${CF_NO_ASYNC_SCHED:-0}" = "1" ] && ASYNC_ARG=()
# CF_CGMODE: force vLLM's cudagraph_mode. ATTRIBUTION ONLY, not a shipping knob.
# A step whose requests have no draft tokens has query length 1, but a spec-configured engine
# sets `uniform_decode_query_len = 1 + num_spec_tokens`, so such a step cannot match a captured
# FULL decode graph and falls back to PIECEWISE. That is a property of turning speculation off
# inside a spec engine, not of how we turn it off -- and the only way to say so rather than
# assert it is to run the BASE arm at PIECEWISE and see whether the gap disappears.
CG_ARG=()
[ -n "${CF_CGMODE:-}" ] && CG_ARG=(--compilation-config "{\"cudagraph_mode\":\"$CF_CGMODE\"}")

# CF_KVBLOCKS: `--num-gpu-blocks-override`. ATTRIBUTION ONLY, not a shipping knob.
# `--speculative-config` shrinks the KV cache an engine gets at the same
# `--gpu-memory-utilization` (4B: 1,112,818 -> 600,425 -> 117,537 tokens), so a base-vs-spec
# ladder compares a BIGGER engine against a SMALLER one and calls the difference "speculation".
# This gives the base arm the spec arm's block count so the two are like-for-like, and any
# remaining gap is speculation itself. Tokens = blocks x the attention block size in the startup
# log, which is NOT the same across arms (4B: 528 base / 544 chain / 688 tree), so the block
# count that matches a token count has to be computed per arm.
KVB_ARG=()
[ -n "${CF_KVBLOCKS:-}" ] && KVB_ARG=(--num-gpu-blocks-override "$CF_KVBLOCKS")

# WAIT FOR THE GPU TO BE EMPTY BEFORE PROFILING.  vLLM sizes the KV cache as
# `gpu_memory_utilization * TOTAL - (everything already resident)`, so starting an arm while the
# previous arm's engine is still tearing down does not slow it down, it SILENTLY SHRINKS ITS KV
# CACHE.  Measured on this box: a 4B chain server started ~0 s after a 4B base server exited got
# "Available KV cache memory: 1.43 GiB / Maximum concurrency 2.53x" against the 41.38 GiB / 73.29x
# the same command gets on an idle GPU -- so the engine could hold about four decoding requests,
# and every ladder level above that measured queueing, not speculation. Nothing in the throughput
# number says so; the only tell is a line in the startup log nobody reads.
for i in $(seq 1 120); do
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
  [ "${USED:-99999}" -lt 1000 ] && break
  [ "$i" = 1 ] && echo "[ladder] GPU $GPU has ${USED} MiB in use; waiting for it to drain" >&2
  sleep 5
done
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")
if [ "${USED:-99999}" -ge 1000 ]; then
  echo "[ladder] REFUSING TO RUN: GPU $GPU still has ${USED} MiB in use after 10 min." >&2
  echo "[ladder] Benchmarking here would size the KV cache against whatever is resident and" >&2
  echo "[ladder] report the result as if it were this arm's. Free the GPU or pick another." >&2
  exit 1
fi

echo "[ladder] $SIZE/$ARM gpu=$GPU port=$PORT tag=$TAG ladder=${CF_LADDER:-default}" >&2
env | grep -E "^CF_(SPEC_MAX_BATCH|COMPILE|BATCH_AUDIT|LADDER)" | sed 's/^/[ladder] /' >&2

"${CF_PY%python}vllm" serve "$CF_MODEL" \
  --seed 42 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "${CF_MAXLEN:-8192}" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "${CF_MAXSEQS:-64}" \
  "${ASYNC_ARG[@]}" \
  "${CG_ARG[@]}" \
  "${KVB_ARG[@]}" \
  "${SPEC_ARGS[@]}" \
  > "$SERVER_LOG" 2>&1 &
SERVE_PID=$!
echo "$SERVE_PID" > "$OUT/serve.pid"
trap 'kill '"$SERVE_PID"' 2>/dev/null; sleep 8; kill -9 '"$SERVE_PID"' 2>/dev/null' EXIT

for i in $(seq 1 240); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[ladder] SERVER DIED"; tail -40 "$SERVER_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
  echo "[ladder] server never became healthy"; tail -60 "$SERVER_LOG"; exit 1; }

curl -sf -m 600 "http://localhost:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CF_MODEL\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}" \
  >/dev/null || { echo "[ladder] first request FAILED"; tail -40 "$SERVER_LOG"; exit 1; }
grep -m1 "(EngineCore pid=.*\[cf-defaults\] ON:" "$SERVER_LOG" | tee "$OUT/cf_defaults.engine.txt"

# THE KV CACHE THIS ENGINE ACTUALLY GOT, asserted against the ladder it is about to be driven
# with.  The pre-flight check above cannot cover a process that lands on the GPU *after* it, and
# an undersized KV cache does not error -- it silently caps how many requests can decode at once,
# so every level above that cap measures the admission queue and reports it as the arm's
# throughput.  (Measured: a 4B chain engine that got 2.53x instead of 73.29x read 505 tok/s at
# "concurrency 8" while its decode batch never once exceeded 4.)
grep -o "Available KV cache memory: .*\|GPU KV cache size: .*\|Maximum concurrency for .*" \
  "$SERVER_LOG" | tee "$OUT/kv_cache.txt"
KVTOK=$(grep -o "GPU KV cache size: [0-9,]* tokens" "$SERVER_LOG" | tail -1 | tr -dc '0-9')
MAXCONC=$(echo "${CF_LADDER:-1:70}" | tr ',' '\n' | cut -d: -f1 | sort -n | tail -1)
# Budget PER REQUEST, not `max_model_len`: vLLM's own "Maximum concurrency for 8,192 tokens per
# request" line assumes every request runs to the context limit, which for this dataset (a few
# hundred prompt tokens + CF_MAXTOK generated) overstates the requirement by ~16x -- using it
# would refuse a perfectly healthy 27B engine.  4x CF_MAXTOK is a deliberately loose bound that
# still catches the failure this guard exists for, which is off by a factor of tens.
NEED=$(( MAXCONC * 4 * ${CF_MAXTOK:-256} ))
if [ -n "$KVTOK" ] && [ "$KVTOK" -lt "$NEED" ]; then
  echo "[ladder] REFUSING TO BENCHMARK: KV cache is ${KVTOK} tokens, but a ladder to" >&2
  echo "[ladder] concurrency ${MAXCONC} at ${CF_MAXTOK:-256} output tokens needs >= ${NEED}." >&2
  echo "[ladder] Levels that do not fit measure the admission queue and report it as this" >&2
  echo "[ladder] arm's throughput. Usually this means another process was resident when this" >&2
  echo "[ladder] engine profiled -- check nvidia-smi and rerun on an idle GPU." >&2
  exit 1
fi

# CF_LADDER = "1:70,4:120,16:256,32:256,64:320"
LADDER=${CF_LADDER:-1:70,4:120,16:256,32:256,64:320}
: > "$OUT/running.txt"
for lvl in ${LADDER//,/ }; do
  C=${lvl%%:*}; N=${lvl##*:}
  # THE DECODE BATCH THE ENGINE ACTUALLY RAN, per level. "concurrency 64" is a property of the
  # CLIENT; if the engine can only admit 15 requests the other 49 are queued and the level
  # measures the admission queue. vLLM already logs `Running: R reqs, Waiting: W reqs` every few
  # seconds -- slicing the log by level is the only thing needed to attribute it, and unlike
  # CF_BATCH_AUDIT it works on the BASE arm, which has no proposer to audit.
  MARK=$(wc -l < "$SERVER_LOG")
  "$CF_PY" $ROOT/vllm/bench_serve_drive.py \
    --base "http://localhost:${PORT}" --model "$CF_MODEL" \
    --concurrency "$C" --requests "$N" \
    --max-tokens "${CF_MAXTOK:-256}" --temperature "${CF_TEMP:-0}" \
    -o "$OUT/c${C}.json" 2>&1 | tee -a "$OUT/drive.log"
  { echo "--- c=$C"
    tail -n +"$MARK" "$SERVER_LOG" |
      grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%" |
      sort | uniq -c | sort -rn
  } >> "$OUT/running.txt"
done

grep -o "\[cf-batch-audit\].*" "$SERVER_LOG" | tail -20 > "$OUT/batch_audit.txt"
grep -o "draft cudagraph captured.*bucket [0-9]*" "$SERVER_LOG" | sort -u > "$OUT/draft_buckets.txt"
echo "[ladder] results -> $OUT"
