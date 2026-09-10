#!/bin/bash
# THE DEFAULT MEASUREMENT PATH: `vllm serve` (the production entry point), driven at fixed
# concurrency, reporting acceptance + throughput in the same shape as bench_cf.sh.
#
#   bench_serve.sh <4b|9b|27b> <base|chain|tree> <gpu> <port> [tag]
#
# vs bench_cf.sh
# --------------
#   bench_cf.sh    offline `LLM()`, VLLM_ENABLE_V1_MULTIPROCESSING=0, batch 1, 7 prompts.
#                  KEEP IT: it is the fast batch-1 regression gate (one process, ~3 min at 4B)
#                  and the only harness that can read the proposer's own counters in-process.
#                  It is NOT the deployment number.
#   bench_serve.sh `vllm serve`, engine core in its own process, continuous batching at
#                  concurrency 1..16.  THIS is the number a deployment sees, and it is the one
#                  to quote.  Flags that are gated on a decode batch of 1 disengage here; see
#                  docs/BENCHMARKING.md ("what disengages under serve").
#
# Three things this script does that serve_cf.sh deliberately does not:
#   1. It does NOT set VLLM_ENABLE_V1_MULTIPROCESSING=0.  Production serving runs the engine
#      core in a spawned subprocess, and the spawn boundary is where provenance was lost once
#      before.  Measuring with it disabled measures a configuration nobody deploys.
#   2. It greps the [cf-defaults] line OUT OF THE ENGINE PROCESS (the `(EngineCore pid=...)`
#      copy), not the API server's, and fails loudly if the engine never printed one.  The API
#      server imports chain_flow too and prints its own line from a process that has no
#      drafter in it -- reading that one tells you nothing about the run.
#   3. It records the completion TEXT for every request so two arms can be diffed before their
#      throughputs are compared.  `bench_serve_diff.py a.json b.json`.
set -uo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
GPU=${3:?gpu}
PORT=${4:?port}
TAG=${5:-}

ROOT=/home/shadeform/chain-flow
OUT=$ROOT/logs/bench_serve/${SIZE}_${ARM}${TAG}
mkdir -p "$OUT"
SERVER_LOG=$OUT/server.log

export CUDA_VISIBLE_DEVICES=$GPU
# The forked venv is the default because TREE needs it. The chain and base arms are
# fork-free: CF_PY=/home/shadeform/vllm-pristine/.venv/bin/python runs them on stock vLLM
# through the `vllm.general_plugins` entry point.
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
export PYTHONPATH=$ROOT/src

# Same capability-gated default table bench_cf.sh evaluates, for the same reason: CF_ASYNC_SPEC
# is read by the forked config/vllm.py before anything imports us, so it cannot be defaulted
# in-process on the fork.  Run as a FILE (no torch import, ~40 ms).
_CF_DEF=$(PYTHONPATH=$ROOT/src "$CF_PY" $ROOT/src/chain_flow/defaults.py --sh)
echo "$_CF_DEF" | sed 's/^/[bench_serve] /' >&2
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
# Async scheduling ON for EVERY arm including base -- otherwise the base arm gives up a
# feature worth +10.5%/+5.5%/+1.6% and every ratio is inflated (docs/BENCHMARKING.md).
# CF_NO_ASYNC_SCHED=1 drops it, which is the ONLY way to run a spec arm with CF_ASYNC_SPEC=0:
# the fork's config/vllm.py refuses `--async-scheduling` for custom_class unless CF_ASYNC_SPEC
# is on, so the two have to be turned off together or the engine will not build at all.
ASYNC_ARG=(--async-scheduling)
[ "${CF_NO_ASYNC_SCHED:-0}" = "1" ] && ASYNC_ARG=()

echo "[bench_serve] $SIZE/$ARM gpu=$GPU port=$PORT model=$CF_MODEL drafter=$CF_DRAFTER_DIR" >&2
echo "[bench_serve] K=${CF_K:-n/a} keep=${CF_TREE_KEEP:-n/a} depth=${CF_TREE_DEPTH:-n/a}" >&2

"${CF_PY%python}vllm" serve "$CF_MODEL" \
  --seed 42 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "${CF_MAXLEN:-8192}" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "${CF_MAXSEQS:-64}" \
  "${ASYNC_ARG[@]}" \
  "${SPEC_ARGS[@]}" \
  > "$SERVER_LOG" 2>&1 &
SERVE_PID=$!
echo "$SERVE_PID" > "$OUT/serve.pid"
# Teardown is BY PID, never pkill -- this box runs other people's jobs.
trap 'kill '"$SERVE_PID"' 2>/dev/null; sleep 8; kill -9 '"$SERVE_PID"' 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[bench_serve] SERVER DIED"; tail -40 "$SERVER_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
  echo "[bench_serve] server never became healthy"; tail -40 "$SERVER_LOG"; exit 1; }
echo "[bench_serve] server ready (pid $SERVE_PID)"

# ---- CONFIRM FROM STATE, NOT FROM ENV ------------------------------------------------
# The engine core is a different process; its [cf-defaults] line is the only statement of
# what actually got built.
#
# IT IS NOT PRINTED AT STARTUP.  `FlowDrafterProposer._build()` is called lazily from the
# FIRST `propose()` (flow_proposer.py:1476), so a server that is `/health`-green has not yet
# resolved a single drafter flag.  Grepping a freshly-started log finds only the API SERVER's
# copy -- printed by `apply()` at import, in a process that has no drafter in it -- which
# states the PROPOSALS and cannot state the outcome.  So: send one request first.
curl -sf -m 300 "http://localhost:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CF_MODEL\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}" \
  >/dev/null || { echo "[bench_serve] first request FAILED"; tail -40 "$SERVER_LOG"; exit 1; }
ENGINE_LINE=$(grep -m1 "(EngineCore pid=.*\[cf-defaults\] ON:" "$SERVER_LOG" || true)
if [ -z "$ENGINE_LINE" ] && [ "$ARM" != "base" ]; then
  echo "[bench_serve] FATAL: the ENGINE process never printed a [cf-defaults] line."
  grep -n "cf-defaults" "$SERVER_LOG" | head; exit 1
fi
echo "$ENGINE_LINE" | tee "$OUT/cf_defaults.engine.txt"
grep -E "\(EngineCore pid=.*\[chain-flow\] (drafter=|shortlist head)" "$SERVER_LOG" \
  | tee "$OUT/cf_build.engine.txt"

RES=$OUT/serve_bench.json
"$CF_PY" $ROOT/vllm/bench_serve_drive.py \
  --base "http://localhost:${PORT}" --model "$CF_MODEL" \
  --concurrency "${CF_CONC:-1,2,4,8,16}" \
  --requests "${CF_REQ:-0}" \
  --max-tokens "${CF_MAXTOK:-256}" \
  --temperature "${CF_TEMP:-0}" \
  -o "$RES" 2>&1 | tee "$OUT/drive.log"

# CF_XCHECK=<conc>: cross-check ONE concurrency point against vLLM's OWN load generator.
# `vllm bench serve` already does most of what bench_serve_drive.py does -- it has
# --max-concurrency, it scrapes the same vllm:spec_decode counters, and --save-detailed keeps
# the generated text.  The wrapper exists for the concurrency LADDER against a single server
# (with a per-level warmup, because our draft cudagraph is captured lazily per batch bucket) and
# for the fixed-output dataset; it is not a claim that the official tool is wrong.  So run it
# once and check the two agree, rather than asserting they do.
# NOT YET RUN: `vllm bench serve`'s dataset loaders need pandas, i.e. `pip install vllm[bench]`,
# which is absent from /home/shadeform/vllm/.venv -- it dies with "Please install vllm[bench]".
# Installing into a shared venv is not this script's call; do it deliberately, then use this.
if [ -n "${CF_XCHECK:-}" ]; then
  "${CF_PY%python}vllm" bench serve --backend openai --host localhost --port "$PORT" \
    --model "$CF_MODEL" --dataset-name custom \
    --dataset-path "$ROOT/logs/specbench/data/subset_fixed256.jsonl" \
    --custom-output-len "${CF_MAXTOK:-256}" --ignore-eos \
    --num-prompts 70 --max-concurrency "$CF_XCHECK" --request-rate inf \
    --temperature 0 --save-result --save-detailed \
    --result-filename "$OUT/vllm_bench_serve_c${CF_XCHECK}.json" \
    2>&1 | tee "$OUT/vllm_bench_serve_c${CF_XCHECK}.log" | tail -30
fi

# CF_SAMPLING_PROBE=1: what a temperature>0 request does to THIS arm.  LAST, because in tree
# mode it is expected to take the engine down (flow_proposer raises rather than emit a tree the
# rejection sampler would reject).
if [ "${CF_SAMPLING_PROBE:-0}" = "1" ]; then
  "$CF_PY" $ROOT/vllm/serve_sampling_probe.py \
    --base "http://localhost:${PORT}" --model "$CF_MODEL" \
    --temperature "${CF_PROBE_TEMP:-0.8}" -o "$OUT/sampling_probe.json" \
    2>&1 | tee "$OUT/sampling_probe.log"
  grep -n "requires greedy sampling\|EngineCore.*Traceback\|EngineDeadError\|must match tensor" \
    "$SERVER_LOG" | head -20 | tee "$OUT/sampling_probe.server_excerpt.txt"
fi

# Which cudagraph batch buckets the DRAFT was captured at -- the direct evidence of how far
# above batch 1 the run actually went.
grep -o "draft cudagraph captured.*bucket [0-9]*" "$SERVER_LOG" | sort -u \
  | tee "$OUT/draft_buckets.txt"
echo "[bench_serve] results -> $RES"
