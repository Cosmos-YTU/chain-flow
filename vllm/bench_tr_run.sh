#!/bin/bash
# Serve ONE arm and measure it on the Turkish prompt sets at CONCURRENCY 1.
#
#   bench_tr_run.sh <4b|9b> <base|chain_v2|chain_tr|chain_v2_trsl> <gpu> <port> [tag]
#
# Arms
#   base           no speculation.  Async scheduling ON, same as every other arm, so the
#                  baseline is not handicapped and the ratio is not inflated.
#   chain_v2       chain K=5 with the ENGLISH v2 parent, shortlist AS SHIPPED (English).
#   chain_tr       chain K=5 with the Turkish fine-tune, shortlist AS SHIPPED WITH IT (Turkish).
#   chain_v2_trsl  the v2 parent forced onto the Turkish shortlist.  Isolates the DRAFT-COST
#                  effect of the bigger Turkish head (77,939 rows vs 62,642) from the weights,
#                  because the Turkish list is accept-neutral on the parent but not cost-neutral.
#
# One measured phase PER PROMPT SET, because acceptance is a Prometheus counter DELTA and a
# delta is only attributable to a window.  Set A and set B are written to separate files and
# are never pooled -- see vllm/bench_tr_data.py.
set -uo pipefail
SIZE=${1:?size}
ARM=${2:?arm}
GPU=${3:?gpu}
PORT=${4:?port}
TAG=${5:-}

ROOT=/home/shadeform/chained-flow
DATA=${CF_TR_DATA:-$ROOT/logs/bench_tr/data}
# CF_TR_IGNORE_EOS=0 measures the NATURAL-EOS condition (what a deployment serves) instead of the
# forced-256 equal-work condition. The two are NOT poolable -- forcing 256 tokens runs the model past
# its natural stop into out-of-distribution text that is harder to draft. Always use a $TAG so the
# two conditions land in different directories.
OUT=$ROOT/logs/bench_tr/${SIZE}_${ARM}${TAG}
mkdir -p "$OUT"
SERVER_LOG=$OUT/server.log
REPEATS=${CF_TR_REPEATS:-2}

export CUDA_VISIBLE_DEVICES=$GPU
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
export PYTHONPATH=$ROOT/src

# An unrecognised SIZE is a HARD EXIT, not a fallback. This was `if 4b ... else <9b>`, so any other
# size -- `27b`, a typo, anything -- was silently served as the 9B stack with the 9B drafters, while
# the output directory, the log lines and the report were all labelled with the size that was asked
# for. That is wrong-and-plausible, which is strictly worse than a crash: a crash costs an hour, a
# plausible wrong number can reach a model card. It produced correct results for 4b and 9b only
# because those were the only two call sites, which is luck, not safety.
case "$SIZE" in
  4b)
    CF_MODEL=Qwen/Qwen3.5-4B; V2=selimaktas/Flow-Drafter-4B-v2; TR=selimaktas/Flow-Drafter-4B-tr
    GMU=${CF_GMU:-0.55} ;;
  9b)
    CF_MODEL=Qwen/Qwen3.5-9B; V2=selimaktas/Flow-Drafter-9B-v2; TR=selimaktas/Flow-Drafter-9B-tr
    GMU=${CF_GMU:-0.70} ;;
  *)
    echo "[bench_tr] FATAL: unsupported size '$SIZE'. This script serves only: 4b, 9b." >&2
    echo "[bench_tr] For 27B use vllm/bench_tr27b_run.sh (different model, drafters and GMU)." >&2
    exit 2 ;;
esac

# CF_SHORTLIST must be decided BEFORE `defaults.py --sh` runs: `was_explicit()` is keyed on
# whether the variable was already in the environment, and an explicit value skips the
# candidate search entirely.
#
# THE TRAP THIS AVOIDS: `shortlist_candidates()` falls back to `out/flow/shortlist_q3527b.pt`
# -- the ENGLISH list -- which exists in this working tree.  A Turkish drafter that did not
# ship its own list would silently pick it up and lose ~0.70 accept, half the training gain,
# with no error and no log line saying anything was wrong.
case "$ARM" in
  base)          DRAFTER="$V2" ;;                                   # unused, but keep the env sane
  chain_v2)      DRAFTER="$V2" ;;                                   # shortlist: default resolution
  chain_tr)      DRAFTER="$TR"; export CF_SHORTLIST=$ROOT/out/flow/shortlist_q3527b_tr.pt ;;
  chain_v2_trsl) DRAFTER="$V2"; export CF_SHORTLIST=$ROOT/out/flow/shortlist_q3527b_tr.pt ;;
  *) echo "bad arm: $ARM"; exit 1 ;;
esac
export CF_DRAFTER_DIR=$DRAFTER

_CF_DEF=$(PYTHONPATH=$ROOT/src "$CF_PY" $ROOT/src/chained_flow/defaults.py --sh)
echo "$_CF_DEF" | sed 's/^/[bench_tr] /' >&2
eval "$_CF_DEF"

SPEC_ARGS=()
if [ "$ARM" != "base" ]; then
  export CF_CUDAGRAPH=1 CF_K=${CF_K:-5}
  SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}")
fi
# ON FOR EVERY ARM INCLUDING BASE.  Dropping it from base alone is worth +10.5%/+5.5% to the
# ratio and every speedup here would be overstated by that much.
export CF_ASYNC_SCHED=1

echo "[bench_tr] $SIZE/$ARM gpu=$GPU port=$PORT model=$CF_MODEL" >&2
echo "[bench_tr] drafter=$CF_DRAFTER_DIR shortlist=${CF_SHORTLIST:-<default resolution>} K=${CF_K:-n/a}" >&2

"${CF_PY%python}vllm" serve "$CF_MODEL" \
  --seed 42 \
  --port "$PORT" \
  --dtype float16 \
  --max-model-len "${CF_MAXLEN:-8192}" \
  --gpu-memory-utilization "$GMU" \
  --max-num-seqs "${CF_MAXSEQS:-64}" \
  --async-scheduling \
  "${SPEC_ARGS[@]}" \
  > "$SERVER_LOG" 2>&1 &
SERVE_PID=$!
echo "$SERVE_PID" > "$OUT/serve.pid"
# Teardown BY PID.  A broad `pkill -f` has self-matched and killed shells on this box.
trap 'kill '"$SERVE_PID"' 2>/dev/null; sleep 8; kill -9 '"$SERVE_PID"' 2>/dev/null' EXIT

for i in $(seq 1 180); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[bench_tr] SERVER DIED"; tail -60 "$SERVER_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
  echo "[bench_tr] server never became healthy"; tail -60 "$SERVER_LOG"; exit 1; }
echo "[bench_tr] server ready (pid $SERVE_PID)"

# The proposer builds LAZILY on the first propose(), so a /health-green server has not yet
# resolved a single flag.  Send one request, THEN read the engine's own lines.
curl -sf -m 300 "http://localhost:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CF_MODEL\",\"prompt\":\"Merhaba\",\"max_tokens\":8,\"temperature\":0}" \
  >/dev/null || { echo "[bench_tr] first request FAILED"; tail -60 "$SERVER_LOG"; exit 1; }

# PROVENANCE FROM THE ENGINE PROCESS, not from this shell's env and not from the API server's
# copy (which imports chained_flow but holds no drafter).
if [ "$ARM" != "base" ]; then
  grep -E "\(EngineCore pid=.*\[cf-defaults\] ON:" "$SERVER_LOG" | head -1 | tee "$OUT/cf_defaults.engine.txt"
  grep -E "\(EngineCore pid=.*\[chained-flow\] (drafter=|shortlist head)" "$SERVER_LOG" \
    | tee "$OUT/cf_build.engine.txt"
  if ! grep -q "shortlist head" "$OUT/cf_build.engine.txt"; then
    echo "[bench_tr] FATAL: engine never reported a shortlist head -- full head or no build"; exit 1
  fi
fi

for rep in $(seq 1 "$REPEATS"); do
  for df in "$DATA"/*.jsonl; do
    name=$(basename "$df" .jsonl)
    [ "$name" = "manifest" ] && continue
    echo "[bench_tr] --- $SIZE/$ARM $name rep$rep ---"
    "$CF_PY" $ROOT/vllm/bench_serve_drive.py \
      --base "http://localhost:${PORT}" --model "$CF_MODEL" \
      --data "$df" \
      --concurrency 1 \
      --ignore-eos "${CF_TR_IGNORE_EOS:-1}" \
      --max-tokens "${CF_MAXTOK:-256}" \
      --temperature 0 \
      --warmup "${CF_TR_WARMUP:-8}" \
      -o "$OUT/${name}_r${rep}.json" 2>&1 | tee -a "$OUT/drive.log"
  done
done

echo "[bench_tr] $SIZE/$ARM COMPLETE -> $OUT"
