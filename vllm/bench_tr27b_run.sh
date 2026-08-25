#!/bin/bash
# Serve ONE 27B arm and measure it on the Turkish prompt sets at CONCURRENCY 1.
#
#   bench_tr27b_run.sh <base|chain_v2|chain_tr|chain_v2_trsl> <gpu> <port> [tag]
#
# A SEPARATE FILE from vllm/bench_tr_run.sh on purpose. That script's size branch is
# `if [ "$SIZE" = "4b" ] ... else`, so ANY size that is not 4b is served as Qwen3.5-9B with the 9B
# drafters -- a 27B run would have silently benchmarked the 9B stack. It is also actively edited by
# another agent, and a merge conflict in the script that produces the headline number is not worth
# saving one case branch. Everything else here is deliberately identical to it so the 27B figures
# are comparable to the published 4B/9B ones.
#
# Arms
#   base           no speculation. Async scheduling ON, same as every other arm, so the baseline is
#                  not handicapped and the ratio is not inflated.
#   chain_v2       chain K=5 with the ENGLISH v2 parent on the English shortlist -- the arm that
#                  showed a net REGRESSION on Turkish at 4B (0.822x) and 9B (0.854x). The fine-tune's
#                  value is turning that loss into a gain, which needs this arm to exist.
#   chain_tr       chain K=5 with the Turkish fine-tune on the Turkish shortlist.
#   chain_v2_trsl  parent forced onto the Turkish shortlist: isolates the DRAFT-COST of the bigger
#                  head (77,939 vs 62,642 rows) from the weights.
set -uo pipefail
ARM=${1:?arm}
GPU=${2:?gpu}
PORT=${3:?port}
TAG=${4:-}

ROOT=/home/shadeform/chained-flow
DATA=${CF_TR_DATA:-$ROOT/logs/bench_tr/data_setA}
OUT=$ROOT/logs/bench_tr/27b_${ARM}${TAG}
mkdir -p "$OUT"
SERVER_LOG=$OUT/server.log
REPEATS=${CF_TR_REPEATS:-2}

export CUDA_VISIBLE_DEVICES=$GPU
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
export PYTHONPATH=$ROOT/src

CF_MODEL=Qwen/Qwen3.5-27B
V2=selimaktas/Flow-Drafter-Qwen3.5-27B-v2
# The Turkish 27B drafter is LOCAL ONLY -- it is deliberately not pushed until both halves of the
# deliverable are measured. CF_DRAFTER_DIR takes a local directory (os.path.isdir short-circuits the
# HF download). Point this at the checkpoint the offline sweep selected.
TR=${CF_TR_CKPT:?set CF_TR_CKPT to the selected checkpoint dir}
GMU=${CF_GMU:-0.85}

# The VAE is NOT read from the drafter dir. tree_vae_flow.py resolves `vae_dir` out of the tree
# config as a MACHINE-ABSOLUTE path, with no env override, and dies with a bare IndexError on a
# checkpoint-* glob if it is missing. Fail loudly here instead.
VAE=$($CF_PY -c "import json,sys;print(json.load(open('$TR/chained_flow_tree_config.json'))['model_args']['vae_dir'])" 2>/dev/null)
[ -f "$VAE/model.safetensors" ] || { echo "[bench27b] FATAL: vae_dir '$VAE' has no model.safetensors"; exit 1; }
echo "[bench27b] vae_dir OK: $VAE" >&2

# CF_SHORTLIST must be decided BEFORE `defaults.py --sh` runs: was_explicit() is keyed on whether
# the variable was already in the environment. Pinned on EVERY spec arm -- including chain_v2, where
# the default resolution happens to land on the right file today (out/flow/shortlist_q3527b.pt,
# 62,642 ids, same count as the packaged list) but is a silent fallback that could drift.
case "$ARM" in
  base)          DRAFTER="$V2" ;;
  chain_v2)      DRAFTER="$V2"; export CF_SHORTLIST=$ROOT/out/flow/shortlist_q3527b.pt ;;
  chain_tr)      DRAFTER="$TR"; export CF_SHORTLIST=$ROOT/out/flow/shortlist_q3527b_tr.pt ;;
  chain_v2_trsl) DRAFTER="$V2"; export CF_SHORTLIST=$ROOT/out/flow/shortlist_q3527b_tr.pt ;;
  *) echo "bad arm: $ARM"; exit 1 ;;
esac
export CF_DRAFTER_DIR=$DRAFTER

_CF_DEF=$(PYTHONPATH=$ROOT/src "$CF_PY" $ROOT/src/chained_flow/defaults.py --sh)
echo "$_CF_DEF" | sed 's/^/[bench27b] /' >&2
eval "$_CF_DEF"

SPEC_ARGS=()
if [ "$ARM" != "base" ]; then
  export CF_CUDAGRAPH=1 CF_K=${CF_K:-5}
  SPEC_ARGS=(--speculative-config "{\"method\":\"custom_class\",\"model\":\"chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer\",\"num_speculative_tokens\":${CF_K}}")
fi
# ON FOR EVERY ARM INCLUDING BASE. Dropping it from base alone inflates every ratio.
export CF_ASYNC_SCHED=1

echo "[bench27b] $ARM gpu=$GPU port=$PORT model=$CF_MODEL" >&2
echo "[bench27b] drafter=$CF_DRAFTER_DIR shortlist=${CF_SHORTLIST:-<default>} K=${CF_K:-n/a} eos=${CF_TR_IGNORE_EOS:-1}" >&2

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
# Teardown BY PID. A broad `pkill -f` has self-matched and killed shells on this box (four times
# in this session alone).
trap 'kill '"$SERVE_PID"' 2>/dev/null; sleep 10; kill -9 '"$SERVE_PID"' 2>/dev/null' EXIT

for i in $(seq 1 240); do
  curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 && break
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "[bench27b] SERVER DIED"; tail -60 "$SERVER_LOG"; exit 1; }
  sleep 10
done
curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || {
  echo "[bench27b] server never became healthy"; tail -60 "$SERVER_LOG"; exit 1; }
echo "[bench27b] server ready (pid $SERVE_PID)"

# The proposer builds LAZILY on the first propose(), so a /health-green server has resolved nothing.
curl -sf -m 600 "http://localhost:${PORT}/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CF_MODEL\",\"prompt\":\"Merhaba\",\"max_tokens\":8,\"temperature\":0}" \
  >/dev/null || { echo "[bench27b] first request FAILED"; tail -60 "$SERVER_LOG"; exit 1; }

# PROVENANCE FROM THE ENGINE PROCESS, not this shell's env.
if [ "$ARM" != "base" ]; then
  grep -E "\(EngineCore pid=.*\[cf-defaults\] ON:" "$SERVER_LOG" | head -1 | tee "$OUT/cf_defaults.engine.txt"
  grep -E "\(EngineCore pid=.*\[chained-flow\] (drafter=|shortlist head)" "$SERVER_LOG" \
    | tee "$OUT/cf_build.engine.txt"
  if ! grep -q "shortlist head" "$OUT/cf_build.engine.txt"; then
    echo "[bench27b] FATAL: engine never reported a shortlist head -- full head or no build"; exit 1
  fi
  # Assert the ROW COUNT the engine actually built, not the env var we asked for.
  case "$ARM" in
    chain_tr|chain_v2_trsl) WANT=77939 ;;
    *)                      WANT=62642 ;;
  esac
  if ! grep -q "shortlist head: $WANT of" "$OUT/cf_build.engine.txt"; then
    echo "[bench27b] FATAL: expected shortlist head $WANT rows; engine reported:"
    grep "shortlist head" "$OUT/cf_build.engine.txt"; exit 1
  fi
  echo "[bench27b] shortlist head verified: $WANT rows"
fi

for rep in $(seq 1 "$REPEATS"); do
  for df in "$DATA"/*.jsonl; do
    name=$(basename "$df" .jsonl)
    [ "$name" = "manifest" ] && continue
    echo "[bench27b] --- $ARM $name rep$rep ---"
    "$CF_PY" $ROOT/vllm/bench_serve_drive.py \
      --base "http://localhost:${PORT}" --model "$CF_MODEL" \
      --data "$df" \
      --concurrency 1 \
      --ignore-eos "${CF_TR_IGNORE_EOS:-0}" \
      --max-tokens "${CF_MAXTOK:-256}" \
      --temperature 0 \
      --warmup "${CF_TR_WARMUP:-8}" \
      -o "$OUT/${name}_r${rep}.json" 2>&1 | tee -a "$OUT/drive.log"
  done
done

echo "[bench27b] $ARM COMPLETE -> $OUT"
