#!/bin/bash
# Is the STALE-TREE FALLBACK reachable?  One arm per invocation.
#
#   cnd_reach.sh <gpu> <size> <tag> [env assignments...]
#
# Starts `serve_cf.sh <size> tree`, reads the engine's REAL attention block size out of its
# startup log (the repro's whole trigger is a prompt of exactly `k*block + 1` tokens, see
# cnd_stale_repro.py), drives cnd_stale_repro.py at it, and then reports the
# CF_TREE_FALLBACK_LOG counters -- which are always on here, because an arm that cannot count
# stale rows cannot answer the question it was run to answer.
#
# Three arms are meaningful, and the FIRST is the one that makes the other two mean anything:
#   cnd_reach.sh 7 4b ctl   CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=0   # positive control
#   cnd_reach.sh 7 4b haz   CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=1   # must DIE
#   cnd_reach.sh 7 4b fix                                                # defaults: must be 0
set -uo pipefail
GPU=${1:?gpu}; SIZE=${2:?size}; TAG=${3:?tag}; shift 3
ROOT=/home/shadeform/chained-flow
PORT=${CF_PORT:-8791}
DIR=$ROOT/logs/cnd
mkdir -p "$DIR"
SRV=$DIR/reach_${SIZE}_${TAG}_server.log
CLI=$DIR/reach_${SIZE}_${TAG}_client.log

MODEL=Qwen/Qwen3.5-4B
[ "$SIZE" = "9b" ] && MODEL=Qwen/Qwen3.5-9B
[ "$SIZE" = "27b" ] && MODEL=Qwen/Qwen3.5-27B

# The counters are the measurement, so they are not optional on any arm.
export CF_TREE_FALLBACK_LOG=1
export CF_GPU=$GPU
export CF_MAXLEN=${CF_MAXLEN:-4096}
for kv in "$@"; do export "${kv?}"; done

echo "[reach] === $SIZE tree / $TAG :: $* ===" | tee "$CLI"
setsid "$ROOT/vllm/serve_cf.sh" "$SIZE" tree "$PORT" "$SRV" &
SPID=$!
# setsid detaches, so the process group to clean up is the server's own; record it.
trap 'kill -TERM -"$(ps -o pgid= "$SPID" 2>/dev/null | tr -d " ")" 2>/dev/null; kill "$SPID" 2>/dev/null' EXIT

# `grep -q` on THIS arm's log as well as the health probe: the port answering is not proof that
# the process answering is the one this script started.
for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
     && grep -q "attention block size to" "$SRV"; then break; fi
  sleep 5
done
if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "[reach] SERVER NEVER CAME UP -- see $SRV" | tee -a "$CLI"
  grep -iE "error|Traceback|RuntimeError" "$SRV" | tail -20 | tee -a "$CLI"
  exit 2
fi

BLOCK=$(grep -oE "attention block size to [0-9]+" "$SRV" | tail -1 | grep -oE "[0-9]+")
BLOCK=${BLOCK:-528}
echo "[reach] server up, attention block size = $BLOCK" | tee -a "$CLI"

"${CF_PY:-/home/shadeform/vllm/.venv/bin/python}" "$ROOT/vllm/cnd_stale_repro.py" \
  "http://127.0.0.1:$PORT" "$MODEL" \
  --block "$BLOCK" --rounds "${CF_ROUNDS:-3}" --maxlen "$CF_MAXLEN" 2>&1 | tee -a "$CLI"
RC=${PIPESTATUS[0]}

# SHUT THE SERVER DOWN BEFORE READING THE COUNTERS.  `note_spec_step` prints on the first few
# stale steps immediately, but the TOTAL (`spec steps N | stale rows M`) only lands from the
# atexit hook -- and a SIGKILLed engine never runs it, so a clean arm produced no total at all
# and "no output" had to be argued rather than read.  TERM and wait.
PGID=$(ps -o pgid= "$SPID" 2>/dev/null | tr -d " ")
[ -n "$PGID" ] && kill -TERM -"$PGID" 2>/dev/null
for _ in $(seq 1 30); do kill -0 "$SPID" 2>/dev/null || break; sleep 2; done
sleep 3
echo "[reach] ---- engine-core accounting ----" | tee -a "$CLI"
grep -E "cf-tree-fallback|spec-slot guard|GDN conv state BUILT|attention block size to" "$SRV" \
  | tail -20 | tee -a "$CLI"
echo "[reach] ---- did the engine die? ----" | tee -a "$CLI"
# Anchored on the RAISE's own first words, not on the phrase "stale-tree fallback" -- the
# spec-slot guard's own startup banner contains that phrase and matched itself.
grep -E "took the stale-tree fallback|EngineDeadError|EngineCore .* died|device-side assert" "$SRV" \
  | tail -8 | tee -a "$CLI"
echo "[reach] $SIZE/$TAG repro rc=$RC" | tee -a "$CLI"
