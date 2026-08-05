#!/bin/bash
# CF_TREE_CONV_NARROW default-ON verification matrix (batch 1, 7 domains, 256 tokens, greedy).
#
#   cnd_matrix.sh <gpu> <size> <arm> <repeat> [extra tag]
#
# One process per (size, arm, repeat).  Losslessness and throughput come out of the SAME run:
# `test_plugin_native.py` dumps the per-domain token ids alongside tok/s, so comparing text
# costs nothing extra and cannot drift from the number it is reported next to.
set -uo pipefail
GPU=${1:?gpu}; SIZE=${2:?size}; ARM=${3:?arm}; REP=${4:?repeat}; EXTRA=${5:-}
ROOT=/home/shadeform/chained-flow
TAG="_cnd${EXTRA}_${SIZE}_${ARM}_r${REP}"
LOG=$ROOT/logs/cnd/${SIZE}_${ARM}${EXTRA}_r${REP}.log
mkdir -p "$ROOT/logs/cnd"

export CF_GPU=$GPU CF_MAXTOK=256 CF_POFF=0,1,2,3,4,5,6
if [ "$ARM" = "tree" ]; then export CF_TREE_KEEP=8 CF_TREE_DEPTH=5; fi
# `_aon` = the DEPLOYMENT baseline (async scheduling on, which is what vLLM gives a
# non-speculative engine by default).  bench_cf.sh defaults the base arm to async OFF so the
# default comparison is like-for-like; both denominators are wanted and they must be labelled.
case "$EXTRA" in *aon*) export CF_ASYNC_SCHED=1 ;; esac
case "$EXTRA" in *stale*) export CF_TREE_FORCE_STALE=1 CF_TREE_FALLBACK_LOG=1 ;; esac
case "$EXTRA" in *wide*) export CF_TREE_CONV_NARROW=0 ;; esac
case "$EXTRA" in *narrow*) export CF_TREE_CONV_NARROW=1 ;; esac

echo "[cnd] === $SIZE / $ARM r$REP ${EXTRA:-} on gpu $GPU -> $LOG ==="
"$ROOT/vllm/bench_cf.sh" "$SIZE" "$ARM" "$TAG" > "$LOG" 2>&1
rc=$?
MODE=base; [ "$ARM" = "base" ] || MODE=spec
cp -f "/tmp/cf_native_${MODE}${TAG}.json" "$ROOT/logs/cnd/res${TAG}.json" 2>/dev/null
echo "[cnd] $SIZE/$ARM r$REP rc=$rc"
grep -E "GDN conv state BUILT|attention block size to|SETS " "$LOG" | tail -5
exit $rc
