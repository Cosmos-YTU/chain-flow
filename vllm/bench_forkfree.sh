#!/bin/bash
# Fork-free validation: run the CHAIN arm against a PRISTINE, unmodified vLLM 0.25.1 and compare
# it with the base arm on the SAME interpreter.
#
# Why this script exists separately from bench_cf.sh: the chain arm's ~1.13x had never been
# MEASURED, only inferred from a forked-vLLM run. Nothing here may quietly borrow the fork.
#
# Three arms per size, and the labels matter (docs/BENCHMARKING.md section 1):
#   base_async_on   vLLM's own default for base decoding      -> the DEPLOYMENT baseline
#   base_async_off  base forced synchronous                   -> the LIKE-FOR-LIKE baseline
#   chain           our arm; the entry point gives it async   -> now like-for-like with base_async_on
#
# Usage: bench_forkfree.sh <4b|9b|27b> [maxtok]
set -euo pipefail
SIZE=${1:?size}
MAXTOK=${2:-256}
GPU=${CF_GPU:-6}
PY=${CF_PY:-/home/shadeform/vllm-pristine/.venv/bin/python}
OUT=${CF_OUT:-/home/shadeform/chained-flow/logs/forkfree}
mkdir -p "$OUT"

# The published drafters do not ship a shortlist, and without one the drafter scores the full
# 248k-row lm_head at every depth (~40% of the draft wasted). The fork-side numbers we are
# comparing against were taken WITH the repo's shortlist, so pass it explicitly rather than let
# the two runs differ in a way that is invisible in the tok/s.
export CF_SHORTLIST=${CF_SHORTLIST:-/home/shadeform/chained-flow/out/flow/shortlist_q3527b.pt}

run () {
  local arm=$1 tag=$2; shift 2
  local log="$OUT/${SIZE}_${tag}.log"
  echo "=== $SIZE $tag -> $log"
  env "$@" CF_GPU="$GPU" CF_PY="$PY" CF_MAXTOK="$MAXTOK" \
      /home/shadeform/chained-flow/vllm/bench_cf.sh "$SIZE" "$arm" "_${SIZE}_${tag}" \
      > "$log" 2>&1 || { echo "FAILED: $tag (see $log)"; tail -30 "$log"; return 1; }
  # The two facts that decide whether the run means anything, pulled out of a 3000-line log.
  grep -hE "Asynchronous scheduling is|\[cf-plugin\]|\[cf-defaults\] ON:" "$log" | tail -3
  grep -hE "^\[(base|spec)\] SETS" "$log"
}

run base  base_async_off CF_ASYNC_SCHED=0
run base  base_async_on  CF_ASYNC_SCHED=1
run chain chain
# The A/B that prices the entry point itself: same arm, guard left alone.
run chain chain_async_off CF_ASYNC_SPEC=0
