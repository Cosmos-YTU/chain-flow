#!/usr/bin/env bash
set -uo pipefail
cd /home/shadeform/chained-flow
V=logs/verify_usage_any.sh
( bash $V Qwen/Qwen3.5-4B  selimaktas/Flow-Drafter-4B-v2  3 8801 0.55 4bv2  62642
  bash $V Qwen/Qwen3.5-9B  selimaktas/Flow-Drafter-9B-v2  3 8802 0.70 9bv2  62642
  bash $V Qwen/Qwen3.5-4B  selimaktas/Flow-Drafter-4B-tr  3 8803 0.55 4btr  77939 ) > logs/v6_a.log 2>&1 &
A=$!
( bash $V Qwen/Qwen3.5-27B selimaktas/Flow-Drafter-Qwen3.5-27B-v2 4 8804 0.85 27bv2 62642
  bash $V Qwen/Qwen3.5-9B  selimaktas/Flow-Drafter-9B-tr  4 8805 0.70 9btr  77939
  bash $V Qwen/Qwen3.5-27B selimaktas/Flow-Drafter-Qwen3.5-27B-tr 4 8806 0.85 27btr 77939 ) > logs/v6_b.log 2>&1 &
B=$!
wait $A; wait $B; echo "ALL SIX DONE"
