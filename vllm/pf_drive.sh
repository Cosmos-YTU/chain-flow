#!/bin/bash
# Namespaced (`pf` = pad-fix) CF_TREE_CONV_NARROW default-ON verification matrix.
cd /home/shadeform/chain-flow
for r in 1 2 3; do for s in 4b 9b 27b; do ./vllm/cnd_matrix.sh 7 $s base $r pf; done; done
for r in 1 2;   do for s in 4b 9b 27b; do ./vllm/cnd_matrix.sh 7 $s base $r pfaon; done; done
for r in 1 2 3; do for s in 4b 9b 27b; do ./vllm/cnd_matrix.sh 7 $s tree $r pf; done; done
for r in 1 2;   do for s in 4b 9b 27b; do ./vllm/cnd_matrix.sh 7 $s tree $r pfwide; done; done
echo "[pf] MATRIX DONE"
