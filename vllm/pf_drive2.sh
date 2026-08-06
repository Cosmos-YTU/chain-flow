#!/bin/bash
# Phase 2 of the `pf` (pad-fix) verification: the CHAIN arm is unaffected, the stale-tree
# fallback is unreachable, and the 4B concurrency ladder still reads what it read.
cd /home/shadeform/chained-flow

# 0. EXTEND THE 27B NULL.  The 27B tree flips domain 4 at token 161 between its own repeats --
#    the tie this project has recorded the 27B BASE arm flipping against itself -- but three
#    base repeats in this session were all identical, so the null as measured here does not yet
#    cover it. Two more base repeats and two more tree repeats say whether d4@161 is a tie both
#    arms land on either side of, or a difference only the tree has.
for r in 4 5; do ./vllm/cnd_matrix.sh 7 27b base $r pf; done
for r in 4 5; do ./vllm/cnd_matrix.sh 7 27b tree $r pf; done

# 1. The chain arm must be untouched.  The slot guard fires on EVERY speculative engine, not
#    just the tree, so the arm that has nothing to do with the tree is the one that says
#    whether it cost anything.
./vllm/cnd_matrix.sh 7 4b chain 1 pf

# 2. REACHABILITY.  Order matters: the positive control runs FIRST, because if it records zero
#    stale rows then neither of the other two arms means anything.
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b ctl CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=0
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b haz CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=1
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b fix
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b fixcut CF_SPEC_MAX_BATCH=2
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b fixnocut CF_SPEC_MAX_BATCH=0

# 3. The 4B concurrency ladder, both arms, narrowing on (now the default).
export CF_LADDER=1:70,4:120,8:180,16:256,32:256,64:320
./vllm/serve_ladder.sh 4b base 7 8794 pfbase > logs/bench_serve/run_pf_base.log 2>&1
./vllm/serve_ladder.sh 4b tree 7 8795 pftree > logs/bench_serve/run_pf_tree.log 2>&1
echo "[pf] PHASE2 DONE"
