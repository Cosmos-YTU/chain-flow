#!/bin/bash
# Phase 3: the two arms that need re-running, plus the A/B that says the guard is inert.
cd /home/shadeform/chain-flow

# The chain arm at batch 1 with the guard OFF. Both halves of the guard are provably unreachable
# here (padding needs a non-empty running batch; truncation needs max_model_len), so this pair
# must agree token-for-token AND in tok/s -- which is how "provably inert" becomes "measured
# inert".
CF_SPEC_SLOT_GUARD=0 ./vllm/cnd_matrix.sh 7 4b chain 2 pfnoguard

# fixcut collided with the previous arm's port and measured `connection refused`; cnd_reach.sh
# now waits for the port and the GPU before it starts. ctl and fix are re-run so their FINAL
# counter totals land (the old teardown SIGKILLed the engine before atexit).
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b fixcut2 CF_SPEC_MAX_BATCH=2
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b ctl2 CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=0
CF_ROUNDS=2 ./vllm/cnd_reach.sh 7 4b fix2
echo "[pf] PHASE3 DONE"

# The 4B ladder's c=4 rung read 0.77x against a recorded 0.95x, with a 565 ms TTFT against 74 ms
# at c=8 -- the shape of a one-off stall, not of a cost. But the slot guard DOES take a step that
# admits a request into a running decode batch off the FULL cudagraph, and c=4 is where joins are
# most frequent relative to the work, so this has to be measured rather than argued: the same
# rungs twice, once with the guard and once without.
export CF_LADDER=1:70,4:120,8:180
./vllm/serve_ladder.sh 4b tree 7 8796 pftree2 > logs/bench_serve/run_pf_tree2.log 2>&1
CF_SPEC_SLOT_GUARD=0 ./vllm/serve_ladder.sh 4b tree 7 8797 pftreeng \
  > logs/bench_serve/run_pf_treeng.log 2>&1
./vllm/serve_ladder.sh 4b base 7 8798 pfbase2 > logs/bench_serve/run_pf_base2.log 2>&1
echo "[pf] PHASE3B DONE"
