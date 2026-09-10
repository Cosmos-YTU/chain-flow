#!/bin/bash
# Wait for the 9bx pipeline to release GPUs 5+6, then run the alignment-fix 4B retrain.
cd /home/shadeform/chain-flow
until [ -f logs/train_9bx_DONE.flag ] || ! pgrep -f "train_tree_flow.py train_configs/recovered/joint_9bx" >/dev/null; do sleep 300; done
sleep 60
echo "[queue] 9bx released GPUs at $(date -u) — starting 4bfix" 
bash logs/train_4bfix.sh
