#!/bin/bash
cd /home/shadeform/chain-flow
until [ -f /tmp/ev9.status ] && grep -q EV9_DONE /tmp/ev9.status; do sleep 60; done
sleep 30
echo "[queue] 9B evals done at $(date -u) — starting 4bfix2 ablation on GPUs 6+7"
bash logs/train_4bfix2.sh
