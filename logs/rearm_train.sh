#!/bin/bash
TAG=$1; GPU=$2
cd /home/shadeform/chain-flow
echo "==== [$(date -u '+%H:%M:%S')] rearm $TAG: waiting for dolly source ===="
while [ ! -d "teacher_states/stage1-$TAG-dolly-chat" ]; do sleep 30; done
sleep 10   # let save finalize
echo "==== [$(date -u '+%H:%M:%S')] dolly present -> launching train pipeline $TAG on GPU$GPU ===="
exec bash logs/train_pipeline.sh $TAG $GPU
