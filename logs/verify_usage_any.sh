#!/usr/bin/env bash
# Run EXACTLY the command a card prints, with every CF_* var unset except CF_DRAFTER_DIR.
#   verify_usage_any.sh <model> <repo> <gpu> <port> <gmu> <tag>
set -uo pipefail
cd /home/shadeform/chain-flow
M=$1; R=$2; GPU=$3; PORT=$4; GMU=$5; TAG=$6; WANT=${7:-77939}
OUT=logs/verify_usage_$TAG; mkdir -p $OUT
CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
env -u CF_SHORTLIST -u CF_K -u CF_ASYNC_SCHED -u CF_CUDAGRAPH -u CF_DRAFTER_DIR -u VLLM_SPEC_TREE \
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/shadeform/chain-flow/src CF_DRAFTER_DIR=$R \
  "${CF_PY%python}vllm" serve "$M" --async-scheduling --port $PORT --gpu-memory-utilization $GMU --max-num-seqs ${MAXSEQS:-64} \
    --speculative-config '{"method":"custom_class","model":"chain_flow.vllm_plugin.flow_proposer.FlowDrafterProposer","num_speculative_tokens":5}' \
    > $OUT/server.log 2>&1 &
PID=$!
trap 'kill '"$PID"' 2>/dev/null; sleep 8; kill -9 '"$PID"' 2>/dev/null' EXIT
for i in $(seq 1 180); do curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "[$TAG] SERVER DIED"; tail -30 $OUT/server.log; exit 1; }; sleep 5; done
curl -sf -m 300 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Merhaba, nasılsın?\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"temperature\":0,\"max_tokens\":24}" \
  > $OUT/response.json 2>&1
echo "[$TAG] reply: $(python3 -c "import json;print(json.load(open('$OUT/response.json'))['choices'][0]['message']['content'][:70])" 2>/dev/null)"
grep -E "\[chain-flow\] (drafter from HF hub|shortlist head)" $OUT/server.log | tee $OUT/provenance.txt
grep -q "shortlist head: $WANT" $OUT/provenance.txt && echo "[$TAG] PASS $WANT rows" || { echo "[$TAG] FAIL expected $WANT"; grep "shortlist head" $OUT/provenance.txt; }
grep -q "drafter checkpoint" $OUT/provenance.txt && echo "[$TAG] PASS auto-pickup" || echo "[$TAG] FAIL provenance"
echo "[$TAG] DONE"
