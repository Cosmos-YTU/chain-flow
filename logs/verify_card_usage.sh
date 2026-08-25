#!/usr/bin/env bash
# Run EXACTLY the command printed on the 4B-tr card and prove the claims it makes.
#
# This box is the adversarial case on purpose: it is a source checkout, so
# out/flow/shortlist_q3527b.pt (62,642 ENGLISH ids) exists and is candidate #2 in
# shortlist_candidates(). If the drafter's own 77,939-id list does not win, the English list is
# picked up silently and costs -0.41 to -0.70 accept with no error. A card that says "picked up
# automatically" has to be tested where it could plausibly fail.
#
# Nothing extra is set in the environment: every CF_* var is explicitly unset except the single
# CF_DRAFTER_DIR the card tells the reader to set. CF_K / CF_ASYNC_SCHED / CF_CUDAGRAPH were
# removed from the card because the library reads none of them meaningfully (K comes from
# num_speculative_tokens; async from --async-scheduling; CUDAGRAPH already defaults to 1).
set -uo pipefail
cd /home/shadeform/chained-flow
PORT=8791
GPU=${VERIFY_GPU:-4}
OUT=logs/verify_card_usage
mkdir -p $OUT

# wait for the 27B campaign to release the GPUs
until grep -q "CAMPAIGN 27B DONE" logs/bench_tr/campaign_27b.log 2>/dev/null; do sleep 60; done
echo "campaign done, verifying card usage block at $(date -u)"

CF_PY=${CF_PY:-/home/shadeform/vllm/.venv/bin/python}
env -u CF_SHORTLIST -u CF_K -u CF_ASYNC_SCHED -u CF_CUDAGRAPH -u CF_DRAFTER_DIR -u VLLM_SPEC_TREE \
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/shadeform/chained-flow/src \
    CF_DRAFTER_DIR=selimaktas/Flow-Drafter-4B-tr \
  "${CF_PY%python}vllm" serve Qwen/Qwen3.5-4B --async-scheduling --port $PORT \
    --gpu-memory-utilization 0.55 \
    --speculative-config '{"method":"custom_class","model":"chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer","num_speculative_tokens":5}' \
    > $OUT/server.log 2>&1 &
PID=$!
trap 'kill '"$PID"' 2>/dev/null; sleep 8; kill -9 '"$PID"' 2>/dev/null' EXIT

for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -40 $OUT/server.log; exit 1; }
  sleep 5
done

# the proposer builds lazily -- one request, with the card's chat_template_kwargs
curl -sf -m 300 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-4B","messages":[{"role":"user","content":"Merhaba, nasılsın?"}],
       "chat_template_kwargs":{"enable_thinking":false},"temperature":0,"max_tokens":32}' \
  | tee $OUT/response.json
echo

echo "=== engine provenance ==="
grep -E "\[chained-flow\] (drafter from HF hub|shortlist head|drafter=)" $OUT/server.log | tee $OUT/provenance.txt

echo "=== VERDICT ==="
grep -q "shortlist head: 77939" $OUT/provenance.txt \
  && echo "PASS: the drafter's own 77,939-id list won" \
  || { echo "FAIL: 77939 not reported -- the English list may have won"; grep "shortlist head" $OUT/provenance.txt; }
grep -q "drafter checkpoint" $OUT/provenance.txt \
  && echo "PASS: provenance is 'drafter checkpoint' (auto-pickup from the Hub snapshot)" \
  || { echo "NOTE: provenance is not 'drafter checkpoint':"; grep "shortlist head" $OUT/provenance.txt; }
echo "VERIFY DONE $(date -u)"
