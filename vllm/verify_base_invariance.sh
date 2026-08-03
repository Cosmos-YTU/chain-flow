#!/bin/bash
# BASE-MODE INVARIANCE GATE.
# Every optimisation in this round touches only tree-verify code paths, which are inert
# unless VLLM_SPEC_TREE=1. This proves it empirically rather than by inspection: run the
# base arm against the PRE-OPTIMISATION vLLM files, then against the current ones, and
# require the generated token ids to be bit-identical.
set -euo pipefail
V=/home/shadeform/vllm/.venv/lib/python3.12/site-packages/vllm
PRE=/home/shadeform/chained-flow/vllm/_vllm_preopt
CUR=/tmp/cf_cur_vllm
FILES="v1/attention/backends/flash_attn.py v1/worker/gpu_model_runner.py \
       v1/sample/rejection_sampler.py v1/spec_decode/tree_state.py"

rm -rf "$CUR"; mkdir -p "$CUR"
for f in $FILES; do mkdir -p "$CUR/$(dirname $f)"; cp "$V/$f" "$CUR/$f"; done
restore() { for f in $FILES; do cp "$CUR/$f" "$V/$f"; done; }
trap restore EXIT

for f in $FILES; do cp "$PRE/$f" "$V/$f"; done
CF_TAG=_inv_pre bash /home/shadeform/chained-flow/vllm/bench_cf.sh "${1:-4b}" base _inv_pre >/tmp/inv_pre.log 2>&1
restore
CF_TAG=_inv_post bash /home/shadeform/chained-flow/vllm/bench_cf.sh "${1:-4b}" base _inv_post >/tmp/inv_post.log 2>&1

/home/shadeform/vllm/.venv/bin/python - <<'PY'
import json
a = json.load(open("/tmp/cf_native_base_inv_pre.json"))
b = json.load(open("/tmp/cf_native_base_inv_post.json"))
ok = all(x["outs"] == y["outs"] for x, y in zip(a["sets"], b["sets"]))
nt = sum(len(o) for s in a["sets"] for o in s["outs"])
print(f"BASE-MODE INVARIANCE: {'PASS' if ok else 'FAIL'} "
      f"({len(a['sets'])} prompt sets, {nt} tokens) | "
      f"pre {sum(s['tokens'] for s in a['sets'])/sum(s['secs'] for s in a['sets']):.1f} t/s "
      f"post {sum(s['tokens'] for s in b['sets'])/sum(s['secs'] for s in b['sets']):.1f} t/s")
PY
