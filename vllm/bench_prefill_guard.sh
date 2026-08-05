#!/bin/bash
# The K x prompt-length sweep that pins the K+1 prefill / uniform-decode cudagraph collision.
# See vllm/test_prefill_guard.py for what the bug is and why the sweep has this shape.
#
#   bench_prefill_guard.sh [size] [K list] [plen list]
#   CF_PY=<venv>/bin/python  selects WHICH vLLM (default: the pristine, unforked one)
#
# One base process for the reference tokens, then one process per K, then a comparison that
# reports the first divergence index per (K, plen) cell -- with the plen == K+1 diagonal called
# out, since that is the only cell the bug ever touched.
set -euo pipefail
SIZE=${1:-4b}
KS=${2:-"3 4 5 6 7 8"}
PLENS=${3:-$(seq -s, 1 32)}
export CUDA_VISIBLE_DEVICES=${CF_GPU:-5}
PY=${CF_PY:-/home/shadeform/vllm-pristine/.venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${CF_OUT:-/home/shadeform/chained-flow/logs/prefill_guard}
mkdir -p "$OUT"

case "$SIZE" in
  4b)  MODEL=Qwen/Qwen3.5-4B;  DR=selimaktas/Flow-Drafter-4B-v2;            GMU=0.55 ;;
  9b)  MODEL=Qwen/Qwen3.5-9B;  DR=selimaktas/Flow-Drafter-9B-v2;            GMU=0.70 ;;
  27b) MODEL=Qwen/Qwen3.5-27B; DR=selimaktas/Flow-Drafter-Qwen3.5-27B-v2;   GMU=0.85 ;;
  *) echo "bad size"; exit 1 ;;
esac
export CF_MODEL=$MODEL CF_GMU=$GMU CF_PLENS=$PLENS
: "${CF_DRAFTER_DIR:=$DR}"; export CF_DRAFTER_DIR

echo "=== base ($MODEL)"
CF_MODE=base "$PY" "$HERE/test_prefill_guard.py" > "$OUT/${SIZE}_base.log" 2>&1 \
  || { tail -30 "$OUT/${SIZE}_base.log"; exit 1; }
grep -h "prefill-guard" "$OUT/${SIZE}_base.log"

RAN=""
for K in $KS; do
  echo "=== spec K=$K"
  if CF_MODE=spec CF_K=$K "$PY" "$HERE/test_prefill_guard.py" > "$OUT/${SIZE}_K${K}.log" 2>&1; then
    RAN="$RAN $K"
  else
    # A K the DRAFTER cannot serve must refuse, clearly, at build time -- that is a separate
    # contract from the cudagraph collision, and one this sweep found (a draft_length=8 drafter
    # cannot emit 8 chain tokens). Report it and keep going; only an unexplained crash is fatal.
    if grep -q "can only emit" "$OUT/${SIZE}_K${K}.log"; then
      echo "  REFUSED at build (expected for this drafter):"
      grep -hoE "num_speculative_tokens=[0-9]+ but this drafter can only emit [0-9]+ chain tokens" \
        "$OUT/${SIZE}_K${K}.log" | head -1 | sed 's/^/    /'
      continue
    fi
    tail -30 "$OUT/${SIZE}_K${K}.log"; exit 1
  fi
  grep -hE "prefill-guard|\[cf-defaults\] ON:" "$OUT/${SIZE}_K${K}.log" | tail -3
done

"$PY" - "$RAN" <<'EOF'
import json, sys
ks = [int(k) for k in sys.argv[1].split()]
base = json.load(open("/tmp/cf_prefill_base.json"))["out"]
bad = degen = 0
# BASE ITSELF degenerates at plen=1 (a single token with no context: the model repeats id 0),
# so "emitted one token over and over" is only evidence of the cudagraph collision when BASE
# did not do it too. Judging it absolutely made the sweep report a false FAIL on every K.
base_degen = {n for n, ids in base.items() if len(set(ids)) == 1 and len(ids) > 4}
if base_degen:
    print(f"note: base degenerates on its own at plen {sorted(base_degen, key=int)} "
          f"-- excluded from the degeneracy test, still compared for token equality")
print(f"\n{'K':>3} {'plen==K+1':>28}   other lengths")
for k in ks:
    d = json.load(open(f"/tmp/cf_prefill_spec_K{k}.json"))["out"]
    diffs = []
    for n, ids in sorted(d.items(), key=lambda kv: int(kv[0])):
        b = base[n]
        if len(set(ids)) == 1 and len(ids) > 4 and n not in base_degen:
            diffs.append((int(n), "DEGENERATE")); degen += 1; continue
        if ids != b:
            i = next((j for j, (x, y) in enumerate(zip(ids, b)) if x != y), min(len(ids), len(b)))
            diffs.append((int(n), f"differs@{i}")); bad += 1
    diag = [f"{n}:{w}" for n, w in diffs if n == k + 1] or ["ok"]
    other = [f"{n}:{w}" for n, w in diffs if n != k + 1] or ["all match base"]
    print(f"{k:>3} {diag[0]:>28}   {', '.join(other)}")
print(f"\ndegenerate cells: {degen}   cells differing from base: {bad}")
print("PASS -- the plen==K+1 diagonal is not special" if degen == 0 else
      "FAIL -- the K+1 collision is back")
sys.exit(1 if degen else 0)
EOF
