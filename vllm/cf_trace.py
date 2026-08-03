"""GPU-busy vs host-wall from a chrome trace: the honest 'is this glue or compute' number."""
import json, sys, collections

f = sys.argv[1]
d = json.load(open(f))
ev = d["traceEvents"]
ker = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
if not ker:
    print("no GPU events"); sys.exit()
t0 = min(e["ts"] for e in ker); t1 = max(e["ts"] + e.get("dur", 0) for e in ker)
wall = t1 - t0
# union of busy intervals (kernels can overlap across streams)
iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in ker)
busy = 0.0; cs, ce = iv[0]
for s, e in iv[1:]:
    if s > ce:
        busy += ce - cs; cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
tot = sum(e.get("dur", 0) for e in ker)
print(f"{f}\n  GPU wall {wall/1000:.1f} ms | GPU BUSY (union) {busy/1000:.1f} ms = {100*busy/wall:.1f}% "
      f"| sum-of-kernels {tot/1000:.1f} ms | idle {100*(1-busy/wall):.1f}%")
agg = collections.Counter(); cnt = collections.Counter()
for e in ker:
    agg[e["name"]] += e.get("dur", 0); cnt[e["name"]] += 1
nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 1
print(f"  per-step (n={nsteps}): GPU busy {busy/1000/nsteps:.3f} ms, wall {wall/1000/nsteps:.3f} ms")
print("  top kernels (us/step, launches/step):")
for k, v in agg.most_common(22):
    print(f"    {v/nsteps:9.1f}  x{cnt[k]/nsteps:7.1f}  {k[:96]}")
print(f"  TOTAL GPU launches/step: {sum(cnt.values())/nsteps:.0f}")

# exact step count from the CF_STEP marker, if present
import json as _j
_st = [e for e in ev if e.get("name") == "CF_STEP"]
if _st:
    n = len(_st)
    print(f"\n  === CF_STEP markers: {n} steps ===")
    print(f"  GPU busy {busy/1000/n:.3f} ms/step | trace wall {wall/1000/n:.3f} ms/step")
    rt = collections.Counter(e["name"] for e in ev if e.get("cat") == "cuda_runtime")
    for k in ("cudaLaunchKernel", "cudaGraphLaunch", "cudaMemcpyAsync", "cudaStreamSynchronize"):
        print(f"  {k:24s} {rt[k]/n:9.1f} /step")
    print(f"  GPU kernels (kernel cat)  {len([e for e in ev if e.get('cat')=='kernel'])/n:9.1f} /step")
