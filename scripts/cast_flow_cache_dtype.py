"""Re-cast a flow window cache's hidden tensor to another dtype, in place.

Exists because `preprocess_flow_dataset.py` defaults `--hidden-dtype` to float32 while every
Turkish cache was built as float16, and `concat_flow_caches.py` rightly refuses to merge the two.
Re-running the preprocess is the obvious fix but needs the teacher states, which the pipeline
deletes as soon as a shard cache exists.

Narrowing is only safe because the collector stores `final_hidden` as float16 already -- the
float32 cache is a widening of float16 data, so casting back recovers the original bits exactly.
That is not assumed: every element is compared after the round trip and the file is left untouched
if a single one differs.

  python scripts/cast_flow_cache_dtype.py --dtype float16 data/flow_cache/_shard_*
"""
from __future__ import annotations

import argparse
import json
import os

import torch

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def convert(d: str, target: torch.dtype, name: str, chunk: int = 2_000_000) -> bool:
    meta_p = os.path.join(d, "metadata.json")
    hid_p = os.path.join(d, "hidden.pt")
    meta = json.load(open(meta_p))
    if meta.get("hidden_dtype") == name:
        print(f"  {os.path.basename(d):<48} already {name}")
        return True
    h = torch.load(hid_p, map_location="cpu")
    # Chunked so a 15 GB tensor does not need three copies resident at once.
    out = torch.empty(h.shape, dtype=target)
    for i in range(0, h.shape[0], chunk):
        sl = h[i:i + chunk]
        out[i:i + chunk] = sl.to(target)
        if not torch.equal(sl, out[i:i + chunk].to(sl.dtype)):
            print(f"  {os.path.basename(d):<48} LOSSY -- refusing, left unchanged")
            return False
    torch.save(out, hid_p + ".tmp")
    os.replace(hid_p + ".tmp", hid_p)          # atomic: a crash cannot leave a half-written cache
    meta["hidden_dtype"] = name
    json.dump(meta, open(meta_p, "w"))
    print(f"  {os.path.basename(d):<48} {h.dtype} -> {target}  "
          f"{h.numel()*h.element_size()/1e9:.1f} -> {out.numel()*out.element_size()/1e9:.1f} GB")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=sorted(DTYPES), required=True)
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    ok = True
    for d in sorted(a.dirs):
        if os.path.isfile(os.path.join(d, "metadata.json")):
            ok &= convert(d, DTYPES[a.dtype], a.dtype)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
