"""Restore a VAE checkpoint's weights from the HF repo that publishes them.

Training configs point `vae_dir` at a LOCAL directory, and those weights are byte-identical to the
`vae/model.safetensors` bundled inside the corresponding published drafter. Keeping a second copy
on disk buys nothing, so the local ones were deleted -- but `vae_dir` still has to resolve when a
training run starts, and discovering that at launch is an expensive way to find out.

  python scripts/restore_vae.py --all          # restore every mapped vae_dir that is missing
  python scripts/restore_vae.py --dir out/vae/ckpts/transformer-hidden-4bx-2560-latent640-fp16

Verifies the downloaded bytes against the sha256 the Hub reports, because a truncated download that
still loads is exactly the failure that would silently change a training run.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil

# local vae_dir basename -> repo whose vae/ holds the identical weights
MAP = {
    "transformer-hidden-4bx-2560-latent640-fp16":    "selimaktas/Flow-Drafter-4B-v2",
    "transformer-hidden-9bx-4096-latent1024-fp16":   "selimaktas/Flow-Drafter-9B-v2",
    "transformer-hidden-q3527bx-5120-latent1024-fp16": "selimaktas/Flow-Drafter-Qwen3.5-27B-v2",
    "transformer-hidden-4b-2560-latent640-fp16":     "selimaktas/Flow-Drafter-4B",
    "transformer-hidden-9b-4096-latent1024-fp16":    "selimaktas/Flow-Drafter-9B",
    "transformer-hidden-27b-5120-latent1024-fp16":   "selimaktas/Flow-Drafter-27B",
    "transformer-hidden-q3527b-5120-latent1024-fp16": "selimaktas/Flow-Drafter-Qwen3.5-27B",
}
ROOT = "out/vae/ckpts"


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def restore(name: str) -> bool:
    from huggingface_hub import HfApi, hf_hub_download

    repo = MAP[name]
    dst = os.path.join(ROOT, name, "model.safetensors")
    if os.path.exists(dst):
        print(f"  {name}: already present")
        return True
    want = None
    for s in HfApi().model_info(repo, files_metadata=True).siblings:
        if s.rfilename == "vae/model.safetensors":
            want = s.lfs.get("sha256") if isinstance(s.lfs, dict) else getattr(s.lfs, "sha256", None)
    src = hf_hub_download(repo, "vae/model.safetensors")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)
    got = sha256(dst)
    if want and got != want:
        os.remove(dst)
        print(f"  {name}: !! sha256 {got[:16]} != published {want[:16]} -- removed")
        return False
    print(f"  {name}: restored from {repo} (sha256 {got[:16]})")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dir", default=None, help="a vae_dir path or its basename")
    a = ap.parse_args()
    names = sorted(MAP) if a.all else ([os.path.basename(a.dir.rstrip("/"))] if a.dir else [])
    if not names:
        ap.error("pass --all or --dir")
    ok = True
    for n in names:
        if n not in MAP:
            print(f"  {n}: no published source known"); ok = False; continue
        ok &= restore(n)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
