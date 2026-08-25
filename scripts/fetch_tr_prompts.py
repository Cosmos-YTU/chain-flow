"""Fetch the Turkish prompt corpora into the layout the collect configs expect.

The prompts are DATA, not source, so a fresh clone does not have them. They are published at
`selimaktas/turkish-flow-drafter-prompts` -- which also means reproducing a run no longer requires
the upstream turkishdspark corpus to be sitting on the machine, as it did when they were built.

  python scripts/fetch_tr_prompts.py            # v1 + v2 + ood, into ./bench_data_tr*/
"""
from __future__ import annotations

import os
import shutil

REPO = "selimaktas/turkish-flow-drafter-prompts"
DIRS = {"v1": "bench_data_tr", "v2": "bench_data_tr_v2", "ood": "bench_data_tr_ood"}


def main() -> int:
    from huggingface_hub import snapshot_download

    src = snapshot_download(REPO, repo_type="dataset")
    n = 0
    for sub, dst in DIRS.items():
        s = os.path.join(src, sub)
        if not os.path.isdir(s):
            print(f"  {sub}: absent in {REPO}")
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(os.listdir(s)):
            d = os.path.join(dst, f)
            if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(os.path.join(s, f)):
                continue
            shutil.copyfile(os.path.join(s, f), d)
            n += 1
        rows = sum(sum(1 for _ in open(os.path.join(dst, f)))
                   for f in os.listdir(dst) if f.endswith(".train.jsonl"))
        print(f"  {dst:<20} {rows:>7,} train rows")
    print(f"  ({n} files copied; identical files skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
