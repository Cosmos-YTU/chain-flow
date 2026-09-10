"""Rebuild THIS CHECKOUT's shortlist (`out/flow/shortlist_q3527b.pt`, and the copy that ships
inside the wheel as `src/chain_flow/data/shortlist_qwen3_5.pt`).

The generic builder now lives in the PACKAGE (`chain_flow.shortlist.build`, exposed as
`chain-flow build-shortlist`), because the fallback warning a pip user sees has to name a
command that user actually has. This file is only the repo-specific invocation: it knows where
this checkout keeps its corpora.

  python scripts/build_shortlist.py                 # -> out/flow/shortlist_q3527b.pt
  python scripts/build_shortlist.py --stats         # coverage curve only

Sources (union -- see chain_flow/shortlist.py for why it is a union and not a top-K):
  * data/flow_cache/*/input_ids.pt   the teacher/bench corpora the drafter was trained on
  * bench_data/*.jsonl prompts       includes `rag` and `translation`, which have no cache
  * every tokenizer special/added token
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from chain_flow.shortlist import build  # noqa: E402

if __name__ == "__main__":
    argv = sys.argv[1:]
    defaults = []
    if not any(a.startswith("--out") for a in argv):
        defaults += ["--out", os.path.join(REPO, "out/flow/shortlist_q3527b.pt")]
    if not any(a.startswith("--tokenizer") for a in argv):
        defaults += ["--tokenizer", "Qwen/Qwen3.5-27B"]
    if not any(a.startswith("--vocab-size") for a in argv):
        defaults += ["--vocab-size", "248320"]
    defaults += ["--ids", os.path.join(REPO, "data/flow_cache/*/input_ids.pt"),
                 "--jsonl", os.path.join(REPO, "bench_data/*.jsonl"), "--text-field", "prompt"]
    raise SystemExit(build(defaults + argv))
