#!/usr/bin/env python3
"""Build the Turkish prompt files for `bench_serve_drive.py`, ONE FILE PER SET.

Why one file per set: acceptance comes from vLLM's global Prometheus counters, so a counter
delta is only attributable to a WINDOW.  Driving one measured phase per set is the only way to
get a per-set acceptance number.  Same reason `specbench_domains.py` splits by domain.

TWO PROMPT COLLECTIONS, REPORTED SEPARATELY -- THEY ARE NOT COMPARABLE
----------------------------------------------------------------------
  setA_*  chain-flow's own Turkish holdouts (`bench_data_tr/*.holdout.jsonl`), built for the
          27B run.  4 structured/instruct domains.  The 4B drafter was evaluated on these.
  setB_*  the turkishdspark benchmark sets (`/root/turkishdspark/data/benchmark/`).  The 9B
          drafter was evaluated on these.
Set A contains structured-JSON domains (function/tool calling) that score far higher than free
prose, so pooling the two produces a mean that describes neither.  Never pool them.

THE `<think>` PREFIX
--------------------
Both Turkish drafters were trained on prompts carrying a baked-in `<think>\\n\\n</think>\\n\\n`
prefix, i.e. the Qwen3.5 chat template rendered with `enable_thinking=False`.  Serving without
it puts the drafter off its training distribution.

`bench_data_tr` rows already ship that rendering in their `prompt` field (verified below: every
row ends with the prefix).  The turkishdspark rows ship raw `messages` (+ `tools`), so they are
rendered HERE with `enable_thinking=False` -- which produces byte-identical text to asking the
server for `chat_template_kwargs={"enable_thinking": false}` on the chat route.

Pre-rendering, rather than passing `chat_template_kwargs` per request, is deliberate:
  * it lets every arm be driven through `/v1/completions`, where the prompt reaching the model
    is the exact bytes in this file and cannot drift between two server processes;
  * it keeps `bench_serve_drive.py` usable unmodified, so the Turkish numbers come out of the
    same harness that produced the English ones in docs/BENCHMARKING.md.
`--verify-route` re-checks the equivalence against a live server.
"""
from __future__ import annotations

import argparse
import json
import pathlib

THINK = "<think>\n\n</think>\n\n"

SET_A = {  # chain-flow Turkish holdouts -- prompt already rendered
    "tr_funccall": "bench_data_tr/tr-function-calling-20k.holdout.jsonl",
    "tr_instruct": "bench_data_tr/tr-instructurca.holdout.jsonl",
    "tr_multiturn": "bench_data_tr/tr-multiturn.holdout.jsonl",
    "tr_toolcall": "bench_data_tr/tr-tool-calling-10k.holdout.jsonl",
}
SET_B = {  # turkishdspark benchmark -- raw messages, rendered here
    "tds_alpaca": "/root/turkishdspark/data/benchmark/turkish_alpaca_100.jsonl",
    "tds_holdout": "/root/turkishdspark/data/benchmark/turkish_holdout_100.jsonl",
    "tds_wikirag": "/root/turkishdspark/data/benchmark/wikirag_tr_100.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/shadeform/chain-flow")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B",
                    help="tokenizer whose chat template renders set B. 4B/9B/27B share it.")
    ap.add_argument("--per-set", type=int, default=50)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    manifest = []

    def emit(name: str, prompts: list[str], src: str) -> None:
        # HARD GATE: the training-distribution prefix must be on every single prompt. A prompt
        # without it silently measures the drafter off-distribution, which is the exact failure
        # the model cards warn about, and it is invisible in the throughput number.
        bad = [i for i, p in enumerate(prompts) if not p.endswith(THINK)]
        if bad:
            raise SystemExit(f"{name}: {len(bad)} prompts lack the <think> prefix "
                             f"(first at row {bad[0]}) -- refusing to write")
        out = outdir / f"{name}.jsonl"
        with out.open("w") as fh:
            for p in prompts:
                fh.write(json.dumps({"prompt": p, "domain": name,
                                     "output_tokens_count": args.output_tokens}) + "\n")
        ntok = [len(tok(p).input_ids) for p in prompts]
        manifest.append({"set": name, "src": src, "n": len(prompts),
                         "prompt_tokens_mean": sum(ntok) / len(ntok),
                         "prompt_tokens_max": max(ntok)})
        print(f"  {name:14s} {len(prompts):3d} prompts  ptok mean {sum(ntok)/len(ntok):6.1f} "
              f"max {max(ntok):5d}  <- {src}")

    print("SET A -- chain-flow Turkish holdouts (prompt pre-rendered upstream)")
    for name, rel in SET_A.items():
        rows = [json.loads(l) for l in (root / rel).open() if l.strip()]
        emit(name, [r["prompt"] for r in rows[: args.per_set]], rel)

    print("SET B -- turkishdspark benchmark (rendered here, enable_thinking=False)")
    for name, path in SET_B.items():
        rows = [json.loads(l) for l in open(path) if l.strip()]
        prompts = [
            tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
            for r in rows[: args.per_set]
        ]
        emit(name, prompts, path)

    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} files -> {outdir}")


if __name__ == "__main__":
    main()
