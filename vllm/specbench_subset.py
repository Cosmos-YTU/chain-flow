#!/usr/bin/env python3
"""Build the guidellm input file for the RedHat AI `speculator_benchmarks` dataset.

The dataset ships SEVEN .jsonl files with INCOMPATIBLE schemas (HumanEval carries
task_id/entry_point/canonical_solution/test; the other six carry
question_id/category/reference).  `load_dataset("RedHatAI/speculator_benchmarks")`
therefore fails -- it tries to concatenate them into one `train` split and raises
DatasetGenerationError ("column names don't match").  Each file has to be loaded on
its own via `data_files=`, which is also what the RedHat harness does (it runs one
guidellm benchmark per file and never pools them).

Only one field is shared by all seven: `prompt`.  guidellm's default column mapper
maps `prompt` -> text_column, so that field alone drives the benchmark.

The dataset specifies NOTHING about generation: no max/expected output length, no
temperature, no reference metrics.  Those come from the harness, not the data.  With
no `output_tokens_count` column guidellm sets no max_tokens at all and every request
runs to EOS (median 1160 / p95 8127 output tokens on Qwen3.5-4B).  Emitting an
explicit `output_tokens_count` makes guidellm send max_completion_tokens=N with
ignore_eos=true, so every request emits exactly N tokens -- equal work per arm.

    specbench_subset.py --per-domain 10 --output-tokens 256 -o subset.jsonl
"""
import argparse
import json
import pathlib

DEFAULT_SRC = (
    "/root/.cache/huggingface/hub/datasets--RedHatAI--speculator_benchmarks/"
    "snapshots/2ae86affa2cb97a972b7fc681dd51c04fbff083e"
)
# The dataset's own file order, used verbatim so the subset is reproducible.
FILES = [
    "HumanEval.jsonl", "math_reasoning.jsonl", "qa.jsonl", "rag.jsonl",
    "summarization.jsonl", "translation.jsonl", "writing.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--per-domain", type=int, default=10,
                    help="first N rows of each file, in file order (no shuffle)")
    ap.add_argument("--output-tokens", type=int, default=0,
                    help="0 = leave unset, i.e. generate to EOS as the harness ships")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    per_domain = []
    for name in FILES:
        with (src / name).open() as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        per_domain.append([(row["prompt"], name[: -len(".jsonl")])
                           for row in lines[: args.per_domain]])

    # Round-robin interleave the seven domains so that ANY prefix of the file (i.e.
    # any guidellm --max-requests N) is domain-balanced. Same convention bench_cf.sh
    # uses for CF_POFF.
    rows = []
    for i in range(args.per_domain):
        for domain_rows in per_domain:
            if i < len(domain_rows):
                prompt, domain = domain_rows[i]
                rec = {"prompt": prompt, "domain": domain}
                if args.output_tokens:
                    rec["output_tokens_count"] = args.output_tokens
                rows.append(rec)

    with open(args.out, "w") as fh:
        for rec in rows:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(rows)} prompts ({args.per_domain}/domain x {len(FILES)}) -> {args.out}")


if __name__ == "__main__":
    main()
