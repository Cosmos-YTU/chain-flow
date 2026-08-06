#!/usr/bin/env python3
"""Build ONE guidellm input file PER DOMAIN of RedHatAI/speculator_benchmarks.

Why per-domain files: the RedHat harness runs one guidellm benchmark per dataset file
and never pools.  Driving one benchmark per file is also the only way to get a per-domain
ACCEPTANCE number, because acceptance comes from vLLM's global Prometheus counters -- a
counter delta can only be attributed to a window, so the window has to be one domain.

Enumerated from the snapshot directory, not from a hardcoded list.  Findings:
  * the README documents 8 files but the snapshot ships 9; `tool_call.jsonl` (200 rows)
    is UNDOCUMENTED.
  * `writing.jsonl` and `question.jsonl` are the SAME BLOB (md5 f776f462, both symlink to
    e493606516e7562a1279a22c7aec5de2c4f38845).  The README calls #4 "MT_bench" and #8
    "Writing"; in this snapshot they are byte-identical MT-bench prompts.  Running both
    would double the cost for zero information, so `writing` is dropped as a duplicate
    and `question` is kept (the README's own name for that content is MT_bench).

Only `prompt` is common to all schemas; guidellm's default column mapper maps it to
text_column.  `output_tokens_count` makes guidellm send max_completion_tokens=N with
ignore_eos=true, so every arm emits exactly N tokens and does identical work.
"""
import argparse
import glob
import hashlib
import json
import pathlib

DEFAULT_SRC_GLOB = (
    "/root/.cache/huggingface/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/*"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None)
    ap.add_argument("--per-domain", type=int, default=10)
    ap.add_argument("--output-tokens", type=int, default=256)
    ap.add_argument("--warmup-per-domain", type=int, default=2)
    ap.add_argument("-o", "--outdir", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.src or sorted(glob.glob(DEFAULT_SRC_GLOB))[-1])
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(p.name for p in src.glob("*.jsonl"))
    # De-duplicate by content hash; keep the first name in sorted order.
    seen: dict[str, str] = {}
    dupes: list[tuple[str, str]] = []
    domains = []
    for name in files:
        h = hashlib.md5((src / name).read_bytes()).hexdigest()
        if h in seen:
            dupes.append((name, seen[h]))
            continue
        seen[h] = name
        domains.append(name)

    manifest = []
    per_domain_rows = {}
    for name in domains:
        dom = name[: -len(".jsonl")]
        with (src / name).open() as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        take = rows[: args.per_domain]
        per_domain_rows[dom] = take
        out = outdir / f"dom_{dom}.jsonl"
        with out.open("w") as fh:
            for r in take:
                rec = {"prompt": r["prompt"], "domain": dom}
                if args.output_tokens:
                    rec["output_tokens_count"] = args.output_tokens
                fh.write(json.dumps(rec) + "\n")
        manifest.append({"domain": dom, "file": name, "rows_in_file": len(rows),
                         "rows_used": len(take), "md5": seen_key(seen, name)})
        print(f"  {dom:18s} {len(take):3d}/{len(rows):4d} prompts -> {out.name}")

    # One small mixed file to warm compile/cudagraph before any measured window.
    warm = outdir / "dom_WARMUP.jsonl"
    with warm.open("w") as fh:
        for i in range(args.warmup_per_domain):
            for dom, rows in per_domain_rows.items():
                if i < len(rows):
                    rec = {"prompt": rows[i]["prompt"], "domain": dom}
                    if args.output_tokens:
                        rec["output_tokens_count"] = args.output_tokens
                    fh.write(json.dumps(rec) + "\n")

    (outdir / "manifest.json").write_text(json.dumps(
        {"src": str(src), "files_found": files, "domains": [m["domain"] for m in manifest],
         "duplicates_dropped": [{"dropped": a, "same_content_as": b} for a, b in dupes],
         "per_domain": args.per_domain, "output_tokens": args.output_tokens,
         "manifest": manifest}, indent=2))
    print(f"\nfiles found in snapshot ({len(files)}): {files}")
    for a, b in dupes:
        print(f"DROPPED DUPLICATE: {a} is byte-identical to {b}")
    print(f"domains used ({len(domains)}): {[m['domain'] for m in manifest]}")


def seen_key(seen, name):
    for h, n in seen.items():
        if n == name:
            return h
    return ""


if __name__ == "__main__":
    main()
