#!/usr/bin/env python3
"""Refresh the THREE v2 model cards with the measured per-domain acceptance + speedup.

Targets the **v2** repos.  NOTE: scripts/push_27b_and_cards.py targets the *v1* repos
(selimaktas/Flow-Drafter-4B / -9B / -27B) and hardcodes base_model Qwen3.6-27B for the
27B card -- it must NOT be reused for these.  Repos here:

    4B  -> selimaktas/Flow-Drafter-4B-v2                (base Qwen/Qwen3.5-4B)
    9B  -> selimaktas/Flow-Drafter-9B-v2                (base Qwen/Qwen3.5-9B)
    27B -> selimaktas/Flow-Drafter-Qwen3.5-27B-v2       (base Qwen/Qwen3.5-27B)

CHAIN ARM ONLY.  The tree arm is not quoted: it is greedy-only in practice and real
serving uses sampling, which the chain handles natively at full speed.

This script only rewrites README.md; it never touches weights.

    push_v2_cards_specbench.py --dry-run     # print the card text, push nothing
    push_v2_cards_specbench.py --diff        # unified diff vs what is live now
    push_v2_cards_specbench.py --push
"""
import argparse
import difflib
import json
import pathlib

ROOT = pathlib.Path("/home/shadeform/chained-flow")
RESULTS = ROOT / "logs/specbench_dom"

# size -> (repo, base_model, card_name, v1_repo, v1_name, train_rows)
REPOS = {
    "4b":  ("selimaktas/Flow-Drafter-4B-v2", "Qwen/Qwen3.5-4B", "Flow-Drafter-4B-v2",
            "selimaktas/Flow-Drafter-4B", "Flow-Drafter-4B", "~46k rows"),
    "9b":  ("selimaktas/Flow-Drafter-9B-v2", "Qwen/Qwen3.5-9B", "Flow-Drafter-9B-v2",
            "selimaktas/Flow-Drafter-9B", "Flow-Drafter-9B", "~46k rows"),
    "27b": ("selimaktas/Flow-Drafter-Qwen3.5-27B-v2", "Qwen/Qwen3.5-27B",
            "Flow-Drafter-Qwen3.5-27B-v2", "selimaktas/Flow-Drafter-Qwen3.5-27B",
            "Flow-Drafter-Qwen3.5-27B", "~30k rows"),
}

# The v1-vs-v2 OFFLINE tree-accept tables already on the live cards; kept, but relabelled
# so they can never be confused with the served numbers this script adds.
OFFLINE = {
    "4b": [("gsm8k held-out", 5.25, 5.51), ("math_reasoning", 5.44, 5.47),
           ("HumanEval (code)", 4.02, 4.22), ("writing (prose)", 3.28, 3.49),
           ("qa (short free-form)", 2.38, 2.67), ("summarization (prose)", 2.05, 2.38)],
    "9b": [("HumanEval (code)", 5.12, 5.10), ("math_reasoning", 5.77, 5.54),
           ("qa (short free-form)", 3.18, 3.31), ("summarization (prose)", 2.78, 3.13),
           ("writing (prose)", 2.87, 3.07)],
    "27b": [("gsm8k held-out", 5.67, 5.69), ("math_reasoning", 5.30, 5.29),
            ("HumanEval (code)", 4.42, 4.52), ("writing (prose)", 3.27, 3.65),
            ("qa (short free-form)", 2.72, 2.83), ("summarization (prose)", 2.29, 2.63)],
}

# Concurrency disclosure. The headline table is a SINGLE-REQUEST measurement, which is the
# favourable end of the range; a reader must not deploy expecting it on a loaded server.
# The ladder figures quoted here are this project's own measurements from
# docs/BENCHMARKING.md (vllm/serve_ladder.sh), NOT part of the specbench run above -- they
# use a different prompt mix, so their concurrency-1 value differs from the table. That is
# stated in the text rather than papered over.
CONCURRENCY_NOTE = {
    "4b": """### These are concurrency-1 numbers, and the gain does not survive load

Every figure above is a **single-request measurement** — one request in flight at all times.
That is the most favourable end of the range.

A speculative step hands the target `(K+1) x batch` query positions instead of `batch`. At
concurrency 1 that is free, because decode is memory-bound and the extra positions ride along in
the same weight read; once the batch is wide enough to be compute-bound it is not free. On this
project's own concurrency ladder at 4B (`vllm/serve_ladder.sh`, a different prompt mix, so its
concurrency-1 value is not the table above) the chain arm measures **1.19x at concurrency 1,
1.00x at 4, 0.90x at 8 and 0.44x at 64** with no cutoff — i.e. **a loaded 4B server is faster
with speculation turned off**. Acceptance is flat across that whole ladder (1.854-1.861), so this
is draft *cost*, not draft quality.

`CF_SPEC_MAX_BATCH` ships **on by default** and disengages speculation above a decode batch of 4
at this size, which holds a loaded server at **0.94-0.96x** of no-speculation instead of 0.44x.
Parity under load is the goal there, not speedup. **Do not deploy this expecting the
concurrency-1 figure on a busy server.**""",
    "9b": """### These are concurrency-1 numbers, and the gain narrows under load

Every figure above is a **single-request measurement** — one request in flight at all times.
That is the most favourable end of the range.

A speculative step hands the target `(K+1) x batch` query positions instead of `batch`. At
concurrency 1 that is free, because decode is memory-bound and the extra positions ride along in
the same weight read; once the batch is wide enough to be compute-bound it is not. The effect is
measured most fully at 4B, where this project's concurrency ladder shows the chain arm going from
1.19x at concurrency 1 to 1.00x at 4 and 0.44x at 64 with no cutoff.

`CF_SPEC_MAX_BATCH` ships **on by default** and resolves to a decode batch of 4 at this size,
disengaging speculation above it so a loaded server stays near parity with no-speculation rather
than falling below it. **Do not deploy this expecting the concurrency-1 figure on a busy
server.**""",
    "27b": """### These are concurrency-1 numbers, and the gain narrows under load

Every figure above is a **single-request measurement** — one request in flight at all times.
That is the most favourable end of the range.

27B holds its advantage further than the smaller sizes, because the target forward is expensive
enough to keep paying for the drafts. On this project's own concurrency ladder
(`vllm/serve_ladder.sh`, a different prompt mix, so its concurrency-1 value is not the table
above) the 27B chain arm measures **1.62x at concurrency 1, 1.46x at 4, 1.27x at 8 and 1.16x at
16** — narrowing, but still above parity.

Above roughly concurrency 16, **do not serve 27B with speculation at all**: `--speculative-config`
more than halves the engine's KV cache, capping the 27B decode batch at ~28 against the base
engine's 64. `CF_SPEC_MAX_BATCH` ships **on by default** and resolves to 16 here, disengaging
speculation past that point. **Do not deploy this expecting the concurrency-1 figure on a busy
server.**""",
}

DOM_ORDER = ["HumanEval", "math_reasoning", "qa", "question", "rag",
             "summarization", "tool_call", "translation"]
DOM_LABEL = {"HumanEval": "HumanEval (code)", "math_reasoning": "math_reasoning",
             "qa": "qa (short free-form)", "question": "question (MT-bench)",
             "rag": "rag", "summarization": "summarization",
             "tool_call": "tool_call", "translation": "translation (de->en)"}


EXPECTED_DOMAINS = set(DOM_ORDER)
REQUIRED_REPEATS = 3


def load(size):
    """The results file is MANDATORY and is validated before a single number is used.

    Every figure in a published card comes from this file. There is no default, no
    fallback and no hardcoded number to fall back to, so a card physically cannot ship
    with placeholder or stale figures: if the run did not measure it, the push aborts.
    """
    p = RESULTS / f"report_{size}.json"
    if not p.exists():
        raise SystemExit(
            f"REFUSING TO BUILD {size} CARD: missing {p}.\n"
            f"Every number in the card must come from a measured run; there is no "
            f"fallback. Run:\n"
            f"  vllm/specbench_dom_report.py --size {size} --json-out {p}")
    r = json.loads(p.read_text())
    problems = []

    doms = set(r.get("domains", {}))
    if doms != EXPECTED_DOMAINS:
        problems.append(f"domains are {sorted(doms)}, expected {sorted(EXPECTED_DOMAINS)}")
    for d, v in r.get("domains", {}).items():
        for arm in ("base", "chain"):
            if arm not in v:
                problems.append(f"domain {d} has no {arm} arm")
            elif v[arm].get("reps", 0) < REQUIRED_REPEATS:
                problems.append(f"domain {d}/{arm} has {v[arm].get('reps')} repeats, "
                                f"need {REQUIRED_REPEATS}")
    for arm in ("base", "chain"):
        if arm not in r.get("pooled", {}):
            problems.append(f"no pooled {arm} row")
        elif r["pooled"][arm].get("complete_repeats", 0) < REQUIRED_REPEATS:
            problems.append(f"pooled {arm} has "
                            f"{r['pooled'][arm].get('complete_repeats')} complete repeats")
    if r.get("equal_work_violations"):
        problems.append(f"equal-work violations: {r['equal_work_violations']}")
    if not r.get("provenance", {}).get("generated_at"):
        problems.append("results file carries no provenance stamp")
    conc = r.get("concurrency", {})
    if not conc or conc.get("max", 99) > 1.01:
        problems.append(f"concurrency not verified as 1 (got {conc})")

    if problems:
        raise SystemExit(f"REFUSING TO BUILD {size} CARD -- results file is not a "
                         f"complete measured run:\n  - " + "\n  - ".join(problems))
    return r


def served_table(r):
    """Per-domain chain acceptance + chain speedup, plus the pooled row."""
    lines = ["| domain | acceptance (tokens/step) | speedup vs base |",
             "|---|---|---|"]
    for d in DOM_ORDER:
        v = r["domains"].get(d)
        if not v or "base" not in v or "chain" not in v:
            continue
        sp = v["chain"]["pooled_tps"] / v["base"]["pooled_tps"]
        lines.append(f"| {DOM_LABEL.get(d, d)} | {v['chain']['accept']:.2f} | {sp:.2f}x |")
    p = r["pooled"]
    sp = p["chain"]["pooled_tps"] / p["base"]["pooled_tps"]
    lines.append(f"| **POOLED (all 8 domains)** | **{p['chain']['accept']:.2f}** "
                 f"| **{sp:.2f}x** |")
    return "\n".join(lines)


def regressions(r, thresh=1.0):
    out = []
    for d in DOM_ORDER:
        v = r["domains"].get(d)
        if not v or "base" not in v or "chain" not in v:
            continue
        sp = v["chain"]["pooled_tps"] / v["base"]["pooled_tps"]
        if sp < thresh:
            out.append((DOM_LABEL.get(d, d).split(" (")[0], sp))
    return out


def honesty(size, r):
    p = r["pooled"]
    sp = p["chain"]["pooled_tps"] / p["base"]["pooled_tps"]
    regs = regressions(r)
    if regs:
        reg_txt = ", ".join(f"{d} {s:.2f}x" for d, s in regs)
        warn = (f"\n**This drafter does not win everywhere.** On {len(regs)} of the 8 domains the chain "
                f"arm is a **regression** — {reg_txt}. Speculative decoding costs a draft pass on every "
                f"step, so where acceptance is low the draft does not pay for itself. The pooled figure "
                f"above ({sp:.2f}x) already includes those losses; quote it rather than the best domain.\n")
    else:
        warn = (f"\nEvery domain measured at or above parity, but the domains differ widely. Quote the "
                f"pooled figure ({sp:.2f}x) rather than the best domain.\n")
    if size == "4b":
        warn += ("\nAt 4B the base model's decode step is cheap, so the drafter has little headroom to "
                 "buy back; this is the size where speculative decoding is hardest to justify. The 9B "
                 "and 27B drafters return more.\n")
    return warn


def card(size):
    repo, base, name, v1repo, v1name, rows = REPOS[size]
    r = load(size)
    p = r["pooled"]
    sp = p["chain"]["pooled_tps"] / p["base"]["pooled_tps"]
    off = "\n".join(f"| {d} | {a:.2f} | {b:.2f} | {b - a:+.2f} |"
                    for d, a, b in OFFLINE[size])
    others = [t for t in ("4b", "9b", "27b") if t != size]
    comp = ", ".join(f"[{REPOS[t][2]}](https://huggingface.co/{REPOS[t][0]})"
                     for t in others)
    spread = max(p[a]["pooled_spread_pct"] for a in p)
    nrep = max(v["chain"]["reps"] for v in r["domains"].values() if "chain" in v)
    nprompt = max(v["chain"]["n_req"] for v in r["domains"].values() if "chain" in v)
    conc = r.get("concurrency", {"mean": 1.0, "max": 1.0})
    div = r.get("divergence", {}).get("chain", {})
    ndiv, ncom = div.get("total_diff", 0), div.get("total_common", 0)
    cconc, cmax = conc["mean"], conc["max"]
    return f"""---
license: apache-2.0
base_model: {base}
tags:
- speculative-decoding
- draft-model
- flow-matching
- chained-flow
---

# {name}

Expanded-data revision of [{v1name}](https://huggingface.co/{v1repo}) — a
speculative-decoding draft model for `{base}` (Chained-Flow, joint-VAE drafter).

**What changed in v2:** trained on a broader, diversity-weighted mix ({rows}, 10 sources) that adds
multi-turn chat (UltraChat), diverse instructions (No-Robots), creative prose (WritingPrompts),
summarization (CNN/DailyMail) and translation (OPUS) on top of the original math/code/STEM core — to lift
acceptance on the low-predictability domains (prose / QA / summarization) toward uniform speedup.

## Measured under vLLM — acceptance and speedup, per domain

> **These are single-request measurements: concurrency 1, one request in flight at all times.**
> They are the favourable end of the range and do **not** hold on a loaded server — see
> "These are concurrency-1 numbers" below.

Benchmark: **RedHatAI/speculator_benchmarks**, all **8 distinct domains** in the published snapshot
(`HumanEval`, `math_reasoning`, `qa`, `question` (MT-bench), `rag`, `summarization`, `tool_call`,
`translation`), **{nprompt} prompts per domain**, run as **one benchmark per domain** — the dataset is
never pooled into a single split. (The snapshot also ships `writing.jsonl`, which is **byte-identical**
to `question.jsonl`, so it is not counted twice.)

Harness: **guidellm 0.6.0** driving `vllm serve` (**vLLM 0.25.1**) over the OpenAI HTTP API.
Arm: **chain, K=5** — a linear 5-token draft, which is the shipping configuration and supports sampling.
**Concurrency 1** (guidellm `synchronous` profile, no request rate; measured request concurrency
mean {cconc:.4f}, max {cmax:.1f}), greedy (`temperature=0`), a **fixed 256 output tokens** per request
(`ignore_eos`) so both arms do identical work, and **asynchronous scheduling enabled for every arm
including the baseline** — the baseline is not handicapped. {nrep} repeats per arm; run-to-run spread on
the pooled figure was ≤{spread:.1f}%.

**Definitions.** *acceptance* = mean accepted length = `1 + num_accepted_tokens / num_drafts`, read from
the delta of vLLM's own Prometheus counters — **the bonus token is included**, so a non-speculative
baseline is 1.00 by definition and the useful range here is 1.00–6.00 for K=5.
*speedup* = chain tok/s ÷ base tok/s, where tok/s is `sum(output tokens) / wall duration` with
**prefill included in the denominator**.

{served_table(r)}
{honesty(size, r)}
{CONCURRENCY_NOTE[size]}

Verification is exact, so the speculative output matches what the base model would have produced. At
fp16 a 1-ULP difference in the target logits can still flip a near-tied token and send the continuation
down a different (equally valid) path; in this run {ncom - ndiv}/{ncom} greedy completions were
byte-identical to the baseline.

## Acceptance — v2 vs v1 (offline harness, NOT the served numbers above)

Offline tree-accept from `scripts/diff_plugin_vs_harness.py`; this is the *training* comparison that
motivated v2, measured with a different (tree) draft geometry, so it reads higher than the served table.
Do not mix the two.

| domain | v1 (5 tech sources) | v2 (10 diverse sources) | Δ |
|---|---|---|---|
{off}

## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, jointly-trained VAE)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint

Drafts all K future hidden states in one flow pass and the base model verifies them. Trained 2-GPU DDP.

_Companion models: {comp}._
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--diff", action="store_true")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    out = {}
    for size in ("4b", "9b", "27b"):
        out[size] = card(size)
        (RESULTS / f"card_{size}_new.md").write_text(out[size])

    if args.dry_run or not (args.diff or args.push):
        for size, txt in out.items():
            print(f"\n{'='*100}\n===== {REPOS[size][0]} README.md\n{'='*100}\n{txt}")

    if args.diff:
        from huggingface_hub import hf_hub_download
        for size, txt in out.items():
            cur = pathlib.Path(hf_hub_download(REPOS[size][0], "README.md")).read_text()
            print(f"\n{'='*100}\n===== DIFF {REPOS[size][0]}\n{'='*100}")
            print("".join(difflib.unified_diff(
                cur.splitlines(True), txt.splitlines(True),
                fromfile="live/README.md", tofile="new/README.md")))

    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        for size, txt in out.items():
            p = RESULTS / f"card_{size}_new.md"
            api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md",
                            repo_id=REPOS[size][0], repo_type="model",
                            commit_message="Card: measured per-domain acceptance + speedup "
                                           "on RedHatAI/speculator_benchmarks (guidellm 0.6.0, "
                                           "vLLM 0.25.1, batch 1, async scheduling on all arms)")
            print(f"pushed -> https://huggingface.co/{REPOS[size][0]}")


if __name__ == "__main__":
    main()
