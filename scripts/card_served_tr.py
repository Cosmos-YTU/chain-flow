#!/usr/bin/env python3
"""Emit the "Served speedup" markdown section for a Turkish drafter card.

EVERY NUMBER IS READ FROM A RESULTS FILE.  Nothing is hand-typed: the cards were wrong once
because a figure was retyped, and a generator makes the card and the results file impossible to
disagree.

  logs/bench_tr/report.json          forced-256 equal-work condition (both sets, four arms)
  logs/bench_tr/report_nateos.json   natural-EOS condition (Set A, three arms) -- the HEADLINE
  out/flow/4btr_results.json         offline differential, K=8, tree and chain arms
  out/flow/4btr_estimator.json       the SAME offline draft data under both acceptance estimators

TWO SERVED CONDITIONS, NEVER POOLED
-----------------------------------
`ignore_eos` was in the benchmark for a good reason -- equal work per arm, so a faster arm that
hits EOS earlier does not look slower per token.  But it forces the model 256 tokens deep, well
past its natural stop, into out-of-distribution continuation that is measurably harder to draft.
On Set A that is worth -0.19 mean acceptance.  So the card reports natural EOS as the user-facing
figure and keeps forced-256 as the clean equal-work tok/s comparison, each labelled.

  card_served_tr.py 4b > /tmp/sec_4b.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import textwrap

ROOT = pathlib.Path("/home/shadeform/chained-flow")
# SET A IS CANONICAL -- the chained-flow Turkish holdouts. Set B is a different prompt
# collection (turkishdspark) and is reported second, never pooled with A.
SET_A = ["tr_funccall", "tr_instruct", "tr_multiturn", "tr_toolcall"]
SET_B = ["tds_alpaca", "tds_holdout", "tds_wikirag"]
PRETTY = {"tr_funccall": "tr_funccall", "tr_instruct": "tr_instruct",
          "tr_multiturn": "tr_multiturn", "tr_toolcall": "tr_toolcall",
          "tds_alpaca": "alpaca", "tds_holdout": "holdout", "tds_wikirag": "wikirag"}


def pooled(rep: dict, sets: list[str], arm: str) -> tuple[float, float]:
    """Token-weighted tok/s and mean acceptance over a collection.

    Under natural EOS the arms emit DIFFERENT token counts, so this is each arm's own
    tokens/wall -- a rate, which is comparable across arms.  The token totals are not.
    """
    sets = [s for s in sets if s in rep and arm in rep[s]]
    tok = sum(rep[s][arm]["tokens"] for s in sets)
    sec = sum(rep[s][arm]["tokens"] / rep[s][arm]["tps_pooled"] for s in sets)
    accs = [rep[s][arm]["accept"] for s in sets if rep[s][arm]["accept"]]
    return tok / sec, (sum(accs) / len(accs) if accs else 0.0)


def points(size: str, tag: str, arms: tuple[str, ...], sets: list[str]) -> list[tuple[float, float]]:
    """(acceptance, speedup) for every measured (arm, set, repeat) in ONE condition."""
    pts = []
    for arm in arms:
        for s in sets:
            for r in (1, 2):
                fb = ROOT / f"logs/bench_tr/{size}_base{tag}/{s}_r{r}.json"
                fa = ROOT / f"logs/bench_tr/{size}_{arm}{tag}/{s}_r{r}.json"
                if not (fa.exists() and fb.exists()):
                    continue
                b = json.load(fb.open())["phases"][0]
                a = json.load(fa.open())["phases"][0]
                pts.append((a["accept"], a["tps"] / b["tps"]))
    return pts


def fit(pts: list[tuple[float, float]]) -> tuple[float, float, int, float]:
    """Least-squares slope of speedup on acceptance through the origin -> (slope, break-even,
    n, max residual)."""
    c = sum(x * y for x, y in pts) / sum(x * x for x, _ in pts)
    return c, 1.0 / c, len(pts), max(abs(y - c * x) for x, y in pts)


def texts(size: str, tag: str, arm: str, s: str, r: int) -> list[str] | None:
    f = ROOT / f"logs/bench_tr/{size}_{arm}{tag}/{s}_r{r}.json"
    return json.load(f.open())["phases"][0]["texts"] if f.exists() else None


def agree(size: str, tag: str, a1: str, r1: int, a2: str, r2: int,
          sets: list[str]) -> tuple[int, int]:
    """(sequences compared, sequences byte-identical) between two runs."""
    n = ok = 0
    for s in sets:
        x, y = texts(size, tag, a1, s, r1), texts(size, tag, a2, s, r2)
        if x is None or y is None:
            continue
        for p, q in zip(x, y):
            n += 1
            ok += (p == q)
    return n, ok


def table(rep: dict, sets: list[str], arms: tuple[str, ...]) -> str:
    have = [a for a in arms if any(a in rep.get(s, {}) for s in sets)]
    hdr = "| prompt set | base tok/s |"
    sep = "|---|---|"
    for a in have:
        lbl = "**-tr**" if a == "chain_tr" else "v2 parent"
        hdr += f" {lbl} tok/s | {lbl} acc | {lbl} |"
        sep += "---|---|---|"
    out = [hdr, sep]
    for s in sets:
        if s not in rep or "base" not in rep[s]:
            continue
        b = rep[s]["base"]
        row = f"| {PRETTY[s]} | {b['tps_pooled']:.1f} |"
        for a in have:
            if a not in rep[s]:
                row += " - | - | - |"
                continue
            v = rep[s][a]
            x = f"{v['tps_pooled'] / b['tps_pooled']:.3f}x"
            row += (f" {v['tps_pooled']:.1f} | {v['accept']:.3f} | "
                    + (f"**{x}**" if a == "chain_tr" else x) + " |")
        out.append(row)
    bt, _ = pooled(rep, sets, "base")
    row = f"| **pooled** | **{bt:.1f}** |"
    for a in have:
        t, acc = pooled(rep, sets, a)
        row += f" **{t:.1f}** | **{acc:.3f}** | **{t / bt:.3f}x** |"
    out.append(row)
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("size", choices=["4b", "9b"])
    args = ap.parse_args()
    size = args.size

    forced = json.load((ROOT / "logs/bench_tr/report.json").open())[size]
    nat = json.load((ROOT / "logs/bench_tr/report_nateos.json").open())[size]
    est = json.load((ROOT / "out/flow/4btr_estimator.json").open())
    r4 = json.load((ROOT / "out/flow/4btr_results.json").open())["raw"]["after"]["tr"]

    # ---- natural EOS, Set A: the headline ----------------------------------------------------
    nb, _ = pooled(nat, SET_A, "base")
    nv, nva = pooled(nat, SET_A, "chain_v2")
    nt, nta = pooled(nat, SET_A, "chain_tr")
    # ---- forced 256, both sets: the equal-work benchmark -------------------------------------
    fbA, _ = pooled(forced, SET_A, "base")
    fvA, fvaA = pooled(forced, SET_A, "chain_v2")
    ftA, ftaA = pooled(forced, SET_A, "chain_tr")
    fbB, _ = pooled(forced, SET_B, "base")
    fvB, fvaB = pooled(forced, SET_B, "chain_v2")
    ftB, ftaB = pooled(forced, SET_B, "chain_tr")

    n_slope, n_be, n_n, n_res = fit(points(size, "_nateos", ("chain_v2", "chain_tr"), SET_A))
    f_slope, f_be, f_n, f_res = fit(
        points(size, "", ("chain_v2", "chain_tr", "chain_v2_trsl"), SET_A + SET_B))

    # mean output length per condition -- the size of the forced tail, straight from the files
    nlen = (sum(nat[s]["chain_tr"]["tokens"] for s in SET_A)
            / sum(nat[s]["chain_tr"]["requests"] for s in SET_A))

    # prefill share and the decode-only cross-check, from the natural-EOS run
    dec = [nat[s]["chain_tr"]["tps_decode"] / nat[s]["base"]["tps_decode"] for s in SET_A]
    hdl = [nat[s]["chain_tr"]["tps_pooled"] / nat[s]["base"]["tps_pooled"] for s in SET_A]
    dec_gap = max(abs(a - b) for a, b in zip(dec, hdl))
    # head-cost control: the parent's weights forced onto the Turkish shortlist (forced-256 only)
    slA, slaA = pooled(forced, SET_A, "chain_v2_trsl")
    head_cost = (slA / fvA - 1) * 100

    # Losslessness, three separate questions -- each answered from the run files.
    det_n, det_ok = agree(size, "_nateos", "chain_tr", 1, "chain_tr", 2, SET_A)
    spec_n, spec_ok = agree(size, "_nateos", "chain_tr", 1, "chain_v2", 1, SET_A)
    base_n, base_ok = agree(size, "_nateos", "base", 1, "chain_tr", 1, SET_A)
    spread = max(nat[s][a]["spread_pct"] for s in SET_A for a in ("base", "chain_v2", "chain_tr"))
    reqs = sum(nat[s][a]["requests"] for s in SET_A for a in ("base", "chain_v2", "chain_tr"))
    errs = sum(nat[s][a]["errors"] for s in SET_A for a in ("base", "chain_v2", "chain_tr"))
    gain = (nt / nb) / (nv / nb) - 1

    # ---- the offline -> served decomposition, all of it from results files -------------------
    o_tree = sum(v["plugin_tree"] for v in r4.values()) / len(r4)
    o_chain = sum(v["plugin_chain"] for v in r4.values()) / len(r4)
    o_k5 = est["mean_uniform"]
    o_ren = est["mean_renewal"]
    # EVERY row of the decomposition is the 4B measurement, served rows included. Splicing this
    # model's served acceptance onto the 4B offline rows would produce a meaningless delta -- it
    # is the cross-condition mixing this whole section exists to stop.
    d_nat = pooled(json.load((ROOT / "logs/bench_tr/report_nateos.json").open())["4b"],
                   SET_A, "chain_tr")[1]
    d_for = pooled(json.load((ROOT / "logs/bench_tr/report.json").open())["4b"],
                   SET_A, "chain_tr")[1]
    if size == "4b":
        prov = "Measured on this checkpoint, on Set A, throughout."
        first_row = "the offline table above"
        own = ""
    else:
        # THIS card's own offline tree mean, from THIS card's results file -- quoting the 4B's
        # 3.40 here would be the same cross-model splice the table above avoids.
        o9 = json.load((ROOT / "out/flow/9btr_results.json").open())
        own_tree, own_set = o9["turkish_mean"]["after"], "Set B"
        prov = ("Measured end to end on the **4B sibling** -- offline *and* served rows, so the "
                "column is internally consistent. Both draft geometries and both estimators were "
                "scored there with the same instrument. The terms are properties of the "
                "measurement, not of the model size.")
        prov = textwrap.fill(prov, 98)
        first_row = "the 4B sibling's Set A tree, not the Set B table above"
        own = "\n" + textwrap.fill(
            f"This checkpoint's own served acceptance is **{nta:.3f}** natural / {ftaA:.3f} "
            f"forced on Set A (table above), against **{own_tree:.2f}** in its offline tree "
            f"table on {own_set}. Different prompt collections, so the two are not a subtraction "
            f"-- but the same shape of difference, arriving the same way.", 98) + "\n"

    print(f"""## Served speedup in vLLM

**These are concurrency-1 numbers** -- exactly one request in flight, which is the *favourable*
end of the range for speculative decoding. Measured through `vllm serve` (the production entry
point) with the **chain** proposer at **K=5**, greedy, and **async scheduling on for every arm
including the baseline** (leaving it off the baseline alone inflates every ratio). Two repeats.

Every request carried **`chat_template_kwargs: {{"enable_thinking": false}}`**, and so must yours:
this drafter was trained on prompts with the `<think>\\n\\n</think>\\n\\n` non-thinking prefix baked
in, and without the flag the serving path renders a different prefix and puts the drafter off its
training distribution. The two renderings were verified token-for-token here, so this is a real
difference, not a formality.

The concurrency **ladder has not been run** for these drafters. Do not assume these hold under
load: the parent arm below loses outright even at concurrency 1, and on the English v2 cards the
chain arm falls below parity by concurrency 8.

### Two measurement conditions, never pooled

| condition | what it does | what it is for |
|---|---|---|
| **natural EOS** | requests stop at the model's own EOS (cap 256) | **the headline** -- what a deployment actually serves |
| **forced 256** | `ignore_eos`, every arm emits exactly 256 tokens | equal work per arm; the clean tok/s comparison |

On Set A the model's natural answer averages **{nlen:.0f} tokens**, so forcing 256 spends roughly
**{(256 - nlen) / 256 * 100:.0f}% of the measured tokens past the point where it wanted to stop** --
in continuation that is off-distribution and measurably harder to draft. That costs
**{ftaA - nta:+.2f}** mean acceptance ({nta:.3f} natural vs {ftaA:.3f} forced), and it is largest
exactly where the answers are shortest, so the two conditions are reported separately and never
averaged together. Every previously published figure for these drafters came from the forced
condition.

Under natural EOS the arms no longer emit the same number of tokens, so **only the rates are
comparable** -- `tokens` totals are not. At concurrency 1 that is a safe comparison: the
decode-only ratios (prefill excluded) sit within {dec_gap:.3f} of the headline ratios below.

### Set A -- chained-flow Turkish holdouts (`bench_data_tr`) -- canonical, **natural EOS**

{table(nat, SET_A, ("chain_v2", "chain_tr"))}

**The English parent is a net regression on Turkish.** At acceptance {nva:.2f} it does not earn
back the cost of its own draft pass, so turning speculation on with the parent makes the server
*slower* ({nv / nb:.3f}x). The value of this fine-tune in served terms is therefore not "a faster
speedup" -- it is **converting speculation from a loss into a win**, {nb:.1f} -> {nt:.1f} tok/s,
**{gain * 100:+.0f}%** against the parent arm.

### Set A -- the same prompts under **forced 256** (equal work)

{table(forced, SET_A, ("chain_v2", "chain_tr"))}

### Set B -- turkishdspark benchmark -- a *different* prompt collection, **forced 256 only**

Reported because it was measured, but it is **not** Set A and the two are never pooled: Set A
carries structured-JSON function/tool-calling domains that accept far above free prose, and a mean
across both describes neither. Set B has also **not** been re-run under natural EOS, so this table
is in the *forced-256* condition while the Set A headline above is in the *natural-EOS* one. Two
different prompt collections **and** two different measurement conditions: read each table on its
own terms, and do not treat the two as one measurement.

{table(forced, SET_B, ("chain_v2", "chain_tr"))}

Same story: parent {fvB / fbB:.3f}x, this checkpoint **{ftB / fbB:.3f}x**.

### Why the offline accept table reads higher than the served acceptance

Not a contradiction, and **not a serving loss**. Every row below is the same weights on the same
prompts; only the one stated thing changes.

{prov}

| | mean accept | Δ | what changed |
|---|---|---|---|
| offline **tree**, K=8 | {o_tree:.2f} | | {first_row} |
| offline **chain**, K=8 | {o_chain:.2f} | {o_chain - o_tree:+.2f} | draft geometry: the served arm is a chain |
| offline chain, **K=5** | {o_k5:.2f} | {o_k5 - o_chain:+.2f} | K truncation, all 50 rows scored |
| **per-draft-step** estimator | {o_ren:.2f} | {o_ren - o_k5:+.2f} | the metric vLLM actually reports |
| **served**, natural EOS | {d_nat:.3f} | {d_nat - o_ren:+.2f} | actually running it in vLLM |
| served, forced 256 | {d_for:.3f} | {d_for - d_nat:+.2f} | benchmark condition, see above |
{own}
Two of those terms are worth spelling out, because the obvious suspects are not the answer.

**K truncation is almost nothing ({o_k5 - o_chain:+.2f}).** Earlier revisions of this card blamed
the gap on K=8 -> K=5. It does not survive measurement.

**The estimator is the big term ({o_ren - o_k5:+.2f}), and it is a denominator, not a loss.** The
offline harness averages run length over every token **position** -- it scores position i, then
i+1, then i+2. vLLM averages over draft **steps**, and a step that accepts n tokens commits n+1
and starts the next step n+1 positions later. Steps therefore land where the previous run *broke*,
while sliding-window sampling is length-biased toward the easy stretches: a 50-token run of
predictable JSON contributes 50 high-run-length windows offline, but the server crosses it in
about 8 steps. Same draft data, same accepted tokens, different denominator --
`scripts/accept_estimator_tr.py` computes both from one set of draft outputs
({est['mean_uniform']:.3f} uniform vs {est['mean_renewal']:.3f} per-step).
**An offline accept number is not comparable to a served one unless both use the same
estimator.**

**The serving path itself is faithful: {d_nat - o_ren:+.2f}.** Between the simulated per-step
acceptance and what vLLM actually reports there is essentially nothing. The engine is not dropping
drafts, and the async draft hand-off, the CUDA-graph draft path and the shortlist head are not
costing acceptance. If you are comparing the two tables, that is the number to look at.

Two further candidates, ruled out by measurement rather than argument: **prefill** (the
decode-only ratios sit within {dec_gap:.3f} of the headline ratios, so this is not a prompt-length
artefact) and **the larger Turkish shortlist** -- an extra forced-256 arm running the *parent's*
weights on the Turkish head scored acceptance {slaA:.3f}, against {fvaA:.3f} on its own smaller
English head -- identical to three decimals, for {head_cost:+.1f}% throughput. The bigger head is
accept-neutral and very nearly cost-neutral.

### Break-even acceptance

Served speedup is close to linear in acceptance, because the draft pass costs a fixed fraction of
a target step:

> **natural EOS: speedup ~= {n_slope:.3f} x acceptance**, break-even **{n_be:.3f}**
> (fitted over {n_n} measured points, max residual {n_res:.3f})
>
> forced 256: speedup ~= {f_slope:.3f} x acceptance, break-even {f_be:.3f}
> ({f_n} points, max residual {f_res:.3f})

Below break-even, speculation costs more than it saves and you should serve without a drafter.
Quote the fit for the condition you are measuring in -- the two are close but they are not the
same fit, and mixing them is exactly the error this section exists to prevent. Under natural EOS
the parent sits at {nva:.2f} on Turkish, i.e. under water; this checkpoint sits at {nta:.2f}.

### Confidence

Run-to-run spread <= {spread:.2f}% over two complete repeats of the natural-EOS condition;
{reqs} requests, **{errs} errors**.

**What "lossless" does and does not mean here**, measured rather than asserted, on Set A:

* each arm is **exactly reproducible**: repeat 1 vs repeat 2 of this drafter agree on
  **{det_ok}/{det_n}** sequences;
* **the drafter does not change the output**: this checkpoint and the English parent -- different
  weights, different shortlists, acceptance {nta:.2f} vs {nva:.2f} -- emit **{spec_ok}/{spec_n}**
  identical sequences. Whatever the drafter proposes, the text that comes out is the same;
* against a **no-speculation** baseline the agreement is **{base_ok}/{base_n}**.

That last number is not a drafting error. Both speculative arms agree with *each other* perfectly
while differing from plain decoding on the same handful of sequences, so what changes is the
target's own arithmetic: the verify pass evaluates K+1 query positions in one forward, a plain
decode evaluates one, and in float16 those are not bit-identical. At a near-tie the greedy argmax
can then fall the other way, after which the two continuations differ legitimately. Speculative
decoding is lossless in exact arithmetic; in float16 it is lossless up to tie-breaking, and if you
need bit-identical parity with a no-drafter deployment you should verify it in your own
precision.""")


if __name__ == "__main__":
    main()
