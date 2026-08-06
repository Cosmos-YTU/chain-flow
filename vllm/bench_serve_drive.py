#!/usr/bin/env python3
"""Drive a running `vllm serve` at a FIXED CONCURRENCY and report throughput + acceptance.

WHY NOT `vllm bench serve` / guidellm
-------------------------------------
Both are rate-based load generators: you ask for N requests/s and they report what the
server managed.  The question here is the opposite one -- "what does the engine do when
exactly C requests are in flight" -- because every flag this repo cares about is gated on
the DECODE BATCH SIZE, and a rate-driven run wanders across batch sizes.  A semaphore of
size C pins it.  (`vllm bench serve --max-concurrency C` does pin it, but it does not keep
the completion TEXT, and see below.)

The second reason is the one that has bitten this project three times: a 1-ULP fp16 tie
sends two arms into different text, and then their throughputs are not comparable because
they did different work.  So this driver keeps every completion and `--diff` compares two
runs token-for-token before any speed number is believed.

DATASET: `logs/specbench/data/subset_fixed256.jsonl` -- the RedHat AI
`speculator_benchmarks` prompts, 10 per domain x 7 domains, round-robin interleaved, with
`output_tokens_count=256`.  We send max_tokens=256 + ignore_eos=true so EVERY arm emits
exactly the same number of tokens and throughput is a pure speed comparison.  Built by
`specbench_subset.py`; kept as a file so a run is reproducible without the HF cache.

ACCEPTANCE comes from the server's own Prometheus counters, delta'd across the measured
window:  accept_len = 1 + accepted/drafts.  That is the same formula speculators'
`run_vllm.py` uses.  It is read from the ENGINE, not from our proposer object, because
under `vllm serve` the proposer lives in the EngineCore subprocess and is unreachable.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import aiohttp

_SPEC_KEYS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")


async def _metrics(sess: aiohttp.ClientSession, base: str) -> dict:
    """The engine's cumulative spec counters right now.  Returns {} for a base arm."""
    try:
        async with sess.get(f"{base}/metrics") as r:
            body = await r.text()
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in body.splitlines():
        for k in _SPEC_KEYS:
            pre = f"vllm:spec_decode_{k}_total"
            if line.startswith(pre + "{"):
                out[k] = out.get(k, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def _accept(a: dict, b: dict) -> float | None:
    if not a or not b:
        return None
    d = b.get("num_drafts", 0) - a.get("num_drafts", 0)
    acc = b.get("num_accepted_tokens", 0) - a.get("num_accepted_tokens", 0)
    return None if d <= 0 else 1.0 + acc / d


def _drafted(a: dict, b: dict) -> float | None:
    """Mean draft tokens OFFERED per step -- how much of the tree/chain was proposed.  A
    tree that quietly falls back to a chain shows up here and nowhere else."""
    if not a or not b:
        return None
    d = b.get("num_drafts", 0) - a.get("num_drafts", 0)
    n = b.get("num_draft_tokens", 0) - a.get("num_draft_tokens", 0)
    return None if d <= 0 else n / d


async def _one(sess, base, model, prompt, args, idx, sem, results, err):
    async with sem:
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            # ignore_eos: equal WORK per arm. Without it a faster arm that hits EOS earlier
            # looks slower per token and the arms are not comparable.
            #
            # DEFAULT ON, and every published forced-256 number was measured with it on. Turn it
            # OFF (--ignore-eos 0) to measure the NATURAL-EOS condition -- what a deployment
            # actually serves. The arms then emit different token counts, so `tokens` is no
            # longer equal across arms and only the RATES (tps, tps_per_req) are comparable;
            # acceptance is unaffected by that since it is a per-draft-step ratio. Forcing 256
            # tokens runs the model well past its natural stop into out-of-distribution
            # continuation, which is measurably harder to draft -- 4B-tr tr_toolcall reads
            # accept 2.247 forced vs 2.546 natural -- so the two conditions must never be mixed.
            "ignore_eos": bool(args.ignore_eos),
        }
        if args.temperature > 0:
            body["seed"] = 1234 + idx
        if args.top_p is not None:
            body["top_p"] = args.top_p
        t0 = time.perf_counter()
        ttft = None
        text = []
        ntok = 0
        try:
            async with sess.post(f"{base}/v1/completions", json=body) as r:
                if r.status != 200:
                    err.append({"idx": idx, "status": r.status,
                                "body": (await r.text())[:2000]})
                    return
                async for raw in r.content:
                    if not raw.startswith(b"data: "):
                        continue
                    chunk = raw[6:].strip()
                    if chunk == b"[DONE]":
                        break
                    try:
                        j = json.loads(chunk)
                    except Exception:
                        continue
                    for c in j.get("choices") or []:
                        t = c.get("text") or ""
                        if t:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            text.append(t)
                    if j.get("usage"):
                        ntok = j["usage"].get("completion_tokens", 0)
        except Exception as e:                                   # noqa: BLE001
            err.append({"idx": idx, "exc": repr(e)})
            return
        dt = time.perf_counter() - t0
        out = "".join(text)
        results.append({"idx": idx, "ttft": ttft, "latency": dt,
                        "ntok": ntok or 0, "text": out})


async def _phase(sess, base, model, prompts, args, conc, tag):
    sem = asyncio.Semaphore(conc)
    results: list[dict] = []
    err: list[dict] = []
    m0 = await _metrics(sess, base)
    t0 = time.perf_counter()
    await asyncio.gather(*[
        _one(sess, base, model, p, args, i, sem, results, err)
        for i, p in enumerate(prompts)])
    wall = time.perf_counter() - t0
    m1 = await _metrics(sess, base)
    results.sort(key=lambda r: r["idx"])
    ntok = sum(r["ntok"] for r in results)
    lat = [r["latency"] for r in results]
    ttft = [r["ttft"] for r in results if r["ttft"] is not None]
    per_req = statistics.mean([r["ntok"] / r["latency"] for r in results if r["latency"] > 0]) \
        if results else 0.0
    return {
        "tag": tag, "concurrency": conc, "requests": len(prompts),
        "completed": len(results), "errors": err[:10], "nerrors": len(err),
        "tokens": ntok, "secs": wall,
        # OUTPUT throughput of the whole server (what a deployment sees) ...
        "tps": ntok / wall if wall > 0 else 0.0,
        # ... and the per-request rate (what a single user sees). At concurrency 1 they
        # are the same number; above that they diverge and BOTH matter.
        "tps_per_req": per_req,
        "accept": _accept(m0, m1), "drafted": _drafted(m0, m1),
        "ttft_mean": statistics.mean(ttft) if ttft else None,
        "ttft_p95": (sorted(ttft)[int(0.95 * (len(ttft) - 1))] if ttft else None),
        "lat_mean": statistics.mean(lat) if lat else None,
        "texts": [r["text"] for r in results],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/home/shadeform/chained-flow/logs/specbench/data/"
                                      "subset_fixed256.jsonl")
    ap.add_argument("--concurrency", default="1,2,4,8,16",
                    help="comma list; one measured phase each, in order")
    ap.add_argument("--requests", type=int, default=0, help="0 = the whole file")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--ignore-eos", type=int, default=1, choices=(0, 1),
                    help="1 (default, and the condition every published forced-256 number was "
                         "measured in) = emit exactly --max-tokens per request, equal work per "
                         "arm. 0 = stop at EOS, the natural-serving condition. Recorded in the "
                         "output json; NEVER pool the two.")
    ap.add_argument("--warmup", type=int, default=8,
                    help="requests to burn before measuring, at EACH concurrency: our draft "
                         "cudagraph is captured lazily PER BATCH BUCKET, so the first step at "
                         "a new bucket pays a capture that is not part of steady state. "
                         "RAISED TO 3*concurrency automatically -- see below")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    prompts = [r["prompt"] for r in rows]
    if args.requests:
        prompts = [prompts[i % len(prompts)] for i in range(args.requests)]

    timeout = aiohttp.ClientTimeout(total=None, sock_read=1200)
    conn = aiohttp.TCPConnector(limit=0)
    phases = []
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        for c in [int(x) for x in args.concurrency.split(",")]:
            # THE WARMUP MUST ACTUALLY REACH CONCURRENCY c.  A fixed 8-request warmup driven at
            # c=16 never puts more than 8 requests in flight, so the batch-16 draft cudagraph --
            # and the inductor autotune that precedes it -- landed in the MEASURED phase instead:
            # 4B chain c=16 read 175.9 tok/s with a 20.0 s mean TTFT, against 826 tok/s at c=8.
            # That was the harness, not the engine.  3x is enough to hold c in flight for a
            # while after ramp-up.
            nwarm = max(args.warmup, 3 * c) if args.warmup else 0
            if nwarm:
                warm = [prompts[i % len(prompts)] for i in range(nwarm)]
                await _phase(sess, args.base, args.model, warm, args, c, f"warmup_c{c}")
            p = await _phase(sess, args.base, args.model, prompts, args, c, f"c{c}")
            phases.append(p)
            print(f"[bench_serve] c={c:<3} {p['tokens']:6d} tok / {p['secs']:7.2f}s = "
                  f"{p['tps']:7.1f} tok/s server, {p['tps_per_req']:6.1f} tok/s/req"
                  + (f" | accept {p['accept']:.3f}" if p["accept"] else " | accept -")
                  + (f" drafted {p['drafted']:.1f}" if p["drafted"] else "")
                  + (f" | TTFT {p['ttft_mean'] * 1000:.0f}ms" if p["ttft_mean"] else "")
                  + (f" | ERRORS {p['nerrors']}" if p["nerrors"] else ""), flush=True)
            if p["nerrors"]:
                print(f"[bench_serve] first error: {p['errors'][0]}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump({"model": args.model, "base": args.base, "data": args.data,
               "max_tokens": args.max_tokens, "temperature": args.temperature,
               "ignore_eos": bool(args.ignore_eos), "phases": phases}, open(args.out, "w"))
    print(f"[bench_serve] wrote {args.out}", flush=True)
    return 1 if any(p["nerrors"] for p in phases) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
