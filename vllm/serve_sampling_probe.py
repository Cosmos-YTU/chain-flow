#!/usr/bin/env python3
"""What does a RUNNING server do when a temperature>0 request arrives?

Offline, `CF_TEMP>0` in `test_plugin_native.py` answers this for one process that then exits.
A server is different in the way that matters: the request is one of many, the engine core is
a separate process, and an exception raised inside `propose()` does not fail the request -- it
propagates out of the engine's step loop.  So the questions are:

  1. does the sampled request itself succeed, fail, or hang?
  2. does the ENGINE survive it -- can a subsequent GREEDY request still be served?
  3. does an INNOCENT greedy request that is merely BATCHED WITH a sampled one also die?
     (`SamplingMetadata.all_greedy` is a property of the whole batch, so this is not
     hypothetical: one sampled request makes the batch non-greedy for everybody.)

Run LAST against a server: by design this may take the engine down.

    serve_sampling_probe.py --base http://localhost:8601 --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time

import aiohttp

PROMPT = "Write a short paragraph about the ocean."


async def _req(sess, base, model, temp, max_tokens=32, timeout=120, prompt=PROMPT):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temp, "ignore_eos": True}
    if temp > 0:
        body["seed"] = 1234
    t0 = time.perf_counter()
    try:
        async with sess.post(f"{base}/v1/completions", json=body,
                             timeout=aiohttp.ClientTimeout(total=timeout)) as r:
            txt = await r.text()
            dt = time.perf_counter() - t0
            if r.status != 200:
                return {"status": r.status, "secs": round(dt, 2), "err": txt[:600]}
            j = json.loads(txt)
            return {"status": 200, "secs": round(dt, 2),
                    "text": j["choices"][0]["text"][:60]}
    except asyncio.TimeoutError:
        return {"status": "TIMEOUT", "secs": round(time.perf_counter() - t0, 2)}
    except Exception as e:                                       # noqa: BLE001
        return {"status": "EXC", "secs": round(time.perf_counter() - t0, 2), "err": repr(e)}


async def _health(sess, base):
    try:
        async with sess.get(f"{base}/health",
                            timeout=aiohttp.ClientTimeout(total=10)) as r:
            return r.status
    except Exception as e:                                       # noqa: BLE001
        return repr(e)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("-o", "--out", default="")
    args = ap.parse_args()
    res = {}
    async with aiohttp.ClientSession() as s:
        res["0_health_before"] = await _health(s, args.base)
        res["1_greedy_before"] = await _req(s, args.base, args.model, 0.0)
        # The main event, on its own so nothing else is in the batch to blame.
        res["2_sampled_alone"] = await _req(s, args.base, args.model, args.temperature)
        res["3_health_after"] = await _health(s, args.base)
        # Did the engine survive?
        res["4_greedy_after"] = await _req(s, args.base, args.model, 0.0)
        # Collateral: a greedy request sharing the batch with a sampled one.
        pair = await asyncio.gather(
            _req(s, args.base, args.model, 0.0, max_tokens=64,
                 prompt="Explain gradient descent simply."),
            _req(s, args.base, args.model, args.temperature, max_tokens=64))
        res["5_greedy_batched_with_sampled"] = pair[0]
        res["5b_the_sampled_one"] = pair[1]
        res["6_health_final"] = await _health(s, args.base)
        res["7_greedy_final"] = await _req(s, args.base, args.model, 0.0)
    for k in sorted(res):
        print(f"[sampling-probe] {k:32} {res[k]}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
