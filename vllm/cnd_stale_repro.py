"""MAKE the stale-tree fallback fire, rather than wait for it.

A STALE ROW is a spec row whose scheduled width is not the width the proposer drafted and
registered it at -- `tree_state.lookup` matches on req_id AND LENGTH, so it misses, and
`_prepare_inputs` synthesises chain parents for that row.  A step in which EVERY spec row is
stale has `TreeStep.branching == False`, which takes the GDN layer off the tree conv kernel and
onto the wide-window `causal_conv1d_update` -- the one step a `CF_TREE_CONV_NARROW` engine
cannot serve.  Stock vLLM 0.25.1 manufactures a stale row in exactly two places, and this
script drives both.

TRIGGER 1 -- `pad_spec_decode` (`v1/core/sched/scheduler.py`).
    A request scheduled out of the WAITING queue with `num_new_tokens == 1` is padded to the
    uniform spec width and handed `[-1] * self.num_spec_tokens` slots.  That is the engine's
    STATIC K, not the cut `num_spec_tokens_to_schedule`, and the request was not in the previous
    step's batch at all, so nothing drafted for it.  With `CF_SPEC_MAX_BATCH` engaged every
    RUNNING request has zero slots, so it is the ONLY spec row and the step is ALL stale.

    `num_new_tokens == 1` means `num_computed_tokens == num_tokens - 1`, and THAT IS WHY THE
    FIRST VERSION OF THIS SCRIPT NEVER FIRED.  The prefix-cache hit is BLOCK ALIGNED
    (`kv_cache_manager.get_computed_blocks` caps the hit at `num_tokens - 1` and then rounds
    DOWN to a block boundary), so a fully-cached prompt only lands on `num_new_tokens == 1`
    when its length is exactly `k * block_size + 1`.  At the 4B tree's 528-token block that is
    a 1-in-528 chance for an arbitrary prompt, and the old script used one arbitrary prompt.
    So `--block` is not a tuning knob here, it is the whole trigger: pass the engine's real
    attention block size, which it prints at startup as
        "Setting attention block size to N tokens ..."
    and the prompts are built at exactly `block + 1`, `2*block + 1`, `3*block + 1` tokens.

TRIGGER 1b -- `pad_spec_decode` WITHOUT ANY PREFIX CACHE, via a ONE-TOKEN PROMPT.
    `num_new_tokens` for a fresh WAITING request is `request.num_tokens - num_computed_tokens`,
    and `num_computed_tokens` is 0 with no cache hit -- so `num_new_tokens == 1` also holds,
    unconditionally, for a prompt of exactly ONE token.  That matters more than trigger 1: vLLM
    turns prefix caching OFF for this hybrid+speculative engine (`enable_prefix_caching=False`
    in the startup config line), which would have made trigger 1 unreachable on this model
    class for a reason that has nothing to do with the fix.  A one-token prompt needs no cache,
    no block-size arithmetic and no priming -- it is one request.

TRIGGER 2 -- SPEC TRUNCATION against `max_model_len` (same file, the RUNNING loop).
    `num_new_tokens` is clamped to `max_model_len - num_computed_tokens - 1` and
    `spec_token_ids` is then SHORTENED to whatever survived, so any request that comes within
    `num_spec_tokens` of the context limit is served a SHORT tree the registry cannot match.
    At a decode batch of 1 that single row is the whole step.  This one owes nothing to the
    cutoff and fires at any K, which is why the script drives it BOTH under load and alone.
    Driven by asking for a completion that runs to the context limit.

WHAT TO RUN, AND WHAT EACH ARM PROVES.  Serve the 4B tree arm three ways and drive each with
this script; `CF_TREE_FALLBACK_LOG=1` in every case (it prints on the first stale steps and at
exit, and prints a banner at import so "no output" cannot be misread as "no stale steps"):

  A. CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=0   -- POSITIVE CONTROL. Stock scheduler
     behaviour on a wide engine: the counters must be NON-ZERO, or this script is not
     reaching the hole and nothing below means anything.
  B. CF_SPEC_SLOT_GUARD=0 CF_TREE_CONV_NARROW=1   -- the hazard, undefended: the engine must
     DIE with the stale-tree RuntimeError. This is what "reachable" costs.
  C. (defaults)                                    -- the fix: zero stale rows, no error.

Usage: cnd_stale_repro.py <base-url> <model> [--block N] [--rounds N] [--maxlen N]
"""
import argparse
import asyncio
import json
import random
import sys
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("base")
p.add_argument("model")
p.add_argument("--block", type=int, default=528,
               help="the engine's attention block size, from its startup log. The cached "
                    "prefix is block-aligned, so prompts must be k*block+1 tokens or "
                    "`pad_spec_decode` is never reached.")
p.add_argument("--rounds", type=int, default=6)
p.add_argument("--maxlen", type=int, default=8192, help="the engine's --max-model-len")
p.add_argument("--load", type=int, default=8, help="concurrent requests to hold the decode "
                                                   "batch above CF_SPEC_MAX_BATCH")
A = p.parse_args()

BASE, MODEL = A.base.rstrip("/"), A.model


def post(payload, timeout=240):
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=json.dumps({"model": MODEL, "temperature": 0, **payload}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:                      # a dead engine answers 500
        return {"error": f"HTTP {e.code}: {e.read()[:400]!r}"}
    except Exception as e:                                   # noqa: BLE001
        return {"error": repr(e)}


def ids(n, seed):
    """`n` token ids, deterministic per seed, in a range every Qwen tokenizer has.

    Token IDS rather than text: the trigger is an EXACT token count, and no amount of
    prompt-writing gets a string to tokenize to exactly `k * block + 1`.
    """
    rng = random.Random(seed)
    return [rng.randrange(1000, 100000) for _ in range(n)]


async def run():
    loop = asyncio.get_running_loop()
    calls = 0
    errors = []

    def go(payload, timeout=240):
        return loop.run_in_executor(None, lambda: post(payload, timeout))

    def dead():
        """One arm of this script is EXPECTED to kill the engine (narrow + guard off), and a
        dead vLLM answers every subsequent request with a 500 after a long wait.  Stop as soon
        as that is established rather than grinding through the remaining rounds."""
        return len(errors) >= 6

    for rnd in range(A.rounds):
        if dead():
            break
        # ---- TRIGGER 1: pad_spec_decode ------------------------------------------------
        # One prompt per block multiple, because only `k * block + 1` lands on
        # `num_new_tokens == 1` and the engine's real block size may not be what was passed.
        for mult in (1, 2, 3):
            if dead():
                break
            n = mult * A.block + 1
            if n + 64 >= A.maxlen:
                continue
            prompt = ids(n, seed=1000 * rnd + mult)
            # 1. Prime: the first pass caches `floor((n-1)/block) * block` tokens.
            await go({"prompt": prompt, "max_tokens": 4})
            calls += 1
            # 2. Load the engine ABOVE the cutoff with long, DISTINCT decodes.
            bg = [go({"prompt": ids(600, seed=7_000_000 + 97 * rnd + i), "max_tokens": 400})
                  for i in range(A.load)]
            await asyncio.sleep(2.0)
            # 3. While they decode, re-send the fully-cached prompt.  It enters from WAITING
            #    with num_new_tokens == 1 -> the padding branch -> 41 undrafted spec slots,
            #    while every running request has been cut to zero.
            hits = [go({"prompt": prompt, "max_tokens": 48}) for _ in range(4)]
            for r in await asyncio.gather(*hits, *bg):
                calls += 1
                if "error" in r:
                    errors.append(r["error"])
            print(f"[repro] round {rnd} pad-trigger mult={mult} (prompt {n} tok) done, "
                  f"{len(errors)} errors so far", flush=True)

        # ---- TRIGGER 1b: a ONE-TOKEN prompt, which needs no prefix cache at all -----------
        if not dead():
            bg = [go({"prompt": ids(600, seed=9_000_000 + 53 * rnd + i), "max_tokens": 400})
                  for i in range(A.load)]
            await asyncio.sleep(2.0)
            # `num_new_tokens = num_tokens - 0 = 1`, so the padding branch fires for every one
            # of these, on any engine, cache or no cache.
            tiny = [go({"prompt": [1000 + rnd * 17 + i], "max_tokens": 48}) for i in range(6)]
            for r in await asyncio.gather(*tiny, *bg):
                calls += 1
                if "error" in r:
                    errors.append(r["error"])
            print(f"[repro] round {rnd} one-token-prompt trigger done, "
                  f"{len(errors)} errors so far", flush=True)

        # ---- TRIGGER 2: spec truncation against max_model_len ---------------------------
        # Ask for more tokens than the context can hold, so the request decodes right up to
        # max_model_len and the last ~num_spec_tokens steps are the ones the scheduler
        # truncates.  Run it ALONE first (decode batch 1 -> that row IS the step, which is the
        # all-stale shape) and then under load.
        if dead():
            break
        tail = ids(A.maxlen - 900, seed=5_000_000 + rnd)
        r = await go({"prompt": tail, "max_tokens": 4000})
        calls += 1
        if "error" in r:
            errors.append(r["error"])
        print(f"[repro] round {rnd} truncation-trigger alone done", flush=True)

        if dead():
            break
        bg = [go({"prompt": ids(600, seed=8_000_000 + 31 * rnd + i), "max_tokens": 300})
              for i in range(A.load)]
        await asyncio.sleep(1.5)
        r = await go({"prompt": ids(A.maxlen - 900, seed=6_000_000 + rnd), "max_tokens": 4000})
        calls += 1
        if "error" in r:
            errors.append(r["error"])
        for x in await asyncio.gather(*bg):
            calls += 1
            if "error" in x:
                errors.append(x["error"])
        print(f"[repro] round {rnd} truncation-trigger under load done", flush=True)

    print(f"[repro] finished: {calls} requests, {len(errors)} errors", flush=True)
    for e in errors[:5]:
        print(f"[repro]   error: {e}", flush=True)
    return 1 if errors else 0


sys.exit(asyncio.run(run()))
