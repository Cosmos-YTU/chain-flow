"""Try to MAKE the stale-tree fallback fire, rather than wait for it.

The fallback needs a step in which EVERY speculative request missed the tree registry.
Reading the scheduler, there is exactly one combination that produces it:

  * `CF_SPEC_MAX_BATCH` is engaged (decode batch above the cutoff), so
    `_update_after_schedule` writes `num_spec_tokens_to_schedule = 0` and every RUNNING
    request gets `spec_token_ids = []` -- no spec slots, hence no spec rows;
  * and one request is scheduled out of the WAITING queue with `num_new_tokens == 1`,
    which takes `Scheduler`'s `pad_spec_decode` branch.  That branch is keyed on the
    ENGINE's static `self.num_spec_tokens`, NOT on the cut `num_spec_tokens_to_schedule`,
    so it hands that request `[-1] * num_spec_tokens` slots -- for a request the proposer
    has never drafted for, so it has no registry entry.

`num_new_tokens == 1` means the whole prompt is already in the prefix cache, so the recipe
is: load the server above the cutoff, then send a prompt it has already seen in full.

Usage: cnd_stale_repro.py <base-url> <model> [rounds]
"""
import asyncio
import json
import sys
import urllib.request

BASE = sys.argv[1]
MODEL = sys.argv[2]
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 6

PROMPT = (
    "Write a detailed technical explanation of how a gated delta network maintains its "
    "recurrent state across a speculative decoding step, covering the convolution state, "
    "the SSM state, and how accepted and rejected draft tokens are reconciled. "
)


def post(prompt, max_tokens):
    req = urllib.request.Request(
        f"{BASE}/v1/completions",
        data=json.dumps(
            {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


async def run():
    loop = asyncio.get_running_loop()
    for rnd in range(ROUNDS):
        # 1. Prime the prefix cache with the exact prompt.
        await loop.run_in_executor(None, post, PROMPT, 8)
        # 2. Load the engine ABOVE the cutoff with long, distinct requests ...
        bg = [
            loop.run_in_executor(None, post, PROMPT + f" Variant {rnd}-{i} follows. ", 400)
            for i in range(8)
        ]
        await asyncio.sleep(1.5)
        # 3. ... and, while they are decoding, send the fully-cached prompt again.  It
        #    enters from the WAITING queue with num_new_tokens == 1.
        hits = [loop.run_in_executor(None, post, PROMPT, 64) for _ in range(4)]
        await asyncio.gather(*hits)
        await asyncio.gather(*bg)
        print(f"[repro] round {rnd} done", flush=True)


asyncio.run(run())
print("[repro] finished", flush=True)
