"""Stop ONE `temperature > 0` request from killing a tree-mode `vllm serve`.

THE FAILURE, MEASURED
---------------------
Tree mode is greedy-only by construction: the fork's tree verify requires `all_greedy` + no
logprobs + no penalties, and a branching tree that reaches the stock linear rejection sampler
raises.  `FlowDrafterProposer.propose()` therefore checks `_tree_ok` and raises a `RuntimeError`
with an actionable message rather than let it fail deeper.

That design is right for the OFFLINE `LLM()` path, where the exception reaches the caller.
Under `vllm serve` it is raised inside the **engine core's step loop**, and measured against a
live 4B tree server on 2026-08-05 the result is:

    greedy request          -> 200 OK
    ONE temperature=0.8 req -> 500  "EngineCore encountered an issue"
    GET /health             -> 503
    every later GREEDY req  -> 500, forever

The engine is dead and does not come back.  **Any client that sends `temperature > 0` to a
tree-mode server denies service to every other client**, which is not a benchmark artefact, it
is a production outage triggered by an ordinary request.

WHY REJECT AT ADMISSION RATHER THAN DEGRADE IN THE PROPOSER
-----------------------------------------------------------
Degrading in the proposer -- "emit no draft for this step" -- is the nicer behaviour, and it is
what `CF_NONGREEDY_CHAIN=1` was reaching for.  It is not currently reachable:

  * the per-step CHAIN fallback is shape-broken (a chain is at most `draft_length` tokens while
    the ring and the draft cudagraph are fixed-shape for `keep*depth+1`), and
  * "no draft at all" is expressible on the synchronous path (return empty lists) but NOT on the
    `CF_ASYNC_SPEC` path, which owes vLLM a `[num_reqs, draft_width]` GPU TENSOR; a short or
    absent draft there scatters the previous step's tokens into `input_ids` silently, which is
    strictly worse than an error.

Making the draft width dynamic is real surgery on the ring, the capture buckets and the tree
hand-off.  Until that exists, the honest production behaviour is to **refuse the request, with
a 4xx and a message that says what to do**, in the API-SERVER process -- where a `ValueError`
becomes a failed request instead of a dead engine, because the engine core never sees it.

So this trades "server dies for everyone" for "the non-greedy request is rejected and says
why".  Greedy traffic is unaffected.  It does NOT make tree mode support sampling.

SCOPE / SAFETY
--------------
* **Default ON.**  This is not a tuning knob -- without it an ordinary request is a
  denial-of-service. `CF_TREE_GREEDY_GUARD=0` opts back into the crash, for reproducing it.
* Only in TREE mode (`VLLM_SPEC_TREE`).  Chain mode handles sampled requests correctly --
  verified against a live server -- and must not be touched.
* Only patches `AsyncLLM`, i.e. the serving path.  The offline `LLM()` path keeps the existing
  raise, where it is the right behaviour.
* Never raises out of `install()`: this runs in every vLLM process that has chained-flow
  installed, including ones with no intention of using it.
"""
from __future__ import annotations

import os

INSTALLED = False
REASON = "not attempted"

_FLAG = "CF_TREE_GREEDY_GUARD"
# ON by default. This is not a tuning knob: without it, one temperature>0 request kills a
# tree-mode server permanently, for every client -- a denial-of-service reachable by an
# ordinary request. Turning a fatal engine crash into a 4xx has no case where it is worse,
# so it defaults on and CF_TREE_GREEDY_GUARD=0 opts back into the crash (for reproducing it).
_DEFAULT = "1"

_MSG = (
    "chained-flow TREE speculative decoding requires greedy sampling: "
    "temperature=0, no logprobs, no frequency/presence/repetition penalties, no bad_words and "
    "no allowed_token_ids. This request asked for {what}. "
    "Retry with temperature=0, or run the server in CHAIN mode (VLLM_SPEC_TREE=0), which is "
    "exact under any sampling params. "
    "(Rejected here on purpose: reaching the engine with these params takes the engine core "
    "down for every client -- set CF_TREE_GREEDY_GUARD=0 to see that happen instead.)"
)


def _truthy(v) -> bool:
    return v is not None and str(v) not in ("0", "", "false", "False", "FALSE", "no", "off", "OFF")


def _sampled_nospec() -> bool:
    """CF_TREE_SAMPLED_NOSPEC: serve sampled requests unspeculated instead of refusing them."""
    try:
        from chained_flow.vllm_plugin import tree_sampling

        return tree_sampling.enabled()
    except Exception:                                       # noqa: BLE001
        return False


def offending(params) -> str:
    """"" if this request is fine for a tree, else a human phrase naming the FIRST problem.

    Mirrors `FlowDrafterProposer._tree_ok`, which mirrors the fork's own rejection-sampler
    preconditions -- but reads `SamplingParams` (what the client sent) rather than
    `SamplingMetadata` (what the batch resolved to), because this runs before the request is
    ever batched.  Kept deliberately conservative: anything it is not sure about, it allows,
    because a false reject is a broken server for a legitimate request.
    """
    try:
        from vllm import SamplingParams
    except Exception:                                       # noqa: BLE001
        return ""
    if not isinstance(params, SamplingParams):
        return ""                                           # pooling / embedding: no drafting
    t = getattr(params, "temperature", 0.0)
    if t is not None and float(t) > 0.0 and not _sampled_nospec():
        # CF_TREE_SAMPLED_NOSPEC=1 (prototype, default off): temperature is no longer a reason
        # to refuse -- the request is admitted and served UNSPECULATED alongside the greedy
        # tree requests. Everything below still refuses, because the tree verify path returns
        # no logprobs and applies no penalties for anyone. See tree_sampling.py.
        return f"temperature={t}"
    if getattr(params, "logprobs", None) is not None:
        return f"logprobs={params.logprobs}"
    if getattr(params, "prompt_logprobs", None) is not None:
        return f"prompt_logprobs={params.prompt_logprobs}"
    for name, neutral in (("frequency_penalty", 0.0), ("presence_penalty", 0.0),
                          ("repetition_penalty", 1.0)):
        v = getattr(params, name, neutral)
        if v is not None and float(v) != neutral:
            return f"{name}={v}"
    if getattr(params, "bad_words", None):
        return "bad_words"
    if getattr(params, "allowed_token_ids", None):
        return "allowed_token_ids"
    return ""


def install() -> None:
    """Wrap `AsyncLLM.add_request` so a non-greedy request is refused in the API server.

    `add_request` is the single funnel every serving front end goes through (OpenAI completions
    and chat, the raw `/generate` route, and `AsyncLLM.generate` itself), so one wrap covers
    them all; and it runs in the API-server process, so a raise here cannot reach the engine.
    """
    global INSTALLED, REASON
    if INSTALLED:
        return
    # Before the guard's own flag checks: the sampled-no-spec prototype must install in the
    # WORKER process too (it patches the rejection sampler and the proposer), and it is what
    # decides whether `offending()` still refuses temperature. Default OFF, cannot raise.
    try:
        from chained_flow.vllm_plugin import tree_sampling

        tree_sampling.install()
    except Exception as e:                                  # noqa: BLE001 - advisory only
        print(f"[cf-plugin] tree sampled-no-spec not installed ({e!r})", flush=True)
    if not _truthy(os.environ.get(_FLAG, _DEFAULT)):
        REASON = f"{_FLAG}=0 (explicitly disabled; the engine WILL die on a non-greedy request)"
        return
    if not _truthy(os.environ.get("VLLM_SPEC_TREE")):
        REASON = "not tree mode; chain mode serves sampled requests correctly"
        return
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as e:                                  # noqa: BLE001
        REASON = f"AsyncLLM not importable ({e!r}); offline path keeps the in-engine raise"
        return

    orig = AsyncLLM.add_request
    if getattr(orig, "_cf_greedy_guard", False):
        INSTALLED = True
        return

    async def add_request(self, request_id, prompt, params, *a, **kw):
        what = offending(params)
        if what:
            # ValueError, not our own type: every vLLM front end already maps it to a 4xx.
            raise ValueError(_MSG.format(what=what))
        return await orig(self, request_id, prompt, params, *a, **kw)

    add_request._cf_greedy_guard = True                      # type: ignore[attr-defined]
    AsyncLLM.add_request = add_request                       # type: ignore[assignment]
    INSTALLED = True
    REASON = ""
    print("[cf-plugin] tree-mode greedy guard INSTALLED: non-greedy requests will be rejected "
          "by the API server with a 4xx instead of taking the engine core down.", flush=True)


def status() -> str:
    return "installed" if INSTALLED else f"not installed ({REASON})"
