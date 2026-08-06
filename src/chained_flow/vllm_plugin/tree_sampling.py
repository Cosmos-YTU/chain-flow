"""PROTOTYPE (default OFF): let a tree-mode server accept `temperature > 0` requests by
serving them UNSPECULATED, in the same batch as greedy requests that keep the full tree.

Set `CF_TREE_SAMPLED_NOSPEC=1` to enable.  Nothing here runs otherwise.

WHAT THIS IS
------------
Tree mode is greedy-only today (see `greedy_guard.py`): the fork's tree verify takes the
`sampling_metadata.all_greedy` branch, and a branching tree reaching the stock LINEAR rejection
sampler raises and kills the engine core.  So `CF_TREE_GREEDY_GUARD` rejects such requests with
a 400 in the API server.

That is a limit of THIS implementation, not of tree speculation.  Route "B" is the cheap fix:
keep the greedy requests on the tree, and give each sampled request exactly ONE correctly
sampled token per step -- i.e. no speedup for the sampled fraction, full speedup for the rest,
and the server accepts all traffic.

HOW IT WORKS, AND WHY IT IS EXACT
---------------------------------
A sampled request still gets a full-width tree drafted and forwarded (that is what keeps the
`CF_ASYNC_SPEC` contract intact -- see below).  Two things then happen in the verify:

  1. Its draft tokens are POISONED to -1 before the descent kernel runs.  The kernel accepts a
     child only when `draft_token == target_argmax`, and `target_argmax >= 0` always, so no
     child can ever match: the descent stops at the root, `accepted_len == 0`, and the request
     commits exactly one token -- the root row's argmax.
  2. That one token is then OVERWRITTEN with a properly sampled draw from the root row, taken
     through vLLM's real `Sampler` (temperature, top-k, top-p, min-p, logit_bias, seeded
     generators -- everything).

Step 2 is the whole correctness argument.  The root row of a request is the row whose logits
predict its next token; for a request with an accepted length of 0 that row is the ONLY row it
consumes, and sampling from it is bit-for-bit what an unspeculated step does.  The tree's
existence is invisible to the output distribution because every one of its nodes was rejected
before it could contribute.

Greedy requests in the same batch are untouched: the descent kernel reads their (unpoisoned)
draft tokens and their own rows, exactly as it does in an all-greedy batch.

WHY THE DRAFT IS STILL FORWARDED (i.e. why this costs throughput)
-----------------------------------------------------------------
The honest alternative -- hand the sampled request ZERO draft tokens -- is expressible on the
synchronous path but not under `CF_ASYNC_SPEC`, which owes vLLM a `[num_reqs, draft_width]` GPU
tensor whose width is uniform (see `FlowDrafterProposer._no_draft`); and `_prelaunch` bails on
`min(num_draft_tokens) <= 0`, so a genuinely zero-width row would also cost every OTHER request
in the batch its draft overlap.  Poisoning a full-width row buys per-request granularity without
touching the width anywhere.  The price is that a sampled request burns `draft_width + 1` target
query positions and one drafter pass per step to produce one token.  At batch 1 that is nearly
free (the target forward is weight-bound); at high concurrency it is not, and the sampled
fraction of the batch is paying real compute for nothing.

WHAT IS STILL REFUSED
---------------------
`logprobs`, penalties, `bad_words`, `allowed_token_ids`, `logprob_token_ids` -- unchanged.  The
tree path returns `logprobs_tensors=None` for the whole batch, so a mixed batch cannot serve a
logprobs request; only `temperature > 0` is unblocked here.  `greedy_guard` keeps rejecting the
rest, which is why this module relaxes that guard rather than removing it.

SCOPE / SAFETY
--------------
* Default OFF.  `CF_TREE_SAMPLED_NOSPEC=1` opts in.
* Pure monkeypatch: it edits no file in the fork and no file another workstream owns.  It wraps
  `RejectionSampler.forward` and relaxes `FlowDrafterProposer._tree_ok`.
* Never raises out of `install()`; on any failure the engine keeps the current greedy-only
  behaviour, which is a working server.
"""
from __future__ import annotations

import os

INSTALLED = False
REASON = "not attempted"

_FLAG = "CF_TREE_SAMPLED_NOSPEC"


def enabled() -> bool:
    v = os.environ.get(_FLAG, "0")
    return v not in ("0", "", "false", "False", "FALSE", "no", "off", "OFF")


# --------------------------------------------------------------------------------------
#  the verify
# --------------------------------------------------------------------------------------
def _mixed_ok(sampling_metadata) -> bool:
    """True when the batch is a tree batch whose ONLY disqualifier is non-greedy temperature.

    Deliberately the same predicate as the fork's own tree gate with `all_greedy` dropped:
    anything that needs logits processors or logprobs is still handed to the stock path (which
    then raises on a branching tree -- the pre-existing behaviour, and why `greedy_guard` still
    rejects those requests at admission).
    """
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    return bool(
        getattr(sampling_metadata, "max_num_logprobs", None) is None
        and not getattr(sampling_metadata, "logprob_token_ids", None)
        and not RejectionSampler._tree_needs_processors(sampling_metadata)
    )


def _mixed_tree_sample(self, metadata, logits, sampling_metadata):
    import torch
    from dataclasses import replace as _replace
    from vllm.v1.sample.rejection_sampler import tree_rejection_sample

    dev = logits.device
    b = len(metadata.num_draft_tokens)
    # [batch] bool: which requests keep their tree.  GREEDY_TEMPERATURE is 0 in the fork, but
    # read the module constant rather than assume it.
    from vllm.v1.sample.rejection_sampler import GREEDY_TEMPERATURE

    greedy = sampling_metadata.temperature == int(GREEDY_TEMPERATURE)

    # --- 1. poison every non-greedy request's draft so no node can be accepted -----------
    draft = metadata.draft_token_ids
    if draft.numel():
        # request index of each flat draft token, without a host sync: draft rows are
        # contiguous per request and `cu_num_draft_tokens` is their exclusive-end index.
        tok_req = torch.bucketize(
            torch.arange(draft.numel(), device=dev, dtype=metadata.cu_num_draft_tokens.dtype),
            metadata.cu_num_draft_tokens,
            right=True,
        )
        draft = draft.masked_fill(~greedy[tok_req.long()], -1)
    md = _replace(metadata, draft_token_ids=draft)

    out = tree_rejection_sample(md, logits)          # [batch, max_spec_len + 1]

    # --- 2. replace each non-greedy request's single token with a real sampled draw ------
    # Root row of request i is `cu_num_sampled_tokens[i-1]` (0 for i == 0): that is the row
    # whose logits predict the request's next token, and with accepted_len == 0 it is the only
    # row the request consumes this step.
    cus = metadata.cu_num_sampled_tokens
    root_rows = torch.cat(
        (torch.zeros(1, dtype=cus.dtype, device=dev), cus[: b - 1])
    ).long()
    sampled = self.sampler(
        logits=logits[root_rows],
        sampling_metadata=sampling_metadata,
    ).sampled_token_ids.view(-1).to(out.dtype)
    out[:, 0] = torch.where(greedy[:b], out[:, 0], sampled[:b])
    return out


def _install_sampler() -> None:
    from vllm.v1.outputs import SamplerOutput
    from vllm.v1.sample.rejection_sampler import RejectionSampler

    orig = RejectionSampler.forward
    if getattr(orig, "_cf_tree_sampling", False):
        return

    def forward(self, metadata, draft_probs, logits, sampling_metadata):
        if (
            metadata.tree_parents is not None
            and not sampling_metadata.all_greedy
            and _mixed_ok(sampling_metadata)
        ):
            return SamplerOutput(
                sampled_token_ids=_mixed_tree_sample(
                    self, metadata, logits, sampling_metadata
                ),
                logprobs_tensors=None,
            )
        return orig(self, metadata, draft_probs, logits, sampling_metadata)

    forward._cf_tree_sampling = True                      # type: ignore[attr-defined]
    RejectionSampler.forward = forward                    # type: ignore[assignment]


# --------------------------------------------------------------------------------------
#  the proposer gate
# --------------------------------------------------------------------------------------
def _install_proposer() -> None:
    """Relax `_tree_ok` so a mixed batch drafts a tree instead of raising.

    `_tree_ok` gates three things: the `propose()` raise (which is what kills the engine), the
    legacy path's branch choice, and `_prelaunch`'s side-stream draft.  Relaxing all three is
    intended -- with the verify above, a mixed batch is a perfectly ordinary tree step for every
    greedy request in it, including the draft overlap.
    """
    from chained_flow.vllm_plugin.flow_proposer import FlowDrafterProposer

    orig = FlowDrafterProposer._tree_ok
    if getattr(orig, "_cf_tree_sampling", False):
        return

    def _tree_ok(sm):
        if orig(sm):
            return True
        if sm is None:
            return False
        try:
            return _mixed_ok(sm)
        except Exception:                                 # noqa: BLE001
            return False

    _tree_ok._cf_tree_sampling = True                     # type: ignore[attr-defined]
    FlowDrafterProposer._tree_ok = staticmethod(_tree_ok)


def install() -> None:
    global INSTALLED, REASON
    if INSTALLED:
        return
    if not enabled():
        REASON = f"{_FLAG}=0 (tree mode stays greedy-only)"
        return
    if not os.environ.get("VLLM_SPEC_TREE"):
        REASON = "not tree mode"
        return
    try:
        _install_sampler()
        _install_proposer()
    except Exception as e:                                # noqa: BLE001
        REASON = f"{e!r}"
        print(f"[cf-plugin] tree sampled-no-spec NOT installed ({e!r}); tree mode stays "
              f"greedy-only", flush=True)
        return
    INSTALLED = True
    REASON = ""
    print("[cf-plugin] tree sampled-no-spec INSTALLED (CF_TREE_SAMPLED_NOSPEC=1): "
          "temperature>0 requests are served UNSPECULATED alongside greedy tree requests. "
          "logprobs/penalties are still refused.", flush=True)


def status() -> str:
    return "installed" if INSTALLED else f"not installed ({REASON})"
