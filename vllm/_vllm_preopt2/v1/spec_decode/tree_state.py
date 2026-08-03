# SPDX-License-Identifier: Apache-2.0
"""Tree speculative decoding support (chained-flow fork of vLLM 0.25.1).

Stock vLLM verifies a LINEAR CHAIN of draft tokens.  This module carries the
extra per-step state needed to verify a draft TREE in a single forward pass:

  * a proposer-side registry mapping ``req_id -> (tokens, parents, depths)``
    (the ``custom_class`` proposer hook only returns a flat token list, so the
    tree shape travels out of band and is re-validated against the tokens the
    scheduler actually scheduled);
  * a per-step ``TreeStep`` object holding the flattened tree layout for the
    current ``execute_model`` call, read by the attention backends and by the
    rejection sampler.

Everything is inert unless ``VLLM_SPEC_TREE=1``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch


def tree_enabled() -> bool:
    return os.environ.get("VLLM_SPEC_TREE", "0") == "1"


def force_tree_attn() -> bool:
    """Run the tree attention fix-up even for chain-shaped trees.

    For a chain the ancestor mask IS the causal mask, so the fix-up is a no-op
    mathematically; forcing it on is how we prove the decomposition is exact.
    """
    return os.environ.get("VLLM_SPEC_TREE_ATTN_FORCE", "0") == "1"


def attn_off() -> bool:
    """Debug: skip the tree attention fix-up (isolation only)."""
    return os.environ.get("VLLM_SPEC_TREE_ATTN_OFF", "0") == "1"


def force_tree_gdn() -> bool:
    """Run the tree GDN kernels even for chain-shaped trees (equivalence test)."""
    return os.environ.get("VLLM_SPEC_TREE_GDN_FORCE", "0") == "1"


# ---------------------------------------------------------------------------
# Proposer -> model-runner side channel
# ---------------------------------------------------------------------------
# req_id -> (tokens tuple, parents list[int] (-1 for root), depths list[int])
_REGISTRY: dict[str, tuple[tuple[int, ...], list[int], list[int]]] = {}


def register(req_id: str, tokens, parents, depths) -> None:
    _REGISTRY[str(req_id)] = (tuple(int(t) for t in tokens), list(parents), list(depths))


def lookup(req_id: str, tokens) -> tuple[list[int], list[int]] | None:
    """Return (parents, depths) if a registered tree matches ``tokens`` exactly.

    The scheduler may truncate/drop speculative tokens (chunked prefill, guided
    decoding rollback).  If the tokens do not match what we registered, the tree
    shape is stale and the caller must fall back to chain semantics.
    """
    ent = _REGISTRY.get(str(req_id))
    if ent is None:
        return None
    toks, parents, depths = ent
    if len(toks) != len(tokens):
        return None
    for a, b in zip(toks, tokens):
        if int(a) != int(b):
            return None
    return parents, depths


def drop(req_ids) -> None:
    live = set(str(r) for r in req_ids)
    for k in list(_REGISTRY):
        if k not in live:
            _REGISTRY.pop(k, None)


# ---------------------------------------------------------------------------
# Per-step state
# ---------------------------------------------------------------------------
@dataclass
class TreeStep:
    """Flattened tree layout for the current execute_model call."""

    # per draft token (flat, concatenated over requests, BFS order per request)
    parents_np: np.ndarray  # local node parent index, -1 for the root
    depths_np: np.ndarray  # local node depth (root == 0)
    num_draft_tokens: np.ndarray  # [num_reqs]
    # request indices (into the input batch) that carry a *branching* tree
    branching: bool = False
    # filled in by the model runner once known
    parents_gpu: torch.Tensor | None = None
    depths_gpu: torch.Tensor | None = None
    # token-row index (into the flat query buffer) of each draft node
    node_rows_np: np.ndarray | None = None
    # [num_reqs] number of tree nodes; 0 for non-spec requests
    # accepted path (node indices) filled by the rejection sampler
    accepted_path: torch.Tensor | None = None
    accepted_len: torch.Tensor | None = None
    # attention plumbing
    spec_req_idx: torch.Tensor | None = None
    ctx_lens: torch.Tensor | None = None
    cu_q: torch.Tensor | None = None
    tree_token_idx: torch.Tensor | None = None
    anc_mask: torch.Tensor | None = None
    max_nodes: int = 0
    extras: dict = field(default_factory=dict)


def build_attn_plumbing(
    ts: "TreeStep",
    *,
    num_reqs: int,
    num_scheduled_tokens: np.ndarray,
    cu_num_tokens: np.ndarray,
    num_computed_tokens: np.ndarray,
    device: torch.device,
) -> None:
    """Precompute everything the full-attention tree fix-up needs.

    Full attention over a tree = attention over the shared PREFIX (paged KV,
    every node sees all of it) merged (log-sum-exp) with a tiny dense attention
    over the request's own tree nodes under an ANCESTOR mask.  This routine
    builds the index / mask tensors for the second part and the varlen
    descriptors for the first.
    """
    parents, depths = ts.parents_np, ts.depths_np
    nd = ts.num_draft_tokens
    off = 0
    req_idx: list[int] = []
    node_rows: list[np.ndarray] = []
    ctx_lens: list[int] = []
    anc: list[np.ndarray] = []
    nmax = 0
    for r in range(num_reqs):
        n = int(nd[r])
        if n == 0:
            continue
        p = parents[off : off + n]
        off += n
        if int(num_scheduled_tokens[r]) != n + 1:
            # chunked prefill + spec: leave to the stock (chain) path
            continue
        start = int(cu_num_tokens[r]) - (n + 1)
        req_idx.append(r)
        node_rows.append(np.arange(start + 1, start + 1 + n, dtype=np.int64))
        # prefix = everything up to and including the last committed token
        ctx_lens.append(int(num_computed_tokens[r]) + 1)
        a = np.zeros((n, n), dtype=bool)
        for i in range(n):
            j = i
            while j >= 0:
                a[i, j] = True
                j = int(p[j])
        anc.append(a)
        nmax = max(nmax, n)
    if not req_idx:
        return
    S = len(req_idx)
    counts = np.array([len(x) for x in node_rows], dtype=np.int32)
    idx_pad = np.zeros((S, nmax), dtype=np.int64)
    valid = np.zeros((S, nmax), dtype=bool)
    mask = np.zeros((S, nmax, nmax), dtype=bool)
    for s in range(S):
        n = int(counts[s])
        idx_pad[s, :n] = node_rows[s]
        idx_pad[s, n:] = node_rows[s][0]
        valid[s, :n] = True
        mask[s, :n, :n] = anc[s]
        for i in range(n, nmax):  # padded rows attend to themselves only
            mask[s, i, 0] = True
    cu = np.zeros(S + 1, dtype=np.int32)
    np.cumsum(counts, out=cu[1:])

    def h2d(a, dtype):
        return torch.from_numpy(np.ascontiguousarray(a)).to(
            device=device, dtype=dtype, non_blocking=True
        )

    ts.spec_req_idx = h2d(np.array(req_idx, dtype=np.int64), torch.int64)
    ts.tree_token_idx = h2d(np.concatenate(node_rows), torch.int64)
    ts.ctx_lens = h2d(np.array(ctx_lens, dtype=np.int32), torch.int32)
    ts.cu_q = h2d(cu, torch.int32)
    ts.anc_mask = h2d(mask, torch.bool)
    ts.extras["idx_pad"] = h2d(idx_pad, torch.int64)
    ts.extras["valid_pad"] = h2d(valid, torch.bool)
    # PADDED-ROW COMPACTION INDEX.  `_tree_attention_fixup` used to compact its
    # [S*nmax, ...] results with BOOLEAN MASK indexing (`x[valid_pad.view(-1)]`),
    # whose output shape is data-dependent and therefore forces a D2H
    # synchronization -- TWICE PER ATTENTION LAYER, i.e. 2 x n_attn_layers host
    # syncs inside every verify forward.  The valid positions are already known
    # HERE, on the host, so publish them as an int index and let the fixup use
    # `index_select` (static shape, no sync).
    ts.extras["valid_idx"] = h2d(np.flatnonzero(valid.reshape(-1)), torch.int64)
    # The attention fix-up masks with `~anc_mask`; negating it per attention layer is
    # n_attn_layers wasted launches on a host-bound path.  Ship the negation.
    ts.extras["not_anc"] = h2d(~mask, torch.bool)
    ts.extras["counts"] = counts
    ts.extras["nmax"] = nmax
    ts.extras["max_ctx"] = int(max(ctx_lens))
    ts.extras["num_spec"] = S
    ts.extras["spec_req_np"] = np.array(req_idx, dtype=np.int64)


CURRENT: TreeStep | None = None


def set_current(step: TreeStep | None) -> None:
    global CURRENT
    CURRENT = step


def get_current() -> TreeStep | None:
    return CURRENT
