# SPDX-License-Identifier: Apache-2.0
"""Tree-aware Gated-DeltaNet (conv1d + recurrence) for spec-decode tree verify.

Stock vLLM's GDN spec-decode kernels assume the drafted tokens form a linear
CHAIN: the recurrence carries one register block across ``for i_t in range(T)``
with no parent indirection, and the conv1d keeps a single sliding window per
request.  Sibling tree nodes at adjacent flat offsets would therefore pollute
each other's state.

Both are fixed here by making every node read its PARENT's state instead of the
previous flat token, exactly as in the chained-flow offline harness
(``chained_flow.vllm_tree.tree_ssm``).  Two facts make this cheap inside vLLM:

  * vLLM already allocates ``num_speculative_tokens + 1`` mamba state slots per
    spec request (``gdn_attn.py``: ``block_table[:, :num_spec+1]``) and selects
    the live one with a per-request integer.  Those slots are exactly the
    per-node state slots a tree needs -- no new allocation.
  * The recurrence is written PATH-PARALLEL (one program per (node, head,
    v-block), each walking its own root->node path from the request's initial
    state) rather than depth-serial, so there is no intra-kernel ordering
    requirement and no read/write race on the initial slot.

Slot assignment: with ``num_speculative_tokens = tree_nodes + 1`` there are
``tree_nodes + 2`` slot columns for ``tree_nodes + 1`` query rows, so the row
that would otherwise land on the request's *initial* column is shifted by one
(``_colmap``); the initial state is therefore never overwritten while other
programs are still reading it.
"""

from __future__ import annotations

import numpy as np
import torch

from vllm.triton_utils import tl, triton


# --------------------------------------------------------------------------- #
#  conv1d over a tree                                                          #
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["T"])
def _tree_conv_kernel(
    x_ptr,  # [T, dim]
    w_ptr,  # [dim, width]
    bias_ptr,
    conv_state_ptr,  # [num_slots, dim, state_len]
    chain_ptr,  # [T, MAXD+1] int32, chain[t,0]=t, clamped at the request root
    depth_ptr,  # [T] int32
    out_slot_ptr,  # [T] int32 absolute conv-state slot to write
    init_slot_ptr,  # [T] int32 absolute conv-state slot holding the history
    o_ptr,  # [T, dim]
    T,
    dim: tl.constexpr,
    stride_x_tok: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_state_slot: tl.int64,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    stride_chain: tl.constexpr,
    stride_o_tok: tl.int64,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    t = tl.program_id(1)
    if t >= T:
        return
    feats = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mw = feats < dim

    w0 = tl.load(w_ptr + feats * stride_w_dim + 0, mw, 0.0)
    w1 = tl.load(w_ptr + feats * stride_w_dim + 1, mw, 0.0)
    w2 = tl.load(w_ptr + feats * stride_w_dim + 2, mw, 0.0)
    w3 = tl.load(w_ptr + feats * stride_w_dim + 3, mw, 0.0)
    if HAS_BIAS:
        acc = tl.load(bias_ptr + feats, mw, 0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_N], tl.float32)

    d = tl.load(depth_ptr + t)
    hbase = (
        conv_state_ptr
        + tl.load(init_slot_ptr + t).to(tl.int64) * stride_state_slot
        + feats * stride_state_dim
    )
    h0 = tl.load(hbase + 0 * stride_state_tok, mw, 0.0)
    h1 = tl.load(hbase + 1 * stride_state_tok, mw, 0.0)
    h2 = tl.load(hbase + 2 * stride_state_tok, mw, 0.0)

    a1 = tl.load(chain_ptr + t * stride_chain + 1).to(tl.int64)
    a2 = tl.load(chain_ptr + t * stride_chain + 2).to(tl.int64)
    a3 = tl.load(chain_ptr + t * stride_chain + 3).to(tl.int64)
    g0 = tl.load(x_ptr + t * stride_x_tok + feats, mw, 0.0)
    xa1 = tl.load(x_ptr + a1 * stride_x_tok + feats, mw, 0.0)
    xa2 = tl.load(x_ptr + a2 * stride_x_tok + feats, mw, 0.0)
    xa3 = tl.load(x_ptr + a3 * stride_x_tok + feats, mw, 0.0)
    # g(k) = token k steps back along this node's path; fall back to history
    g1 = tl.where(d >= 1, xa1, h2)
    g2 = tl.where(d >= 2, xa2, tl.where(d >= 1, h2, h1))
    g3 = tl.where(d >= 3, xa3, tl.where(d >= 2, h2, tl.where(d >= 1, h1, h0)))

    acc += g3 * w0
    acc += g2 * w1
    acc += g1 * w2
    acc += g0 * w3
    if SILU:
        acc = acc / (1 + tl.exp(-acc))
    tl.store(o_ptr + t * stride_o_tok + feats, acc, mask=mw)

    obase = (
        conv_state_ptr
        + tl.load(out_slot_ptr + t).to(tl.int64) * stride_state_slot
        + feats * stride_state_dim
    )
    tl.store(obase + 0 * stride_state_tok, g2, mask=mw)
    tl.store(obase + 1 * stride_state_tok, g1, mask=mw)
    tl.store(obase + 2 * stride_state_tok, g0, mask=mw)


@torch.inference_mode()
def tree_causal_conv1d(x, conv_state, weight, bias, meta, activation="silu"):
    """x: [T, dim] pre-conv.  Returns [T, dim] post-conv (+ silu)."""
    T, dim = x.shape
    assert weight.shape[1] == 4, "tree conv currently assumes conv width 4"
    o = torch.empty_like(x)
    BLOCK_N = 256
    _tree_conv_kernel[(triton.cdiv(dim, BLOCK_N), T)](
        x,
        weight,
        bias if bias is not None else x,
        conv_state,
        meta.chain,
        meta.depth,
        meta.out_slot,
        meta.init_slot,
        o,
        T,
        dim,
        x.stride(0),
        weight.stride(0),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        meta.chain.stride(0),
        o.stride(0),
        bias is not None,
        activation == "silu",
        BLOCK_N,
        num_warps=4,
        num_stages=1,
    )
    return o


# --------------------------------------------------------------------------- #
#  gated delta recurrence over a tree (path-parallel)                          #
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["T", "MAXD"])
def _tree_gdn_kernel(
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    o,
    state,  # [num_slots, HV, V, K]
    chain,  # [T, MAXD+1]
    depth,  # [T]
    out_slot,  # [T]
    init_slot,  # [T]
    scale,
    T,
    MAXD,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_state_slot: tl.int64,
    stride_chain: tl.constexpr,
    THRESHOLD: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    i_v, i_hv, t = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if t >= T:
        return
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)

    p_h0 = (
        state
        + tl.load(init_slot + t).to(tl.int64) * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    d_t = tl.load(depth + t)
    # walk root -> node: apply ancestors from the deepest offset down to 0
    for kk in range(MAXD, -1, -1):
        active = kk <= d_t
        idx = tl.where(active, kk, 0)
        anc = tl.load(chain + t * stride_chain + idx).to(tl.int64)
        b_k = tl.load(k + (anc * H + i_h) * K + o_k, mask=mask_k, other=0).to(
            tl.float32
        )
        b_v = tl.load(v + (anc * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(
            tl.float32
        )
        if USE_QK_L2NORM:
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        x = tl.load(a + anc * HV + i_hv).to(tl.float32) + dt_bias_val
        softplus_x = tl.where(x <= THRESHOLD, tl.log(1 + tl.exp(x)), x)
        b_g = -tl.exp(A_log_val) * softplus_x
        b_beta = tl.sigmoid(tl.load(b + anc * HV + i_hv).to(tl.float32))
        h2 = b_h * tl.exp(b_g)
        v2 = b_v - tl.sum(h2 * b_k[None, :], 1)
        v2 = v2 * b_beta
        h2 = h2 + v2[:, None] * b_k[None, :]
        b_h = tl.where(active, h2, b_h)

    b_q = tl.load(q + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)
    p_o = o + (t * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = (
        state
        + tl.load(out_slot + t).to(tl.int64) * stride_state_slot
        + i_hv * V * K
        + o_v[:, None] * K
        + o_k[None, :]
    )
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@torch.inference_mode()
def tree_gated_delta_rule_update(
    A_log, a, b, dt_bias, q, k, v, state, meta, scale=None, use_qk_l2norm=True
):
    """q,k: [1,T,H,K]  v: [1,T,HV,V]  a,b: [T,HV].  Returns o: [T,HV,V]."""
    T = q.shape[1]
    H, Kd = q.shape[2], q.shape[3]
    HV, Vd = v.shape[2], v.shape[3]
    if scale is None:
        scale = Kd**-0.5
    BK = triton.next_power_of_2(Kd)
    BV = min(triton.next_power_of_2(Vd), 32)
    NV = triton.cdiv(Vd, BV)
    o = torch.empty(T, HV, Vd, device=q.device, dtype=q.dtype)
    _tree_gdn_kernel[(NV, HV, T)](
        A_log,
        a.contiguous(),
        b.contiguous(),
        dt_bias,
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        o,
        state,
        meta.chain,
        meta.depth,
        meta.out_slot,
        meta.init_slot,
        scale,
        T,
        meta.maxd,
        H,
        HV,
        Kd,
        Vd,
        BK,
        BV,
        state.stride(0),
        meta.chain.stride(0),
        20.0,
        use_qk_l2norm,
        num_warps=1,
        num_stages=1,
    )
    return o


# --------------------------------------------------------------------------- #
#  LEVEL-SYNCHRONOUS tree recurrence (O(nodes) work)                           #
#  The path-parallel kernel above re-walks each node's whole ancestor path, so  #
#  its work is O(nodes x depth).  Here we launch ONCE PER DEPTH: every node     #
#  reads its PARENT's already-written state slot and applies exactly its own    #
#  update.  A level only reads slots written by strictly earlier levels (and    #
#  out_slot is unique per row and never equals an init slot), so there is no    #
#  race; same-stream launch order provides the barrier.                         #
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["NROW"])
def _tree_gdn_level_kernel(
    A_log, a, b, dt_bias, q, k, v, o, state,
    rows,          # [NROW] global query-row index of each node at this depth
    parent_slot,   # [T] state slot holding this row's PARENT state
    out_slot,      # [T]
    scale, NROW,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr,
    stride_state_slot: tl.int64,
    THRESHOLD: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
):
    i_v, i_hv, j = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if j >= NROW:
        return
    t = tl.load(rows + j).to(tl.int64)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    p_h0 = (state + tl.load(parent_slot + t).to(tl.int64) * stride_state_slot
            + i_hv * V * K + o_v[:, None] * K + o_k[None, :])
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    b_k = tl.load(k + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(v + (t * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
    x = tl.load(a + t * HV + i_hv).to(tl.float32) + dt_bias_val
    softplus_x = tl.where(x <= THRESHOLD, tl.log(1 + tl.exp(x)), x)
    b_g = -tl.exp(A_log_val) * softplus_x
    b_beta = tl.sigmoid(tl.load(b + t * HV + i_hv).to(tl.float32))
    b_h = b_h * tl.exp(b_g)
    b_v = b_v - tl.sum(b_h * b_k[None, :], 1)
    b_v = b_v * b_beta
    b_h = b_h + b_v[:, None] * b_k[None, :]

    b_q = tl.load(q + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)
    p_o = o + (t * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    p_ht = (state + tl.load(out_slot + t).to(tl.int64) * stride_state_slot
            + i_hv * V * K + o_v[:, None] * K + o_k[None, :])
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@torch.inference_mode()
def tree_gated_delta_rule_update_level(
    A_log, a, b, dt_bias, q, k, v, state, meta, scale=None, use_qk_l2norm=True
):
    """O(nodes) level-synchronous twin of tree_gated_delta_rule_update."""
    T = q.shape[1]
    H, Kd = q.shape[2], q.shape[3]
    HV, Vd = v.shape[2], v.shape[3]
    if scale is None:
        scale = Kd**-0.5
    BK = triton.next_power_of_2(Kd)
    BV = min(triton.next_power_of_2(Vd), 32)
    NV = triton.cdiv(Vd, BV)
    o = torch.empty(T, HV, Vd, device=q.device, dtype=q.dtype)
    qc, kc, vc, ac, bc = (q.contiguous(), k.contiguous(), v.contiguous(),
                          a.contiguous(), b.contiguous())
    for rows in meta.levels:
        n = rows.numel()
        if n == 0:
            continue
        _tree_gdn_level_kernel[(NV, HV, n)](
            A_log, ac, bc, dt_bias, qc, kc, vc, o, state,
            rows, meta.parent_slot, meta.out_slot, scale, n,
            H, HV, Kd, Vd, BK, BV, state.stride(0), 20.0, use_qk_l2norm,
            num_warps=1, num_stages=1,
        )
    return o


# --------------------------------------------------------------------------- #
#  per-step metadata                                                           #
# --------------------------------------------------------------------------- #
class TreeGDNMeta:
    """Per-layer slot tensors + the layer-independent tree layout."""

    __slots__ = (
        "chain",
        "depth",
        "out_slot",
        "init_slot",
        "parent_slot",
        "levels",
        "maxd",
        "num_rows",
    )


class TreeGDNLayout:
    """Layer-INDEPENDENT part of the tree layout (built once per step).

    NOTE: every GDN layer has its OWN mamba block ids (``spec_state_indices``
    differs per layer), so only the ancestry/column arithmetic can be shared;
    the absolute slot ids must be gathered per layer.
    """

    __slots__ = (
        "chain", "depth", "seq", "loc", "maxd", "num_rows", "levels", "hoist"
    )


def build_gdn_layout(
    *,
    parents_np: np.ndarray,
    depths_np: np.ndarray,
    num_draft_tokens: np.ndarray,
    spec_req_order: np.ndarray,
    device: torch.device,
) -> TreeGDNLayout:
    """Flatten the per-request trees into the query-row layout the kernels use.

    Query row 0 of a request is the previously committed token; row ``1 + i`` is
    tree node ``i``.  Ancestry: row 0 -> the request's initial state, row
    ``1 + i`` -> row 0 when node ``i`` is a tree root, else row
    ``1 + parents[i]``.
    """
    rows_parent: list[int] = []
    rows_depth: list[int] = []
    rows_seq: list[int] = []
    rows_local: list[int] = []
    starts = np.zeros(len(num_draft_tokens) + 1, dtype=np.int64)
    np.cumsum(num_draft_tokens, out=starts[1:])
    base = 0
    for s, r in enumerate(spec_req_order):
        r = int(r)
        n = int(num_draft_tokens[r])
        off = int(starts[r])
        p = parents_np[off : off + n]
        d = depths_np[off : off + n]
        rows_parent.append(-1)
        rows_depth.append(0)
        rows_seq.append(s)
        rows_local.append(0)
        for i in range(n):
            rows_parent.append(base + (0 if int(p[i]) < 0 else 1 + int(p[i])))
            rows_depth.append(1 + int(d[i]))
            rows_seq.append(s)
            rows_local.append(1 + i)
        base += n + 1
    T = len(rows_parent)
    par = np.asarray(rows_parent, dtype=np.int64)
    dep = np.asarray(rows_depth, dtype=np.int64)
    maxd = int(dep.max())

    self_idx = np.arange(T, dtype=np.int64)
    par_idx = np.where(par < 0, self_idx, par)
    # width >= 4: the width-4 conv1d kernel always reads chain[:, 0..3]
    chain_w = max(maxd, 3) + 1
    chain = np.empty((T, chain_w), dtype=np.int32)
    chain[:, 0] = self_idx
    cur = self_idx
    for kk in range(1, chain_w):
        cur = par_idx[cur]
        chain[:, kk] = cur

    lay = TreeGDNLayout()
    lay.chain = torch.from_numpy(chain).to(device, non_blocking=True)
    lay.depth = torch.from_numpy(dep.astype(np.int32)).to(device, non_blocking=True)
    lay.seq = torch.from_numpy(np.asarray(rows_seq, dtype=np.int64)).to(
        device, non_blocking=True
    )
    lay.loc = torch.from_numpy(np.asarray(rows_local, dtype=np.int64)).to(
        device, non_blocking=True
    )
    lay.maxd = maxd
    lay.num_rows = T
    # row indices grouped by depth, for the level-synchronous scan
    lay.levels = [
        torch.from_numpy(np.nonzero(dep == d)[0].astype(np.int32)).to(
            device, non_blocking=True
        )
        for d in range(maxd + 1)
    ]
    return lay


def layer_meta(
    lay: TreeGDNLayout,
    state_indices: torch.Tensor,  # [num_spec, num_spec_cfg + 1] THIS LAYER's slots
    num_accepted: torch.Tensor,  # [num_spec] initial column + 1
) -> TreeGDNMeta:
    """Resolve per-layer absolute state slots (pure GPU, no host sync).

    Column remap: the request's *initial* column is skipped so a node never
    clobbers the state its siblings are still reading (``_colmap``).
    """
    # HOT PATH: this runs once per GDN layer (28 on Qwen3.5-4B, more on 27B) on every
    # tree step, and the tree verify forward has no full cudagraph -- so every op here
    # is host dispatch on the critical path.  Everything except the two slot gathers is
    # LAYER-INDEPENDENT (`num_accepted` is the same buffer for every layer), so it is
    # computed once and cached on the per-step layout object.  The two survivors use a
    # flat index_select instead of 2-D advanced indexing (1 kernel instead of ~3).
    # Values are unchanged; verify with CF_GDN_META_CHECK=1.
    W = int(state_indices.shape[1])
    h = getattr(lay, "hoist", None)
    if h is None or h[0] != W:
        ic = num_accepted.to(torch.int64) - 1  # [S]
        ic_row = ic[lay.seq]  # [T]
        col = torch.where(lay.loc < ic_row, lay.loc, lay.loc + 1)
        h = (
            W,
            lay.seq * W + col,  # flat index of each row's OUT state slot
            lay.seq * W + ic_row,  # flat index of each row's INIT state slot
            lay.chain[:, 1].to(torch.int64),  # parent row (chain[:, 1])
            lay.depth == 0,  # depth-0 rows start from the INIT slot
        )
        lay.hoist = h
    _, flat_out, flat_init, parent_row, is_root = h
    si = state_indices.reshape(-1)
    out_slot = si.index_select(0, flat_out).to(torch.int32)
    init_slot = si.index_select(0, flat_init).to(torch.int32)
    m = TreeGDNMeta()
    m.chain = lay.chain
    m.depth = lay.depth
    m.maxd = lay.maxd
    m.num_rows = lay.num_rows
    m.out_slot = out_slot
    m.init_slot = init_slot
    # parent's state slot: depth-0 rows start from the request's initial slot,
    # deeper rows from the slot their PARENT row wrote (chain[:,1] = parent row)
    m.parent_slot = torch.where(
        is_root, init_slot, out_slot.index_select(0, parent_row)
    )
    m.levels = lay.levels
    import os as _os

    if _os.environ.get("CF_GDN_META_CHECK", "0") == "1":
        # equivalence gate against the pre-hoist formulation
        _ic = num_accepted.to(torch.int64) - 1
        _ic_row = _ic[lay.seq]
        _col = torch.where(lay.loc < _ic_row, lay.loc, lay.loc + 1)
        _out = state_indices[lay.seq, _col].to(torch.int32)
        _init = state_indices[lay.seq, _ic_row].to(torch.int32)
        _par = torch.where(
            lay.depth == 0, _init, _out[lay.chain[:, 1].to(torch.int64)]
        ).to(torch.int32)
        assert torch.equal(m.out_slot, _out), "tree GDN hoist: out_slot mismatch"
        assert torch.equal(m.init_slot, _init), "tree GDN hoist: init_slot mismatch"
        assert torch.equal(m.parent_slot, _par), "tree GDN hoist: parent_slot mismatch"
    return m


def next_init_column(
    loc_leaf: torch.Tensor, init_col: torch.Tensor
) -> torch.Tensor:
    """Column that will hold the carried-forward state (same remap as above)."""
    return torch.where(loc_leaf < init_col, loc_leaf, loc_leaf + 1)
