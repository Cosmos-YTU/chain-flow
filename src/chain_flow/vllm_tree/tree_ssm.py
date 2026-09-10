"""Tree-aware Gated-DeltaNet SSM ops for vLLM, built ENTIRELY from vLLM's existing kernels.

Key insight (validated bit-exact, 0.000e+00 vs linear per-path): vLLM's `packed_decode` recurrence and
`causal_conv1d_update` both read *and* write the same state slot addressed by `*_state_indices`. So a
draft TREE is verified without any new kernel — process the tree depth-by-depth, and at each depth
copy every node's PARENT-slot state into the node's own slot, then call vLLM's kernel in-place with
`state_indices = node_slots`. The kernel reads the (parent) state and writes the node's new state.

Slot convention: slot 0 = zero sentinel (kernel special-cases `state_idx<=0`), other slots hold live
state (prefix + one per tree node). Depth order guarantees a parent's final state is ready before its
children are processed. Cost = D kernel calls per SSM layer (D = tree depth), each cheap — the whole
tree verify is ~one model forward, not D forwards.
"""
from __future__ import annotations

import torch

from vllm.model_executor.layers.fla.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update
from vllm.triton_utils import tl, triton


# ------------------------------------------------------------------------- #
#  Fused tree-scan Gated-DeltaNet recurrence — ONE kernel launch per layer   #
#  instead of D per-depth packed_decode calls. Replicates vLLM's             #
#  fused_recurrent_gated_delta_rule_packed_decode math exactly, but each     #
#  (v-slice, head) program scans nodes 0..N-1 sequentially, reading its       #
#  parent slot's state (BFS order => parent index < node index, and each      #
#  program only touches its own (head, v-slice) region => race-free). This    #
#  collapses the ~D*18 launch-latency-bound micro-kernels to 1 (the SoL fix). #
# ------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["N"])
def _tree_scan_gdn_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, ssm_state, node_slots, parent_slots, N, scale,
    stride_mixed_tok, stride_a_tok, stride_b_tok, stride_state_token,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
):
    i_v, i_hv = tl.program_id(0), tl.program_id(1)
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_n in range(N):
        pslot = tl.load(parent_slots + i_n).to(tl.int64)
        nslot = tl.load(node_slots + i_n).to(tl.int64)
        p_h0 = ssm_state + pslot * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        p_mixed = mixed_qkv + i_n * stride_mixed_tok
        b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(tl.bfloat16).to(tl.float32)
        b_h *= tl.exp(g_val)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= beta_val
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        p_o = o + (i_n * HV + i_hv) * V + o_v
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)
        p_ht = ssm_state + nslot * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(ssm_state.dtype.element_ty), mask=mask_h)


@torch.inference_mode()
def tree_scan_gdn_recurrence(
    mixed_qkv, a, b, A_log, dt_bias, ssm_state, node_slots, parent_slots, scale, *, use_qk_l2norm=True,
):
    """One-launch tree recurrence. Bit-exact with the per-depth packed_decode version; nodes must be
    in BFS/topological order (parent index < node index). Writes each node's state into its slot."""
    N, HV = a.shape
    V, K = ssm_state.shape[-2:]
    conv_dim = mixed_qkv.shape[1]
    H = (conv_dim - HV * V) // 2 // K
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    out = torch.empty(N, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    _tree_scan_gdn_kernel[(NV, HV)](
        mixed_qkv, a, b, A_log, dt_bias, out, ssm_state, node_slots, parent_slots, N, scale,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), ssm_state.stride(0),
        H, HV, K, V, BK, BV, 20.0, use_qk_l2norm, num_warps=1, num_stages=1,
    )
    return out


# --------------------------------------------------------------------------- #
#  DEPTH-PARALLEL tree-scan (v2): the v1 fused kernel above serialises over ALL #
#  N nodes inside one launch (`for i_n in range(N)`), but siblings at the same  #
#  depth are INDEPENDENT — the only true dependency is parent->child (depth).   #
#  v2 keeps the exact same per-node math but puts the node index on a GRID axis  #
#  and launches ONCE PER DEPTH (D launches, D<<N). Each launch runs a whole     #
#  depth's nodes fully in parallel; same-stream ordering makes each depth's     #
#  parent states ready before the next. Static `depth_starts` => cudagraph-safe. #
#  Bit-exact with v1 (identical arithmetic; only the loop became a grid axis).   #
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["N", "base"])
def _tree_depth_gdn_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, ssm_state, node_slots, parent_slots, N, base, scale,
    stride_mixed_tok, stride_a_tok, stride_b_tok, stride_state_token,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
):
    i_v, i_hv, i_j = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n = base + i_j
    if i_n >= N:
        return
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    pslot = tl.load(parent_slots + i_n).to(tl.int64)
    nslot = tl.load(node_slots + i_n).to(tl.int64)
    p_h0 = ssm_state + pslot * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    p_mixed = mixed_qkv + i_n * stride_mixed_tok
    b_q = tl.load(p_mixed + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    a_val = tl.load(a + i_n * stride_a_tok + i_hv).to(tl.float32)
    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    x = a_val + dt_bias_val
    softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
    g_val = -tl.exp(A_log_val) * softplus_x
    beta_val = tl.sigmoid(b_val).to(tl.bfloat16).to(tl.float32)
    b_h *= tl.exp(g_val)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    p_o = o + (i_n * HV + i_hv) * V + o_v
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)
    p_ht = ssm_state + nslot * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(ssm_state.dtype.element_ty), mask=mask_h)


@torch.inference_mode()
def tree_scan_gdn_recurrence_v2(
    mixed_qkv, a, b, A_log, dt_bias, ssm_state, node_slots, parent_slots, depth_starts, scale,
    *, use_qk_l2norm=True,
):
    """Depth-parallel tree recurrence: D launches (one per depth), each fully parallel over its nodes.
    ``depth_starts`` = static list of node-index boundaries per depth (e.g. [0,8,16,...,N]); nodes must be
    in BFS order. Bit-exact with tree_scan_gdn_recurrence. Writes each node's state into its slot."""
    N, HV = a.shape
    V, K = ssm_state.shape[-2:]
    conv_dim = mixed_qkv.shape[1]
    H = (conv_dim - HV * V) // 2 // K
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    out = torch.empty(N, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    for d in range(len(depth_starts) - 1):
        base = depth_starts[d]
        w = depth_starts[d + 1] - base
        if w <= 0:
            continue
        _tree_depth_gdn_kernel[(NV, HV, w)](
            mixed_qkv, a, b, A_log, dt_bias, out, ssm_state, node_slots, parent_slots, N, base, scale,
            mixed_qkv.stride(0), a.stride(0), b.stride(0), ssm_state.stride(0),
            H, HV, K, V, BK, BV, 20.0, use_qk_l2norm, num_warps=1, num_stages=1,
        )
    return out


@triton.jit(do_not_specialize=["N", "base"])
def _tree_depth_conv_kernel(
    x, conv_state, weight, bias, o, node_slots, parent_slots, N, base, dim,
    stride_x_tok, stride_state_slot, stride_state_dim, stride_w_dim, stride_o_tok,
    HAS_BIAS: tl.constexpr, SILU: tl.constexpr, BLOCK_N: tl.constexpr,
):
    i_n = base + tl.program_id(1)
    if i_n >= N:
        return
    feats = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = feats < dim
    w0 = tl.load(weight + feats * stride_w_dim + 0, mask, 0.0)
    w1 = tl.load(weight + feats * stride_w_dim + 1, mask, 0.0)
    w2 = tl.load(weight + feats * stride_w_dim + 2, mask, 0.0)
    w3 = tl.load(weight + feats * stride_w_dim + 3, mask, 0.0)
    bz = tl.load(bias + feats, mask, 0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_N], tl.float32)
    pslot = tl.load(parent_slots + i_n).to(tl.int64)
    nslot = tl.load(node_slots + i_n).to(tl.int64)
    pbase = conv_state + pslot * stride_state_slot + feats * stride_state_dim
    s0 = tl.load(pbase + 0, mask, 0.0)
    s1 = tl.load(pbase + 1, mask, 0.0)
    s2 = tl.load(pbase + 2, mask, 0.0)
    s3 = tl.load(pbase + 3, mask, 0.0)
    xi = tl.load(x + i_n * stride_x_tok + feats, mask, 0.0)
    acc = bz + s0 * w0 + s1 * w1 + s2 * w2 + xi * w3
    if SILU:
        acc = acc * tl.sigmoid(acc)
    tl.store(o + i_n * stride_o_tok + feats, acc.to(o.dtype.element_ty), mask)
    nbase = conv_state + nslot * stride_state_slot + feats * stride_state_dim
    tl.store(nbase + 0, s1, mask)
    tl.store(nbase + 1, s2, mask)
    tl.store(nbase + 2, xi, mask)
    tl.store(nbase + 3, s3, mask)


@torch.inference_mode()
def tree_scan_causal_conv1d_v2(x, conv_state, weight, bias, node_slots, parent_slots, depth_starts,
                               *, activation="silu"):
    """Depth-parallel tree causal conv (width 4): D launches, each parallel over its depth's nodes.
    Bit-exact with tree_scan_causal_conv1d. Returns conv output [N, dim]; writes each node's state."""
    N, dim = x.shape
    BLOCK_N = 256
    NB = triton.cdiv(dim, BLOCK_N)
    o = torch.empty_like(x)
    for d in range(len(depth_starts) - 1):
        base = depth_starts[d]
        w = depth_starts[d + 1] - base
        if w <= 0:
            continue
        _tree_depth_conv_kernel[(NB, w)](
            x, conv_state, weight, bias if bias is not None else x, o, node_slots, parent_slots, N, base, dim,
            x.stride(0), conv_state.stride(0), conv_state.stride(1), weight.stride(0), o.stride(0),
            bias is not None, activation == "silu", BLOCK_N, num_warps=2, num_stages=1,
        )
    return o


# --------------------------------------------------------------------------- #
#  PATH-PARALLEL tree-scan (v3, SoL): the v2 depth kernels are launch/occupancy #
#  bound (ncu: ~10% occupancy, 288 tiny 1-warp launches, 6 per layer). Heads    #
#  are INDEPENDENT and the only serial dependency is parent->child, so instead   #
#  of a depth-serial split we launch ONCE per layer over grid (N x HV x V-blk)   #
#  = ~6144 programs (65% occ). Each (node,head) computes its OWN state by walking #
#  its root->node path from the PREFIX state (slot 1) — no intermediate parent    #
#  reads, no depth barrier. bf16-round after each step to stay bit-exact with the #
#  bf16-intermediate depth-serial reference (else the tree verify is not lossless).#
# --------------------------------------------------------------------------- #
@triton.jit(do_not_specialize=["N", "MAXD"])
def _tree_path_gdn_kernel(
    mixed_qkv, a, b, A_log, dt_bias, o, ssm_state, node_slots, chain, depth, N, MAXD, scale,
    stride_mixed_tok, stride_a_tok, stride_b_tok, stride_state_token, stride_chain,
    H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BK: tl.constexpr, BV: tl.constexpr, SOFTPLUS_THRESHOLD: tl.constexpr, USE_QK_L2NORM: tl.constexpr,
):
    i_n, i_hv, i_v = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if i_n >= N:
        return
    i_h = i_hv // (HV // H)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_val = tl.load(dt_bias + i_hv).to(tl.float32)
    # start from the PREFIX state (slot 1) — shared by every node's path
    p_pref = ssm_state + 1 * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_pref, mask=mask_h, other=0).to(tl.float32)
    d_i = tl.load(depth + i_n)
    # walk root->node: chain[i,kk] = kk-th ancestor (chain[i,0]=i, chain[i,d_i]=root); apply kk=d_i..0
    for kk in range(MAXD, -1, -1):
        active = kk <= d_i
        idx = tl.where(active, kk, 0)
        anc = tl.load(chain + i_n * stride_chain + idx)
        p_mixed = mixed_qkv + anc * stride_mixed_tok
        b_k = tl.load(p_mixed + H * K + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_mixed + 2 * H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM:
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        a_val = tl.load(a + anc * stride_a_tok + i_hv).to(tl.float32)
        b_val = tl.load(b + anc * stride_b_tok + i_hv).to(tl.float32)
        x = a_val + dt_bias_val
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        g_val = -tl.exp(A_log_val) * softplus_x
        beta_val = tl.sigmoid(b_val).to(tl.bfloat16).to(tl.float32)
        h2 = b_h * tl.exp(g_val)
        v2 = b_v - tl.sum(h2 * b_k[None, :], 1)
        v2 = v2 * beta_val
        h2 = h2 + v2[:, None] * b_k[None, :]
        # bf16-round intermediate states (matches the stored-slot round-trip); node's own (kk==0) stays fp32 for output
        h2r = tl.where(kk > 0, h2.to(tl.bfloat16).to(tl.float32), h2)
        b_h = tl.where(active, h2r, b_h)
    # output uses the node's own q (chain[i,0] = i_n)
    p_node = mixed_qkv + i_n * stride_mixed_tok
    b_q = tl.load(p_node + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
    if USE_QK_L2NORM:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_q = b_q * scale
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(o + (i_n * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
    nslot = tl.load(node_slots + i_n).to(tl.int64)
    p_ht = ssm_state + nslot * stride_state_token + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(ssm_state.dtype.element_ty), mask=mask_h)


def build_ancestor_chain(parents, depths, device, max_depth):
    """Precompute [N, max_depth+1] ancestor chain (chain[i,0]=i, chain[i,k]=k-th ancestor, clamped at root)
    and [N] tree depths, for the path-parallel scan. ``parents`` = node-index parents (-1 for roots)."""
    N = len(parents)
    pt = torch.as_tensor(parents, device=device, dtype=torch.long)
    self_idx = torch.arange(N, device=device)
    par_idx = torch.where(pt < 0, self_idx, pt)
    chain = torch.empty(N, max_depth + 1, dtype=torch.long, device=device)
    chain[:, 0] = self_idx
    cur = self_idx
    for kk in range(1, max_depth + 1):
        cur = par_idx[cur]
        chain[:, kk] = cur
    dep = torch.as_tensor(depths, device=device, dtype=torch.long)
    return chain, dep


@torch.inference_mode()
def tree_scan_gdn_recurrence_path(
    mixed_qkv, a, b, A_log, dt_bias, ssm_state, node_slots, chain, depth, max_depth, scale,
    *, use_qk_l2norm=True,
):
    """SoL path-parallel tree recurrence: ONE launch over (N x HV x V-blocks). Bit-exact with
    tree_scan_gdn_recurrence (bf16-intermediate). ``chain``/``depth`` from build_ancestor_chain."""
    N, HV = a.shape
    V, K = ssm_state.shape[-2:]
    conv_dim = mixed_qkv.shape[1]
    H = (conv_dim - HV * V) // 2 // K
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    out = torch.empty(N, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    _tree_path_gdn_kernel[(N, HV, NV)](
        mixed_qkv, a, b, A_log, dt_bias, out, ssm_state, node_slots, chain, depth, N, max_depth, scale,
        mixed_qkv.stride(0), a.stride(0), b.stride(0), ssm_state.stride(0), chain.stride(0),
        H, HV, K, V, BK, BV, 20.0, use_qk_l2norm, num_warps=1, num_stages=1,
    )
    return out


def _depth_groups(depths: torch.Tensor, max_depth: int) -> list[torch.Tensor]:
    return [(depths == d).nonzero(as_tuple=True)[0] for d in range(max_depth + 1)]


@torch.inference_mode()
def tree_gated_delta_recurrence(
    mixed_qkv: torch.Tensor,      # [N, conv_dim] post-conv, per node
    a: torch.Tensor,             # [N, HV]
    b: torch.Tensor,             # [N, HV]
    A_log: torch.Tensor,         # [HV]
    dt_bias: torch.Tensor,       # [HV]
    ssm_state: torch.Tensor,     # [num_slots, HV, V, K]  (slot 0 = zero sentinel; mutated in place)
    node_slots: torch.Tensor,    # [N] int  slot assigned to each node (>0, unique)
    parent_slots: torch.Tensor,  # [N] int  slot of each node's parent (prefix slot for roots)
    groups: list,                # PRECOMPUTED per-depth node-index tensors (avoids .nonzero syncs)
    scale: float,
    *,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Per-node gated-delta output [N, HV, V]; writes each node's state into its slot of ``ssm_state``."""
    N, HV = a.shape
    V, K = ssm_state.shape[-2:]
    out = torch.zeros(N, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
    for idx in groups:
        if idx.numel() == 0:
            continue
        ns = node_slots[idx]
        ssm_state[ns.long()] = ssm_state[parent_slots[idx].long()]   # parent -> node slot (disjoint, no clone)
        o = torch.zeros(idx.numel(), 1, HV, V, device=mixed_qkv.device, dtype=mixed_qkv.dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv[idx].contiguous(), a[idx].contiguous(), b[idx].contiguous(),
            A_log, dt_bias, scale, ssm_state, o, ns.to(torch.int32),
            use_qk_l2norm_in_kernel=use_qk_l2norm,
        )
        out[idx] = o[:, 0]
    return out


@triton.jit(do_not_specialize=["N"])
def _tree_scan_conv_kernel(
    x, conv_state, weight, bias, o, node_slots, parent_slots, N, dim,
    stride_x_tok, stride_state_slot, stride_state_dim, stride_w_dim, stride_o_tok,
    HAS_BIAS: tl.constexpr, SILU: tl.constexpr, BLOCK_N: tl.constexpr,
):
    feats = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = feats < dim
    w0 = tl.load(weight + feats * stride_w_dim + 0, mask, 0.0)
    w1 = tl.load(weight + feats * stride_w_dim + 1, mask, 0.0)
    w2 = tl.load(weight + feats * stride_w_dim + 2, mask, 0.0)
    w3 = tl.load(weight + feats * stride_w_dim + 3, mask, 0.0)
    bz = tl.load(bias + feats, mask, 0.0).to(tl.float32) if HAS_BIAS else tl.zeros([BLOCK_N], tl.float32)
    for i_n in range(N):
        pslot = tl.load(parent_slots + i_n).to(tl.int64)
        nslot = tl.load(node_slots + i_n).to(tl.int64)
        pbase = conv_state + pslot * stride_state_slot + feats * stride_state_dim
        s0 = tl.load(pbase + 0, mask, 0.0)
        s1 = tl.load(pbase + 1, mask, 0.0)
        s2 = tl.load(pbase + 2, mask, 0.0)
        s3 = tl.load(pbase + 3, mask, 0.0)
        xi = tl.load(x + i_n * stride_x_tok + feats, mask, 0.0)
        acc = bz + s0 * w0 + s1 * w1 + s2 * w2 + xi * w3       # bf16 products, fp32 accumulate
        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(o + i_n * stride_o_tok + feats, acc.to(o.dtype.element_ty), mask)
        nbase = conv_state + nslot * stride_state_slot + feats * stride_state_dim
        tl.store(nbase + 0, s1, mask)
        tl.store(nbase + 1, s2, mask)
        tl.store(nbase + 2, xi, mask)
        tl.store(nbase + 3, s3, mask)


@torch.inference_mode()
def tree_scan_causal_conv1d(x, conv_state, weight, bias, node_slots, parent_slots, *, activation="silu"):
    """One-launch tree causal conv (width 4). Bit-exact with the per-depth causal_conv1d_update version;
    nodes must be in BFS order. Returns conv output [N, dim]; writes each node's rolled state to its slot."""
    N, dim = x.shape
    BLOCK_N = 256
    o = torch.empty_like(x)
    _tree_scan_conv_kernel[(triton.cdiv(dim, BLOCK_N),)](
        x, conv_state, weight, bias if bias is not None else x, o, node_slots, parent_slots, N, dim,
        x.stride(0), conv_state.stride(0), conv_state.stride(1), weight.stride(0), o.stride(0),
        bias is not None, activation == "silu", BLOCK_N, num_warps=2, num_stages=1,
    )
    return o


@torch.inference_mode()
def tree_causal_conv1d(
    x: torch.Tensor,             # [N, conv_dim] pre-conv, per node (mutated in place -> conv output)
    conv_state: torch.Tensor,    # [num_slots, conv_dim, width]  (slot 0 sentinel; mutated in place)
    weight: torch.Tensor,        # [conv_dim, width]
    bias: torch.Tensor | None,
    node_slots: torch.Tensor,    # [N] int
    parent_slots: torch.Tensor,  # [N] int
    groups: list,                # PRECOMPUTED per-depth node-index tensors
    *,
    activation: str = "silu",
) -> torch.Tensor:
    """Tree causal conv: each node's conv sees its ancestor context. Returns conv output [N, conv_dim]."""
    out = torch.empty_like(x)
    for idx in groups:
        if idx.numel() == 0:
            continue
        ns = node_slots[idx].to(torch.int32)
        conv_state[ns.long()] = conv_state[parent_slots[idx].long()]
        xi = x[idx].clone()
        causal_conv1d_update(xi, conv_state, weight, bias, activation, conv_state_indices=ns)
        out[idx] = xi
    return out


# ------------------------------------------------------------------------- #
#  torch.library custom-op wrappers so torch.compile treats the tree-scan    #
#  Triton kernels as opaque ops WITHOUT a graph break — inductor then fuses   #
#  all the surrounding dense/norm/residual work into memory-bound regions     #
#  (the same design as vLLM's `splitting_ops` around its GDN core).           #
# ------------------------------------------------------------------------- #
@torch.library.custom_op("cflow::tree_gdn", mutates_args={"ssm_state"})
def tree_gdn_op(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                A_log: torch.Tensor, dt_bias: torch.Tensor, ssm_state: torch.Tensor,
                node_slots: torch.Tensor, parent_slots: torch.Tensor,
                scale: float, use_qk_l2norm: bool) -> torch.Tensor:
    return tree_scan_gdn_recurrence(mixed_qkv, a, b, A_log, dt_bias, ssm_state,
                                    node_slots, parent_slots, scale, use_qk_l2norm=use_qk_l2norm)


@tree_gdn_op.register_fake
def _tree_gdn_fake(mixed_qkv, a, b, A_log, dt_bias, ssm_state, node_slots, parent_slots, scale, use_qk_l2norm):
    N, HV = a.shape
    V = ssm_state.shape[-2]
    return mixed_qkv.new_empty(N, HV, V)


@torch.library.custom_op("cflow::tree_conv", mutates_args={"conv_state"})
def tree_conv_op(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor, node_slots: torch.Tensor, parent_slots: torch.Tensor) -> torch.Tensor:
    return tree_scan_causal_conv1d(x, conv_state, weight, bias, node_slots, parent_slots, activation="silu")


@tree_conv_op.register_fake
def _tree_conv_fake(x, conv_state, weight, bias, node_slots, parent_slots):
    return torch.empty_like(x)


# ------------------------------------------------------------------------- #
#  v2 custom ops: same opaque-to-torch.compile wrappers, but the depth-       #
#  parallel kernels (one launch per depth). `depth_starts` is a STATIC list   #
#  of per-depth node boundaries (canonical tree => invariant => cudagraph-     #
#  safe). Drop-in replacements for tree_gdn_op / tree_conv_op, ~2.5x cheaper.  #
# ------------------------------------------------------------------------- #
@torch.library.custom_op("cflow::tree_gdn_v2", mutates_args={"ssm_state"})
def tree_gdn_op_v2(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
                   A_log: torch.Tensor, dt_bias: torch.Tensor, ssm_state: torch.Tensor,
                   node_slots: torch.Tensor, parent_slots: torch.Tensor,
                   depth_starts: list[int], scale: float, use_qk_l2norm: bool) -> torch.Tensor:
    return tree_scan_gdn_recurrence_v2(mixed_qkv, a, b, A_log, dt_bias, ssm_state,
                                       node_slots, parent_slots, depth_starts, scale,
                                       use_qk_l2norm=use_qk_l2norm)


@tree_gdn_op_v2.register_fake
def _tree_gdn_v2_fake(mixed_qkv, a, b, A_log, dt_bias, ssm_state, node_slots, parent_slots,
                      depth_starts, scale, use_qk_l2norm):
    N, HV = a.shape
    V = ssm_state.shape[-2]
    return mixed_qkv.new_empty(N, HV, V)


@torch.library.custom_op("cflow::tree_conv_v2", mutates_args={"conv_state"})
def tree_conv_op_v2(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
                    bias: torch.Tensor, node_slots: torch.Tensor, parent_slots: torch.Tensor,
                    depth_starts: list[int]) -> torch.Tensor:
    return tree_scan_causal_conv1d_v2(x, conv_state, weight, bias, node_slots, parent_slots,
                                      depth_starts, activation="silu")


@tree_conv_op_v2.register_fake
def _tree_conv_v2_fake(x, conv_state, weight, bias, node_slots, parent_slots, depth_starts):
    return torch.empty_like(x)
