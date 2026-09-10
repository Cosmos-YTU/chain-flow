"""Tree-parallel Gated-DeltaNet recurrence — the core of the SoL tree-verify kernel.

The backbone's linear-attention recurrence (exact, from the model's torch reference) is, per token:

    S = S * exp(g)                      # gated decay      (S: [H, Dk, Dv])
    S = S + k ⊗ ((v - Sᵀk) * beta)      # rank-1 delta update
    o = Sᵀ q                            # readout

A draft *tree* needs, for every node, this recurrence applied along the node's ROOT→node path. The
only change vs the sequential form is that each node reads its PARENT's state, not the previous
token's. Nodes at the same depth are independent, so the whole tree is computed in `depth` vectorized
steps (each a handful of tensor ops over all nodes at that depth) rather than one model forward per
node/path. This is what collapses the correct tree verify toward single-forward cost.

Implemented in torch first and bit-validated against the sequential per-path recurrence; a Triton port
replaces only the inner depth step, against this as the oracle.
"""
from __future__ import annotations

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def _delta_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One recurrence step, batched over a leading node dim. S:[N,H,Dk,Dv] q,k:[N,H,Dk] v:[N,H,Dv]
    g,beta:[N,H]. Returns (new S, output o:[N,H,Dv]). Matches torch_recurrent_gated_delta_rule."""
    gt = g.exp()[..., None, None]                     # [N,H,1,1]
    bt = beta[..., None]                              # [N,H,1]
    S = S * gt
    kv_mem = (S * k[..., None]).sum(dim=-2)           # Sᵀk -> [N,H,Dv]
    delta = (v - kv_mem) * bt                         # [N,H,Dv]
    S = S + k[..., None] * delta[..., None, :]        # rank-1 update
    o = (S * q[..., None]).sum(dim=-2)                # Sᵀq -> [N,H,Dv]
    return S, o


def tree_gated_delta_scan(query, key, value, g, beta, parent, depth, initial_state,
                          *, use_qk_l2norm: bool = True):
    """Gated-delta recurrence over a tree.

    query,key: [N, H, Dk]   value: [N, H, Dv]   g,beta: [N, H]
    parent: [N] long (parent node index; roots use -1)   depth: [N] long (0-based)
    initial_state: [H, Dk, Dv] — the committed prefix's per-head state, shared by all roots.
    Returns core_attn_out [N, H, Dv] and per-node states [N, H, Dk, Dv].
    """
    q = query.float(); k = key.float(); v = value.float(); g = g.float(); beta = beta.float()
    if use_qk_l2norm:
        q = _l2norm(q); k = _l2norm(k)
    N, H, Dk = q.shape
    Dv = v.shape[-1]
    q = q * (Dk ** -0.5)
    dev = q.device
    states = torch.zeros(N, H, Dk, Dv, dtype=torch.float32, device=dev)
    out = torch.zeros(N, H, Dv, dtype=torch.float32, device=dev)
    init = initial_state.to(torch.float32).to(dev)

    max_depth = int(depth.max().item()) if N else -1
    for d in range(max_depth + 1):
        idx = (depth == d).nonzero(as_tuple=True)[0]          # nodes at this depth
        if idx.numel() == 0:
            continue
        if d == 0:
            S_parent = init.unsqueeze(0).expand(idx.numel(), -1, -1, -1)
        else:
            S_parent = states[parent[idx]]                    # gather parent states
        S_new, o = _delta_step(S_parent, q[idx], k[idx], v[idx], g[idx], beta[idx])
        states[idx] = S_new
        out[idx] = o
    return out, states


# ---- reference: sequential per-path recurrence (the correctness oracle) --------------------------

def sequential_path_output(query, key, value, g, beta, path_ids, initial_state, *, use_qk_l2norm=True):
    """Run the plain sequential recurrence along one path (list of node indices, root..node) and
    return the LAST node's output — what a per-path forward would produce. For validation."""
    q = query.float(); k = key.float(); v = value.float(); gg = g.float(); bb = beta.float()
    if use_qk_l2norm:
        q = _l2norm(q); k = _l2norm(k)
    Dk = q.shape[-1]
    q = q * (Dk ** -0.5)
    S = initial_state.to(torch.float32).clone().unsqueeze(0)   # [1,H,Dk,Dv]
    o = None
    for n in path_ids:
        S, o = _delta_step(S, q[n:n+1], k[n:n+1], v[n:n+1], gg[n:n+1], bb[n:n+1])
    return o[0]   # [H, Dv]
