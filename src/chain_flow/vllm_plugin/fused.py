"""Fusions for the Chained-Flow draft path. Applied as monkeypatches to a loaded drafter — no fork.

The draft is launch-bound, not compute-bound: on 27B `predict_hidden` measures 8.43 ms against a
~0.3 ms memory roofline (~28x off) because it runs ~240 tiny eager kernels. Cudagraphs remove launch
latency but cannot remove the memory round-trips between micro-kernels — these fusions do.
"""
from __future__ import annotations

import torch


def fuse_path_head(drafter) -> None:
    """8 sequential Linear(D,D) + 8 masked muls + 7 adds  ->  one bmm + mul + sum.

    Original (tree_flow.py::_residual_from_lastp):
        h = sum_j  offset_proj[j](emb[:, j, :]) * mask[:, j:j+1]
        return mlp(norm(h))
    """
    ph = drafter.path_head
    order = ph.order
    W = torch.stack([ph.offset_proj[j].weight for j in range(order)], 0).contiguous()  # [o, D, D]
    b = torch.stack([ph.offset_proj[j].bias for j in range(order)], 0).contiguous()    # [o, D]
    drafter._fused_W, drafter._fused_b = W, b

    def _residual_fused(self, lastp: torch.Tensor, nlive: int | None = None) -> torch.Tensor:
        # `nlive` trims the batched GEMM to the offsets that can actually have an ancestor at this
        # depth (see TreeFlowDrafter._residual_from_lastp) -- the rest are masked to zero anyway,
        # so this is bit-exact and saves (order - nlive) x D x D of weight traffic per depth.
        o = order if nlive is None else min(nlive, order)
        lastp = lastp[:, :o]
        mask = (lastp >= 0).to(self._dtype)                       # [N, o]
        emb = self._embed(lastp.clamp_min(0))                     # [N, o, D]
        # [o, N, D] @ [o, D, D]^T -> [o, N, D]   (single batched GEMM over the live offsets)
        t = torch.baddbmm(self._fused_b[:o].unsqueeze(1), emb.transpose(0, 1),
                          self._fused_W[:o].transpose(1, 2))
        h = (t * mask.transpose(0, 1).unsqueeze(-1)).sum(0)        # [N, D]
        return self.path_head.mlp(self.path_head.norm(h))

    drafter._residual_from_lastp = _residual_fused.__get__(drafter, type(drafter))


def compile_flow(drafter, mode: str = "max-autotune-no-cudagraphs",
                 fullgraph: bool = False) -> None:
    """Fuse the flow net (predict_hidden): ~16 tiny layer-passes -> compiled/fused kernels.

    cudagraph-friendly mode by default (we capture our own graph over the whole draft, so we do not
    want inductor to also manage graphs).
    """
    inner = drafter.predict_hidden
    compiled = torch.compile(inner, mode=mode, dynamic=False, fullgraph=fullgraph)
    drafter._eager_predict_hidden = inner
    drafter.predict_hidden = compiled


def apply_all(drafter, compile_flow_net: bool = True) -> None:
    fuse_path_head(drafter)
    if compile_flow_net:
        compile_flow(drafter)
