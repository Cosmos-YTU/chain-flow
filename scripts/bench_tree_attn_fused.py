"""Microbench + correctness harness for the tree-attention suffix/merge fix-up.

Compares, on the 4B shipping tree shape (S=1, N=41, H=16, HKV=4, D=256, bf16):
  * eager   -- the ~16-op torch chain in _tree_attention_fixup (the reference)
  * triton  -- the previous single-Triton-kernel version (kept here verbatim)
  * cuda    -- vllm/v1/spec_decode/tree_attn_fused.cu

Timing is 8 launches inside a replayed CUDA graph, i.e. exactly how the fix-up
runs in the real forward (one launch per full-attention layer, 8 at 4B).

  CUDA_VISIBLE_DEVICES=7 python scripts/bench_tree_attn_fused.py
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, "/home/shadeform/vllm/.venv/lib/python3.12/site-packages")

from vllm.triton_utils import tl, triton  # noqa: E402


# --------------------------------------------------------------------------
# the previous Triton implementation, verbatim, for the A/B
# --------------------------------------------------------------------------
@triton.jit
def _triton_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, po_ptr, plse_ptr, idx_ptr, mask_ptr, scale,
    sq_t, sq_h, sk_t, sk_h, sv_t, sv_h, so_t, so_h, spo_t, spo_h,
    N: tl.constexpr, M: tl.constexpr, KV_REP: tl.constexpr, NPOW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, D: tl.constexpr,
    QK_IEEE: tl.constexpr,
):
    s = tl.program_id(0)
    mb = tl.program_id(1)
    h = tl.program_id(2)
    offs_m = mb * BLOCK_M + tl.arange(0, BLOCK_M)
    mok = offs_m < N
    offs_n = tl.arange(0, NPOW)
    nok = offs_n < N
    offs_d = tl.arange(0, D)
    rq = tl.load(idx_ptr + s * N + offs_m, mask=mok, other=0)
    rk = tl.load(idx_ptr + s * N + offs_n, mask=nok, other=0)
    kh = h // KV_REP
    q = tl.load(q_ptr + rq[:, None] * sq_t + h * sq_h + offs_d[None, :],
                mask=mok[:, None], other=0.0)
    k = tl.load(k_ptr + rk[:, None] * sk_t + kh * sk_h + offs_d[None, :],
                mask=nok[:, None], other=0.0)
    if QK_IEEE:
        scores = tl.dot(q.to(tl.float32), tl.trans(k).to(tl.float32),
                        out_dtype=tl.float32, input_precision="ieee") * scale
    else:
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
    masked = tl.load(mask_ptr + s * N * N + offs_m[:, None] * N + offs_n[None, :],
                     mask=mok[:, None] & nok[None, :], other=1)
    scores = tl.where(masked != 0, float("-inf"), scores)
    smax = tl.max(scores, 1)
    p = tl.exp(scores - smax[:, None])
    den = tl.sum(p, 1)
    s_lse = smax + tl.log(den)
    probs = p / den[:, None]
    mrow = s * N + offs_m
    p_lse = tl.load(plse_ptr + h * M + mrow, mask=mok, other=0.0)
    p_lse = tl.where(p_lse == float("inf"), float("-inf"), p_lse)
    s_lse = tl.where(s_lse == float("inf"), float("-inf"), s_lse)
    max_lse = tl.maximum(p_lse, s_lse)
    p_se = tl.exp(p_lse - max_lse)
    s_se = tl.exp(s_lse - max_lse)
    out_se = p_se + s_se
    p_scale = (p_se / out_se)[:, None]
    s_scale = (s_se / out_se)[:, None]
    for d0 in tl.range(0, D, BLOCK_D):
        offs_dc = d0 + tl.arange(0, BLOCK_D)
        v = tl.load(v_ptr + rk[:, None] * sv_t + kh * sv_h + offs_dc[None, :],
                    mask=nok[:, None], other=0.0)
        s_out = tl.dot(probs, v.to(tl.float32), out_dtype=tl.float32,
                       input_precision="ieee")
        s_out = s_out.to(o_ptr.dtype.element_ty).to(tl.float32)
        p_out = tl.load(po_ptr + mrow[:, None] * spo_t + h * spo_h + offs_dc[None, :],
                        mask=mok[:, None], other=0.0)
        out = p_out.to(tl.float32) * p_scale + s_out * s_scale
        tl.store(o_ptr + rq[:, None] * so_t + h * so_h + offs_dc[None, :],
                 out.to(o_ptr.dtype.element_ty), mask=mok[:, None])


def triton_run(t, scale, S, N):
    q, k, v, o, po, plse, idx, na = t
    H, D = q.shape[1], q.shape[2]
    HKV = k.shape[1]
    npow = max(16, triton.next_power_of_2(N))
    bm = 16 if N > 16 else npow
    bd = min(64, D)
    _triton_kernel[(S, triton.cdiv(N, bm), H)](
        q, k, v, o, po, plse, idx, na, scale,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
        v.stride(0), v.stride(1), o.stride(0), o.stride(1),
        po.stride(0), po.stride(1),
        N, S * N, H // HKV, npow, bm, bd, D, True,
        num_warps=8, num_stages=1,
    )


# --------------------------------------------------------------------------
# the eager reference (a transcription of _tree_attention_fixup's tail)
# --------------------------------------------------------------------------
# NOTE: the dispatching wrapper, NOT triton_merge_attn_states -- on CUDA with a
# %8 head size it routes to the custom CUDA op, whose arithmetic differs.
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: E402


def eager_run(t, scale, S, N):
    q, k, v, o, po, plse, idx, na = t
    H, D = q.shape[1], q.shape[2]
    Hkv = k.shape[1]
    rep = H // Hkv
    flat = idx.view(-1)
    qp = q.index_select(0, flat).view(S, N, H, D)
    kp = k.index_select(0, flat).view(S, N, Hkv, D)
    vp = v.index_select(0, flat).view(S, N, Hkv, D)
    if rep > 1:
        kp = kp.repeat_interleave(rep, dim=2)
        vp = vp.repeat_interleave(rep, dim=2)
    scores = torch.einsum("snhd,smhd->shnm", qp.float(), kp.float()) * scale
    scores.masked_fill_(na.unsqueeze(1), float("-inf"))
    smax = scores.amax(dim=-1, keepdim=True)
    exp = torch.exp(scores - smax)
    denom = exp.sum(dim=-1, keepdim=True)
    suffix_lse = (smax + torch.log(denom)).squeeze(-1)
    probs = exp / denom
    suffix_out = torch.einsum("shnm,smhd->snhd", probs, vp.float())
    # the `index_select` compaction is what makes suffix_out contiguous -- the
    # bare reshape leaves stride(1)=S*H*D and merge_attn_states reads the suffix
    # with the PREFIX's head stride, so dropping it silently reads garbage.
    valid = torch.arange(S * N, device=o.device)
    suffix_out = suffix_out.reshape(S * N, H, D).index_select(0, valid).to(o.dtype)
    suffix_lse = (
        suffix_lse.permute(0, 2, 1).reshape(S * N, H).index_select(0, valid).transpose(0, 1)
    ).contiguous()
    merged = torch.empty_like(po)
    merge_attn_states(merged, po, plse, suffix_out, suffix_lse.float())
    o.index_copy_(0, flat, merged.to(o.dtype))


from vllm.v1.spec_decode import tree_attn_fused  # noqa: E402


def cuda_run(t, scale, S, N):
    q, k, v, o, po, plse, idx, na = t
    ok = tree_attn_fused.fused_suffix_merge(
        query=q, key=k, value=v, output=o, prefix_out=po, prefix_lse=plse,
        idx_pad=idx, not_anc=na, scale=scale, S=S, N=N)
    assert ok, "cuda path declined this shape"


# --------------------------------------------------------------------------
def make(S, N, H, HKV, D, keep, depth, dtype, dev, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = (torch.arange(S, device=dev)[:, None] * (N + 1) + 1
           + torch.arange(N, device=dev)[None, :]).to(torch.int64)
    T = int(idx.max().item()) + 1
    q = torch.randn(T, H, D, generator=g, device=dev, dtype=torch.float32).to(dtype)
    k = torch.randn(T, HKV, D, generator=g, device=dev, dtype=torch.float32).to(dtype)
    v = torch.randn(T, HKV, D, generator=g, device=dev, dtype=torch.float32).to(dtype)
    o = torch.zeros(T, H, D, device=dev, dtype=dtype)
    po = torch.randn(S * N, H, D, generator=g, device=dev, dtype=torch.float32).to(dtype)
    plse = (torch.rand(H, S * N, generator=g, device=dev) * 4 + 2).contiguous()
    # parents: node 0 is the root, then `depth` levels of `keep` nodes
    na = torch.ones(S, N, N, dtype=torch.bool, device=dev)
    import random
    rnd = random.Random(seed)
    for s in range(S):
        par = [-1] * N
        prev = [0]
        nxt = 1
        for _ in range(depth):
            lvl = []
            for _ in range(keep):
                if nxt >= N:
                    break
                par[nxt] = rnd.choice(prev)
                lvl.append(nxt)
                nxt += 1
            prev = lvl or prev
        for i in range(N):
            j = i
            while j >= 0:
                na[s, i, j] = False
                j = par[j]
    return (q, k, v, o, po, plse, idx, na)


def bench(fn, t, scale, S, N, launches=8, iters=200):
    for _ in range(5):
        fn(t, scale, S, N)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(t, scale, S, N)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(launches):
            fn(t, scale, S, N)
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    per_fwd = e0.elapsed_time(e1) * 1e3 / iters  # us for `launches` launches
    return per_fwd, per_fwd / launches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=41)
    ap.add_argument("--S", type=int, default=1)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--HKV", type=int, default=4)
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--dtype", default="bf16", choices=("bf16", "fp16"))
    a = ap.parse_args()
    dev = "cuda"
    dtype = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    scale = a.D ** -0.5

    args = (a.S, a.N, a.H, a.HKV, a.D, a.keep, a.depth, dtype, dev)
    ref = make(*args)
    eager_run(ref, scale, a.S, a.N)
    oref = ref[3].clone()

    names = {"triton": triton_run, "cuda": cuda_run}
    for nm, fn in names.items():
        t = make(*args)
        fn(t, scale, a.S, a.N)
        oo = t[3]
        rows = ref[6].view(-1)
        A = oref.index_select(0, rows).float()
        B = oo.index_select(0, rows).float()
        same = (oref.index_select(0, rows) == oo.index_select(0, rows)).float().mean().item()
        mx = (A - B).abs().max().item()
        ulp = ((A - B).abs() / A.abs().clamp_min(1e-6)).max().item()
        print(f"{nm:7s} vs eager: exact={same * 100:.4f}%  maxabs={mx:.3e}  maxrel={ulp:.3e}")

    print()
    print(f"shape S={a.S} N={a.N} H={a.H} HKV={a.HKV} D={a.D} dtype={dtype}")
    ancs = (~ref[7][0]).sum(1).float()
    print(f"ancestors/row: mean={ancs.mean().item():.2f} max={int(ancs.max().item())}"
          f"  dense would be {a.N}")
    print()
    for nm, fn in [("eager", eager_run), ("triton", triton_run), ("cuda", cuda_run)]:
        t = make(*args)
        fwd, lay = bench(fn, t, scale, a.S, a.N, iters=a.iters)
        print(f"{nm:7s}  {lay:8.2f} us/layer   {fwd:9.2f} us/forward (8 layers)")


if __name__ == "__main__":
    main()
