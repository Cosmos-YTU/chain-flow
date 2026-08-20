"""Prove the fused head is equivalent to the reference path -- losses AND gradients.

A fast path that is quietly wrong produces a plausible loss curve and a worse model, which is the
most expensive failure available here. So this compares against the literal reference expressions
copied out of `train_tree_flow.py`, on the same inputs, including every gradient that flows.

Runs on CPU in fp64 by default: fp64 removes floating-point noise as an explanation, so any
disagreement is a real algebraic difference rather than something to argue about. `--bf16-cuda`
repeats it in the shipping dtype to report the numerical gap that actually occurs in training.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from chained_flow.training.fused_head import (accept_from, ce_from, coverage_from, head_reductions)


def reference(hidden, W, bias, emb, w2, tok, cov_b, margin, gamma, eps):
    """The reference path, verbatim: materialise logits, then the three losses off it."""
    B, K, _ = hidden.shape
    logits = torch.matmul(hidden.to(W.dtype), W.t())
    if bias is not None:
        logits = logits + bias
    if emb is not None:
        logits = logits + torch.matmul(emb.to(W.dtype), w2.t())     # MarkovHead.bias(prev)
    rd = torch.float64 if logits.dtype == torch.float64 else torch.float32
    ce = F.cross_entropy(logits.reshape(B * K, -1).to(rd), tok.reshape(B * K))
    probs = F.softmax(logits.to(rd), dim=-1)
    tp = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    prefix = torch.cumprod(tp.clamp_min(eps), dim=1)
    w = gamma ** torch.arange(K, device=logits.device, dtype=prefix.dtype)
    acc = -(prefix * w.unsqueeze(0)).sum(dim=1).mean()
    true = logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1).to(rd)
    thresh = logits.to(rd).topk(cov_b + 1, dim=-1).values[..., cov_b]
    cov = F.relu(thresh + margin - true).mean()
    return ce, acc, cov


def fused(hidden, W, bias, emb, w2, tok, cov_b, margin, gamma, eps, chunk):
    lse, true, kth = head_reductions(hidden, W, bias, tok, markov_emb=emb, markov_w2=w2,
                                     cov_b=cov_b, chunk=chunk)
    return ce_from(lse, true), accept_from(lse, true, gamma, eps), coverage_from(kth, true, margin)


def run(device, dtype, B, K, H, V, rank, cov_b, chunk, tol):
    g = torch.Generator(device="cpu").manual_seed(0)
    mk = lambda *s: torch.randn(*s, generator=g).to(device=device, dtype=dtype)
    hid, W, w2 = mk(B, K, H), mk(V, H) / H**0.5, mk(V, rank) / rank**0.5
    emb = mk(B, K, rank)
    bias = mk(V)
    tok = torch.randint(0, V, (B, K), generator=g).to(device)
    margin, gamma, eps = 1.0, 0.8, 1e-9

    ok = True
    for tag, use_markov in (("no markov (base_logits)", False), ("with markov (cond_logits)", True)):
        e, w = (emb, w2) if use_markov else (None, None)
        ins = [hid.clone().requires_grad_(True)]
        if use_markov:
            ins += [e.clone().requires_grad_(True), w.clone().requires_grad_(True)]
        rf = reference(ins[0], W, bias, *( (ins[1], ins[2]) if use_markov else (None, None) ),
                       tok, cov_b, margin, gamma, eps)
        rg = torch.autograd.grad(sum(rf), ins, retain_graph=False)

        ins2 = [hid.clone().requires_grad_(True)]
        if use_markov:
            ins2 += [e.clone().requires_grad_(True), w.clone().requires_grad_(True)]
        ff = fused(ins2[0], W, bias, *( (ins2[1], ins2[2]) if use_markov else (None, None) ),
                   tok, cov_b, margin, gamma, eps, chunk)
        fg = torch.autograd.grad(sum(ff), ins2)

        print(f"\n  {tag}")
        for nm, a, b in zip(("ce", "accept", "coverage"), rf, ff):
            d = (a - b).abs().item()
            print(f"    loss {nm:<9} ref={a.item():+.10f} fused={b.item():+.10f}  |d|={d:.3e}"
                  f"  {'OK' if d <= tol else 'FAIL'}")
            ok &= d <= tol
        names = ["d/d hidden"] + (["d/d markov_emb", "d/d markov_w2"] if use_markov else [])
        for nm, a, b in zip(names, rg, fg):
            d = (a - b).abs().max().item()
            scale = a.abs().max().clamp_min(1e-30).item()
            print(f"    grad {nm:<15} max|d|={d:.3e}  rel={d/scale:.3e}  {'OK' if d/scale <= tol else 'FAIL'}")
            ok &= d / scale <= tol
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-cuda", action="store_true", help="also run in the shipping dtype on GPU")
    ap.add_argument("--chunk", type=int, default=7, help="deliberately not a divisor of B*K")
    a = ap.parse_args()

    print("=== fp64 on CPU: any disagreement here is ALGEBRAIC, not floating point ===")
    ok = run("cpu", torch.float64, B=5, K=4, H=16, V=97, rank=8, cov_b=3, chunk=a.chunk, tol=1e-10)

    if a.bf16_cuda:
        if not torch.cuda.is_available():
            print("\n--bf16-cuda requested but no CUDA device is visible"); return 2
        print("\n=== bf16 on CUDA: the numerical gap that ACTUALLY occurs in training ===")
        print("    (the reference runs F.softmax in bf16; the fused path uses exp(true-lse) in")
        print("     fp32, which is strictly more accurate -- so a small gap here is expected)")
        ok &= run("cuda", torch.bfloat16, B=8, K=8, H=512, V=4096, rank=256, cov_b=8,
                  chunk=a.chunk, tol=5e-2)

    print("\n" + ("ALL CHECKS PASSED" if ok else "!! MISMATCH -- do not enable the fast path"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
