"""Fused linear+CE fast path for training: compute the lm_head reductions without ever
materialising a [rows, 248320] tensor.

WHY THIS EXISTS
---------------
At 27B the head dominates training: a microbatch of 64 windows x draft_length 8 scores 512
positions over a 248,320-token vocabulary, and the reference path materialises that tensor
FOUR times per drafter forward --

    logits            [512, 248320] bf16   254 MB   (saved for backward)
    _ce               -> F.cross_entropy upcasts internally
    _accept           -> F.softmax(logits)          254 MB, and its grad
    _coverage         -> logits.float().topk(...)   509 MB fp32 copy

...for `cond_logits`, and again for `base_logits`.  Measured share of the training step: 38% of
FLOPs and by far the largest activation.  It is why the 27B microbatch is 64 while 4B and 9B run
at 512, which in turn starves every OTHER GEMM in the step of batch.

THE OBSERVATION
---------------
All three consumers reduce each row to THREE SCALARS:

    lse   = logsumexp(logits)            -> CE = lse - true_logit
    true  = logits[tok]                  -> accept prob = exp(true - lse)      (EXACTLY softmax[tok])
    kth   = (cov_b+1)-th largest logit   -> coverage hinge = relu(kth + margin - true)

`_accept` never needed the softmax: `softmax(x)[i] == exp(x_i - logsumexp(x))` identically.  So the
full-vocab tensor is a temporary, and a row-chunked pass can produce the three reductions and throw
each chunk away.

EXACTNESS
---------
This is not an approximation and not a shortlist:

  * every vocabulary entry participates in the logsumexp and the topk, exactly as before;
  * the backward is the analytic gradient, which is what autograd would have built --
        d lse  / d logits = softmax(logits)
        d true / d logits = onehot(tok)
        d kth  / d logits = onehot(argkth)      (the same subgradient torch.topk propagates)
  * reductions run in fp32 regardless of the logits dtype.

The one INTENTIONAL numerical difference: the reference `_accept` runs `F.softmax` in the logits'
own dtype (bf16), while `exp(true - lse)` here is computed in fp32.  That is strictly more accurate,
not less, but it IS a change -- `scripts/verify_fused_head.py` reports it rather than hiding it.

COST
----
Backward recomputes the chunk's logits instead of reading them back, so the head costs 3 matmuls
instead of 2.  That is deliberate: it trades ~25% more head FLOPs for the removal of ~1.5 GB of
activation per microbatch, which is what allows a much larger microbatch -- and the larger batch
speeds up the flow and VAE GEMMs too, which the head's own FLOPs never touched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _rdt(t):
    """Reduction dtype: fp32 for half/bf16/fp32 inputs, but fp64 stays fp64.

    Hard-coding `.float()` would silently downcast a float64 check to float32 and cap its
    resolution at ~1e-7 -- which is precisely the band an algebraic error would hide in. The
    equivalence test runs in fp64 to rule floating point OUT as an explanation, so the reduction
    has to honour that.
    """
    return torch.float64 if t.dtype == torch.float64 else torch.float32


def _chunk_logits(x, W, bias, emb, w2):
    """logits = x @ W.T (+bias) (+ emb @ w2.T) for ONE chunk.

    The markov bias is folded in HERE rather than added by the caller.  `MarkovHead.bias(prev)`
    is `w1(prev) @ w2.T`, a [rows, 248320] tensor in its own right -- materialising it would put
    back most of the activation this module exists to remove.  As a sum of two matmuls into the
    same chunk buffer it costs no extra memory at all.
    """
    logits = torch.matmul(x, W.t())
    if bias is not None:
        logits = logits + bias
    if emb is not None:
        logits = logits + torch.matmul(emb, w2.t())
    return logits


def _chunk_reduce(x, W, bias, emb, w2, tok, cov_b):
    """One chunk -> (lse, true, kth, argkth). No autograd, nothing retained."""
    logits = _chunk_logits(x, W, bias, emb, w2)
    lf = logits.to(_rdt(logits))
    lse = torch.logsumexp(lf, dim=-1)
    true = lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
    if cov_b is None:
        return lse, true, None, None
    # (cov_b+1)-th largest, plus WHICH entry it was so the backward can place the subgradient.
    tv, ti = lf.topk(cov_b + 1, dim=-1)
    return lse, true, tv[..., cov_b], ti[..., cov_b]


class _FusedHeadReduce(torch.autograd.Function):
    """Row-chunked lm_head reductions with an analytic, recompute-based backward."""

    @staticmethod
    def forward(ctx, x, W, bias, emb, w2, tok, cov_b, chunk):
        rows = x.shape[0]
        chunk = max(1, min(chunk, rows))
        lse_l, true_l, kth_l, ki_l = [], [], [], []
        with torch.no_grad():
            for a in range(0, rows, chunk):
                b = min(a + chunk, rows)
                e = None if emb is None else emb[a:b]
                lse, true, kth, ki = _chunk_reduce(x[a:b], W, bias, e, w2, tok[a:b], cov_b)
                lse_l.append(lse); true_l.append(true)
                if cov_b is not None:
                    kth_l.append(kth); ki_l.append(ki)
        E = torch.empty(0)
        ctx.save_for_backward(x, W, bias if bias is not None else E, tok,
                              torch.cat(ki_l) if cov_b is not None else E.long(),
                              emb if emb is not None else E, w2 if w2 is not None else E)
        ctx.cov_b, ctx.chunk = cov_b, chunk
        ctx.has_bias, ctx.has_markov = bias is not None, emb is not None
        out_kth = torch.cat(kth_l) if cov_b is not None else torch.zeros_like(torch.cat(lse_l))
        return torch.cat(lse_l), torch.cat(true_l), out_kth

    @staticmethod
    def backward(ctx, g_lse, g_true, g_kth):
        x, W, bias, tok, ki, emb, w2 = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        emb, w2 = (emb, w2) if ctx.has_markov else (None, None)
        rows, chunk = x.shape[0], ctx.chunk
        gx = torch.empty_like(x)
        gemb = torch.empty_like(emb) if emb is not None else None
        # w2 IS trainable, so its gradient accumulates across chunks. Full-vocab but only
        # [V, markov_rank] = 63.6M at rank 256, the same size as the parameter itself.
        gw2 = torch.zeros_like(w2, dtype=_rdt(x)) if w2 is not None else None
        for a in range(0, rows, chunk):
            b = min(a + chunk, rows)
            e = None if emb is None else emb[a:b]
            logits = _chunk_logits(x[a:b], W, bias, e, w2)
            # d lse/d logits = softmax; d true/d logits = onehot(tok); d kth/d logits = onehot(argkth)
            gl = torch.softmax(logits.to(_rdt(logits)), dim=-1) * g_lse[a:b].unsqueeze(-1)
            gl.scatter_add_(-1, tok[a:b].unsqueeze(-1), g_true[a:b].unsqueeze(-1))
            if ctx.cov_b is not None:
                gl.scatter_add_(-1, ki[a:b].unsqueeze(-1), g_kth[a:b].unsqueeze(-1))
            glw = gl.to(W.dtype)
            gx[a:b] = torch.matmul(glw, W)
            if emb is not None:
                gemb[a:b] = torch.matmul(glw, w2)
                gw2 += torch.matmul(gl.t(), e.to(gl.dtype))
        # lm_head W/bias are frozen buffers -- never optimised, so no gradient is produced.
        return gx, None, None, gemb, (None if gw2 is None else gw2.to(w2.dtype)), None, None, None


def head_reductions(hidden, W, bias, tok, *, markov_emb=None, markov_w2=None,
                    cov_b=None, chunk=64):
    """[B, K, H] hidden -> (lse, true, kth), each [B, K].

    `markov_emb`/`markov_w2` fold in the MarkovHead bias (`cond_logits` carries it, `base_logits`
    does not).  `cov_b=None` skips the topk for callers that only need cross-entropy.
    """
    B, K, H = hidden.shape
    x = hidden.reshape(B * K, H).to(W.dtype)
    e = None if markov_emb is None else markov_emb.reshape(B * K, markov_emb.shape[-1]).to(W.dtype)
    lse, true, kth = _FusedHeadReduce.apply(x, W, bias, e, markov_w2,
                                            tok.reshape(B * K), cov_b, chunk)
    return lse.view(B, K), true.view(B, K), kth.view(B, K)


def ce_from(lse, true):
    """Identical to F.cross_entropy(logits, tok): mean over rows of (logsumexp - true logit)."""
    return (lse - true).mean()


def accept_from(lse, true, gamma, eps):
    """Identical to the reference `_accept`, with softmax[tok] written as exp(true - lse)."""
    tp = torch.exp(true - lse)
    prefix = torch.cumprod(tp.clamp_min(eps), dim=1)
    w = gamma ** torch.arange(lse.shape[1], device=lse.device, dtype=prefix.dtype)
    return -(prefix * w.unsqueeze(0)).sum(dim=1).mean()


def coverage_from(kth, true, margin):
    """Identical to the reference `_coverage`: hinge of the (b+1)-th largest against the true logit."""
    return F.relu(kth + margin - true).mean()
