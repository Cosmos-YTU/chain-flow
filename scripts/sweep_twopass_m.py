"""Accept-vs-time curve for the two-pass candidate head (CF_TWOPASS_M).

Loads the target + drafter ONCE and, for each rescoring width M, measures on the SAME windows:
  * plugin CHAIN accept  (comparable to diff_plugin_vs_harness.py's `plugin chain` column)
  * plugin TREE  accept  (the shipping path: lagged context, keep x depth branching draft)
  * draft ms at batch 1, compiled + CUDA-graphed with CF_CUDA_BLOCK=1 (the shipping baseline)

M=0 is the one-pass head, i.e. the baseline row.

  CUDA_VISIBLE_DEVICES=5 python scripts/sweep_twopass_m.py \
      --ckd out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8 --model Qwen/Qwen3.5-4B \
      --shortlist out/flow/shortlist_q3527b.pt --per_domain 200 \
      --M 0,128,256,512,1024,2048,4096
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def bench(fn, n=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0


def graphed(fn, warmup=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def tree_accept(bt, N, tgt):
    tk, tp = bt[:, :N].tolist(), bt[:, N:].tolist()
    out = 0
    for j, t in enumerate(tgt):
        kids = {}
        for ni, pa in enumerate(tp[j]):
            kids.setdefault(int(pa), []).append(ni)
        cur, d = -1, 0
        while d < len(t):
            nx = next((c for c in kids.get(cur, []) if int(tk[j][c]) == int(t[d])), None)
            if nx is None:
                break
            cur, d = nx, d + 1
        out += d
    return out


def chain_accept(ch, tgt):
    out = 0
    for j, t in enumerate(tgt):
        d = 0
        while d < len(ch[j]) and d < len(t) and int(ch[j][d]) == int(t[d]):
            d += 1
        out += d
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--states", default="teacher_states/bench-4b-*")
    ap.add_argument("--shortlist", default="out/flow/shortlist_q3527b.pt")
    ap.add_argument("--per_domain", type=int, default=200)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--topb", type=int, default=8)
    ap.add_argument("--M", default="0,128,256,512,1024,2048,4096")
    ap.add_argument("--seed_cond", default="0", help="CF_TWOPASS_SEED variants to try")
    ap.add_argument("--shared", action="store_true", help="CF_TWOPASS_SHARED")
    ap.add_argument("--path_trim", action="store_true", help="CF_PATH_TRIM")
    ap.add_argument("--no_time", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("CF_CUDA_BLOCK", "1")
    from diff_plugin_vs_harness import build_proposer

    dev, dtype = "cuda", torch.float16
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    embed_w, lm_w = m.get_input_embeddings().weight, m.lm_head.weight
    p, dcfg = build_proposer(args.ckd, embed_w, lm_w, args.K, args.width, dev, dtype)
    d = p.drafter
    for name in ("integrate", "flow_velocity"):     # drop the python counters (they break compile)
        d.__dict__.pop(name, None)
    del m
    torch.cuda.empty_cache()

    V = lm_w.shape[0]
    if args.shortlist:
        sl = torch.load(args.shortlist, map_location="cpu").flatten().long()
        sl = sl[(sl >= 0) & (sl < V)].unique().to(dev)
        p._sl, p._hw = sl, lm_w[sl].contiguous()
        p._w2 = d.markov.w2.weight[sl].contiguous()
    p.tree_keep, p.tree_topb, p.tree_depth = args.keep, args.topb, args.depth
    p.twopass_shared, p.path_trim = args.shared, args.path_trim
    Vs, N = p._hw.shape[0], args.keep * args.depth
    C, K = p.ctx_size, args.K
    print(f"ckpt={os.path.basename(args.ckd)} head rows={Vs} (of {V}) K={K} width={args.width} "
          f"tree keep={args.keep} depth={args.depth} topb={args.topb} batch={args.batch}")

    # ---- windows -----------------------------------------------------------------------
    from datasets import load_from_disk
    doms = []
    for path in sorted(glob.glob(args.states)):
        dom = os.path.basename(path).split("-", 2)[-1]
        ds = load_from_disk(path)
        wins = []
        for ex in ds:
            ids = ex["input_ids"]
            fh = torch.tensor(ex["final_hidden"], dtype=dtype)
            T = min(len(ids), fh.shape[0])
            start = max(C, int(ex.get("prompt_length", 0)))
            for i in range(start, T - K - 1):
                wins.append((fh[i - C: i + 1], ids[i], ids[i + 1: i + 1 + K]))
                if len(wins) >= args.per_domain:
                    break
            if len(wins) >= args.per_domain:
                break
        if wins:
            doms.append((dom, wins))
    print(f"domains: {[(x[0], len(x[1])) for x in doms]}\n")

    variants = []
    Ms = [int(x) for x in args.M.split(",")]
    seeds = [int(x) for x in args.seed_cond.split(",")]
    for M in Ms:
        if M == 0:
            variants.append((0, 1))
        else:
            for s in seeds:
                variants.append((M, s))

    hdr = f"{'M':>6} {'seed':>5} " + " ".join(f"{dm[:6]:>7}" for dm, _ in doms)
    print("CHAIN accept (plugin arm)")
    print(hdr + f" {'MEAN':>7}")
    rows = {}
    for M, s in variants:
        p.twopass_m, p.twopass_seed = M, bool(s)
        p.feedback = False
        means = []
        for dom, wins in doms:
            acc, n = 0, 0
            for b0 in range(0, len(wins), args.batch):
                ch = wins[b0: b0 + args.batch]
                H = torch.stack([w[0] for w in ch]).to(dev)
                k0 = torch.tensor([w[1] for w in ch], device=dev)
                tgt = [w[2] for w in ch]
                ctx_p = H[:, :-1].contiguous()
                acc += chain_accept(p._beam_chains(ctx_p, k0).tolist(), tgt)
                n += len(ch)
            means.append(1 + acc / n)
        rows[("c", M, s)] = means
        print(f"{M if M else 'off':>6} {s if M else '-':>5} "
              + " ".join(f"{v:>7.3f}" for v in means)
              + f" {sum(means) / len(means):>7.3f}", flush=True)

    print("\nTREE accept (plugin arm, the shipping path)")
    print(hdr + f" {'MEAN':>7} {'worst':>7}")
    for M, s in variants:
        p.twopass_m, p.twopass_seed = M, bool(s)
        means = []
        for dom, wins in doms:
            acc, n = 0, 0
            for b0 in range(0, len(wins), args.batch):
                ch = wins[b0: b0 + args.batch]
                H = torch.stack([w[0] for w in ch]).to(dev)
                k0 = torch.tensor([w[1] for w in ch], device=dev)
                tgt = [w[2] for w in ch]
                ctx_p = H[:, :-1].contiguous()
                acc += tree_accept(p._beam_tree(ctx_p, k0), N, tgt)
                n += len(ch)
            means.append(1 + acc / n)
        rows[("t", M, s)] = means
        print(f"{M if M else 'off':>6} {s if M else '-':>5} "
              + " ".join(f"{v:>7.3f}" for v in means)
              + f" {sum(means) / len(means):>7.3f} {min(means):>7.3f}", flush=True)

    if args.no_time:
        return

    # ---- draft time, compiled + cudagraphed (the shipping baseline) ---------------------
    from chained_flow.vllm_plugin import fused
    fused.fuse_path_head(d)
    fused.compile_flow(d, mode="max-autotune-no-cudagraphs")
    ctx1 = torch.randn(1, C, d.hidden_size, device=dev, dtype=dtype)
    k01 = torch.randint(0, 1000, (1,), device=dev)
    print("\ndraft ms (batch 1, compiled + cudagraphed, CF_CUDA_BLOCK=1)")
    print(f"{'M':>6} {'seed':>5} {'tree draft ms':>14} {'vs one-pass':>12} {'TREE accept':>12}")
    base_ms = None
    for M, s in variants:
        p.twopass_m, p.twopass_seed = M, bool(s)
        rep = graphed(lambda: p._beam_tree(ctx1, k01))
        ms = bench(rep, n=200)
        if base_ms is None:
            base_ms = ms
        acc = sum(rows[("t", M, s)]) / len(rows[("t", M, s)])
        print(f"{M if M else 'off':>6} {s if M else '-':>5} {ms:>14.3f} "
              f"{ms - base_ms:>+12.3f} {acc:>12.3f}", flush=True)


if __name__ == "__main__":
    main()
