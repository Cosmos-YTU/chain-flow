"""Controlled differential: is the plugin's low accept an IMPLEMENTATION bug or the one-step lag?

Same cached teacher hiddens/tokens for every arm, so the only variable is how the context is built:

  harness  ctx = h[i-C+1 .. i]      draft depths 0..K-1   (offline harness: it re-runs the model on
                                                           each committed token, so it HAS h(t_i))
  plugin   ctx = h[i-C .. i-1]      draft depths 1..K     (vLLM never computes h(t_i) in the step
                                    + seed token t_i       where we must propose)

Both predict the same targets t_{i+1}..t_{i+K}, so accept is directly comparable. Also reports the
harness TREE accept as an anchor against the published per-domain numbers.

Runs the plugin's REAL _beam_chains (imported, not reimplemented) so an implementation bug shows up.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chained_flow.vllm_plugin.flow_proposer import FlowDrafterProposer  # noqa: E402 (no vllm import)


def build_proposer(ckd, embed_w, lm_w, K, width, dev, dtype, flow_steps=None):
    """Construct the real proposer object without __init__ (which needs a live vLLM config)."""
    import torch.nn.functional as F
    from types import SimpleNamespace
    from safetensors.torch import load_file
    from chained_flow.drafters.tree_vae_flow import TreeVAEFlowDrafter, TreeVAEFlowConfig

    class Emb:
        def __init__(s, w): s.w = w
        def __call__(s, i): return F.embedding(i, s.w)

    class Head(torch.nn.Module):
        def __init__(s, w): super().__init__(); s.weight = w
        def forward(s, h): return h.to(s.weight.dtype) @ s.weight.T

    class SM:
        def __init__(s):
            s.config = SimpleNamespace(hidden_size=embed_w.shape[1], vocab_size=embed_w.shape[0])
            s._lm, s._e = Head(lm_w), Emb(embed_w)
        @property
        def lm_head(s): return s._lm
        def get_input_embeddings(s): return s._e

    class Stub:
        def __init__(s): s.model = SM()
        def lm_head(s, h): return s.model.lm_head(h)

    cfgj = json.load(open(f"{ckd}/chained_flow_tree_config.json"))["model_args"]
    fields = {f.name for f in dataclasses.fields(TreeVAEFlowConfig)}
    dcfg = TreeVAEFlowConfig(**{k: v for k, v in cfgj.items() if k in fields})
    # inference-only ODE-solver step count (see flow_proposer._build / CF_FLOW_STEPS)
    fs = flow_steps if flow_steps is not None else os.environ.get("CF_FLOW_STEPS")
    if fs:
        print(f"[diff] num_flow_steps {dcfg.num_flow_steps} -> {int(fs)}", flush=True)
        dcfg.num_flow_steps = int(fs)
    d = TreeVAEFlowDrafter(Stub(), dcfg).to(dev).to(dtype).eval()
    d._dtype = dtype
    assert d.config is dcfg
    # VERIFY the integrator really takes the override: count flow_velocity calls per integrate().
    _n = {"v": 0, "i": 0}
    _fv, _ig = d.flow_velocity, d.integrate
    def _fv_c(*a, **k):
        _n["v"] += 1
        return _fv(*a, **k)
    def _ig_c(*a, **k):
        _n["i"] += 1
        return _ig(*a, **k)
    d.flow_velocity, d.integrate = _fv_c, _ig_c
    d._flowcnt = _n
    sd = load_file(f"{ckd}/model.safetensors")
    sub = {k[len("drafter."):]: v for k, v in sd.items() if k.startswith("drafter.")}
    res = d.load_state_dict(sub, strict=False)
    assert not res.missing_keys, f"missing drafter weights: {res.missing_keys[:5]}"

    p = FlowDrafterProposer.__new__(FlowDrafterProposer)
    p.drafter, p.ctx_size, p.order = d, dcfg.context_size, dcfg.path_order
    p.K, p.width, p.dev, p.dtype = K, width, dev, dtype
    p._sl, p._hw, p._w2 = None, lm_w, d.markov.w2.weight
    p.prof, p.use_cg, p.feedback = False, False, False
    # two-pass candidate head (CF_TWOPASS_M) -- same env knob the plugin reads, so the offline
    # accept measured here is the accept the engine gets.
    p.twopass_m = int(os.environ.get("CF_TWOPASS_M", "0"))
    p.twopass_seed = os.environ.get("CF_TWOPASS_SEED", "0") == "1"
    p.path_trim = os.environ.get("CF_PATH_TRIM", "0") == "1"
    p.twopass_shared = os.environ.get("CF_TWOPASS_SHARED", "0") == "1"
    p._t = {"flow": 0.0, "beam": 0.0}
    return p, dcfg


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--states", default="teacher_states/bench-4b-*")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--per_domain", type=int, default=400, help="windows sampled per domain")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--flow_steps", type=int, default=None,
                    help="override the checkpoint's num_flow_steps (inference-only ODE steps)")
    ap.add_argument("--lag_ckd", default=None,
                    help="lag-trained ckpt: adds a 4th arm using build_context(h[..t-1], token t)")
    ap.add_argument("--shortlist", default=None,
                    help="CF_SHORTLIST .pt: restrict the head to these token ids (shipping config)")
    ap.add_argument("--plugin_tree", action="store_true",
                    help="also score the plugin's REAL _beam_tree draft with tree acceptance "
                         "(the shipping path: lagged context + branching)")
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--topb", type=int, default=8)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.float16
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    embed_w = m.get_input_embeddings().weight
    lm_w = m.lm_head.weight
    p, dcfg = build_proposer(args.ckd, embed_w, lm_w, args.K, args.width, dev, dtype,
                             flow_steps=args.flow_steps)
    pl = None
    if args.lag_ckd:
        pl, _ = build_proposer(args.lag_ckd, embed_w, lm_w, args.K, args.width, dev, dtype)
        assert pl.drafter.lag_proj is not None, "lag_ckd was not trained with lag_context=true"
        pl.feedback = True   # lag ctx already represents position t -> draft depths 0..K-1
    if args.shortlist:
        V = lm_w.shape[0]
        sl = torch.load(args.shortlist, map_location="cpu").flatten().long()
        sl = sl[(sl >= 0) & (sl < V)].unique().to(dev)
        p._sl = sl
        p._hw = lm_w[sl].contiguous()
        p._w2 = p.drafter.markov.w2.weight[sl].contiguous()
        print(f"[diff] shortlist head: {sl.numel()} of {V} rows", flush=True)
    if p.twopass_m:
        print(f"[diff] TWO-PASS candidate head: M={p.twopass_m} of {p._hw.shape[0]} rows, "
              f"seed_cond={'on' if p.twopass_seed else 'off'}", flush=True)
    p.tree_keep, p.tree_topb, p.tree_depth = args.keep, args.topb, args.depth
    C, K = p.ctx_size, args.K
    print(f"ctx_size={C} draft_length={dcfg.draft_length} K={K} width={args.width} "
          f"num_flow_steps={dcfg.num_flow_steps}", flush=True)

    from datasets import load_from_disk
    rows = []
    for path in sorted(glob.glob(args.states)):
        dom = os.path.basename(path).split("-", 2)[-1]
        ds = load_from_disk(path)
        rows.append((dom, ds))

    print(f"\n{'domain':<18} {'n':>6} {'harness TREE':>12} {'harness chain':>14} "
          f"{'plugin chain':>13} {'delta':>7}")
    print("-" * 76)
    tot = {"h": [], "p": []}
    for dom, ds in rows:
        wins = []
        for ex in ds:
            ids = ex["input_ids"]
            fh = torch.tensor(ex["final_hidden"], dtype=dtype)
            T = min(len(ids), fh.shape[0])
            # need h[i-C .. i] (plugin arm reaches one further back) and targets t_{i+1..i+K}
            # Score GENERATED tokens only. Prompt-region tokens are not model-generated and are far
            # harder to predict; including them depresses every arm and breaks comparison with the
            # published per-domain accept, which is measured during generation.
            start = max(C, int(ex.get("prompt_length", 0)))
            for i in range(start, T - K - 1):
                wins.append((fh[i - C: i + 1], ids[i], ids[i + 1: i + 1 + K]))
                if len(wins) >= args.per_domain:
                    break
            if len(wins) >= args.per_domain:
                break
        if not wins:
            continue

        acc = {"h": 0, "p": 0, "t": 0, "l": 0}
        n = 0
        for b0 in range(0, len(wins), args.batch):
            chunk = wins[b0: b0 + args.batch]
            H = torch.stack([w[0] for w in chunk]).to(dev)            # [B, C+1, D]
            k0 = torch.tensor([w[1] for w in chunk], device=dev)       # committed token t_i
            tgt = [w[2] for w in chunk]

            # harness arm: context ENDS at h(t_i); draft depths 0..K-1
            ctx_h = torch.stack([p.drafter._context(p._DS(H[j:j + 1, 1:]))[0]
                                 for j in range(H.shape[0])], 0)
            p.feedback = True                                          # feedback=True => draft depth 0
            ch_h = p._beam_chains(ctx_h, k0).tolist()

            # plugin arm: context ends at h(t_{i-1}); seed t_i; draft depths 1..K
            ctx_p = torch.stack([p.drafter._context(p._DS(H[j:j + 1, :-1]))[0]
                                 for j in range(H.shape[0])], 0)
            p.feedback = False
            ch_p = p._beam_chains(ctx_p, k0).tolist()

            if args.plugin_tree:
                # PLUGIN TREE arm: the shipping path -- lagged context, branching draft, tree
                # acceptance. This is the arm the two-pass candidate head has to not regress.
                N = args.keep * args.depth
                bt = p._beam_tree(ctx_p, k0)
                tk, tp = bt[:, :N].tolist(), bt[:, N:].tolist()
                for j, t in enumerate(tgt):
                    kids = {}
                    for ni, pa in enumerate(tp[j]):
                        kids.setdefault(int(pa), []).append(ni)
                    cur, dd = -1, 0
                    while dd < len(t):
                        nx = next((c for c in kids.get(cur, []) if int(tk[j][c]) == int(t[dd])), None)
                        if nx is None:
                            break
                        cur, dd = nx, dd + 1
                    acc["pt"] = acc.get("pt", 0) + dd

            if pl is not None:
                # LAG arm: exactly what the vLLM proposer can supply -- h(..t-1) plus token t,
                # with the missing slot synthesized by the trained lag_proj.
                ctx_l = pl.drafter.build_context(H[:, :-1].to(dtype), k0)
                ch_l = pl._beam_chains(ctx_l, k0).tolist()
                for j, t in enumerate(tgt):
                    d = 0
                    while d < len(ch_l[j]) and d < len(t) and int(ch_l[j][d]) == int(t[d]):
                        d += 1
                    acc["l"] = acc.get("l", 0) + d

            # harness TREE arm: multi-branch acceptance (the published per-domain numbers). Anchor
            # for the whole offline setup -- if this reproduces them, the arms above are trustworthy.
            for j, t in enumerate(tgt):
                tree = p.drafter.build_tree_fast(p._DS(H[j:j + 1, 1:]), top_b=8, max_nodes=8,
                                                 max_depth=5)
                toks, pars = list(tree.tokens), list(tree.parents)
                kids = {}
                for ni, pa in enumerate(pars):
                    kids.setdefault(pa, []).append(ni)
                cur, d = -1, 0
                while d < len(t):
                    nxt = next((c for c in kids.get(cur, []) if int(toks[c]) == int(t[d])), None)
                    if nxt is None:
                        break
                    cur, d = nxt, d + 1
                acc["t"] += d

            for j, t in enumerate(tgt):
                n += 1
                for key, ch in (("h", ch_h[j]), ("p", ch_p[j])):
                    d = 0
                    while d < len(ch) and d < len(t) and int(ch[d]) == int(t[d]):
                        d += 1
                    acc[key] += d
        ah, apl, at = 1 + acc["h"] / n, 1 + acc["p"] / n, 1 + acc["t"] / n
        tot["h"].append(ah); tot["p"].append(apl); tot.setdefault("t", []).append(at)
        extra = ""
        if args.plugin_tree:
            apt = 1 + acc.get("pt", 0) / n
            tot.setdefault("pt", []).append(apt)
            extra += f"   PLUGIN-TREE {apt:>5.2f}"
        if pl is not None:
            al = 1 + acc["l"] / n
            tot.setdefault("l", []).append(al)
            extra = f"   LAG {al:>5.2f}  (vs plugin {al - apl:+.2f})"
        print(f"{dom:<18} {n:>6} {at:>12.2f} {ah:>14.2f} {apl:>13.2f} {apl - ah:>+7.2f}{extra}",
              flush=True)

    if tot["h"]:
        mh = sum(tot["h"]) / len(tot["h"]); mp = sum(tot["p"]) / len(tot["p"])
        mt = sum(tot["t"]) / len(tot["t"])
        print("-" * 76)
        print(f"{'MEAN':<18} {'':>6} {mt:>12.2f} {mh:>14.2f} {mp:>13.2f} {mp - mh:>+7.2f}")
        if "pt" in tot and tot["pt"]:
            mpt = sum(tot["pt"]) / len(tot["pt"])
            print(f"{'MEAN PLUGIN-TREE':<18} {'':>6} {mpt:>12.2f}"
                  f"   (keep={args.keep} depth={args.depth} topb={args.topb}"
                  f", twopass_M={p.twopass_m or 'off'}; worst domain {min(tot['pt']):.2f})")
        if "l" in tot and tot["l"]:
            ml = sum(tot["l"]) / len(tot["l"])
            print(f"{'MEAN LAG':<18} {'':>6} {'':>12} {'':>14} {ml:>13.2f} "
                  f"(vs plugin {ml - mp:+.2f}, vs harness chain {ml - mh:+.2f})")
        print(f"\nworst domain: tree {min(tot['t']):.2f} | harness chain {min(tot['h']):.2f} "
              f"| plugin {min(tot['p']):.2f}")
        cnt = p.drafter._flowcnt
        print(f"\n[verify] integrate() calls={cnt['i']} flow_velocity() calls={cnt['v']} "
              f"=> {cnt['v'] / max(cnt['i'], 1):.2f} velocity evals per integrate "
              f"(config num_flow_steps={dcfg.num_flow_steps})")
        print(f"\ndecomposition of the gap (mean): tree {mt:.2f} "
              f"-[chain: {mh-mt:+.2f}]-> {mh:.2f} -[lag: {mp-mh:+.2f}]-> {mp:.2f}")


if __name__ == "__main__":
    main()
