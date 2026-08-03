"""Is the flow actually doing anything?

Training logs show flow.mse 25.09 -> 0.0274 while hidden.rel_mse PLATEAUS at 0.63, which is what an
algebraic shortcut decoupled from the real hidden prediction would look like. This measures the
flow's contribution directly, inference-only, no retrain:

  * accept with the full flow (num_flow_steps=2, as shipped)
  * accept with 1 Euler step
  * accept with the flow DISABLED -- decode z0, the deterministic delta extrapolation
    z0[i] = h_last + (i+1)*(h_last - h_prev)  (init_mode: delta)
  * geometry: how far the flow actually moves z0 (relative L2 + cosine)

If accept is flat across all three, the flow machinery is not earning its ~10 ms.
Also probes the tau mismatch: training samples tau ~ U(0,1) but inference visits only {0, 0.5}.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so the sibling import resolves
from diff_plugin_vs_harness import build_proposer  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckd", default="out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--states", default="teacher_states/bench-4b-*")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--per_domain", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    dev, dtype = "cuda", torch.float16
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    p, dcfg = build_proposer(args.ckd, m.get_input_embeddings().weight, m.lm_head.weight,
                             args.K, args.width, dev, dtype)
    d = p.drafter
    C = p.ctx_size
    print(f"ckpt={os.path.basename(args.ckd)} ctx={C} draft_len={dcfg.draft_length} "
          f"num_flow_steps={dcfg.num_flow_steps} init_mode={dcfg.init_mode}", flush=True)

    from datasets import load_from_disk
    wins = []
    for path in sorted(glob.glob(args.states)):
        ds = load_from_disk(path)
        n0 = len(wins)
        for ex in ds:
            ids = ex["input_ids"]
            fh = torch.tensor(ex["final_hidden"], dtype=dtype)
            T = min(len(ids), fh.shape[0])
            for i in range(max(C, int(ex.get("prompt_length", 0))), T - args.K - 1):
                wins.append((fh[i - C: i + 1], ids[i], ids[i + 1: i + 1 + args.K]))
                if len(wins) - n0 >= args.per_domain:
                    break
            if len(wins) - n0 >= args.per_domain:
                break
    print(f"windows: {len(wins)}", flush=True)

    orig_integrate = d.integrate

    def make_integrate(steps):
        if steps == 0:
            return lambda context, z0: z0           # flow disabled: z0 straight through
        def f(context, z0):
            z, dt = z0, 1.0 / steps
            for s in range(steps):
                tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=d._dtype)
                z = z + dt * d.flow_velocity(z, tau, context)
            return z
        return f

    # geometry: how far does the flow move z0?
    H = torch.stack([w[0] for w in wins[:args.batch]]).to(dev)
    ctx = torch.stack([d._context(p._DS(H[j:j + 1, :-1]))[0] for j in range(H.shape[0])], 0).to(dtype)
    ctx_lat = d._encode(ctx.to(d._dtype))
    z0 = d.init_latents(ctx_lat)
    d.integrate = orig_integrate
    zf = d.integrate(ctx_lat, z0)
    rel = ((zf - z0).norm(dim=-1) / z0.norm(dim=-1).clamp_min(1e-6)).mean().item()
    cos = torch.nn.functional.cosine_similarity(zf.float(), z0.float(), dim=-1).mean().item()
    print(f"\nflow displacement in LATENT space: relative L2 {rel:.4f} | cosine(z_final, z0) {cos:.4f}")
    h0, hf = d._decode(z0), d._decode(zf)
    relh = ((hf - h0).norm(dim=-1) / h0.norm(dim=-1).clamp_min(1e-6)).mean().item()
    cosh = torch.nn.functional.cosine_similarity(hf.float(), h0.float(), dim=-1).mean().item()
    print(f"flow displacement in HIDDEN space: relative L2 {relh:.4f} | cosine(h_final, h_0) {cosh:.4f}")

    print(f"\n{'flow steps':>12} {'accept':>8}   (plugin arm: lagged ctx + known0 seed)")
    print("-" * 46)
    for steps in (2, 1, 0):
        d.integrate = make_integrate(steps)
        tot, n = 0, 0
        for b0 in range(0, len(wins), args.batch):
            chunk = wins[b0: b0 + args.batch]
            Hb = torch.stack([w[0] for w in chunk]).to(dev)
            k0 = torch.tensor([w[1] for w in chunk], device=dev)
            cx = torch.stack([d._context(p._DS(Hb[j:j + 1, :-1]))[0] for j in range(Hb.shape[0])], 0)
            p.feedback = False
            ch = p._beam_chains(cx, k0).tolist()
            for j, (_, _, t) in enumerate(chunk):
                n += 1
                q = 0
                while q < len(ch[j]) and q < len(t) and int(ch[j][q]) == int(t[q]):
                    q += 1
                tot += q
        lbl = "DISABLED (z0)" if steps == 0 else str(steps)
        print(f"{lbl:>12} {1 + tot / n:8.2f}")
    d.integrate = orig_integrate


if __name__ == "__main__":
    main()
