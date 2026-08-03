"""Eval for the TwoPass-flow drafter (V8): cached per-position accept@i (teacher-forced AR,
matching training) + optional live fla+fold speedup. Reuses chunked-flow eval helpers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.context import ChainedFlowContext
from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.generation import generate_with_drafter
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.eval_chunked_flow import (
    collect_speedup_prompts,
    generate_greedy_baseline,
    per_token_flow_metrics,
    summarize_metric,
    torch_dtype_from_string,
)
from chained_flow.training.train_twopass_flow import load_twopass_module
from chained_flow.training.window_dataset import TeacherWindowDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an EAGLE-style drafter checkpoint.")
    p.add_argument("--flow_dir", required=True)
    p.add_argument("--dataset_path", default="data/flow_cache/gsm8k_1k_test")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_batches", type=int, default=8)
    p.add_argument("--window_seed", type=int, default=0)
    p.add_argument("--measure_speedup", action="store_true")
    p.add_argument("--speedup_num_prompts", type=int, default=16)
    p.add_argument("--speedup_max_new_tokens", type=int, default=32)
    p.add_argument("--output_path", default=None)
    return p.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    frozen_lm, _ = FrozenLMWrapper.from_pretrained(
        args.model_id, device=args.device, dtype=torch_dtype_from_string(args.dtype), local_files_only=False
    )
    module, config = load_twopass_module(args.flow_dir, frozen_lm=frozen_lm, device=device)
    drafter = module.drafter
    ctx_size = drafter.config.context_size
    draft_len = drafter.config.draft_length

    dataset = TeacherWindowDataset.from_path(
        args.dataset_path, split=args.dataset_split, context_size=ctx_size, draft_length=draft_len,
        windows_per_epoch=None, seed=args.window_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_teacher_windows)

    accum: dict[str, list[torch.Tensor]] = {}
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        context_hidden = batch["context_hidden"].to(device)
        target_hidden = batch["target_hidden"].to(device)
        future_tokens = batch["future_tokens"].to(device)
        # cached eval = INFERENCE path: pass-1 parallel flow -> own tokens -> causal pass-2 refine.
        # Pass 2 conditions on the drafter's OWN pass-1 tokens (shifted), so NO target leak; matches propose().
        b, k = future_tokens.shape
        ctx = context_hidden.to(drafter._dtype)
        z0 = drafter.init_latents(ctx)
        h1 = drafter._integrate_pass1(ctx, z0)
        t1 = module.lm_head(h1).argmax(dim=-1)
        prev_emb = drafter._shift_emb(t1)
        h2 = drafter._integrate_pass2(ctx, h1, prev_emb)
        pred_hidden = h2
        drafter_logits = module.lm_head(h2)
        teacher_logits = module.lm_head(target_hidden)
        metrics = per_token_flow_metrics(
            pred_hidden=pred_hidden, target_hidden=target_hidden,
            pred_latent=pred_hidden, target_latent=target_hidden,  # no VAE: latent == hidden
            drafter_logits=drafter_logits, teacher_logits=teacher_logits, future_tokens=future_tokens,
        )
        for k, v in metrics.items():
            accum.setdefault(k, []).append(v.detach().float().cpu())

    summary = {k: summarize_metric(torch.cat(v)) for k, v in accum.items()}
    out: dict = {"flow_dir": args.flow_dir, "dataset_path": args.dataset_path, "metrics": summary}

    if args.measure_speedup:
        cctx = ChainedFlowContext(frozen_lm)
        _, prompts = collect_speedup_prompts(dataset, num_prompts=args.speedup_num_prompts, warmup_prompts=2)
        bt = st = 0.0; btok = stok = 0; acc = 0.0; steps = 0
        for prompt in prompts[:2]:
            generate_with_drafter(cctx, drafter, prompt, max_new_tokens=args.speedup_max_new_tokens, draft_len=draft_len)
        for prompt in prompts:
            base = generate_greedy_baseline(frozen_lm, prompt, max_new_tokens=args.speedup_max_new_tokens)
            bt += float(base["seconds"]); btok += int(base["generated_tokens"])
            flow = generate_with_drafter(cctx, drafter, prompt, max_new_tokens=args.speedup_max_new_tokens, draft_len=draft_len)
            st += float(flow.timings.get("total_generation")); stok += int(flow.generated_token_count)
            for s in flow.step_stats:
                acc += float(s.accepted_len); steps += 1
        btps = btok / bt if bt else 0.0; stps = stok / st if st else 0.0
        out["speedup"] = {
            "backbone_tps": btps, "drafter_verifier_tps": stps,
            "real_speedup": (stps / btps) if btps else 0.0,
            "mean_accept_len": (acc / steps) if steps else 0.0,
        }

    g = lambda k: summary[k]["mean"]
    print(f"greedy_prefix_len={g('accept.greedy_prefix_len'):.3f}  "
          + " ".join(f"acc@{i}={g(f'accept.rate@{i}'):.3f}" for i in range(1, draft_len + 1)))
    if "speedup" in out:
        print(f"real_speedup={out['speedup']['real_speedup']:.3f}  live_accept={out['speedup']['mean_accept_len']:.3f}")
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
