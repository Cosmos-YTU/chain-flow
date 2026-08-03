"""Eval for the tree-flow drafter (V9).

Reports, leak-safe:
  * accept@i on the path-conditioned CHAIN (own tokens; drop-in comparison to V6/EAGLE chain).
  * top-b coverage -> TREE-CEILING accepted-length, for the MARGINAL distributions (comparable to V6)
    and for the PATH-CONDITIONED distributions (the tree-relevant number V9 was trained to raise).
    Ceiling accepted-length = E[ sum_j 1{true token in top-b at every position <= j} ] over windows.
  * optional --measure_tree: REAL greedy tree accepted-length, verifying against the backbone's own
    argmax along the accepted path (D backbone forwards, membership test per depth).

The coverage ceiling conditions each position on the CORRECT parents (the honest "at a correct-so-far
tree node, is the true child in my top-b" question) — the same construction used for V6's ceiling.
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

from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.eval_chunked_flow import (
    collect_speedup_prompts,
    per_token_flow_metrics,
    summarize_metric,
    torch_dtype_from_string,
)
from chained_flow.training.train_tree_flow import load_tree_module
from chained_flow.training.window_dataset import TeacherWindowDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a tree-flow (V9) drafter checkpoint.")
    p.add_argument("--flow_dir", required=True)
    p.add_argument("--dataset_path", default="data/flow_cache/gsm8k_1k_test")
    p.add_argument("--dataset_split", default="train")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--max_batches", type=int, default=8)
    p.add_argument("--window_seed", type=int, default=0)
    p.add_argument("--coverage_bs", default="1,2,4,8,16")
    p.add_argument("--measure_tree", action="store_true")
    p.add_argument("--tree_num_prompts", type=int, default=16)
    p.add_argument("--tree_max_steps", type=int, default=16)
    p.add_argument("--tree_top_b", type=int, default=None)
    p.add_argument("--tree_max_nodes", type=int, default=None)
    p.add_argument("--output_path", default=None)
    return p.parse_args()


def _ceiling(contains: torch.Tensor) -> float:
    """contains [N, K] in {0,1} -> mean accepted-length under 'true token stays in top-b' acceptance."""
    prefix = torch.cumprod(contains, dim=1)
    return float(prefix.sum(dim=1).mean())


def _topb_contains(logits: torch.Tensor, tok: torch.Tensor, b: int) -> torch.Tensor:
    topb = logits.topk(min(b, logits.shape[-1]), dim=-1).indices
    return (topb == tok.unsqueeze(-1)).any(dim=-1).float()


@torch.inference_mode()
def _backbone_greedy_next(frozen_lm, seq: torch.Tensor) -> int:
    state, _ = frozen_lm.prefill(seq)
    logits = frozen_lm.lm_head(state.final_hidden[:, -1, :])
    return int(logits.argmax(dim=-1).item())


@torch.inference_mode()
def measure_tree_accept(frozen_lm, drafter, prompt, *, max_steps, top_b, max_nodes) -> tuple[float, int]:
    """Greedy tree accepted-length: at each depth accept the backbone's own argmax iff it is among the
    tree's children of the accepted node. Returns (tree_tokens_accepted, decode_steps)."""
    seq = prompt.to(frozen_lm.device)
    total, steps = 0, 0
    for _ in range(max_steps):
        state, _ = frozen_lm.prefill(seq)
        tree = drafter.build_tree(state, top_b=top_b, max_nodes=max_nodes)
        children: dict[int, list[int]] = {}
        for i, par in enumerate(tree.parents):
            children.setdefault(par, []).append(i)
        cur_parent, cur_seq, accepted = -1, seq, 0
        while True:
            cand = children.get(cur_parent, [])
            if not cand:
                break
            tok2node = {tree.tokens[n]: n for n in cand}
            g = _backbone_greedy_next(frozen_lm, cur_seq)
            if g in tok2node:
                accepted += 1
                cur_parent = tok2node[g]
                cur_seq = torch.cat([cur_seq, torch.tensor([[g]], device=cur_seq.device)], dim=1)
            else:
                break
        total += accepted
        steps += 1
        # advance one step regardless (spec-decoding always commits the backbone token)
        g = _backbone_greedy_next(frozen_lm, cur_seq)
        seq = torch.cat([cur_seq, torch.tensor([[g]], device=cur_seq.device)], dim=1)
    return total, steps


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    frozen_lm, _ = FrozenLMWrapper.from_pretrained(
        args.model_id, device=args.device, dtype=torch_dtype_from_string(args.dtype), local_files_only=False
    )
    module, config = load_tree_module(args.flow_dir, frozen_lm=frozen_lm, device=device)
    drafter = module.drafter
    ctx_size = drafter.config.context_size
    draft_len = drafter.config.draft_length
    cov_bs = [int(x) for x in str(args.coverage_bs).split(",") if x]

    dataset = TeacherWindowDataset.from_path(
        args.dataset_path, split=args.dataset_split, context_size=ctx_size, draft_length=draft_len,
        windows_per_epoch=None, seed=args.window_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_teacher_windows)

    accum: dict[str, list[torch.Tensor]] = {}
    cov_marg: dict[int, list[torch.Tensor]] = {b: [] for b in cov_bs}
    cov_cond: dict[int, list[torch.Tensor]] = {b: [] for b in cov_bs}
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        context_hidden = batch["context_hidden"].to(device)
        target_hidden = batch["target_hidden"].to(device)
        future_tokens = batch["future_tokens"].to(device)
        b, k = future_tokens.shape

        pred_hidden = drafter.predict_hidden(context_hidden)          # one parallel flow pass
        base_logits = module.lm_head(pred_hidden)                     # marginal [B,K,V]

        # --- leak-safe path-conditioned CHAIN (own tokens), matches propose() ---
        chain_logits = base_logits.clone()
        path: list[torch.Tensor] = []
        for pos in range(k):
            if path:
                committed = torch.stack(path, dim=1)
                res = drafter._position_residual(committed, pos)
                chain_logits[:, pos, :] = module.lm_head(pred_hidden[:, pos, :] + res) + drafter.markov.bias(path[-1])
            path.append(chain_logits[:, pos, :].argmax(dim=-1))
        teacher_logits = module.lm_head(target_hidden)
        metrics = per_token_flow_metrics(
            pred_hidden=pred_hidden, target_hidden=target_hidden,
            pred_latent=pred_hidden, target_latent=target_hidden,
            drafter_logits=chain_logits, teacher_logits=teacher_logits, future_tokens=future_tokens,
        )
        for kk, vv in metrics.items():
            accum.setdefault(kk, []).append(vv.detach().float().cpu())

        # --- path-CONDITIONED logits with CORRECT parents (for the tree ceiling) ---
        residual = drafter._path_residual(future_tokens)
        cond_logits = module.lm_head(pred_hidden + residual)
        prev = torch.zeros_like(future_tokens); prev[:, 1:] = future_tokens[:, :-1]
        bias = drafter.markov.bias(prev); bias[:, 0, :] = 0.0
        cond_logits = cond_logits + bias
        for bb in cov_bs:
            cov_marg[bb].append(_topb_contains(base_logits, future_tokens, bb).cpu())
            cov_cond[bb].append(_topb_contains(cond_logits, future_tokens, bb).cpu())

    summary = {k: summarize_metric(torch.cat(v)) for k, v in accum.items()}
    ceil_marg = {b: _ceiling(torch.cat(cov_marg[b])) for b in cov_bs}
    ceil_cond = {b: _ceiling(torch.cat(cov_cond[b])) for b in cov_bs}
    out: dict = {"flow_dir": args.flow_dir, "dataset_path": args.dataset_path, "metrics": summary,
                 "tree_ceiling": {"marginal": ceil_marg, "conditioned": ceil_cond}}

    if args.measure_tree:
        _, prompts = collect_speedup_prompts(dataset, num_prompts=args.tree_num_prompts, warmup_prompts=0)
        top_b = args.tree_top_b if args.tree_top_b is not None else drafter.config.tree_top_b
        max_nodes = args.tree_max_nodes if args.tree_max_nodes is not None else drafter.config.tree_max_nodes
        tot, stp = 0, 0
        for prompt in prompts:
            a, s = measure_tree_accept(frozen_lm, drafter, prompt, max_steps=args.tree_max_steps,
                                       top_b=top_b, max_nodes=max_nodes)
            tot += a; stp += s
        out["tree_accept"] = {"mean_accept_len": (tot / stp) if stp else 0.0, "top_b": top_b,
                              "max_nodes": max_nodes, "steps": stp}

    g = lambda k: summary[k]["mean"]
    print(f"CHAIN greedy_prefix_len={g('accept.greedy_prefix_len'):.3f}  "
          + " ".join(f"acc@{i}={g(f'accept.rate@{i}'):.3f}" for i in range(1, draft_len + 1)))
    print("TREE-CEILING (accepted-len):")
    for b in cov_bs:
        print(f"  b={b:>2}  marginal={ceil_marg[b]:.3f}  conditioned={ceil_cond[b]:.3f}")
    if "tree_accept" in out:
        print(f"LIVE tree_accept={out['tree_accept']['mean_accept_len']:.3f} "
              f"(top_b={out['tree_accept']['top_b']}, nodes={out['tree_accept']['max_nodes']})")
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
