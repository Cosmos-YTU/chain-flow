"""Wall-clock: baseline vs V9-chain vs EAGLE-chain (real live speedup on this hybrid backbone),
plus a component-measured projection for the V9 draft TREE.

Why a projection for the tree: a faithful single-pass tree verify needs a tree-attention mask, which
works on the 6 full-attention layers but NOT the 18 linear-attention/SSM layers (a sequential pass
over a flattened tree lets each node's recurrent state absorb non-ancestor siblings). A correct
single-pass SSM tree verify needs a state-forking kernel (SGLang-style) we have not built. So here we
MEASURE the real cost of processing N node-positions through the backbone (what such a kernel targets)
and combine it with the measured tree accepted-length to project the tree's wall-clock ceiling. The
chain numbers are fully real (they reuse the existing verify path).

speedup = accepted_tokens_per_step * T_baseline_step / (T_drafter + T_verify + T_commit)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chain_flow.context import ChainedFlowContext
from chain_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chain_flow.generation import generate_with_drafter
from chain_flow.training.eval_chunked_flow import (
    collect_speedup_prompts,
    generate_greedy_baseline,
    torch_dtype_from_string,
)
from chain_flow.training.train_tree_flow import load_tree_module
from chain_flow.training.train_eagle_flow import load_eagle_module
from chain_flow.training.window_dataset import TeacherWindowDataset


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _time_call(fn, *, iters, warmup, dev):
    for _ in range(warmup):
        fn()
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(dev)
    return (time.perf_counter() - t0) / iters


@torch.inference_mode()
def chain_speedup(frozen_lm, drafter, prompts, *, draft_len, max_new_tokens):
    cctx = ChainedFlowContext(frozen_lm)
    for p in prompts[:2]:                       # warmup
        generate_with_drafter(cctx, drafter, p, max_new_tokens=max_new_tokens, draft_len=draft_len)
    st = 0.0; stok = 0; acc = 0.0; steps = 0
    for p in prompts:
        r = generate_with_drafter(cctx, drafter, p, max_new_tokens=max_new_tokens, draft_len=draft_len)
        st += float(r.timings.get("total_generation")); stok += int(r.generated_token_count)
        for s in r.step_stats:
            acc += float(s.accepted_len); steps += 1
    return {"tps": stok / st if st else 0.0, "mean_accept_len": acc / steps if steps else 0.0}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree_dir", default="out/flow/ckpts/tree-hiddenkv-k4-l8-ffn6-rank256-o4-cov8")
    ap.add_argument("--eagle_dir", default="out/flow/ckpts/eagle-mix6-55k-k4-l8-ffn6")
    ap.add_argument("--dataset_path", default="data/flow_cache/gsm8k_1k_test")
    ap.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--num_prompts", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--tree_top_b", type=int, default=8)
    ap.add_argument("--tree_max_nodes", type=int, default=48)
    ap.add_argument("--output_path", default="out/flow/evals/tree_v9_wallclock.json")
    args = ap.parse_args()
    dev = torch.device(args.device)

    frozen_lm, _ = FrozenLMWrapper.from_pretrained(
        args.model_id, device=args.device, dtype=torch_dtype_from_string(args.dtype), local_files_only=False)
    tree_mod, _ = load_tree_module(args.tree_dir, frozen_lm=frozen_lm, device=dev)
    tree_drafter = tree_mod.drafter
    eagle_mod, _ = load_eagle_module(args.eagle_dir, frozen_lm=frozen_lm, device=dev)
    eagle_drafter = eagle_mod.drafter
    K = tree_drafter.config.draft_length

    dataset = TeacherWindowDataset.from_path(
        args.dataset_path, split="train", context_size=tree_drafter.config.context_size,
        draft_length=K, windows_per_epoch=None, seed=0)
    _, prompts = collect_speedup_prompts(dataset, num_prompts=args.num_prompts, warmup_prompts=2)

    out: dict = {"model": args.model_id, "K": K, "num_prompts": len(prompts)}

    # --- baseline decode tok/s (batch 1) ---
    bt = 0.0; btok = 0
    for p in prompts[:2]:
        generate_greedy_baseline(frozen_lm, p, max_new_tokens=args.max_new_tokens)
    for p in prompts:
        b = generate_greedy_baseline(frozen_lm, p, max_new_tokens=args.max_new_tokens)
        bt += float(b["seconds"]); btok += int(b["generated_tokens"])
    base_tps = btok / bt if bt else 0.0
    T_base_step = 1.0 / base_tps
    out["baseline"] = {"tps": base_tps, "T_step_ms": T_base_step * 1e3}

    # --- real chain wall-clock: V9 chain and EAGLE chain ---
    out["v9_chain"] = chain_speedup(frozen_lm, tree_drafter, prompts, draft_len=K, max_new_tokens=args.max_new_tokens)
    out["v9_chain"]["real_speedup"] = out["v9_chain"]["tps"] / base_tps if base_tps else 0.0
    out["eagle_chain"] = chain_speedup(frozen_lm, eagle_drafter, prompts, draft_len=K, max_new_tokens=args.max_new_tokens)
    out["eagle_chain"]["real_speedup"] = out["eagle_chain"]["tps"] / base_tps if base_tps else 0.0

    # --- tree component latencies ---
    # a representative mid-length prefill state to time against
    probe = prompts[len(prompts) // 2].to(dev)
    state, _ = frozen_lm.prefill(probe)

    T_build = _time_call(lambda: tree_drafter.build_tree(state, top_b=args.tree_top_b, max_nodes=args.tree_max_nodes),
                         iters=20, warmup=5, dev=dev)
    # verify-block scaling: cost of pushing N node-positions through the backbone with the prompt cache.
    # This is the target cost of a correct single-pass tree-verify kernel (N = #tree nodes).
    import copy
    def verify_block(n):
        vs = copy.deepcopy(state.past_key_values)
        toks = torch.zeros((1, n), dtype=torch.long, device=dev)
        from chain_flow.frozen_lm import LMState
        s2 = LMState(input_ids=state.input_ids, past_key_values=vs, final_hidden=state.final_hidden,
                     logits=state.logits, position=state.position)
        frozen_lm.forward_with_cache(toks, s2, use_cache=True)
    tree_nodes = tree_drafter.build_tree(state, top_b=args.tree_top_b, max_nodes=args.tree_max_nodes).num_nodes()
    T_verify = {n: _time_call(lambda n=n: verify_block(n), iters=15, warmup=4, dev=dev) for n in (K, 24, tree_nodes)}
    T_commit = _time_call(lambda: verify_block(K), iters=15, warmup=4, dev=dev)  # commit accepted prefix (~K)

    out["tree_components_ms"] = {"build_tree": T_build * 1e3, "commit": T_commit * 1e3,
                                 "verify_block": {str(n): v * 1e3 for n, v in T_verify.items()},
                                 "tree_nodes": tree_nodes}

    # --- projected tree wall-clock under an ideal single-pass tree-verify kernel ---
    # accepted length from the live tree eval (measured separately); allow override / read json.
    a_tree = None
    gj = Path("out/flow/evals/tree_v9_gsm8k.json")
    if gj.exists():
        a_tree = json.load(open(gj)).get("tree_accept", {}).get("mean_accept_len")
    a_tree = a_tree or 3.54
    step_cost_ideal = T_build + T_verify[tree_nodes] + T_commit
    out["tree_projection"] = {
        "accept_len_used": a_tree,
        "step_cost_ideal_ms": step_cost_ideal * 1e3,
        "speedup_ideal_kernel": a_tree * T_base_step / step_cost_ideal,
        "note": "ideal kernel = correct single-pass SSM tree verify at N-node-position cost (not yet built)",
    }

    b = out["baseline"]; v = out["v9_chain"]; e = out["eagle_chain"]; tp = out["tree_projection"]
    print(f"baseline: {b['tps']:.1f} tok/s  (T_step={b['T_step_ms']:.2f} ms)")
    print(f"V9  chain: {v['tps']:.1f} tok/s  speedup={v['real_speedup']:.3f}  accept={v['mean_accept_len']:.2f}")
    print(f"EAGLE chain: {e['tps']:.1f} tok/s  speedup={e['real_speedup']:.3f}  accept={e['mean_accept_len']:.2f}")
    tc = out["tree_components_ms"]
    print(f"tree components: build={tc['build_tree']:.2f}ms  verify[{tc['tree_nodes']}nodes]="
          f"{tc['verify_block'][str(tc['tree_nodes'])]:.2f}ms  commit={tc['commit']:.2f}ms  "
          f"(verify[K={args.tree_top_b and K}]={tc['verify_block'][str(K)]:.2f}ms)")
    print(f"V9 TREE projected (ideal kernel): speedup={tp['speedup_ideal_kernel']:.3f}  "
          f"(accept={tp['accept_len_used']:.2f}, step={tp['step_cost_ideal_ms']:.2f}ms)")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output_path, "w"), indent=2)
    print(f"saved: {args.output_path}")


if __name__ == "__main__":
    main()
