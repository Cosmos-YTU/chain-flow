from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.context import ChainedFlowContext
from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.eval_chunked_flow import find_flow_config_dir, load_flow_training_module, torch_dtype_from_string
from chained_flow.training.train_chunked_flow import ChunkedFlowModelArguments
from chained_flow.training.window_dataset import FlowWindowCacheDataset, TeacherWindowDataset
from chained_flow.verifier import SpeculativeVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe cached teacher windows against live Qwen/drafter behavior.")
    parser.add_argument("--flow_dir", required=True)
    parser.add_argument("--dataset_path", default="data/flow_cache/gsm8k_1k_test")
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_prefix_tokens", type=int, default=256, help="Skip sampled windows with longer prefixes.")
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def load_model_args(flow_dir: str | Path) -> ChunkedFlowModelArguments:
    config_dir = find_flow_config_dir(flow_dir)
    with (config_dir / "chained_flow_chunked_flow_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    return ChunkedFlowModelArguments(**config["model_args"])


def row_tensors(dataset, row_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, FlowWindowCacheDataset):
        offset = int(dataset.row_offsets[row_idx].item())
        length = int(dataset.row_lengths[row_idx].item())
        return dataset.input_ids[offset : offset + length].long(), dataset.hidden[offset : offset + length].float()
    if getattr(dataset, "row_tensors", None) is not None:
        input_ids, hidden = dataset.row_tensors[row_idx]
        return input_ids.long(), hidden.float()
    row = dataset.dataset[row_idx]
    return torch.as_tensor(row["input_ids"], dtype=torch.long), torch.as_tensor(row["final_hidden"], dtype=torch.float32)


def cached_context_hidden(hidden: torch.Tensor, *, t: int, context_size: int) -> torch.Tensor:
    context_start = t - context_size + 1
    if context_start >= 0:
        context = hidden[context_start : t + 1]
    else:
        pad = hidden[:1].expand(-context_start, -1)
        context = torch.cat([pad, hidden[: t + 1]], dim=0)
    return context.unsqueeze(0)


def ids_list(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def decode_tokens(frozen_lm, token_ids: list[int]) -> str:
    return frozen_lm.decode(token_ids, skip_special_tokens=False)


def greedy_future_from_state(frozen_lm, state, *, k: int) -> tuple[list[int], object]:
    tokens: list[int] = []
    cur = state
    for _ in range(k):
        token, _ = frozen_lm.next_token(cur)
        tokens.append(int(token.item()))
        cur, _ = frozen_lm.forward_with_cache(token, cur, use_cache=True)
    return tokens, cur


def prefix_match_len(a: list[int], b: list[int]) -> int:
    total = 0
    for x, y in zip(a, b):
        if x != y:
            break
        total += 1
    return total


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_examples < 1:
        raise ValueError("num_examples must be >= 1")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch_dtype_from_string(args.dtype)
    model_args = load_model_args(args.flow_dir)

    print(f"loading Qwen backbone: {args.model_id} dtype={args.dtype} device={device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    print(f"loading flow checkpoint: {args.flow_dir}", flush=True)
    module, _ = load_flow_training_module(args.flow_dir, frozen_lm=frozen_lm, device=device)
    verifier = SpeculativeVerifier(frozen_lm)

    dataset = TeacherWindowDataset.from_path(
        args.dataset_path,
        split=args.dataset_split,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=None,
        seed=args.seed,
        materialize_rows=True,
    )
    rng = random.Random(args.seed)
    samples = list(dataset.valid_rows)
    rng.shuffle(samples)

    rows = []
    attempted = 0
    for row_idx, min_t, max_t in samples:
        if len(rows) >= args.num_examples:
            break
        t = rng.randint(min_t, max_t)
        attempted += 1
        prefix_len = t + 1
        if args.max_prefix_tokens is not None and prefix_len > args.max_prefix_tokens:
            continue

        input_ids, hidden = row_tensors(dataset, int(row_idx))
        if t + model_args.draft_length + 1 > int(input_ids.shape[0]):
            continue

        prefix = input_ids[:prefix_len].unsqueeze(0).to(device)
        cached_future = ids_list(input_ids[t + 1 : t + model_args.draft_length + 1])
        cached_context = cached_context_hidden(hidden, t=t, context_size=model_args.context_size).to(device)
        cached_hidden_pred = module.drafter.decode_latent(module.drafter.predict_latent_from_context(cached_context))
        cached_tokens = ids_list(module.lm_head(cached_hidden_pred).argmax(dim=-1))

        live_future_state, _ = frozen_lm.prefill(prefix)
        live_future, _ = greedy_future_from_state(frozen_lm, live_future_state, k=model_args.draft_length)

        live_draft_state, _ = frozen_lm.prefill(prefix)
        live_context = module.drafter._context(live_draft_state)
        live_hidden_pred = module.drafter.decode_latent(module.drafter.predict_latent_from_context(live_context))
        live_tokens = ids_list(module.lm_head(live_hidden_pred).argmax(dim=-1))

        live_verify_state, _ = frozen_lm.prefill(prefix)

        hidden_delta = (cached_context.float() - live_context.float()).pow(2).mean().sqrt().item()
        hidden_scale = cached_context.float().pow(2).mean().sqrt().clamp_min(1e-8).item()
        hidden_rel_rmse = hidden_delta / hidden_scale
        hidden_cos = torch.nn.functional.cosine_similarity(
            cached_context.float().reshape(-1, cached_context.shape[-1]),
            live_context.float().reshape(-1, live_context.shape[-1]),
            dim=-1,
        ).mean().item()

        verify_result = verifier.verify(live_verify_state, torch.tensor([live_tokens], dtype=torch.long, device=device))

        row = {
            "row_idx": int(row_idx),
            "t": int(t),
            "prefix_len": int(prefix_len),
            "cached_future": cached_future,
            "live_future": live_future,
            "cached_draft": cached_tokens,
            "live_draft": live_tokens,
            "cached_future_eq_live_future": cached_future == live_future,
            "cached_draft_eq_live_draft": cached_tokens == live_tokens,
            "cached_live_context_rel_rmse": hidden_rel_rmse,
            "cached_live_context_cosine": hidden_cos,
            "cached_draft_vs_cached_future_prefix": prefix_match_len(cached_tokens, cached_future),
            "live_draft_vs_live_future_prefix": prefix_match_len(live_tokens, live_future),
            "verifier_accept_len": int(verify_result.acceptance.accepted_len),
            "verifier_matches": ids_list(verify_result.acceptance.matches.long()),
            "cached_future_text": decode_tokens(frozen_lm, cached_future),
            "live_future_text": decode_tokens(frozen_lm, live_future),
            "cached_draft_text": decode_tokens(frozen_lm, cached_tokens),
            "live_draft_text": decode_tokens(frozen_lm, live_tokens),
        }
        rows.append(row)

        print("", flush=True)
        print(f"example {len(rows)} row={row_idx} t={t} prefix_len={prefix_len}", flush=True)
        print(f"cached future == live future : {row['cached_future_eq_live_future']}", flush=True)
        print(f"cached draft  == live draft  : {row['cached_draft_eq_live_draft']}", flush=True)
        print(f"cached/live context rel RMSE  : {hidden_rel_rmse:.6f}", flush=True)
        print(f"cached/live context cosine    : {hidden_cos:.6f}", flush=True)
        print(f"cached draft prefix vs cached future: {row['cached_draft_vs_cached_future_prefix']}", flush=True)
        print(f"live draft prefix vs live future    : {row['live_draft_vs_live_future_prefix']}", flush=True)
        print(f"verifier accepted len              : {row['verifier_accept_len']}", flush=True)
        print(f"cached future ids: {cached_future}", flush=True)
        print(f"live future ids  : {live_future}", flush=True)
        print(f"cached draft ids : {cached_tokens}", flush=True)
        print(f"live draft ids   : {live_tokens}", flush=True)

    if not rows:
        raise RuntimeError(f"no examples collected after {attempted} attempts; increase max_prefix_tokens or check dataset")

    summary = {
        "num_examples": len(rows),
        "cached_future_live_future_match_rate": sum(r["cached_future_eq_live_future"] for r in rows) / len(rows),
        "cached_draft_live_draft_match_rate": sum(r["cached_draft_eq_live_draft"] for r in rows) / len(rows),
        "mean_cached_prefix_match": sum(r["cached_draft_vs_cached_future_prefix"] for r in rows) / len(rows),
        "mean_live_prefix_match": sum(r["live_draft_vs_live_future_prefix"] for r in rows) / len(rows),
        "mean_verifier_accept_len": sum(r["verifier_accept_len"] for r in rows) / len(rows),
        "mean_cached_live_context_rel_rmse": sum(r["cached_live_context_rel_rmse"] for r in rows) / len(rows),
        "mean_cached_live_context_cosine": sum(r["cached_live_context_cosine"] for r in rows) / len(rows),
    }
    result = {"metadata": vars(args), "summary": summary, "examples": rows}

    print("", flush=True)
    print("summary", flush=True)
    print("-------", flush=True)
    for key, value in summary.items():
        print(f"{key}: {value}", flush=True)

    if args.output_path is not None:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"probe saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
