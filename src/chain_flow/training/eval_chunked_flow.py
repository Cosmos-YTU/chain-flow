from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from chain_flow.context import ChainedFlowContext
from chain_flow.frozen_lm import DEFAULT_MODEL_ID
from chain_flow.generation import generate_with_drafter
from chain_flow.timing import synchronize_if_needed
from chain_flow.training.collators import collate_teacher_windows
from chain_flow.training.train_chunked_flow import (
    ChunkedFlowModelArguments,
    FlowLossArguments,
    SingleExpertFlowTrainingModule,
    flow_config_from_args,
)
from chain_flow.training.window_dataset import TeacherWindowDataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is normally present through datasets/transformers.
    tqdm = None


@dataclass
class ChunkedFlowEvalArguments:
    flow_dir: str
    dataset_path: str
    dataset_split: str = "train"
    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None
    dtype: str = "float16"
    batch_size: int = 32
    max_batches: int | None = None
    windows_per_epoch: int | None = None
    window_seed: int = 0
    materialize_rows: bool = True
    local_files_only: bool = False
    output_path: str | None = None
    eval_all_checkpoints: bool = False
    checkpoint_stride: int = 1
    noise_seed: int = 0
    measure_speedup: bool = False
    measure_speedup_lower_bound: bool = False
    speedup_num_prompts: int = 16
    speedup_warmup_prompts: int = 2
    speedup_max_new_tokens: int = 32


def torch_dtype_from_string(dtype: str | None) -> torch.dtype | None:
    if dtype is None:
        return None
    normalized = dtype.lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"unsupported dtype: {dtype}")
    return mapping[normalized]


def summarize_metric(values: torch.Tensor) -> dict[str, float]:
    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }


def summarize_scalar_metric(value: float) -> dict[str, float]:
    return {
        "mean": float(value),
        "std": 0.0,
        "min": float(value),
        "max": float(value),
        "p50": float(value),
        "p90": float(value),
        "p95": float(value),
        "p99": float(value),
    }


def dataset_eval_slug(dataset_path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_path).strip("._-")
    return slug or "dataset"


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def _has_weights(path: Path) -> bool:
    return (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()


def discover_flow_checkpoints(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    checkpoints = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-") and _has_weights(path)
    ]
    checkpoints.sort(key=lambda path: (_checkpoint_step(path), path.name))
    if not checkpoints:
        raise FileNotFoundError(f"no flow checkpoints found in {run_dir}")
    return checkpoints


def select_checkpoint_stride(checkpoints: list[Path], *, stride: int = 1) -> list[Path]:
    if stride < 1:
        raise ValueError("checkpoint_stride must be >= 1")
    return checkpoints[::stride]


def find_flow_config_dir(flow_dir: str | Path) -> Path:
    flow_dir = Path(flow_dir)
    for candidate in (flow_dir, flow_dir.parent):
        if (candidate / "chain_flow_chunked_flow_config.json").exists():
            return candidate
    raise FileNotFoundError(f"missing flow config in {flow_dir} or {flow_dir.parent}")


def checkpoint_eval_output_path(
    args: ChunkedFlowEvalArguments,
    checkpoint_dir: str | Path,
    *,
    run_dir: str | Path,
) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    run_dir = Path(run_dir)
    checkpoint_name = checkpoint_dir.name
    if checkpoint_name == run_dir.name:
        checkpoint_name = "final"
    dataset_slug = dataset_eval_slug(args.dataset_path)

    if args.output_path is None:
        return run_dir / f"flow_eval_metrics_{dataset_slug}_{checkpoint_name}.json"

    output_path = Path(args.output_path)
    if output_path.suffix == ".json":
        return output_path.with_name(f"{output_path.stem}_{dataset_slug}_{checkpoint_name}{output_path.suffix}")
    return output_path / f"flow_eval_metrics_{dataset_slug}_{checkpoint_name}.json"


def single_eval_output_path(args: ChunkedFlowEvalArguments) -> Path:
    dataset_slug = dataset_eval_slug(args.dataset_path)
    if args.output_path is not None:
        output_path = Path(args.output_path)
        if output_path.suffix == ".json":
            return output_path.with_name(f"{output_path.stem}_{dataset_slug}{output_path.suffix}")
        return output_path / f"flow_eval_metrics_{dataset_slug}.json"
    return Path(args.flow_dir) / f"flow_eval_metrics_{dataset_slug}.json"


def load_flow_training_module(
    flow_dir: str | Path,
    *,
    frozen_lm,
    device: torch.device,
) -> tuple[SingleExpertFlowTrainingModule, dict[str, Any]]:
    flow_dir = Path(flow_dir)
    config_dir = find_flow_config_dir(flow_dir)
    with (config_dir / "chain_flow_chunked_flow_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_args = ChunkedFlowModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = SingleExpertFlowTrainingModule(frozen_lm, flow_config_from_args(model_args), loss_args)

    safetensors_path = flow_dir / "model.safetensors"
    bin_path = flow_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing flow weights in {flow_dir}")

    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    return module, config


def load_flow_eval_dataloader(
    args: ChunkedFlowEvalArguments,
    *,
    context_size: int,
    draft_length: int,
) -> DataLoader:
    dataset = TeacherWindowDataset.from_path(
        args.dataset_path,
        split=args.dataset_split,
        context_size=context_size,
        draft_length=draft_length,
        windows_per_epoch=args.windows_per_epoch,
        seed=args.window_seed,
        materialize_rows=args.materialize_rows,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_teacher_windows)
    print(
        f"flow eval dataset loaded: windows={len(dataset)} context_size={context_size} draft_length={draft_length}",
        flush=True,
    )
    return dataloader


def per_token_flow_metrics(
    *,
    pred_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    pred_latent: torch.Tensor,
    target_latent: torch.Tensor,
    drafter_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    future_tokens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if pred_hidden.shape != target_hidden.shape:
        raise ValueError("pred_hidden and target_hidden must have the same shape")
    if pred_latent.shape != target_latent.shape:
        raise ValueError("pred_latent and target_latent must have the same shape")
    if drafter_logits.shape != teacher_logits.shape:
        raise ValueError("drafter_logits and teacher_logits must have the same shape")
    if drafter_logits.shape[:2] != future_tokens.shape:
        raise ValueError("logits prefix shape must match future_tokens")

    eps = 1e-12
    hidden_mse = F.mse_loss(pred_hidden, target_hidden, reduction="none").mean(dim=-1)
    target_power = target_hidden.float().pow(2).mean(dim=-1).clamp_min(eps)
    hidden_rel_mse = hidden_mse.float() / target_power
    hidden_rel_rmse = hidden_mse.float().sqrt() / target_power.sqrt()

    pred_norm = pred_hidden.norm(dim=-1)
    target_norm = target_hidden.norm(dim=-1)
    both_zero = (pred_norm == 0) & (target_norm == 0)
    cosine = F.cosine_similarity(pred_hidden, target_hidden, dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)

    latent_mse = F.mse_loss(pred_latent, target_latent, reduction="none").mean(dim=-1)

    drafter_probs = F.softmax(drafter_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    drafter_log_probs = drafter_probs.clamp_min(eps).log()
    teacher_log_probs = teacher_probs.clamp_min(eps).log()
    mixture_probs = (0.5 * (drafter_probs + teacher_probs)).clamp_min(eps)
    mixture_log_probs = mixture_probs.log()

    logit_ce = -drafter_log_probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
    logit_js_div = 0.5 * (teacher_probs * (teacher_log_probs - mixture_log_probs)).sum(dim=-1)
    logit_js_div = logit_js_div + 0.5 * (drafter_probs * (drafter_log_probs - mixture_log_probs)).sum(dim=-1)

    teacher_token_prob = teacher_probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
    drafter_token_prob = drafter_probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
    prob_ratio = drafter_token_prob / teacher_token_prob.clamp_min(eps)

    pred_tokens = drafter_logits.argmax(dim=-1)
    token_top1_match = (pred_tokens == future_tokens).float()

    def topk_match(k: int) -> torch.Tensor:
        topk = min(k, drafter_logits.shape[-1])
        drafter_topk = drafter_logits.topk(topk, dim=-1).indices
        return (drafter_topk == future_tokens.unsqueeze(-1)).any(dim=-1).float()

    future_token_score = drafter_logits.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
    teacher_rank = (drafter_logits > future_token_score.unsqueeze(-1)).sum(dim=-1).float() + 1.0

    prefix_matches = torch.cumprod(token_top1_match, dim=1)
    greedy_prefix_len = prefix_matches.sum(dim=1)
    sequence_match = prefix_matches[:, -1]

    metrics: dict[str, torch.Tensor] = {
        "hidden.rel_mse": hidden_rel_mse.reshape(-1),
        "hidden.rel_rmse": hidden_rel_rmse.reshape(-1),
        "hidden.cosine_similarity": cosine.reshape(-1),
        "latent.mse": latent_mse.reshape(-1),
        "logit.ce": logit_ce.reshape(-1),
        "logit.js_div_to_teacher": logit_js_div.reshape(-1),
        "logit.teacher_prob": drafter_token_prob.reshape(-1),
        "logit.prob_ratio_vs_teacher": prob_ratio.reshape(-1),
        "token.top1_match": token_top1_match.reshape(-1),
        "token.top5_contains": topk_match(5).reshape(-1),
        "token.top10_contains": topk_match(10).reshape(-1),
        "token.teacher_rank": teacher_rank.reshape(-1),
        "token.sequence_match": sequence_match,
        "accept.greedy_prefix_len": greedy_prefix_len,
    }
    for slot in range(future_tokens.shape[1]):
        suffix = f"@{slot + 1}"
        metrics[f"token.top1_match{suffix}"] = token_top1_match[:, slot]
        metrics[f"token.top5_contains{suffix}"] = topk_match(5)[:, slot]
        metrics[f"token.top10_contains{suffix}"] = topk_match(10)[:, slot]
        metrics[f"token.teacher_rank{suffix}"] = teacher_rank[:, slot]
        metrics[f"logit.teacher_prob{suffix}"] = drafter_token_prob[:, slot]
        metrics[f"logit.prob_ratio_vs_teacher{suffix}"] = prob_ratio[:, slot]
        metrics[f"accept.rate{suffix}"] = prefix_matches[:, slot]
    return metrics


def _slice_row_prompt(dataset: Any, row_idx: int, end: int) -> torch.Tensor:
    if hasattr(dataset, "row_offsets") and hasattr(dataset, "input_ids"):
        offset = int(dataset.row_offsets[row_idx].item())
        length = int(dataset.row_lengths[row_idx].item())
        row_input_ids = dataset.input_ids[offset : offset + length]
        return row_input_ids[:end].long().unsqueeze(0)

    row_tensors = getattr(dataset, "row_tensors", None)
    if row_tensors is not None:
        input_ids, _ = row_tensors[row_idx]
        return input_ids[:end].long().unsqueeze(0)

    hf_dataset = getattr(dataset, "dataset", None)
    if hf_dataset is not None:
        row = hf_dataset[row_idx]
        return torch.as_tensor(row["input_ids"], dtype=torch.long)[:end].unsqueeze(0)

    raise TypeError("speedup measurement requires a TeacherWindowDataset or FlowWindowCacheDataset")


def collect_speedup_prompts(dataset: Any, *, num_prompts: int, warmup_prompts: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    total = max(0, num_prompts) + max(0, warmup_prompts)
    prompts: list[torch.Tensor] = []
    seen: set[tuple[int, int]] = set()
    index = 0
    max_attempts = max(total * 20, total + 100)
    while len(prompts) < total and index < max_attempts:
        row_idx, t = dataset._sample_row_and_t(index)
        key = (int(row_idx), int(t))
        index += 1
        if key in seen:
            continue
        seen.add(key)
        prompt = _slice_row_prompt(dataset, int(row_idx), int(t) + 1)
        if prompt.numel() > 0:
            prompts.append(prompt.contiguous())
    if len(prompts) < total:
        raise ValueError(f"could only collect {len(prompts)} speedup prompts; requested {total}")
    return prompts[:warmup_prompts], prompts[warmup_prompts:]


@torch.inference_mode()
def generate_greedy_baseline(frozen_lm, prompt: torch.Tensor, *, max_new_tokens: int) -> dict[str, float]:
    synchronize_if_needed(frozen_lm.device)
    start = time.perf_counter()
    input_ids = prompt.to(frozen_lm.device)
    state, prefill_timings = frozen_lm.prefill(input_ids)
    decode_seconds = 0.0
    generated = 0
    eos_token_id = frozen_lm.eos_token_id
    while generated < max_new_tokens:
        token, next_timings = frozen_lm.next_token(state)
        state, forward_timings = frozen_lm.forward_with_cache(token, state, use_cache=True)
        decode_seconds += next_timings.get("next_token") + forward_timings.get("forward_with_cache")
        generated += 1
        if eos_token_id is not None and int(token.item()) == int(eos_token_id):
            break
    synchronize_if_needed(frozen_lm.device)
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "prefill_seconds": prefill_timings.get("prefill"),
        "decode_seconds": decode_seconds,
        "generated_tokens": float(generated),
        "tokens_per_second": float(generated / seconds) if seconds > 0 else 0.0,
        "decode_tokens_per_second": float(generated / decode_seconds) if decode_seconds > 0 else 0.0,
    }


def _accumulate_flow_speed(
    flow,
    *,
    totals: dict[str, float],
) -> None:
    flow_time = flow.timings.get("total_generation")
    flow_prefill_time = flow.timings.get("prefill")
    totals["seconds"] += flow_time
    totals["decode_seconds"] += max(0.0, flow_time - flow_prefill_time)
    totals["tokens"] += float(flow.generated_token_count)
    for step in flow.step_stats:
        totals["accepted"] += float(step.accepted_len)
        totals["drafted"] += float(step.draft_len)
        totals["steps"] += 1.0


def _flow_speed_summary(
    *,
    prefix: str,
    totals: dict[str, float],
    baseline_tps: float,
    baseline_decode_tps: float,
) -> dict[str, float | int]:
    flow_tps = totals["tokens"] / totals["seconds"] if totals["seconds"] > 0 else 0.0
    flow_decode_tps = totals["tokens"] / totals["decode_seconds"] if totals["decode_seconds"] > 0 else 0.0
    mean_accept = totals["accepted"] / totals["steps"] if totals["steps"] > 0 else 0.0
    mean_draft = totals["drafted"] / totals["steps"] if totals["steps"] > 0 else 0.0
    return {
        f"{prefix}real": flow_tps / baseline_tps if baseline_tps > 0 else 0.0,
        f"{prefix}decode_only": flow_decode_tps / baseline_decode_tps if baseline_decode_tps > 0 else 0.0,
        f"{prefix}acceptance_proxy": 1.0 + mean_accept,
        f"{prefix}flow_tokens_per_second": flow_tps,
        f"{prefix}flow_decode_tokens_per_second": flow_decode_tps,
        f"{prefix}flow_seconds": totals["seconds"],
        f"{prefix}flow_decode_seconds": totals["decode_seconds"],
        f"{prefix}flow_generated_tokens": int(totals["tokens"]),
        f"{prefix}mean_accept_len": mean_accept,
        f"{prefix}mean_draft_len": mean_draft,
        f"{prefix}draft_steps": int(totals["steps"]),
    }


@torch.inference_mode()
def measure_generation_speedup(
    args: ChunkedFlowEvalArguments,
    *,
    module: SingleExpertFlowTrainingModule,
    frozen_lm,
    dataloader: DataLoader,
) -> dict[str, Any]:
    if args.speedup_num_prompts < 1:
        raise ValueError("speedup_num_prompts must be >= 1 when measure_speedup is enabled")
    if args.speedup_max_new_tokens < 1:
        raise ValueError("speedup_max_new_tokens must be >= 1 when measure_speedup is enabled")

    warmup_prompts, prompts = collect_speedup_prompts(
        dataloader.dataset,
        num_prompts=args.speedup_num_prompts,
        warmup_prompts=args.speedup_warmup_prompts,
    )
    draft_len = module.drafter.config.draft_length
    print(
        f"measuring real flow speedup: prompts={len(prompts)} warmup={len(warmup_prompts)} "
        f"max_new_tokens={args.speedup_max_new_tokens} draft_len={draft_len}",
        flush=True,
    )

    for prompt in warmup_prompts:
        generate_greedy_baseline(frozen_lm, prompt, max_new_tokens=args.speedup_max_new_tokens)
        generate_with_drafter(
            ChainedFlowContext(frozen_lm),
            module.drafter,
            prompt,
            max_new_tokens=args.speedup_max_new_tokens,
            draft_len=draft_len,
        )
        if args.measure_speedup_lower_bound:
            generate_with_drafter(
                ChainedFlowContext(frozen_lm),
                module.drafter,
                prompt,
                max_new_tokens=args.speedup_max_new_tokens,
                draft_len=draft_len,
                force_zero_accept=True,
            )

    baseline_seconds = 0.0
    baseline_decode_seconds = 0.0
    baseline_tokens = 0.0
    flow_totals = {"seconds": 0.0, "decode_seconds": 0.0, "tokens": 0.0, "accepted": 0.0, "drafted": 0.0, "steps": 0.0}
    lower_bound_totals = {"seconds": 0.0, "decode_seconds": 0.0, "tokens": 0.0, "accepted": 0.0, "drafted": 0.0, "steps": 0.0}
    iterator = prompts
    if tqdm is not None:
        iterator = tqdm(prompts, total=len(prompts), desc="measuring speedup", unit="prompt")
    for prompt in iterator:
        baseline = generate_greedy_baseline(frozen_lm, prompt, max_new_tokens=args.speedup_max_new_tokens)
        baseline_seconds += baseline["seconds"]
        baseline_decode_seconds += baseline["decode_seconds"]
        baseline_tokens += baseline["generated_tokens"]

        flow = generate_with_drafter(
            ChainedFlowContext(frozen_lm),
            module.drafter,
            prompt,
            max_new_tokens=args.speedup_max_new_tokens,
            draft_len=draft_len,
        )
        _accumulate_flow_speed(flow, totals=flow_totals)

        if args.measure_speedup_lower_bound:
            lower_bound_flow = generate_with_drafter(
                ChainedFlowContext(frozen_lm),
                module.drafter,
                prompt,
                max_new_tokens=args.speedup_max_new_tokens,
                draft_len=draft_len,
                force_zero_accept=True,
            )
            _accumulate_flow_speed(lower_bound_flow, totals=lower_bound_totals)

    baseline_tps = baseline_tokens / baseline_seconds if baseline_seconds > 0 else 0.0
    baseline_decode_tps = baseline_tokens / baseline_decode_seconds if baseline_decode_seconds > 0 else 0.0
    summary = {
        "baseline_tokens_per_second": baseline_tps,
        "baseline_decode_tokens_per_second": baseline_decode_tps,
        "baseline_seconds": baseline_seconds,
        "baseline_decode_seconds": baseline_decode_seconds,
        "baseline_generated_tokens": int(baseline_tokens),
        "num_prompts": len(prompts),
        "warmup_prompts": len(warmup_prompts),
        "max_new_tokens": args.speedup_max_new_tokens,
    }
    summary.update(_flow_speed_summary(prefix="", totals=flow_totals, baseline_tps=baseline_tps, baseline_decode_tps=baseline_decode_tps))
    if args.measure_speedup_lower_bound:
        summary.update(
            _flow_speed_summary(
                prefix="lower_bound_",
                totals=lower_bound_totals,
                baseline_tps=baseline_tps,
                baseline_decode_tps=baseline_decode_tps,
            )
        )
    return summary


@torch.inference_mode()
def evaluate_flow_checkpoint(
    args: ChunkedFlowEvalArguments,
    *,
    flow_dir: str | Path,
    frozen_lm,
    dataloader: DataLoader,
    device: torch.device,
    output_path: str | Path,
) -> dict[str, Any]:
    flow_dir = Path(flow_dir)
    print(f"loading flow checkpoint: {flow_dir}", flush=True)
    module, config = load_flow_training_module(flow_dir, frozen_lm=frozen_lm, device=device)
    print(f"flow checkpoint loaded: device={next(module.parameters()).device}", flush=True)

    metric_values: dict[str, list[torch.Tensor]] = {}
    total_windows = 0
    total_batches = len(dataloader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    generator = torch.Generator(device=device).manual_seed(args.noise_seed)

    iterator = enumerate(dataloader)
    progress = None
    if tqdm is not None:
        progress = tqdm(iterator, total=total_batches, desc=f"evaluating {flow_dir.name}", unit="batch")
        iterator = progress
    for step, batch in iterator:
        if args.max_batches is not None and step >= args.max_batches:
            break
        context_hidden = batch["context_hidden"].to(device)
        target_hidden = batch["target_hidden"].to(device)
        future_tokens = batch["future_tokens"].to(device)

        z_ctx = module.drafter.encode_hidden(context_hidden)
        z_target = module.drafter.encode_hidden(target_hidden)
        z0 = module.drafter.init_latents(z_ctx, generator=generator)
        pred_latent = module.drafter.integrate_latents(z_ctx, z0=z0)
        pred_hidden = module.drafter.decode_latent(pred_latent)
        drafter_logits = module.lm_head(pred_hidden)
        teacher_logits = module.lm_head(target_hidden)

        batch_metrics = per_token_flow_metrics(
            pred_hidden=pred_hidden,
            target_hidden=target_hidden,
            pred_latent=pred_latent,
            target_latent=z_target,
            drafter_logits=drafter_logits,
            teacher_logits=teacher_logits,
            future_tokens=future_tokens,
        )
        batch_size = int(context_hidden.shape[0])
        total_windows += batch_size
        if progress is not None:
            progress.set_postfix(windows=total_windows)
        for name, value in batch_metrics.items():
            metric_values.setdefault(name, []).append(value.detach().cpu())

    metrics = {name: summarize_metric(torch.cat(values, dim=0)) for name, values in metric_values.items()}
    speedup: dict[str, Any] | None = None
    if args.measure_speedup:
        speedup = measure_generation_speedup(
            args,
            module=module,
            frozen_lm=frozen_lm,
            dataloader=dataloader,
        )
        for name, value in speedup.items():
            if isinstance(value, (int, float)):
                metrics[f"speedup.{name}"] = summarize_scalar_metric(float(value))

    result = {
        "flow_dir": str(flow_dir),
        "checkpoint": flow_dir.name,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "model_id": args.model_id,
        "device": str(device),
        "num_windows": total_windows,
        "noise_seed": args.noise_seed,
        "config": config,
        "metrics": metrics,
    }
    if speedup is not None:
        result["speedup"] = speedup
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"flow eval metrics saved: {output_path}", flush=True)
    return result


@torch.inference_mode()
def evaluate_flow(args: ChunkedFlowEvalArguments) -> dict[str, Any] | list[dict[str, Any]]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch_dtype_from_string(args.dtype)
    print(f"loading flow eval LM: {args.model_id} dtype={args.dtype} device={device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    config_dir = find_flow_config_dir(args.flow_dir)
    with (config_dir / "chain_flow_chunked_flow_config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_args = ChunkedFlowModelArguments(**config["model_args"])
    print(f"loading flow eval dataset: {args.dataset_path} split={args.dataset_split}", flush=True)
    dataloader = load_flow_eval_dataloader(
        args,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
    )

    if not args.eval_all_checkpoints:
        output_path = single_eval_output_path(args)
        return evaluate_flow_checkpoint(
            args,
            flow_dir=args.flow_dir,
            frozen_lm=frozen_lm,
            dataloader=dataloader,
            device=device,
            output_path=output_path,
        )

    run_dir = Path(args.flow_dir)
    if run_dir.name.startswith("checkpoint-"):
        run_dir = run_dir.parent
    checkpoints = discover_flow_checkpoints(run_dir)
    selected_checkpoints = select_checkpoint_stride(checkpoints, stride=args.checkpoint_stride)
    print(
        f"found {len(checkpoints)} flow checkpoints in {run_dir}; "
        f"selected {len(selected_checkpoints)} with checkpoint_stride={args.checkpoint_stride}",
        flush=True,
    )

    results = []
    for checkpoint_dir in selected_checkpoints:
        output_path = checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir)
        results.append(
            evaluate_flow_checkpoint(
                args,
                flow_dir=checkpoint_dir,
                frozen_lm=frozen_lm,
                dataloader=dataloader,
                device=device,
                output_path=output_path,
            )
        )
    return results


__all__ = [
    "ChunkedFlowEvalArguments",
    "checkpoint_eval_output_path",
    "dataset_eval_slug",
    "discover_flow_checkpoints",
    "evaluate_flow",
    "evaluate_flow_checkpoint",
    "find_flow_config_dir",
    "load_flow_eval_dataloader",
    "load_flow_training_module",
    "measure_generation_speedup",
    "per_token_flow_metrics",
    "select_checkpoint_stride",
    "single_eval_output_path",
    "summarize_metric",
    "summarize_scalar_metric",
    "torch_dtype_from_string",
]
