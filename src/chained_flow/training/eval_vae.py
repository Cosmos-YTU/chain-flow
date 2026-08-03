from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset

from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.train_vae import HiddenVAETrainingModule, VAELossArguments, VAEModelArguments
from chained_flow.training.vae_dataset import HiddenTokenTensorDataset, TeacherHiddenTokenDataset, collate_hidden_tokens

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is normally present through datasets/transformers.
    tqdm = None


@dataclass
class VAEEvalArguments:
    vae_dir: str
    dataset_path: str
    dataset_split: str = "train"
    model_id: str = DEFAULT_MODEL_ID
    device: str | None = None
    dtype: str = "float16"
    batch_size: int = 32
    max_batches: int | None = None
    response_only: bool = True
    local_files_only: bool = False
    output_path: str | None = None
    eval_all_checkpoints: bool = False
    checkpoint_stride: int = 1


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


def logit_kl_divergence(
    real_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if real_logits.shape != recon_logits.shape:
        raise ValueError(
            f"real_logits and recon_logits must have the same shape, got "
            f"{tuple(real_logits.shape)} and {tuple(recon_logits.shape)}"
        )
    real_probs = F.softmax(real_logits.float() / temperature, dim=-1)
    recon_log_probs = F.log_softmax(recon_logits.float() / temperature, dim=-1)
    return F.kl_div(recon_log_probs, real_probs, reduction="batchmean") * (temperature**2)


def per_token_logit_kl_divergence(
    real_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    if real_logits.shape != recon_logits.shape:
        raise ValueError(
            f"real_logits and recon_logits must have the same shape, got "
            f"{tuple(real_logits.shape)} and {tuple(recon_logits.shape)}"
        )
    real_probs = F.softmax(real_logits.float() / temperature, dim=-1)
    recon_log_probs = F.log_softmax(recon_logits.float() / temperature, dim=-1)
    real_log_probs = real_probs.clamp_min(1e-12).log()
    return (real_probs * (real_log_probs - recon_log_probs)).sum(dim=-1) * (temperature**2)


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


def find_vae_config_dir(vae_dir: str | Path) -> Path:
    vae_dir = Path(vae_dir)
    for candidate in (vae_dir, vae_dir.parent):
        if (candidate / "chained_flow_vae_config.json").exists():
            return candidate
    raise FileNotFoundError(f"missing VAE config in {vae_dir} or {vae_dir.parent}")


def load_vae_training_module(vae_dir: str | Path, *, device: torch.device) -> HiddenVAETrainingModule:
    vae_dir = Path(vae_dir)
    config_dir = find_vae_config_dir(vae_dir)
    config_path = config_dir / "chained_flow_vae_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    model_args = VAEModelArguments(**config["model_args"])
    loss_args = VAELossArguments(**config["loss_args"])
    module = HiddenVAETrainingModule(model_args, loss_args)

    safetensors_path = vae_dir / "model.safetensors"
    bin_path = vae_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing VAE weights in {vae_dir}")

    module.load_state_dict(state_dict)
    module.to(device)
    module.eval()
    return module


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.removeprefix("checkpoint-"))
    except ValueError:
        return -1


def discover_vae_checkpoints(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    checkpoints = [
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and path.name.startswith("checkpoint-")
        and ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())
    ]
    checkpoints.sort(key=lambda path: (_checkpoint_step(path), path.name))
    if not checkpoints:
        raise FileNotFoundError(f"no VAE checkpoints found in {run_dir}")
    return checkpoints


def select_checkpoint_stride(checkpoints: list[Path], *, stride: int = 1) -> list[Path]:
    if stride < 1:
        raise ValueError("checkpoint_stride must be >= 1")
    return checkpoints[::stride]


def dataset_eval_slug(dataset_path: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_path).strip("._-")
    return slug or "dataset"


def checkpoint_eval_output_path(args: VAEEvalArguments, checkpoint_dir: str | Path, *, run_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    run_dir = Path(run_dir)
    checkpoint_name = checkpoint_dir.name
    if checkpoint_name == run_dir.name:
        checkpoint_name = "final"
    dataset_slug = dataset_eval_slug(args.dataset_path)

    if args.output_path is None:
        return run_dir / f"vae_eval_metrics_{dataset_slug}_{checkpoint_name}.json"

    output_path = Path(args.output_path)
    if output_path.suffix == ".json":
        return output_path.with_name(f"{output_path.stem}_{dataset_slug}_{checkpoint_name}{output_path.suffix}")
    return output_path / f"vae_eval_metrics_{dataset_slug}_{checkpoint_name}.json"


def single_eval_output_path(args: VAEEvalArguments) -> Path:
    if args.output_path is not None:
        output_path = Path(args.output_path)
        if output_path.suffix == ".json":
            dataset_slug = dataset_eval_slug(args.dataset_path)
            return output_path.with_name(f"{output_path.stem}_{dataset_slug}{output_path.suffix}")
        return output_path / f"vae_eval_metrics_{dataset_eval_slug(args.dataset_path)}.json"
    return Path(args.vae_dir) / f"vae_eval_metrics_{dataset_eval_slug(args.dataset_path)}.json"


def per_token_vae_metrics(
    recon_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    *,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    real_logits: torch.Tensor,
    recon_logits: torch.Tensor,
    free_bits: float = 0.0,
) -> dict[str, torch.Tensor]:
    if recon_hidden.shape != target_hidden.shape:
        raise ValueError(
            f"recon_hidden and target_hidden must have identical shape, got "
            f"{tuple(recon_hidden.shape)} and {tuple(target_hidden.shape)}"
        )
    if real_logits.shape != recon_logits.shape:
        raise ValueError(
            f"real_logits and recon_logits must have identical shape, got "
            f"{tuple(real_logits.shape)} and {tuple(recon_logits.shape)}"
        )

    eps = 1e-12
    hidden_mse = F.mse_loss(recon_hidden, target_hidden, reduction="none").mean(dim=-1)
    target_power = target_hidden.float().pow(2).mean(dim=-1).clamp_min(eps)
    hidden_rel_mse = hidden_mse.float() / target_power
    hidden_rel_rmse = hidden_mse.float().sqrt() / target_power.sqrt()

    recon_norm = recon_hidden.norm(dim=-1)
    target_norm = target_hidden.norm(dim=-1)
    both_zero = (recon_norm == 0) & (target_norm == 0)
    cosine = F.cosine_similarity(recon_hidden, target_hidden, dim=-1)
    cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
    hidden_cos = 1.0 - cosine
    hidden_norm = (recon_norm - target_norm).pow(2)

    latent_kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0.0:
        latent_kl = latent_kl.clamp_min(free_bits)
    latent_kl = latent_kl.sum(dim=-1)

    real_probs = F.softmax(real_logits.float(), dim=-1)
    recon_probs = F.softmax(recon_logits.float(), dim=-1)
    real_log_probs = real_probs.clamp_min(eps).log()
    recon_log_probs = recon_probs.clamp_min(eps).log()
    mixture_probs = (0.5 * (real_probs + recon_probs)).clamp_min(eps)
    mixture_log_probs = mixture_probs.log()

    logit_kl = (real_probs * (real_log_probs - recon_log_probs)).sum(dim=-1)
    logit_js_div = 0.5 * (real_probs * (real_log_probs - mixture_log_probs)).sum(dim=-1)
    logit_js_div = logit_js_div + 0.5 * (recon_probs * (recon_log_probs - mixture_log_probs)).sum(dim=-1)

    real_top1 = real_logits.argmax(dim=-1)
    recon_top1 = recon_logits.argmax(dim=-1)
    token_top1_match = (real_top1 == recon_top1).float()

    def topk_match(k: int) -> torch.Tensor:
        topk = min(k, recon_logits.shape[-1])
        recon_topk = recon_logits.topk(topk, dim=-1).indices
        return (recon_topk == real_top1.unsqueeze(-1)).any(dim=-1).float()

    recon_score_on_real_top1 = recon_logits.gather(dim=-1, index=real_top1.unsqueeze(-1)).squeeze(-1)
    token_real_top1_rank_in_recon = (recon_logits > recon_score_on_real_top1.unsqueeze(-1)).sum(dim=-1).float() + 1.0
    real_top1_prob = real_probs.gather(dim=-1, index=real_top1.unsqueeze(-1)).squeeze(-1)
    recon_prob_on_real_top1 = recon_probs.gather(dim=-1, index=real_top1.unsqueeze(-1)).squeeze(-1)
    prob_ratio_on_real_top1 = recon_prob_on_real_top1 / real_top1_prob.clamp_min(eps)
    logit_ce_delta = -recon_prob_on_real_top1.clamp_min(eps).log() + real_top1_prob.clamp_min(eps).log()

    return {
        "hidden.mse": hidden_mse,
        "hidden.rel_mse": hidden_rel_mse,
        "hidden.rel_rmse": hidden_rel_rmse,
        "hidden.cos": hidden_cos,
        "hidden.cosine_similarity": cosine,
        "hidden.norm": hidden_norm,
        "latent.kl": latent_kl,
        "logit.kl": logit_kl,
        "logit.js_div": logit_js_div,
        "logit.ce_delta": logit_ce_delta,
        "token.match": token_top1_match,
        "token.top1_match": token_top1_match,
        "token.top5_match": topk_match(5),
        "token.top10_match": topk_match(10),
        "token.real_top1_rank_in_recon": token_real_top1_rank_in_recon,
        "token.recon_prob_on_real_top1": recon_prob_on_real_top1,
        "token.prob_ratio_on_real_top1": prob_ratio_on_real_top1,
    }



class FlowCacheHiddenTokenDataset(TorchDataset):
    def __init__(self, hidden_tokens: torch.Tensor):
        if hidden_tokens.ndim != 2:
            raise ValueError("hidden_tokens must have shape [tokens, hidden_size]")
        self.hidden_tokens = hidden_tokens.contiguous()

    def __len__(self) -> int:
        return int(self.hidden_tokens.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"hidden": self.hidden_tokens[index].float()}


def load_flow_cache_hidden_tokens(path: str | Path, *, response_only: bool = True) -> torch.Tensor:
    from chained_flow.training.window_dataset import FLOW_CACHE_FILES, is_flow_window_cache

    cache_dir = Path(path)
    if not is_flow_window_cache(cache_dir):
        raise ValueError(f"not a flow window cache: {cache_dir}")
    hidden = torch.load(cache_dir / FLOW_CACHE_FILES["hidden"], map_location="cpu")
    row_offsets = torch.load(cache_dir / FLOW_CACHE_FILES["row_offsets"], map_location="cpu").long()
    row_lengths = torch.load(cache_dir / FLOW_CACHE_FILES["row_lengths"], map_location="cpu").long()
    prompt_lengths = torch.load(cache_dir / FLOW_CACHE_FILES["prompt_lengths"], map_location="cpu").long()

    chunks: list[torch.Tensor] = []
    for row_idx in range(int(row_offsets.numel())):
        offset = int(row_offsets[row_idx].item())
        length = int(row_lengths[row_idx].item())
        start = int(prompt_lengths[row_idx].item()) if response_only else 0
        if length > start:
            chunks.append(hidden[offset + start : offset + length].float())
    if not chunks:
        raise ValueError(f"flow cache contains no hidden tokens: {cache_dir}")
    return torch.cat(chunks, dim=0).contiguous()

def load_vae_eval_dataloader(args: VAEEvalArguments) -> DataLoader:
    from chained_flow.training.window_dataset import is_flow_window_cache

    if is_flow_window_cache(args.dataset_path):
        hidden_tokens = load_flow_cache_hidden_tokens(args.dataset_path, response_only=args.response_only)
        eval_dataset = FlowCacheHiddenTokenDataset(hidden_tokens)
        source = "flow_cache"
    else:
        token_dataset = TeacherHiddenTokenDataset.from_path(
            args.dataset_path,
            split=args.dataset_split,
            response_only=args.response_only,
        )
        eval_dataset = HiddenTokenTensorDataset(token_dataset.hidden_tokens, sample=False)
        source = "teacher_dataset"
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_hidden_tokens)
    print(
        f"VAE eval dataset loaded: source={source} tokens={len(eval_dataset)} response_only={args.response_only}",
        flush=True,
    )
    return dataloader


def load_eval_lm_head(args: VAEEvalArguments, *, device: torch.device) -> torch.nn.Module:
    dtype = torch_dtype_from_string(args.dtype)
    print(f"loading LM head: {args.model_id} dtype={args.dtype} device={device}", flush=True)
    wrapper, _ = FrozenLMWrapper.from_pretrained(
        args.model_id,
        device=device,
        dtype=dtype,
        local_files_only=args.local_files_only,
    )
    lm_head = wrapper.model.lm_head.eval()
    print(f"LM head loaded: device={next(lm_head.parameters()).device}", flush=True)
    return lm_head


@torch.inference_mode()
def evaluate_vae_checkpoint(
    args: VAEEvalArguments,
    *,
    vae_dir: str | Path,
    dataloader: DataLoader,
    lm_head: torch.nn.Module,
    device: torch.device,
    output_path: str | Path,
) -> dict[str, Any]:
    vae_dir = Path(vae_dir)
    print(f"loading VAE checkpoint: {vae_dir}", flush=True)
    vae_module = load_vae_training_module(vae_dir, device=device)
    print(f"VAE checkpoint loaded: device={next(vae_module.parameters()).device}", flush=True)
    lm_head_dtype = next(lm_head.parameters()).dtype

    metric_values: dict[str, list[torch.Tensor]] = {}
    total_examples = 0
    total_batches = len(dataloader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    iterator = enumerate(dataloader)
    progress = None
    if tqdm is not None:
        progress = tqdm(iterator, total=total_batches, desc=f"evaluating {vae_dir.name}", unit="batch")
        iterator = progress
    for step, batch in iterator:
        if args.max_batches is not None and step >= args.max_batches:
            break
        hidden = batch["hidden"].to(device)
        output = vae_module.vae(hidden)
        real_logits = lm_head(hidden.to(dtype=lm_head_dtype))
        recon_logits = lm_head(output.recon_hidden.to(dtype=lm_head_dtype))
        batch_metrics = per_token_vae_metrics(
            output.recon_hidden,
            hidden,
            mu=output.mu,
            logvar=output.logvar,
            real_logits=real_logits,
            recon_logits=recon_logits,
            free_bits=vae_module.loss_config.free_bits,
        )
        batch_size = int(hidden.shape[0])
        total_examples += batch_size
        if progress is not None:
            progress.set_postfix(tokens=total_examples)
        for name, value in batch_metrics.items():
            metric_values.setdefault(name, []).append(value.detach().cpu())

    metrics = {name: summarize_metric(torch.cat(values, dim=0)) for name, values in metric_values.items()}
    result = {
        "vae_dir": str(vae_dir),
        "checkpoint": vae_dir.name,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "model_id": args.model_id,
        "device": str(device),
        "num_tokens": total_examples,
        "metrics": metrics,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"VAE eval metrics saved: {output_path}", flush=True)
    return result


@torch.inference_mode()
def evaluate_vae(args: VAEEvalArguments) -> dict[str, Any] | list[dict[str, Any]]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"loading VAE eval dataset: {args.dataset_path} split={args.dataset_split}", flush=True)
    dataloader = load_vae_eval_dataloader(args)
    lm_head = load_eval_lm_head(args, device=device)

    if not args.eval_all_checkpoints:
        output_path = single_eval_output_path(args)
        return evaluate_vae_checkpoint(
            args,
            vae_dir=args.vae_dir,
            dataloader=dataloader,
            lm_head=lm_head,
            device=device,
            output_path=output_path,
        )

    run_dir = Path(args.vae_dir)
    if run_dir.name.startswith("checkpoint-"):
        run_dir = run_dir.parent
    checkpoints = discover_vae_checkpoints(run_dir)
    selected_checkpoints = select_checkpoint_stride(checkpoints, stride=args.checkpoint_stride)
    print(
        f"found {len(checkpoints)} VAE checkpoints in {run_dir}; "
        f"selected {len(selected_checkpoints)} with checkpoint_stride={args.checkpoint_stride}",
        flush=True,
    )

    results = []
    for checkpoint_dir in selected_checkpoints:
        output_path = checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir)
        results.append(
            evaluate_vae_checkpoint(
                args,
                vae_dir=checkpoint_dir,
                dataloader=dataloader,
                lm_head=lm_head,
                device=device,
                output_path=output_path,
            )
        )
    return results


__all__ = [
    "VAEEvalArguments",
    "checkpoint_eval_output_path",
    "dataset_eval_slug",
    "discover_vae_checkpoints",
    "evaluate_vae",
    "evaluate_vae_checkpoint",
    "find_vae_config_dir",
    "load_eval_lm_head",
    "load_vae_eval_dataloader",
    "load_flow_cache_hidden_tokens",
    "load_vae_training_module",
    "single_eval_output_path",
    "select_checkpoint_stride",
    "logit_kl_divergence",
    "per_token_logit_kl_divergence",
    "per_token_vae_metrics",
    "summarize_metric",
    "torch_dtype_from_string",
]
