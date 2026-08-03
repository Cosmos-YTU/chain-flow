"""Training for the embedding-space flow matcher (V5). Additive; reuses the existing Trainer,
collator, dataset, and loss/data dataclasses. Works on single-layer (context_feature_dim=1024)
or multi-layer (4096) caches.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import TrainingArguments

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.embed_flow import EmbedFlowConfig, EmbedFlowDrafter
from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.train_chunked_flow import (
    ComponentLoggingTrainer,
    FlowLossArguments,
    TeacherDataArguments,
)
from chained_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class EmbedModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 8
    draft_length: int = 4
    expert_dim: int = 1024
    context_feature_dim: int = 1024
    num_heads: int = 8
    ffn_multiplier: int = 4
    num_drafter_layers: int = 8
    num_flow_steps: int = 4
    init_mode: str = "noise"
    noise_scale: float = 0.6
    drafter_dropout: float = 0.0
    architecture: str = "embed_flow"
    local_files_only: bool = False
    device: str | None = None


def embed_config_from_args(args: EmbedModelArguments) -> EmbedFlowConfig:
    return EmbedFlowConfig(
        context_size=args.context_size,
        draft_length=args.draft_length,
        expert_dim=args.expert_dim,
        context_feature_dim=args.context_feature_dim,
        num_heads=args.num_heads,
        ffn_multiplier=args.ffn_multiplier,
        num_drafter_layers=args.num_drafter_layers,
        num_flow_steps=args.num_flow_steps,
        init_mode=args.init_mode,
        noise_scale=args.noise_scale,
        drafter_dropout=args.drafter_dropout,
        architecture=args.architecture,
    )


class EmbedTrainingModule(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, drafter_config: EmbedFlowConfig, loss_config: FlowLossArguments):
        super().__init__()
        self.drafter = EmbedFlowDrafter(frozen_lm, drafter_config)
        self.loss_config = loss_config
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        if bias is None:
            self.lm_head_bias = None
        else:
            self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=self.lm_head_weight.dtype)
        logits = torch.matmul(x, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def _pos_weights(self, k: int, device, dtype) -> torch.Tensor:
        # exp(-pos/pos_decay) (DSpark-style, gentle) if pos_decay>0, else uniform.
        pd = self.loss_config.pos_decay
        if pd and pd > 0:
            return torch.exp(-torch.arange(k, device=device, dtype=dtype) / float(pd))
        return torch.ones(k, device=device, dtype=dtype)

    def _token_ce(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        b, k, v = logits.shape
        return F.cross_entropy(logits.reshape(b * k, v), future_tokens.reshape(b * k))

    def _distribution_l1(self, draft_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        # DSpark dominant term: L1 between drafter and teacher softmax distributions, position-weighted.
        draft = F.softmax(draft_logits.float(), dim=-1)
        teacher = F.softmax(teacher_logits.float(), dim=-1)
        l1 = (draft - teacher).abs().sum(dim=-1)  # [B, K]
        w = self._pos_weights(l1.shape[1], l1.device, l1.dtype)
        return (l1 * w.unsqueeze(0)).sum(dim=1).mean() / w.sum()

    def _expected_accept(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        tp = probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
        prefix = torch.cumprod(tp.clamp_min(self.loss_config.eps), dim=1)
        w = self.loss_config.gamma ** torch.arange(logits.shape[1], device=logits.device, dtype=prefix.dtype)
        return -(prefix * w.unsqueeze(0)).sum(dim=1).mean()

    def forward(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,  # unused (kept for collator/dataset compatibility)
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if future_tokens.shape[1] != self.drafter.config.draft_length:
            raise ValueError(
                f"future_tokens must have draft length {self.drafter.config.draft_length}, got {future_tokens.shape[1]}"
            )
        out = self.drafter.forward_teacher(context_hidden, future_tokens)
        logits = self.lm_head(out["pred_embed"])
        cfg = self.loss_config
        components = {
            "flow.mse": F.mse_loss(out["v_pred"], out["v_star"]),
            "embed.mse": F.mse_loss(out["pred_embed"], out["e_target"]),
            "logit.ce": self._token_ce(logits, future_tokens),
            "verifier.expected_accept": self._expected_accept(logits, future_tokens),
        }
        # Flow-matching trajectory term is always kept; this stays a flow matcher.
        total = cfg.lambda_flow * components["flow.mse"] + cfg.lambda_ce * components["logit.ce"]

        if cfg.lambda_dist > 0.0:
            # DSpark-style distribution endpoint loss. Requires target_hidden to be the FINAL
            # layer (single-layer cache) so lm_head(target_hidden) is the true teacher distribution.
            if target_hidden.shape[-1] != self.lm_head_weight.shape[-1]:
                raise ValueError(
                    "lambda_dist>0 requires the single-layer cache (target_hidden must be final-layer "
                    f"hidden of size {self.lm_head_weight.shape[-1]}, got {target_hidden.shape[-1]})"
                )
            teacher_logits = self.lm_head(target_hidden).detach()
            components["dist.l1"] = self._distribution_l1(logits, teacher_logits)
            total = total + cfg.lambda_dist * components["dist.l1"]
        else:
            # legacy geometry term (default behavior unchanged)
            total = total + cfg.lambda_hidden * components["embed.mse"] \
                          + cfg.lambda_accept * components["verifier.expected_accept"]

        output: dict[str, torch.Tensor] = {"loss": total, "pred_embed": out["pred_embed"]}
        for name, value in components.items():
            output[f"loss_component/{name}"] = value.detach()
        return output


def train_embed_with_trainer(model_args, data_args, loss_args, training_args) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(f"loading embed-flow backbone: model_id={model_args.model_id} device={model_args.device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        model_args.model_id, device=model_args.device, local_files_only=model_args.local_files_only
    )
    frozen_lm = context.frozen_lm
    print(f"embed-flow backbone loaded: device={frozen_lm.device}", flush=True)

    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path, split=data_args.dataset_split,
        context_size=model_args.context_size, draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch, seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    print(f"embed-flow dataset: windows_per_epoch={len(dataset)} valid_rows={len(dataset.valid_rows)}", flush=True)

    model = EmbedTrainingModule(frozen_lm, embed_config_from_args(model_args), loss_args)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"embed-flow model initialized: parameters={total} trainable={trainable} "
        f"context_feature_dim={model_args.context_feature_dim} steps={model_args.num_flow_steps} init={model_args.init_mode}",
        flush=True,
    )

    eval_dataset = None
    if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval:
        # small fixed eval subset so eval stays fast regardless of (uncapped) train size.
        import os
        from torch.utils.data import Subset

        n_eval = int(os.environ.get("EMBED_EVAL_SAMPLES", "8192"))
        n_eval = min(n_eval, len(dataset))
        eval_dataset = Subset(dataset, list(range(n_eval)))
        print(f"embed-flow eval subset: {n_eval} windows", flush=True)
    trainer = ComponentLoggingTrainer(
        model=model, args=training_args, train_dataset=dataset, eval_dataset=eval_dataset,
        data_collator=collate_teacher_windows,
    )
    print("embed-flow training started", flush=True)
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("embed-flow training finished", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_embed_config.json").open("w", encoding="utf-8") as f:
        json.dump({"model_args": asdict(model_args), "data_args": asdict(data_args), "loss_args": asdict(loss_args)}, f, indent=2)
    return {"output_dir": training_args.output_dir, "global_step": trainer.state.global_step, "metrics": train_result.metrics}


def load_embed_module(flow_dir: str | Path, *, frozen_lm: FrozenLMWrapper, device):
    flow_dir = Path(flow_dir)
    config_path = flow_dir / "chained_flow_embed_config.json"
    if not config_path.exists():
        config_path = flow_dir.parent / "chained_flow_embed_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_args = EmbedModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = EmbedTrainingModule(frozen_lm, embed_config_from_args(model_args), loss_args)
    safetensors_path = flow_dir / "model.safetensors"
    bin_path = flow_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing embed-flow weights in {flow_dir}")
    missing, _ = module.load_state_dict(state_dict, strict=False)
    trained_missing = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("token_embedding_weight") or m.endswith("lm_head_bias"))]
    if trained_missing:
        raise RuntimeError(f"missing trained params: {trained_missing}")
    module.to(device)
    module.eval()
    return module, config


__all__ = [
    "EmbedModelArguments", "EmbedTrainingModule", "embed_config_from_args",
    "train_embed_with_trainer", "load_embed_module",
]
