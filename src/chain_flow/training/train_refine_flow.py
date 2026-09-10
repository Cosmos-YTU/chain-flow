"""Training for the token-conditioned bidirectional flow refiner (V3). Additive: reuses the
existing Trainer, collator, dataset, and loss/data argument dataclasses.
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

from chain_flow.context import ChainedFlowContext
from chain_flow.drafters.refine_flow import RefineFlowConfig, RefineFlowDrafter
from chain_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chain_flow.training.collators import collate_teacher_windows
from chain_flow.training.train_chunked_flow import (
    ComponentLoggingTrainer,
    FlowLossArguments,
    TeacherDataArguments,
)
from chain_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class RefineModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 8
    draft_length: int = 4
    expert_dim: int = 1024
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_refine_steps: int = 4
    init_mode: str = "delta"
    noise_scale: float = 1.0
    self_cond_prob: float = 0.5
    drafter_dropout: float = 0.0
    architecture: str = "refine_flow"
    local_files_only: bool = False
    device: str | None = None


def refine_config_from_args(args: RefineModelArguments) -> RefineFlowConfig:
    return RefineFlowConfig(
        context_size=args.context_size,
        draft_length=args.draft_length,
        expert_dim=args.expert_dim,
        num_heads=args.num_heads,
        ffn_multiplier=args.ffn_multiplier,
        num_drafter_layers=args.num_drafter_layers,
        num_refine_steps=args.num_refine_steps,
        init_mode=args.init_mode,
        noise_scale=args.noise_scale,
        self_cond_prob=args.self_cond_prob,
        drafter_dropout=args.drafter_dropout,
        architecture=args.architecture,
    )


class RefineTrainingModule(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, drafter_config: RefineFlowConfig, loss_config: FlowLossArguments):
        super().__init__()
        self.drafter = RefineFlowDrafter(frozen_lm, drafter_config)
        self.loss_config = loss_config
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        if bias is None:
            self.lm_head_bias = None
        else:
            self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.lm_head_weight.dtype)
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def _relative_hidden_mse(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)
        target_power = target.float().pow(2).mean(dim=-1).clamp_min(self.loss_config.eps)
        return (mse.float() / target_power).mean()

    def _hidden_cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1, pred.shape[-1])
        target = target.reshape(-1, target.shape[-1])
        return (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()

    def _token_ce_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        b, l, v = logits.shape
        return F.cross_entropy(logits.reshape(b * l, v), future_tokens.reshape(b * l))

    def _expected_acceptance_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        token_probs = probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
        prefix = torch.cumprod(token_probs.clamp_min(self.loss_config.eps), dim=1)
        weights = self.loss_config.gamma ** torch.arange(logits.shape[1], device=logits.device, dtype=prefix.dtype)
        return -(prefix * weights.unsqueeze(0)).sum(dim=1).mean()

    def forward(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if target_hidden.shape[1] != self.drafter.config.draft_length:
            raise ValueError(
                f"target_hidden must have draft length {self.drafter.config.draft_length}, got {target_hidden.shape[1]}"
            )
        out = self.drafter.forward_teacher(context_hidden, target_hidden, future_tokens)
        v_pred, v_star, pred_hidden = out["v_pred"], out["v_star"], out["pred_hidden"]
        logits = self.lm_head(pred_hidden)

        cfg = self.loss_config
        components = {
            "flow.mse": F.mse_loss(v_pred, v_star),
            "hidden.rel_mse": self._relative_hidden_mse(pred_hidden, target_hidden),
            "hidden.cos": self._hidden_cosine_loss(pred_hidden, target_hidden),
            "logit.ce": self._token_ce_loss(logits, future_tokens),
            "verifier.expected_accept": self._expected_acceptance_loss(logits, future_tokens),
        }
        total = (
            cfg.lambda_flow * components["flow.mse"]
            + cfg.lambda_hidden * components["hidden.rel_mse"]
            + cfg.lambda_cos * components["hidden.cos"]
            + cfg.lambda_ce * components["logit.ce"]
            + cfg.lambda_accept * components["verifier.expected_accept"]
        )
        output: dict[str, torch.Tensor] = {"loss": total, "pred_hidden": pred_hidden}
        for name, value in components.items():
            output[f"loss_component/{name}"] = value.detach()
        return output


def train_refine_with_trainer(
    model_args: RefineModelArguments,
    data_args: TeacherDataArguments,
    loss_args: FlowLossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(f"loading refine backbone: model_id={model_args.model_id} device={model_args.device}", flush=True)
    context = ChainedFlowContext.from_pretrained(
        model_args.model_id, device=model_args.device, local_files_only=model_args.local_files_only
    )
    frozen_lm = context.frozen_lm
    print(f"refine backbone loaded: device={frozen_lm.device}", flush=True)

    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path,
        split=data_args.dataset_split,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch,
        seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    print(f"refine dataset: windows_per_epoch={len(dataset)} valid_rows={len(dataset.valid_rows)}", flush=True)

    model = RefineTrainingModule(frozen_lm, refine_config_from_args(model_args), loss_args)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"refine model initialized: parameters={total} trainable={trainable} "
        f"refine_steps={model_args.num_refine_steps} init={model_args.init_mode} self_cond={model_args.self_cond_prob}",
        flush=True,
    )

    eval_dataset = dataset if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval else None
    trainer = ComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_teacher_windows,
    )
    print("refine training started", flush=True)
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("refine training finished", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chain_flow_refine_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"model_args": asdict(model_args), "data_args": asdict(data_args), "loss_args": asdict(loss_args)},
            f,
            indent=2,
        )
    return {"output_dir": training_args.output_dir, "global_step": trainer.state.global_step, "metrics": train_result.metrics}


def load_refine_module(flow_dir: str | Path, *, frozen_lm: FrozenLMWrapper, device) -> tuple[RefineTrainingModule, dict[str, Any]]:
    flow_dir = Path(flow_dir)
    config_path = flow_dir / "chain_flow_refine_config.json"
    if not config_path.exists():
        config_path = flow_dir.parent / "chain_flow_refine_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    model_args = RefineModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = RefineTrainingModule(frozen_lm, refine_config_from_args(model_args), loss_args)

    safetensors_path = flow_dir / "model.safetensors"
    bin_path = flow_dir / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    elif bin_path.exists():
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing refine weights in {flow_dir}")
    missing, _ = module.load_state_dict(state_dict, strict=False)
    trained_missing = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("token_embedding_weight") or m.endswith("lm_head_bias"))]
    if trained_missing:
        raise RuntimeError(f"missing trained params: {trained_missing}")
    module.to(device)
    module.eval()
    return module, config


__all__ = [
    "RefineModelArguments",
    "RefineTrainingModule",
    "refine_config_from_args",
    "train_refine_with_trainer",
    "load_refine_module",
]
