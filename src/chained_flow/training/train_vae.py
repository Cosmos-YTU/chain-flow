from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from chained_flow.training.vae_dataset import TeacherHiddenSequenceDataset, TeacherHiddenTokenDataset, collate_hidden_tokens
from chained_flow.training.vae_losses import HiddenVAELossConfig, compute_hidden_vae_loss
from chained_flow.vae import HiddenVAEConfig, build_hidden_vae


@dataclass
class VAEModelArguments:
    vae_type: str = "residual_mlp"
    hidden_size: int = 1024
    latent_size: int = 256
    intermediate_size: int = 512
    num_layers: int = 2
    num_heads: int = 4
    max_sequence_length: int = 128
    dropout: float = 0.0
    device: str | None = None


@dataclass
class VAEDataArguments:
    dataset_path: str = "teacher_states/gsm8k-qwen35-08b-smoke"
    dataset_split: str = "train"
    tokens_per_epoch: int | None = None
    token_seed: int = 0
    response_only: bool = True
    validation_fraction: float = 0.1
    split_seed: int = 0
    sequence_length: int = 1


@dataclass
class VAELossArguments:
    lambda_mse: float = 1.0
    lambda_cos: float = 0.2
    lambda_norm: float = 0.05
    beta: float = 1e-4
    free_bits: float = 0.0


class HiddenVAETrainingModule(nn.Module):
    def __init__(
        self,
        model_args: VAEModelArguments,
        loss_args: VAELossArguments,
    ):
        super().__init__()
        self.vae = build_hidden_vae(
            model_args.vae_type,
            HiddenVAEConfig(
                hidden_size=model_args.hidden_size,
                latent_size=model_args.latent_size,
                intermediate_size=model_args.intermediate_size,
                num_layers=model_args.num_layers,
                num_heads=model_args.num_heads,
                max_sequence_length=model_args.max_sequence_length,
                dropout=model_args.dropout,
            ),
        )
        self.loss_config = HiddenVAELossConfig(
            lambda_mse=float(loss_args.lambda_mse),
            lambda_cos=float(loss_args.lambda_cos),
            lambda_norm=float(loss_args.lambda_norm),
            beta=float(loss_args.beta),
            free_bits=float(loss_args.free_bits),
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.vae(hidden)
        loss_output = compute_hidden_vae_loss(
            output.recon_hidden,
            hidden,
            mu=output.mu,
            logvar=output.logvar,
            config=self.loss_config,
        )
        result: dict[str, torch.Tensor] = {
            "loss": loss_output.total,
            "recon_hidden": output.recon_hidden,
            "z": output.z,
        }
        for name, value in loss_output.components.items():
            result[f"loss_component/{name}"] = value.detach()
        return result


class VAEComponentLoggingTrainer(Trainer):
    def _component_logs(self, outputs, *, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}{key.replace('loss_component/', '')}": float(value.detach().mean().cpu())
            for key, value in outputs.items()
            if key.startswith("loss_component/")
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        component_logs = self._component_logs(outputs)
        if component_logs and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log(component_logs)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            outputs = model(**inputs)
            loss = outputs["loss"]
        component_logs = self._component_logs(outputs, prefix="eval_")
        if component_logs:
            self.log(component_logs)
        return loss.detach(), None, None


def configure_epoch_eval(training_args: TrainingArguments) -> TrainingArguments:
    training_args.eval_strategy = IntervalStrategy.EPOCH
    training_args.save_strategy = SaveStrategy.EPOCH
    training_args.do_eval = True
    return training_args


def train_vae_with_trainer(
    model_args: VAEModelArguments,
    data_args: VAEDataArguments,
    loss_args: VAELossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(
        f"loading VAE dataset: {data_args.dataset_path} split={data_args.dataset_split}",
        flush=True,
    )
    if data_args.sequence_length > 1:
        dataset = TeacherHiddenSequenceDataset.from_path(
            data_args.dataset_path,
            split=data_args.dataset_split,
            sequence_length=data_args.sequence_length,
            windows_per_epoch=data_args.tokens_per_epoch,
            seed=data_args.token_seed,
            response_only=data_args.response_only,
        )
        print(
            f"sequence VAE dataset initialized: windows_per_epoch={len(dataset)} "
            f"rows={len(dataset.dataset)} sequence_length={data_args.sequence_length} "
            f"response_only={data_args.response_only}",
            flush=True,
        )
    else:
        dataset = TeacherHiddenTokenDataset.from_path(
            data_args.dataset_path,
            split=data_args.dataset_split,
            tokens_per_epoch=data_args.tokens_per_epoch,
            seed=data_args.token_seed,
            response_only=data_args.response_only,
        )
        print(
            f"VAE dataset initialized: tokens_per_epoch={len(dataset)} "
            f"rows={len(dataset.dataset)} response_only={data_args.response_only}",
            flush=True,
        )
    train_dataset, eval_dataset = dataset.train_val_split(
        val_fraction=data_args.validation_fraction,
        seed=data_args.split_seed,
    )
    print(
        f"VAE validation split initialized: train_items={len(train_dataset)} "
        f"val_items={len(eval_dataset)} val_fraction={data_args.validation_fraction}",
        flush=True,
    )
    print(
        f"initializing VAE model: type={model_args.vae_type} "
        f"hidden_size={model_args.hidden_size} latent_size={model_args.latent_size} "
        f"intermediate_size={model_args.intermediate_size} num_layers={model_args.num_layers} "
        f"num_heads={model_args.num_heads} max_sequence_length={model_args.max_sequence_length}",
        flush=True,
    )
    model = HiddenVAETrainingModule(model_args, loss_args)
    if model_args.device:
        model = model.to(torch.device(model_args.device))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"VAE model initialized: parameters={parameter_count}", flush=True)
    model_device = next(model.parameters()).device
    print(f"VAE model device: {model_device}", flush=True)
    print(f"initializing VAE trainer: output_dir={training_args.output_dir}", flush=True)
    configure_epoch_eval(training_args)
    print(
        f"VAE trainer strategies: eval_strategy={training_args.eval_strategy} "
        f"save_strategy={training_args.save_strategy} do_eval={training_args.do_eval} "
        f"train_batch_size={training_args.per_device_train_batch_size} "
        f"eval_batch_size={training_args.per_device_eval_batch_size}",
        flush=True,
    )
    trainer = VAEComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_hidden_tokens,
    )
    print("VAE trainer initialized", flush=True)
    print("VAE training started", flush=True)
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("VAE training finished", flush=True)
    print(f"saving VAE model: {training_args.output_dir}", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print(f"VAE training state saved: {training_args.output_dir}", flush=True)

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_vae_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_args": asdict(model_args),
                "data_args": asdict(data_args),
                "loss_args": asdict(loss_args),
            },
            f,
            indent=2,
        )
    return {
        "output_dir": training_args.output_dir,
        "global_step": trainer.state.global_step,
        "metrics": train_result.metrics,
    }
