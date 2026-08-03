from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, TrainingArguments

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.hidden_mlp import HiddenMLPConfig
from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.losses import DrafterLossConfig
from chained_flow.training.trainer_module import HiddenMLPTrainingModule
from chained_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class HiddenMLPModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 4
    draft_length: int = 4
    hidden_multiplier: int = 2
    vae_dir: str | None = None
    local_files_only: bool = False
    device: str | None = None


@dataclass
class TeacherDataArguments:
    dataset_path: str = "teacher_states/gsm8k-qwen35-08b-smoke"
    windows_per_epoch: int | None = None
    window_seed: int = 0


@dataclass
class LossArguments:
    lambda_mse: float = 1.0
    lambda_cos: float = 0.2
    lambda_norm: float = 0.05
    lambda_latent_mse: float = 0.0
    lambda_latent_cos: float = 0.0
    lambda_ce: float = 0.2
    lambda_expected_accept: float = 0.1
    position_gamma: float = 0.8


class ComponentLoggingTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        component_logs = {
            key.replace("loss_component/", ""): float(value.detach().mean().cpu())
            for key, value in outputs.items()
            if key.startswith("loss_component/")
        }
        if component_logs and self.state.global_step % max(1, self.args.logging_steps) == 0:
            self.log(component_logs)
        return (loss, outputs) if return_outputs else loss


def loss_config_from_args(args: LossArguments) -> DrafterLossConfig:
    return DrafterLossConfig(
        lambda_mse=args.lambda_mse,
        lambda_cos=args.lambda_cos,
        lambda_norm=args.lambda_norm,
        lambda_latent_mse=args.lambda_latent_mse,
        lambda_latent_cos=args.lambda_latent_cos,
        lambda_ce=args.lambda_ce,
        lambda_expected_accept=args.lambda_expected_accept,
        position_gamma=args.position_gamma,
    )


def train_hidden_mlp_with_trainer(
    model_args: HiddenMLPModelArguments,
    data_args: TeacherDataArguments,
    loss_args: LossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    context = ChainedFlowContext.from_pretrained(
        model_args.model_id,
        device=model_args.device,
        local_files_only=model_args.local_files_only,
    )
    frozen_lm = context.frozen_lm

    dataset = TeacherWindowDataset.from_disk(
        data_args.dataset_path,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch,
        seed=data_args.window_seed,
    )

    model = HiddenMLPTrainingModule(
        frozen_lm,
        HiddenMLPConfig(
            context_size=model_args.context_size,
            draft_length=model_args.draft_length,
            hidden_multiplier=model_args.hidden_multiplier,
            vae_dir=model_args.vae_dir,
        ),
        loss_config_from_args(loss_args),
    )
    trainer = ComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_teacher_windows,
    )
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_train_config.json").open("w", encoding="utf-8") as f:
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
