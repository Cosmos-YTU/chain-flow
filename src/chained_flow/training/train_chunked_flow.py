from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments

from chained_flow.context import ChainedFlowContext
from chained_flow.drafters.chunked_flow import SingleExpertFlowConfig, SingleExpertFlowDrafter
from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class ChunkedFlowModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 4
    draft_length: int = 2
    chunk_size: int = 2
    vae_dir: str | None = None
    expert_dim: int = 128
    num_heads: int = 4
    ffn_multiplier: int = 4
    num_flow_steps: int = 1
    noise_scale: float = 1.0
    init_mode: str = "noise"
    train_vae: bool = False
    architecture: str = "independent_experts"
    num_drafter_layers: int = 2
    drafter_dropout: float = 0.0
    detach_previous_chunks: bool = True
    vae_learning_rate_multiplier: float = 0.05
    local_files_only: bool = False
    device: str | None = None


@dataclass
class TeacherDataArguments:
    dataset_path: str = "teacher_states/gsm8k-qwen35-08b-smoke"
    dataset_split: str = "train"
    windows_per_epoch: int | None = None
    window_seed: int = 0
    materialize_rows: bool = True


@dataclass
class FlowLossArguments:
    lambda_flow: float = 1.0
    lambda_latent: float = 0.5
    lambda_hidden: float = 0.5
    lambda_cos: float = 0.1
    lambda_ce: float = 0.2
    lambda_accept: float = 0.1
    gamma: float = 0.8
    eps: float = 1e-8
    # DSpark-style distribution-matching (embed-flow): if lambda_dist>0, add L1 between the
    # drafter's and teacher's softmax token distributions. pos_decay>0 enables exp(-pos/pos_decay)
    # position weighting. Both default off, so existing trainers/configs are unchanged.
    lambda_dist: float = 0.0
    pos_decay: float = 0.0
    # Per-position CE upweighting: weight_i = (i+1)**ce_pos_power, normalized to mean 1.
    # 0.0 = uniform (current/default). >0 emphasizes DEEP draft positions (the acc@3-4 weak spot).
    ce_pos_power: float = 0.0


@dataclass
class FlowLossOutput:
    total: torch.Tensor
    components: dict[str, torch.Tensor]


class SingleExpertFlowTrainingModule(nn.Module):
    def __init__(
        self,
        frozen_lm: FrozenLMWrapper,
        drafter_config: SingleExpertFlowConfig,
        loss_config: FlowLossArguments,
    ):
        super().__init__()
        self.drafter = SingleExpertFlowDrafter(frozen_lm, drafter_config)
        self.loss_config = loss_config
        self.include_vae_in_state_dict = drafter_config.train_vae
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        if bias is None:
            self.lm_head_bias = None
        else:
            self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.include_vae_in_state_dict:
            return state
        return {key: value for key, value in state.items() if not key.startswith("drafter.vae.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        full_state = super().state_dict()
        full_state.update(state_dict)
        return super().load_state_dict(full_state, strict=strict, assign=assign)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.lm_head_weight.dtype)
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def _relative_hidden_mse(self, pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred_hidden, target_hidden, reduction="none").mean(dim=-1)
        target_power = target_hidden.float().pow(2).mean(dim=-1).clamp_min(self.loss_config.eps)
        return (mse.float() / target_power).mean()

    def _hidden_cosine_loss(self, pred_hidden: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        pred = pred_hidden.reshape(-1, pred_hidden.shape[-1])
        target = target_hidden.reshape(-1, target_hidden.shape[-1])
        pred_norm = pred.norm(dim=-1)
        target_norm = target.norm(dim=-1)
        both_zero = (pred_norm == 0) & (target_norm == 0)
        cosine = F.cosine_similarity(pred, target, dim=-1)
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        return (1.0 - cosine).mean()

    def _token_ce_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        batch, length, vocab = logits.shape
        power = getattr(self.loss_config, "ce_pos_power", 0.0)
        if power and power != 0.0:
            ce = F.cross_entropy(
                logits.reshape(batch * length, vocab), future_tokens.reshape(batch * length), reduction="none"
            ).reshape(batch, length)
            pos = torch.arange(length, device=logits.device, dtype=ce.dtype) + 1.0
            w = pos ** power
            w = w / w.mean()  # normalize so overall CE scale is unchanged
            return (ce * w.unsqueeze(0)).mean()
        return F.cross_entropy(logits.reshape(batch * length, vocab), future_tokens.reshape(batch * length))

    def _expected_acceptance_loss(self, logits: torch.Tensor, future_tokens: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        token_probs = probs.gather(dim=-1, index=future_tokens.unsqueeze(-1)).squeeze(-1)
        prefix_probs = torch.cumprod(token_probs.clamp_min(self.loss_config.eps), dim=1)
        weights = self.loss_config.gamma ** torch.arange(logits.shape[1], device=logits.device, dtype=prefix_probs.dtype)
        return -(prefix_probs * weights.unsqueeze(0)).sum(dim=1).mean()

    def compute_flow_loss(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> tuple[FlowLossOutput, torch.Tensor, torch.Tensor]:
        if target_hidden.shape[1] != self.drafter.config.draft_length:
            raise ValueError(
                f"target_hidden must have draft length {self.drafter.config.draft_length}, got {target_hidden.shape[1]}"
            )
        z_ctx = self.drafter.encode_hidden(context_hidden)
        z_target = self.drafter.encode_hidden(target_hidden)
        z0 = self.drafter.init_latents(z_ctx)
        batch = z_target.shape[0]
        tau = torch.rand(batch, 1, 1, device=z_target.device, dtype=z_target.dtype)
        z_tau = (1.0 - tau) * z0 + tau * z_target
        v_star = z_target - z0
        v_pred = self.drafter.flow_velocity(z_tau, tau, z_ctx, previous_latents=z_target)

        components: dict[str, torch.Tensor] = {}
        components["flow.mse"] = F.mse_loss(v_pred, v_star)

        z_pred = self.drafter.integrate_latents(z_ctx, z0=z0)
        components["latent.mse"] = F.mse_loss(z_pred, z_target)

        pred_hidden = self.drafter.decode_latent(z_pred)
        components["hidden.rel_mse"] = self._relative_hidden_mse(pred_hidden, target_hidden)
        components["hidden.cos"] = self._hidden_cosine_loss(pred_hidden, target_hidden)

        logits = self.lm_head(pred_hidden)
        components["logit.ce"] = self._token_ce_loss(logits, future_tokens)
        components["verifier.expected_accept"] = self._expected_acceptance_loss(logits, future_tokens)

        total = pred_hidden.new_zeros(())
        total = total + self.loss_config.lambda_flow * components["flow.mse"]
        total = total + self.loss_config.lambda_latent * components["latent.mse"]
        total = total + self.loss_config.lambda_hidden * components["hidden.rel_mse"]
        total = total + self.loss_config.lambda_cos * components["hidden.cos"]
        total = total + self.loss_config.lambda_ce * components["logit.ce"]
        total = total + self.loss_config.lambda_accept * components["verifier.expected_accept"]
        return FlowLossOutput(total=total, components=components), z_pred, pred_hidden

    def forward(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_output, pred_latent, pred_hidden = self.compute_flow_loss(context_hidden, target_hidden, future_tokens)
        output: dict[str, torch.Tensor] = {
            "loss": loss_output.total,
            "pred_latent": pred_latent,
            "pred_hidden": pred_hidden,
        }
        for name, value in loss_output.components.items():
            output[f"loss_component/{name}"] = value.detach()
        return output


class ComponentLoggingTrainer(Trainer):
    def __init__(self, *args, vae_learning_rate_multiplier: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.vae_learning_rate_multiplier = vae_learning_rate_multiplier

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        trainable = [param for param in self.model.parameters() if param.requires_grad]
        vae_params = [param for name, param in self.model.named_parameters() if param.requires_grad and ".vae." in name]
        vae_param_ids = {id(param) for param in vae_params}
        flow_params = [param for param in trainable if id(param) not in vae_param_ids]
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        groups = []
        if flow_params:
            groups.append({"params": flow_params})
        if vae_params:
            groups.append({"params": vae_params, "lr": self.args.learning_rate * self.vae_learning_rate_multiplier})
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

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


def flow_config_from_args(args: ChunkedFlowModelArguments) -> SingleExpertFlowConfig:
    return SingleExpertFlowConfig(
        context_size=args.context_size,
        draft_length=args.draft_length,
        chunk_size=args.chunk_size,
        vae_dir=args.vae_dir,
        expert_dim=args.expert_dim,
        num_heads=args.num_heads,
        ffn_multiplier=args.ffn_multiplier,
        num_flow_steps=args.num_flow_steps,
        noise_scale=args.noise_scale,
        init_mode=args.init_mode,
        train_vae=args.train_vae,
        architecture=args.architecture,
        num_drafter_layers=args.num_drafter_layers,
        drafter_dropout=args.drafter_dropout,
        detach_previous_chunks=args.detach_previous_chunks,
    )


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def train_chunked_flow_with_trainer(
    model_args: ChunkedFlowModelArguments,
    data_args: TeacherDataArguments,
    loss_args: FlowLossArguments,
    training_args: TrainingArguments,
) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(
        f"loading flow backbone: model_id={model_args.model_id} "
        f"device={model_args.device} local_files_only={model_args.local_files_only}",
        flush=True,
    )
    context = ChainedFlowContext.from_pretrained(
        model_args.model_id,
        device=model_args.device,
        local_files_only=model_args.local_files_only,
    )
    frozen_lm = context.frozen_lm
    print(f"flow backbone loaded: device={frozen_lm.device}", flush=True)

    print(
        f"loading flow dataset: {data_args.dataset_path} split={data_args.dataset_split}",
        flush=True,
    )
    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path,
        split=data_args.dataset_split,
        context_size=model_args.context_size,
        draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch,
        seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    dataset_rows = getattr(dataset, "num_rows", None)
    if dataset_rows is None:
        dataset_rows = len(dataset.dataset)
    print(
        f"flow dataset initialized: windows_per_epoch={len(dataset)} "
        f"available_windows={dataset.available_windows} rows={dataset_rows} "
        f"valid_rows={len(dataset.valid_rows)} window_seed={data_args.window_seed} "
        f"materialize_rows={data_args.materialize_rows}",
        flush=True,
    )

    print(
        f"initializing flow drafter: context_size={model_args.context_size} "
        f"draft_length={model_args.draft_length} chunk_size={model_args.chunk_size} "
        f"expert_dim={model_args.expert_dim} num_heads={model_args.num_heads} "
        f"ffn_multiplier={model_args.ffn_multiplier} num_flow_steps={model_args.num_flow_steps} "
        f"noise_scale={model_args.noise_scale} init_mode={model_args.init_mode} train_vae={model_args.train_vae} "
        f"vae_lr_multiplier={model_args.vae_learning_rate_multiplier} vae_dir={model_args.vae_dir}",
        flush=True,
    )
    model = SingleExpertFlowTrainingModule(
        frozen_lm,
        flow_config_from_args(model_args),
        loss_args,
    )
    total_parameters, trainable_parameters = _parameter_counts(model)
    print(
        f"flow model initialized: parameters={total_parameters} "
        f"trainable_parameters={trainable_parameters}",
        flush=True,
    )
    model_device = next(model.parameters()).device
    print(f"flow model device: {model_device}", flush=True)
    print(f"initializing flow trainer: output_dir={training_args.output_dir}", flush=True)
    print(
        f"flow trainer strategies: eval_strategy={training_args.eval_strategy} "
        f"save_strategy={training_args.save_strategy} do_eval={training_args.do_eval} "
        f"train_batch_size={training_args.per_device_train_batch_size} "
        f"gradient_accumulation_steps={training_args.gradient_accumulation_steps} "
        f"logging_steps={training_args.logging_steps}",
        flush=True,
    )
    eval_dataset = dataset if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval else None
    trainer = ComponentLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_teacher_windows,
        vae_learning_rate_multiplier=model_args.vae_learning_rate_multiplier,
    )
    print("flow trainer initialized", flush=True)
    print("flow training started", flush=True)
    train_result = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("flow training finished", flush=True)
    print(f"saving flow model: {training_args.output_dir}", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print(f"flow training state saved: {training_args.output_dir}", flush=True)

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "chained_flow_chunked_flow_config.json").open("w", encoding="utf-8") as f:
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


__all__ = [
    "ChunkedFlowModelArguments",
    "ComponentLoggingTrainer",
    "FlowLossArguments",
    "FlowLossOutput",
    "_parameter_counts",
    "SingleExpertFlowTrainingModule",
    "TeacherDataArguments",
    "flow_config_from_args",
    "train_chunked_flow_with_trainer",
]
