"""Training for the Markov-v2 (order-2, hidden-gated) flow drafter (V7). Additive; reuses Trainer,
collator, dataset, loss/data dataclasses. Single-layer cache (context_feature_dim == hidden).
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
from chain_flow.drafters.markov2_flow import Markov2FlowConfig, Markov2FlowDrafter
from chain_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chain_flow.training.collators import collate_teacher_windows
from chain_flow.training.train_chunked_flow import ComponentLoggingTrainer, FlowLossArguments, TeacherDataArguments
from chain_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class Markov2ModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4
    expert_dim: int = 1024
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_flow_steps: int = 2
    init_mode: str = "delta"
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    noise_scale: float = 1.0
    markov_rank: int = 512
    markov_order: int = 2
    markov_hidden_gate: bool = True
    architecture: str = "markov2_flow"
    local_files_only: bool = False
    device: str | None = None


def markov2_config_from_args(a: Markov2ModelArguments) -> Markov2FlowConfig:
    return Markov2FlowConfig(
        context_size=a.context_size, draft_length=a.draft_length, chunk_size=a.chunk_size,
        expert_dim=a.expert_dim, num_heads=a.num_heads, ffn_multiplier=a.ffn_multiplier,
        num_drafter_layers=a.num_drafter_layers, num_flow_steps=a.num_flow_steps,
        init_mode=a.init_mode, detach_previous_chunks=a.detach_previous_chunks,
        drafter_dropout=a.drafter_dropout, noise_scale=a.noise_scale, markov_rank=a.markov_rank, markov_order=a.markov_order, markov_hidden_gate=a.markov_hidden_gate,
        architecture=a.architecture,
    )


class Markov2TrainingModule(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, dcfg: Markov2FlowConfig, loss_cfg: FlowLossArguments):
        super().__init__()
        self.drafter = Markov2FlowDrafter(frozen_lm, dcfg)
        self.loss_config = loss_cfg
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        self.lm_head_bias = None if bias is None else self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.to(dtype=self.lm_head_weight.dtype)
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def _rel_mse(self, p, t):
        mse = F.mse_loss(p, t, reduction="none").mean(dim=-1)
        return (mse.float() / t.float().pow(2).mean(dim=-1).clamp_min(self.loss_config.eps)).mean()

    def _ce(self, logits, tok):
        b, k, v = logits.shape
        return F.cross_entropy(logits.reshape(b * k, v), tok.reshape(b * k))

    def _accept(self, logits, tok):
        probs = F.softmax(logits, dim=-1)
        tp = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        prefix = torch.cumprod(tp.clamp_min(self.loss_config.eps), dim=1)
        w = self.loss_config.gamma ** torch.arange(logits.shape[1], device=logits.device, dtype=prefix.dtype)
        return -(prefix * w.unsqueeze(0)).sum(dim=1).mean()

    def forward(self, context_hidden, target_hidden, future_tokens) -> dict[str, torch.Tensor]:
        if target_hidden.shape[1] != self.drafter.config.draft_length:
            raise ValueError(f"target_hidden draft length mismatch: {target_hidden.shape[1]}")
        out = self.drafter.forward_teacher(context_hidden, target_hidden, future_tokens)
        cfg = self.loss_config
        comp = {
            "flow.mse": F.mse_loss(out["v_pred"], out["v_star"]),
            "hidden.rel_mse": self._rel_mse(out["pred_hidden"], target_hidden.to(out["pred_hidden"].dtype)),
            # CE on the markov-corrected logits trains both flow head + markov head jointly
            "logit.ce": self._ce(out["markov_logits"], future_tokens),
            # also keep base-logit CE so the flow head stays good on its own (position 0 has no markov)
            "logit.ce_base": self._ce(out["base_logits"], future_tokens),
            "verifier.expected_accept": self._accept(out["markov_logits"], future_tokens),
        }
        total = (cfg.lambda_flow * comp["flow.mse"] + cfg.lambda_hidden * comp["hidden.rel_mse"]
                 + cfg.lambda_ce * comp["logit.ce"] + 0.2 * cfg.lambda_ce * comp["logit.ce_base"]
                 + cfg.lambda_accept * comp["verifier.expected_accept"])
        output = {"loss": total, "pred_hidden": out["pred_hidden"]}
        for n, v in comp.items():
            output[f"loss_component/{n}"] = v.detach()
        return output


def train_markov2_with_trainer(model_args, data_args, loss_args, training_args) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(f"loading markov-flow backbone: {model_args.model_id} device={model_args.device}", flush=True)
    ctx = ChainedFlowContext.from_pretrained(model_args.model_id, device=model_args.device, local_files_only=model_args.local_files_only)
    frozen_lm = ctx.frozen_lm
    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path, split=data_args.dataset_split,
        context_size=model_args.context_size, draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch, seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    print(f"markov-flow dataset windows={len(dataset)} valid_rows={len(dataset.valid_rows)}", flush=True)
    model = Markov2TrainingModule(frozen_lm, markov2_config_from_args(model_args), loss_args)
    tot = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"markov-flow params={tot} trainable={tr} markov_rank={model_args.markov_rank}", flush=True)
    eval_ds = None
    if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval:
        from torch.utils.data import Subset
        eval_ds = Subset(dataset, list(range(min(8192, len(dataset)))))
    trainer = ComponentLoggingTrainer(model=model, args=training_args, train_dataset=dataset,
                                      eval_dataset=eval_ds, data_collator=collate_teacher_windows)
    print("markov-flow training started", flush=True)
    res = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("markov-flow training finished", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", res.metrics); trainer.save_metrics("train", res.metrics); trainer.save_state()
    out = Path(training_args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "chain_flow_markov2_config.json").open("w") as f:
        json.dump({"model_args": asdict(model_args), "data_args": asdict(data_args), "loss_args": asdict(loss_args)}, f, indent=2)
    return {"output_dir": training_args.output_dir, "global_step": trainer.state.global_step, "metrics": res.metrics}


def load_markov2_module(flow_dir, *, frozen_lm, device):
    flow_dir = Path(flow_dir)
    cfgp = flow_dir / "chain_flow_markov2_config.json"
    if not cfgp.exists():
        cfgp = flow_dir.parent / "chain_flow_markov2_config.json"
    config = json.load(open(cfgp))
    model_args = Markov2ModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = Markov2TrainingModule(frozen_lm, markov2_config_from_args(model_args), loss_args)
    sp = flow_dir / "model.safetensors"; bp = flow_dir / "pytorch_model.bin"
    if sp.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sp), device="cpu")
    elif bp.exists():
        sd = torch.load(bp, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing markov weights in {flow_dir}")
    missing, _ = module.load_state_dict(sd, strict=False)
    tm = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("lm_head_bias"))]
    if tm:
        raise RuntimeError(f"missing trained params: {tm}")
    module.to(device); module.eval()
    return module, config


__all__ = ["Markov2ModelArguments", "Markov2TrainingModule", "markov2_config_from_args",
           "train_markov2_with_trainer", "load_markov2_module"]
