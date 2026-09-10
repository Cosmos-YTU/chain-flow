"""Training for the two-pass causal token-conditioned flow drafter (V8). Additive; reuses the
existing Trainer, collator, dataset, loss/data dataclasses. Single-layer cache.
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
from chain_flow.drafters.twopass_flow import TwoPassFlowConfig, TwoPassFlowDrafter
from chain_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chain_flow.training.collators import collate_teacher_windows
from chain_flow.training.train_chunked_flow import ComponentLoggingTrainer, FlowLossArguments, TeacherDataArguments
from chain_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class TwoPassModelArguments:
    model_id: str = DEFAULT_MODEL_ID
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4
    expert_dim: int = 1024
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_refine_layers: int = 4
    num_flow_steps: int = 2
    num_refine_steps: int = 1
    init_mode: str = "delta"
    noise_scale: float = 0.0
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    architecture: str = "twopass_flow"
    local_files_only: bool = False
    device: str | None = None


def twopass_config_from_args(a: TwoPassModelArguments) -> TwoPassFlowConfig:
    return TwoPassFlowConfig(
        context_size=a.context_size, draft_length=a.draft_length, chunk_size=a.chunk_size,
        expert_dim=a.expert_dim, num_heads=a.num_heads, ffn_multiplier=a.ffn_multiplier,
        num_drafter_layers=a.num_drafter_layers, num_refine_layers=a.num_refine_layers,
        num_flow_steps=a.num_flow_steps, num_refine_steps=a.num_refine_steps,
        init_mode=a.init_mode, noise_scale=a.noise_scale,
        detach_previous_chunks=a.detach_previous_chunks, drafter_dropout=a.drafter_dropout,
        architecture=a.architecture,
    )


class TwoPassTrainingModule(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, dcfg: TwoPassFlowConfig, loss_cfg: FlowLossArguments):
        super().__init__()
        self.drafter = TwoPassFlowDrafter(frozen_lm, dcfg)
        self.loss_config = loss_cfg
        lm = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm.weight.detach().clone(), persistent=False)
        bias = getattr(lm, "bias", None)
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
        th = target_hidden.to(out["h2"].dtype)
        cfg = self.loss_config
        comp = {
            "flow.mse": F.mse_loss(out["v1_pred"], out["v1_star"]),          # pass-1 velocity
            "flow.mse2": F.mse_loss(out["v2_pred"], out["v2_star"]),         # pass-2 velocity
            "hidden.rel_mse": self._rel_mse(out["h2"], th),                  # refined hidden
            "logit.ce": self._ce(out["logits2"], future_tokens),            # final (refined) CE
            "logit.ce1": self._ce(out["logits1"], future_tokens),           # keep pass-1 strong
            "verifier.expected_accept": self._accept(out["logits2"], future_tokens),
        }
        total = (cfg.lambda_flow * (comp["flow.mse"] + comp["flow.mse2"])
                 + cfg.lambda_hidden * comp["hidden.rel_mse"]
                 + cfg.lambda_ce * comp["logit.ce"] + 0.3 * cfg.lambda_ce * comp["logit.ce1"]
                 + cfg.lambda_accept * comp["verifier.expected_accept"])
        output = {"loss": total, "pred_hidden": out["h2"]}
        for n, v in comp.items():
            output[f"loss_component/{n}"] = v.detach()
        return output


def train_twopass_with_trainer(model_args, data_args, loss_args, training_args) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(f"loading twopass backbone: {model_args.model_id} device={model_args.device}", flush=True)
    ctx = ChainedFlowContext.from_pretrained(model_args.model_id, device=model_args.device, local_files_only=model_args.local_files_only)
    frozen_lm = ctx.frozen_lm
    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path, split=data_args.dataset_split,
        context_size=model_args.context_size, draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch, seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    print(f"twopass dataset windows={len(dataset)} valid_rows={len(dataset.valid_rows)}", flush=True)
    model = TwoPassTrainingModule(frozen_lm, twopass_config_from_args(model_args), loss_args)
    tot = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"twopass params={tot} trainable={tr} refine_layers={model_args.num_refine_layers} noise={model_args.noise_scale}", flush=True)
    eval_ds = None
    if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval:
        from torch.utils.data import Subset
        eval_ds = Subset(dataset, list(range(min(8192, len(dataset)))))
    trainer = ComponentLoggingTrainer(model=model, args=training_args, train_dataset=dataset,
                                      eval_dataset=eval_ds, data_collator=collate_teacher_windows)
    print("twopass training started", flush=True)
    res = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("twopass training finished", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", res.metrics); trainer.save_metrics("train", res.metrics); trainer.save_state()
    out = Path(training_args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "chain_flow_twopass_config.json").open("w") as f:
        json.dump({"model_args": asdict(model_args), "data_args": asdict(data_args), "loss_args": asdict(loss_args)}, f, indent=2)
    return {"output_dir": training_args.output_dir, "global_step": trainer.state.global_step, "metrics": res.metrics}


def load_twopass_module(flow_dir, *, frozen_lm, device):
    flow_dir = Path(flow_dir)
    cfgp = flow_dir / "chain_flow_twopass_config.json"
    if not cfgp.exists():
        cfgp = flow_dir.parent / "chain_flow_twopass_config.json"
    config = json.load(open(cfgp))
    model_args = TwoPassModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = TwoPassTrainingModule(frozen_lm, twopass_config_from_args(model_args), loss_args)
    sp = flow_dir / "model.safetensors"; bp = flow_dir / "pytorch_model.bin"
    if sp.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sp), device="cpu")
    elif bp.exists():
        sd = torch.load(bp, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing twopass weights in {flow_dir}")
    missing, _ = module.load_state_dict(sd, strict=False)
    tm = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("lm_head_bias") or m.endswith("token_embedding_weight"))]
    if tm:
        raise RuntimeError(f"missing trained params: {tm}")
    module.to(device); module.eval()
    return module, config


__all__ = ["TwoPassModelArguments", "TwoPassTrainingModule", "twopass_config_from_args",
           "train_twopass_with_trainer", "load_twopass_module"]
