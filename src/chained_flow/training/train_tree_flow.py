"""Training for the tree-flow drafter (V9). Additive; reuses Trainer, collator, dataset, and the
shared loss/data dataclasses. Single-layer cache (context_feature_dim == hidden).

New vs V6 (markov): a path-conditioned CE (path head + markov head, teacher-forced on the real parent
tokens) plus a top-b COVERAGE loss that trains the drafter FOR the tree — it pushes the true token
inside the top-b at each conditioned position so the draft tree has the right candidate to branch to.
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
from chained_flow.drafters.tree_flow import TreeFlowConfig, TreeFlowDrafter
from chained_flow.frozen_lm import DEFAULT_MODEL_ID, FrozenLMWrapper
from chained_flow.training.collators import collate_teacher_windows
from chained_flow.training.train_chunked_flow import ComponentLoggingTrainer, FlowLossArguments, TeacherDataArguments
from chained_flow.training.window_dataset import TeacherWindowDataset


@dataclass
class TreeModelArguments:
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
    markov_rank: int = 256
    path_order: int = 4
    path_ffn_multiplier: int = 2
    lambda_cov: float = 0.3
    cov_b: int = 8
    cov_margin: float = 1.0
    tree_top_b: int = 8
    tree_max_nodes: int = 48
    architecture: str = "tree_flow"
    local_files_only: bool = False
    device: str | None = None
    # --- VAE latent-flow variant (set vae_dir to enable tree_vae_flow) ---
    vae_dir: str | None = None
    latent_size: int = 640
    vae_type: str = "transformer_hidden"
    vae_intermediate_size: int = 1920
    vae_num_layers: int = 2
    vae_num_heads: int = 8
    vae_max_sequence_length: int = 16
    train_vae: bool = False
    lambda_vae_recon: float = 0.5
    lag_context: bool = False
    lag_extend: bool = False
    anchor_token: bool = False
    sched_sampling_p: float = 0.0
    prev_token_cond: bool = True
    # Warm start: a finished drafter checkpoint (local dir or HF repo id) whose weights -- drafter
    # AND the jointly-trained VAE, which lives in the same model.safetensors -- are loaded into the
    # freshly-built module before training. Not `resume_from_checkpoint`: no optimizer/scheduler
    # state is restored, so this is a fine-tune from step 0 with a new dataset and LR.
    init_from: str | None = None


def tree_config_from_args(a: TreeModelArguments) -> TreeFlowConfig:
    common = dict(
        context_size=a.context_size, draft_length=a.draft_length, chunk_size=a.chunk_size,
        expert_dim=a.expert_dim, num_heads=a.num_heads, ffn_multiplier=a.ffn_multiplier,
        num_drafter_layers=a.num_drafter_layers, num_flow_steps=a.num_flow_steps,
        init_mode=a.init_mode, detach_previous_chunks=a.detach_previous_chunks,
        drafter_dropout=a.drafter_dropout, noise_scale=a.noise_scale, markov_rank=a.markov_rank,
        path_order=a.path_order, path_ffn_multiplier=a.path_ffn_multiplier,
        lambda_cov=a.lambda_cov, cov_b=a.cov_b, cov_margin=a.cov_margin,
        tree_top_b=a.tree_top_b, tree_max_nodes=a.tree_max_nodes, architecture=a.architecture,
    )
    if a.vae_dir:
        from chained_flow.drafters.tree_vae_flow import TreeVAEFlowConfig
        return TreeVAEFlowConfig(
            **common, vae_dir=a.vae_dir, latent_size=a.latent_size, vae_type=a.vae_type,
            vae_intermediate_size=a.vae_intermediate_size, vae_num_layers=a.vae_num_layers,
            vae_num_heads=a.vae_num_heads, vae_max_sequence_length=a.vae_max_sequence_length,
            train_vae=a.train_vae, lambda_vae_recon=a.lambda_vae_recon,
            lag_context=a.lag_context, lag_extend=a.lag_extend, anchor_token=a.anchor_token,
            sched_sampling_p=a.sched_sampling_p, prev_token_cond=a.prev_token_cond,
        )
    return TreeFlowConfig(**common)


class TreeFlowTrainingModule(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, dcfg: TreeFlowConfig, loss_cfg: FlowLossArguments):
        super().__init__()
        from chained_flow.drafters.tree_vae_flow import TreeVAEFlowConfig, TreeVAEFlowDrafter
        if isinstance(dcfg, TreeVAEFlowConfig):
            self.drafter = TreeVAEFlowDrafter(frozen_lm, dcfg)
        else:
            self.drafter = TreeFlowDrafter(frozen_lm, dcfg)
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

    def _tv(self, logits, target_hidden):
        """Total-variation distance to the TARGET's next-token distribution, position-weighted.

        Under rejection sampling the acceptance rate IS the distributional overlap:
            alpha = sum_v min(p_v, q_v) = 1 - d_TV(p, q)
        so minimising d_TV optimises acceptance DIRECTLY, whereas CE/KL only do so indirectly
        (they chase the argmax and the tails respectively). This is the objective DSpark trains on
        ({ce: 0.1, tv: 0.9}); we previously put ~7% of the loss on an accept surrogate and ~40% on
        hidden reconstruction.

        Chunked over the draft dimension: a [B, K, V] teacher tensor at V=248320 would be ~2 GB on
        top of the drafter's own logits, so build one [B, V] position at a time and accumulate.
        Teacher side is no-grad -- it is a fixed reference, not a thing to fit.
        """
        K = logits.shape[1]
        # flatter than the 0.8^i used elsewhere: deep positions are exactly where accept is lost
        pd = float(getattr(self.loss_config, "pos_decay", 0.0)) or 4.0
        w = torch.exp(-torch.arange(K, device=logits.device, dtype=torch.float32) / pd)
        w = w / w.sum()
        acc = logits.new_zeros((), dtype=torch.float32)
        for k in range(K):
            q = F.softmax(logits[:, k].float(), dim=-1)
            with torch.no_grad():
                p = F.softmax(self.lm_head(target_hidden[:, k]).float(), dim=-1)
            acc = acc + w[k] * torch.minimum(p, q).sum(-1).mean()
        return 1.0 - acc                                   # = weighted d_TV

    def _coverage(self, logits, tok):
        """Top-b hinge: push the true token above the (b+1)-th largest logit so it lands in the top-b
        the tree branches over. Zero once the true token is already inside the top-b by the margin."""
        b = self.drafter.config.cov_b
        margin = self.drafter.config.cov_margin
        true_score = logits.gather(-1, tok.unsqueeze(-1)).squeeze(-1).float()   # [B, K]
        thresh = logits.float().topk(b + 1, dim=-1).values[..., b]              # (b+1)-th largest
        return F.relu(thresh + margin - true_score).mean()

    def forward(self, context_hidden, target_hidden, future_tokens,
                lag_token=None, prev_token=None) -> dict[str, torch.Tensor]:
        # ANCHOR: emb(last committed token) conditions the flow itself. The lag dataset already
        # supplies exactly what is needed -- context ending at h(t-1) plus input_ids[t] -- so the
        # model TRAINS UNDER THE LAG rather than being handed h(t), which is what every competing
        # system does (EAGLE cnets.py dataprepare, DFlash randomised anchors, DeepSeek-V3 MTP).
        anchor_token = lag_token if getattr(self.drafter.config, "anchor_token", False) else None
        if float(getattr(self.loss_config, "lambda_dist", 0.0)) > 0.0 \
                and target_hidden.shape[-1] != self.lm_head_weight.shape[-1]:
            raise ValueError(
                "lambda_dist>0 needs the SINGLE-LAYER cache: target_hidden must be the final-layer "
                f"hidden of size {self.lm_head_weight.shape[-1]}, got {target_hidden.shape[-1]} -- "
                "otherwise lm_head(target_hidden) is not the true teacher distribution.")
        if target_hidden.shape[1] != self.drafter.config.draft_length:
            raise ValueError(f"target_hidden draft length mismatch: {target_hidden.shape[1]}")
        context_hidden = self.drafter.build_context(context_hidden, lag_token)
        if not getattr(self.drafter.config, "prev_token_cond", True):
            prev_token = None
        out = self.drafter.forward_teacher(context_hidden, target_hidden, future_tokens, prev_token,
                                           anchor_token=anchor_token)
        cfg = self.loss_config
        th = target_hidden.to(out["pred_hidden"].dtype)
        comp = {
            "flow.mse": F.mse_loss(out["v_pred"], out["v_star"]),
            "hidden.rel_mse": self._rel_mse(out["pred_hidden"], th),
            "hidden.rel_mse_cond": self._rel_mse(out["cond_hidden"], th),   # path-conditioned hidden
            "logit.ce": self._ce(out["cond_logits"], future_tokens),        # path-conditioned (main)
            "logit.ce_base": self._ce(out["base_logits"], future_tokens),   # keep marginal decodable
            "tree.coverage": self._coverage(out["cond_logits"], future_tokens),
            "verifier.expected_accept": self._accept(out["cond_logits"], future_tokens),
        }
        lam_tv = float(getattr(cfg, "lambda_dist", 0.0))
        if lam_tv > 0.0:
            comp["logit.tv"] = self._tv(out["cond_logits"], target_hidden)
        total = (cfg.lambda_flow * comp["flow.mse"]
                 + cfg.lambda_hidden * (comp["hidden.rel_mse"] + comp["hidden.rel_mse_cond"])
                 + cfg.lambda_ce * comp["logit.ce"] + 0.2 * cfg.lambda_ce * comp["logit.ce_base"]
                 + self.drafter.config.lambda_cov * comp["tree.coverage"]
                 + cfg.lambda_accept * comp["verifier.expected_accept"])
        if lam_tv > 0.0:
            total = total + lam_tv * comp["logit.tv"]
        if "recon_hidden" in out:                                       # jointly-trained VAE anchor
            comp["vae.recon"] = self._rel_mse(out["recon_hidden"], th)
            total = total + getattr(self.drafter.config, "lambda_vae_recon", 0.5) * comp["vae.recon"]
        output = {"loss": total, "pred_hidden": out["pred_hidden"]}
        for n, v in comp.items():
            output[f"loss_component/{n}"] = v.detach()
        return output


def warm_start_from(module: nn.Module, init_from: str, *, local_files_only: bool = False) -> str:
    """Load a finished drafter's weights into `module` and PROVE which file they came from.

    Prints the sha256 of the actual `model.safetensors` bytes, not the config string that named
    it: a config can point anywhere, and a stale default that silently trained from scratch is
    exactly the failure this print exists to make impossible. Every parameter that does NOT get
    overwritten is listed, because a silent shape/name drift here is a from-scratch run wearing a
    warm-start config.
    """
    import hashlib

    src = Path(init_from)
    if not (src / "model.safetensors").exists():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(init_from, local_files_only=local_files_only))
    weights = src / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"init_from={init_from!r} resolved to {src}, which has no model.safetensors")

    h = hashlib.sha256()
    with weights.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()

    from safetensors.torch import load_file
    sd = load_file(str(weights), device="cpu")
    missing, unexpected = module.load_state_dict(sd, strict=False)
    # lm_head_* are non-persistent buffers rebuilt from the frozen LM; they are never in a ckpt.
    missing = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("lm_head_bias"))]

    print(f"WARM START from {init_from}", flush=True)
    print(f"  resolved       : {weights}", flush=True)
    print(f"  sha256         : {digest}", flush=True)
    print(f"  tensors loaded : {len(sd) - len(unexpected)} of {len(sd)} in file", flush=True)
    if unexpected:
        print(f"  UNEXPECTED ({len(unexpected)}, ignored): {unexpected[:8]}", flush=True)
    if missing:
        print(f"  NOT INITIALISED ({len(missing)}, random): {missing[:8]}", flush=True)
    else:
        print("  every trainable parameter was initialised from the checkpoint", flush=True)
    if len(sd) and len(unexpected) == len(sd):
        raise RuntimeError(
            f"warm start loaded NOTHING: all {len(sd)} tensors in {weights} were unexpected. "
            f"The checkpoint architecture does not match this config.")
    return digest


def train_tree_with_trainer(model_args, data_args, loss_args, training_args) -> dict[str, Any]:
    torch.manual_seed(training_args.seed)
    print(f"loading tree-flow backbone: {model_args.model_id} device={model_args.device}", flush=True)
    ctx = ChainedFlowContext.from_pretrained(model_args.model_id, device=model_args.device, local_files_only=model_args.local_files_only)
    frozen_lm = ctx.frozen_lm
    dataset = TeacherWindowDataset.from_path(
        data_args.dataset_path, split=data_args.dataset_split,
        context_size=model_args.context_size, draft_length=model_args.draft_length,
        windows_per_epoch=data_args.windows_per_epoch, seed=data_args.window_seed,
        materialize_rows=data_args.materialize_rows,
    )
    # anchoring needs the SAME windows as lag mode: ctx ends at h(t-1) + the committed token
    dataset.lag_context = bool(getattr(model_args, "lag_context", False)) or \
                          bool(getattr(model_args, "anchor_token", False))
    if dataset.lag_context:
        print("LAG MODE: contexts end at h(t-1) + learned slot from emb(token t) -- matches what "
              "the vLLM proposer can actually supply at draft time", flush=True)
    print(f"tree-flow dataset windows={len(dataset)} valid_rows={len(dataset.valid_rows)}", flush=True)
    model = TreeFlowTrainingModule(frozen_lm, tree_config_from_args(model_args), loss_args)
    if getattr(model_args, "init_from", None):
        warm_start_from(model, model_args.init_from, local_files_only=model_args.local_files_only)
    tot = sum(p.numel() for p in model.parameters()); tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"tree-flow params={tot} trainable={tr} path_order={model_args.path_order} "
          f"lambda_cov={model_args.lambda_cov} cov_b={model_args.cov_b}", flush=True)
    eval_ds = None
    if str(training_args.eval_strategy) != "IntervalStrategy.NO" and training_args.do_eval:
        from torch.utils.data import Subset
        eval_ds = Subset(dataset, list(range(min(8192, len(dataset)))))
    trainer = ComponentLoggingTrainer(model=model, args=training_args, train_dataset=dataset,
                                      eval_dataset=eval_ds, data_collator=collate_teacher_windows)
    print("tree-flow training started", flush=True)
    res = trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    print("tree-flow training finished", flush=True)
    trainer.save_model(training_args.output_dir)
    trainer.log_metrics("train", res.metrics); trainer.save_metrics("train", res.metrics); trainer.save_state()
    out = Path(training_args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "chained_flow_tree_config.json").open("w") as f:
        json.dump({"model_args": asdict(model_args), "data_args": asdict(data_args), "loss_args": asdict(loss_args)}, f, indent=2)
    return {"output_dir": training_args.output_dir, "global_step": trainer.state.global_step, "metrics": res.metrics}


def load_tree_module(flow_dir, *, frozen_lm, device):
    flow_dir = Path(flow_dir)
    cfgp = flow_dir / "chained_flow_tree_config.json"
    if not cfgp.exists():
        cfgp = flow_dir.parent / "chained_flow_tree_config.json"
    config = json.load(open(cfgp))
    model_args = TreeModelArguments(**config["model_args"])
    loss_args = FlowLossArguments(**config["loss_args"])
    module = TreeFlowTrainingModule(frozen_lm, tree_config_from_args(model_args), loss_args)
    sp = flow_dir / "model.safetensors"; bp = flow_dir / "pytorch_model.bin"
    if sp.exists():
        from safetensors.torch import load_file
        sd = load_file(str(sp), device="cpu")
    elif bp.exists():
        sd = torch.load(bp, map_location="cpu")
    else:
        raise FileNotFoundError(f"missing tree weights in {flow_dir}")
    missing, _ = module.load_state_dict(sd, strict=False)
    tm = [m for m in missing if not (m.endswith("lm_head_weight") or m.endswith("lm_head_bias"))]
    if tm:
        raise RuntimeError(f"missing trained params: {tm}")
    module.to(device); module.eval()
    return module, config


__all__ = ["TreeModelArguments", "TreeFlowTrainingModule", "tree_config_from_args", "warm_start_from",
           "train_tree_with_trainer", "load_tree_module"]
