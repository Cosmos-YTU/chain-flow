"""Tree-native VAE flow drafter.

Merges the two proven pieces:
  * TreeFlowDrafter (tree_flow.py): the path-conditioned tree machinery — build_tree_fast,
    PathHead residual, MarkovHead bias, lm_head — all operating in the base HIDDEN space.
  * The VAE latent flow (a frozen HiddenVAE): shrink base_hidden -> latent, run the flow-matching
    expert in the SMALL latent space (expert_dim = latent_size), decode back to hidden.

Why: the flow (~80% of the draft cost) scales with expert_dim^2. Running it at latent 640 instead
of hidden 2560 is ~16x cheaper AND base-decoupled. The frozen VAE reconstructs top-1 at ~0.948
(held-out), so it imposes ~no extra accept cap.

Only predict_hidden / forward_teacher change (encode -> latent flow -> decode). init_latents,
integrate, flow_velocity are dim-agnostic and reused; the tree machinery is unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from chain_flow.drafters.chunked_flow import HiddenKVFlowExpert
from chain_flow.drafters.tree_flow import MarkovHead, PathHead, TreeFlowConfig, TreeFlowDrafter
from chain_flow.frozen_lm import FrozenLMWrapper
from chain_flow.vae.base import HiddenVAEConfig
from chain_flow.vae.registry import build_hidden_vae


@dataclass
class TreeVAEFlowConfig(TreeFlowConfig):
    vae_dir: str | None = None            # trained HiddenVAE checkpoint dir
    latent_size: int = 640                # the flow's working dim (expert_dim is forced to this)
    vae_type: str = "transformer_hidden"
    vae_intermediate_size: int = 1920
    vae_num_layers: int = 2
    vae_num_heads: int = 8
    vae_max_sequence_length: int = 16
    train_vae: bool = False               # jointly fine-tune the VAE end-to-end for accept (not frozen)
    lambda_vae_recon: float = 0.5         # autoencoder anchor weight when train_vae
    lag_context: bool = False             # train for the ONE-STEP-LAGGED context vLLM actually gives
    lag_extend: bool = False              # append the synthetic slot instead of evicting h[t-C]
    anchor_token: bool = False            # feed emb(committed token) INTO every flow block
    sched_sampling_p: float = 0.0         # mix own argmax into the path/markov conditioning
    prev_token_cond: bool = True          # give position 0 its REAL parent (the committed token)


def _load_frozen_vae(cfg: TreeVAEFlowConfig, hidden_size: int) -> nn.Module:
    vae_cfg = HiddenVAEConfig(
        hidden_size=hidden_size, latent_size=cfg.latent_size,
        intermediate_size=cfg.vae_intermediate_size, num_layers=cfg.vae_num_layers,
        num_heads=cfg.vae_num_heads, max_sequence_length=cfg.vae_max_sequence_length,
    )
    vae = build_hidden_vae(cfg.vae_type, vae_cfg)
    if cfg.vae_dir:
        import os
        from safetensors.torch import load_file

        # CF_VAE_DIR overrides the config. `vae_dir` is written into every checkpoint as a
        # MACHINE-ABSOLUTE path at training time (e.g. /home/shadeform/chain-flow/out/vae/...),
        # and the published repos carry it verbatim, so the first person to clone this on another
        # box -- or run it in a container with a different root -- has no other way to point it
        # somewhere real. The `vae/` directory shipped inside a drafter repo is for humans; nothing
        # here reads it.
        override = os.environ.get("CF_VAE_DIR")
        d = Path(override or cfg.vae_dir)
        sp = d / "model.safetensors"
        if not sp.exists():
            cks = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            if not cks:
                # This used to be a bare IndexError on cks[-1], which says nothing about which
                # path was tried or why. Name both.
                src = "CF_VAE_DIR" if override else f"vae_dir in the checkpoint config"
                raise FileNotFoundError(
                    f"VAE not found for the drafter.\n"
                    f"  looked in : {d}   (from {src})\n"
                    f"  expected  : {d/'model.safetensors'}, or a checkpoint-N/model.safetensors under it\n"
                    f"  exists    : {'yes, but neither of the above is in it' if d.is_dir() else 'no -- the directory itself is missing'}\n"
                    f"  fix       : set CF_VAE_DIR to a directory holding the VAE for this drafter."
                )
            sp = cks[-1] / "model.safetensors"
        sd = load_file(str(sp))
        vsd = {k[len("vae."):]: v for k, v in sd.items() if k.startswith("vae.")}
        vae.load_state_dict(vsd or sd, strict=True)
    return vae


class TreeVAEFlowDrafter(TreeFlowDrafter):
    """TreeFlowDrafter with the flow moved into a frozen VAE's latent space."""

    def __init__(self, frozen_lm: FrozenLMWrapper, config: TreeVAEFlowConfig):
        nn.Module.__init__(self)                      # bypass TreeFlowDrafter's expert_dim==hidden check
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        self.latent_size = config.latent_size
        self.num_chunks = config.draft_length // config.chunk_size

        # VAE (encode: hidden->latent, decode: latent->hidden). Frozen by default; when
        # train_vae is set, it fine-tunes jointly with the flow for the ACCEPT objective
        # (a reconstruction anchor in the loss keeps the latent decodable).
        self.vae = _load_frozen_vae(config, self.hidden_size)
        self.train_vae = bool(getattr(config, "train_vae", False))
        for p in self.vae.parameters():
            p.requires_grad = self.train_vae
        self.vae.train(self.train_vae)

        # flow experts operate in LATENT space (expert_dim == latent_size)
        def make_expert():
            return HiddenKVFlowExpert(
                hidden_size=self.latent_size, context_size=config.context_size,
                draft_length=config.draft_length, chunk_size=config.chunk_size,
                expert_dim=self.latent_size, num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier, num_layers=config.num_drafter_layers,
                dropout=config.drafter_dropout,
                anchor=bool(getattr(config, "anchor_token", False)),
            )
        self.expert = make_expert()
        self.extra_experts = nn.ModuleList([make_expert() for _ in range(self.num_chunks - 1)])

        # tree machinery stays in HIDDEN space (unchanged from TreeFlowDrafter)
        vocab = frozen_lm.model.config.vocab_size
        self.markov = MarkovHead(vocab, config.markov_rank)
        # ANCHOR: emb(token) lives in TARGET hidden space; the flow works in expert_dim (== latent_size
        # for the VAE variant), so project into the flow's working dim before conditioning.
        self.anchor_in = (nn.Linear(self.hidden_size, config.expert_dim)
                          if getattr(config, "anchor_token", False) else None)
        self.path_head = PathHead(self.hidden_size, config.path_order, config.path_ffn_multiplier,
                                  dropout=config.drafter_dropout)
        self._dtype = next(self.expert.parameters()).dtype
        # Lag mode: at propose time vLLM has h(t-1) and the committed token t, but NOT h(t) -- it
        # never runs a forward pass with t as input in that step. Predicting from h(t-1) instead of
        # h(t) costs ~0.68 accept (scripts/diff_plugin_vs_harness.py). So learn the missing slot from
        # the token embedding, the way EAGLE conditions on hidden + next-token embedding.
        # EAGLE's formulation: h_hat(t) = f([h(t-1) ; emb(t)]). BOTH inputs matter -- a projection of
        # the token embedding ALONE is a token-generic vector (identical in every context), which is
        # why the first lag run was a wash: it helped only where the token id alone predicts what
        # follows (math/code) and hurt elsewhere.
        self.lag_proj = (nn.Linear(2 * self.hidden_size, self.hidden_size)
                         if config.lag_context else None)

    def build_context(self, context_hidden: torch.Tensor,
                      lag_token: torch.Tensor | None = None) -> torch.Tensor:
        """Non-lag: context already ends at h(t). Lag: context ends at h(t-1), and vLLM cannot give us
        h(t) -- so estimate it by fusing the last real hidden with the committed token's embedding,
        residually (h(t) ~= h(t-1) + delta, so the residual form starts from a sensible point)."""
        if self.lag_proj is None or lag_token is None:
            return context_hidden
        ctx = context_hidden.to(self._dtype)
        h_prev = ctx[:, -1]                                        # h(t-1): the other half of the input
        emb = self._embed(lag_token).to(self._dtype)
        slot = (h_prev + self.lag_proj(torch.cat([h_prev, emb], dim=-1))).unsqueeze(1)
        if self.config.lag_extend:
            # Keep every real hidden and ADD the slot (ctx_size+1). Evicting h[t-C] to make room cost
            # more than the slot bought on free-form text (writing -0.32, summ -0.25) while still
            # winning on structured text (math +0.29) -- so stop trading a real hidden away.
            return torch.cat([ctx, slot], dim=1)
        return torch.cat([ctx[:, 1:], slot], dim=1)

    # ---- VAE bridge ------------------------------------------------------
    def _encode(self, hidden: torch.Tensor) -> torch.Tensor:
        vdt = next(self.vae.parameters()).dtype
        return self.vae.encode_sequence(hidden.to(vdt)).mu.to(self._dtype)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        vdt = next(self.vae.parameters()).dtype
        return self.vae.decode_sequence(z.to(vdt)).to(self._dtype)

    # ---- flow in latent space --------------------------------------------
    def predict_hidden(self, context: torch.Tensor, anchor=None) -> torch.Tensor:
        ctx_lat = self._encode(context.to(self._dtype))          # [B, ctx, latent]
        z = self.integrate(ctx_lat, self.init_latents(ctx_lat), anchor=anchor)
        return self._decode(z)                                   # [B, K, hidden]

    def forward_teacher(self, context: torch.Tensor, target_hidden: torch.Tensor,
                        future_tokens: torch.Tensor, prev_token: torch.Tensor | None = None,
                        anchor_token: torch.Tensor | None = None, *, fused_head: bool = False):
        anchor = self.anchor_embed(anchor_token)
        ctx_lat = self._encode(context.to(self._dtype))
        tgt_lat = self._encode(target_hidden.to(self._dtype))
        z0 = self.init_latents(ctx_lat)
        b = z0.shape[0]
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=self._dtype)
        z_tau = (1.0 - tau) * z0 + tau * tgt_lat
        v_star = tgt_lat - z0                                    # flow matching in LATENT space
        v_pred = self.flow_velocity(z_tau, tau, ctx_lat, previous=tgt_lat, anchor=anchor)
        pred_hidden = self._decode(self.integrate(ctx_lat, z0, anchor=anchor))  # decoded marginal hidden
        # FUSED-HEAD FAST PATH. `base_logits`/`cond_logits` are [B, K, 248320]; at 27B that is
        # 254 MB apiece in bf16, and the markov bias is a third one. The three losses that consume
        # them only need per-row reductions (logsumexp, the true-token logit, the (b+1)-th largest),
        # so when the loss module asks for it we hand back the HIDDENS and the markov factor and
        # let it reduce in chunks -- see training/fused_head.py.
        if fused_head:
            if self.training and float(getattr(self.config, "sched_sampling_p", 0.0)) > 0.0:
                raise RuntimeError(
                    "fused_head cannot serve sched_sampling_p>0: scheduled sampling needs "
                    "base_logits.argmax over the full vocabulary. Set CF_FUSED_HEAD=0 for that run.")
            residual = self._path_residual(future_tokens, prev_token)
            prev = torch.zeros_like(future_tokens)
            prev[:, 1:] = future_tokens[:, :-1]
            if prev_token is not None:
                prev = prev.clone()
                prev[:, 0] = prev_token
            emb = self.markov.w1(prev)
            if prev_token is None:
                # reference sets bias[:, 0, :] = 0; w2 is a Linear WITHOUT bias, so a zero
                # embedding row reproduces that exactly rather than approximately.
                emb = emb.clone()
                emb[:, 0, :] = 0.0
            out = {"v_pred": v_pred, "v_star": v_star, "pred_hidden": pred_hidden,
                   "cond_hidden": pred_hidden + residual, "markov_emb": emb}
            if self.train_vae:
                out["recon_hidden"] = self._decode(tgt_lat)
            return out

        base_logits = self.lm_head(pred_hidden)

        path_tokens = future_tokens
        p_ss = float(getattr(self.config, "sched_sampling_p", 0.0))
        if self.training and p_ss > 0.0:
            own = base_logits.argmax(dim=-1)                     # exposure-gap fix, no extra forward
            flip = torch.rand_like(own, dtype=torch.float32) < p_ss
            path_tokens = torch.where(flip, own, future_tokens)
        residual = self._path_residual(path_tokens, prev_token)  # HIDDEN-space path conditioning
        cond_hidden = pred_hidden + residual
        cond_logits = self.lm_head(cond_hidden)
        prev = torch.zeros_like(path_tokens)
        prev[:, 1:] = path_tokens[:, :-1]
        bias = self.markov.bias(prev)
        if prev_token is None:
            bias[:, 0, :] = 0.0
        else:
            bias[:, 0, :] = self.markov.bias(prev_token.unsqueeze(1))[:, 0, :]
        cond_logits = cond_logits + bias
        out = {"v_pred": v_pred, "v_star": v_star, "pred_hidden": pred_hidden,
               "cond_hidden": cond_hidden, "base_logits": base_logits, "cond_logits": cond_logits}
        if self.train_vae:
            # autoencoder anchor: decode(encode(target)) must reconstruct the target hidden,
            # so the jointly-trained latent stays decodable while the flow/CE reshape it for accept
            out["recon_hidden"] = self._decode(tgt_lat)
        return out
