from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.hidden_mlp import HiddenMLPConfig, HiddenMLPDrafter
from chained_flow.frozen_lm import FrozenLMWrapper
from chained_flow.training.losses import DrafterLossConfig, compute_drafter_loss


@dataclass
class HiddenMLPTrainingOutput:
    loss: torch.Tensor
    pred_hidden: torch.Tensor
    pred_latent: torch.Tensor | None
    components: dict[str, torch.Tensor]


class HiddenMLPTrainingModule(nn.Module):
    def __init__(
        self,
        frozen_lm: FrozenLMWrapper,
        drafter_config: HiddenMLPConfig,
        loss_config: DrafterLossConfig,
    ):
        super().__init__()
        self.drafter = HiddenMLPDrafter(frozen_lm, drafter_config)
        self.loss_config = loss_config
        lm_head = frozen_lm.model.lm_head
        self.register_buffer("lm_head_weight", lm_head.weight.detach().clone(), persistent=False)
        bias = getattr(lm_head, "bias", None)
        if bias is None:
            self.lm_head_bias = None
        else:
            self.register_buffer("lm_head_bias", bias.detach().clone(), persistent=False)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = torch.matmul(hidden_states, self.lm_head_weight.t())
        if self.lm_head_bias is not None:
            logits = logits + self.lm_head_bias
        return logits

    def forward(
        self,
        context_hidden: torch.Tensor,
        target_hidden: torch.Tensor,
        future_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_latent = None
        target_latent = None
        if self.drafter.uses_vae:
            pred_latent = self.drafter.predict_latent_from_context(context_hidden)
            pred_hidden = self.drafter.decode_latent(pred_latent)
            target_latent = self.drafter.encode_hidden(target_hidden)
        else:
            pred_hidden = self.drafter.predict_from_context(context_hidden)
        loss_output = compute_drafter_loss(
            pred_hidden,
            target_hidden,
            pred_latent=pred_latent,
            target_latent=target_latent,
            future_tokens=future_tokens,
            lm_head=self.lm_head,
            config=self.loss_config,
        )
        output: dict[str, torch.Tensor] = {
            "loss": loss_output.total,
            "pred_hidden": pred_hidden,
        }
        if pred_latent is not None:
            output["pred_latent"] = pred_latent
        for name, value in loss_output.components.items():
            output[f"loss_component/{name}"] = value.detach()
        return output
