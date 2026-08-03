from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chained_flow.timing import TimingStats, timed_section


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"


@dataclass
class LMState:
    input_ids: torch.Tensor
    past_key_values: Any
    final_hidden: torch.Tensor
    logits: torch.Tensor
    position: int


class FrozenLMWrapper:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, model_id: str = DEFAULT_MODEL_ID):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_id = model_id
        for param in self.model.parameters():
            param.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ) -> tuple["FrozenLMWrapper", TimingStats]:
        timings = TimingStats()
        model_kwargs = dict(kwargs)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device is not None and str(device) == "auto":
            model_kwargs.setdefault("device_map", "auto")
        with timed_section(timings, "model_load"):
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            if device is not None and str(device) != "auto":
                model = model.to(device)
        return cls(model, tokenizer, model_id=model_id), timings

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def eos_token_id(self) -> int | None:
        value = getattr(self.tokenizer, "eos_token_id", None)
        if value is None:
            value = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        return value

    def tokenize(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(text, return_tensors="pt")
        return encoded.input_ids.to(self.device)

    def decode(self, input_ids: torch.Tensor | list[int], **kwargs: Any) -> str:
        if isinstance(input_ids, torch.Tensor):
            ids = input_ids.detach().cpu().tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
        else:
            ids = input_ids
        return self.tokenizer.decode(ids, **kwargs)

    def _forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Any = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Any:
        return self.model(
            input_ids=input_ids.to(self.device),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> tuple[LMState, TimingStats]:
        timings = TimingStats()
        with timed_section(timings, "prefill", self.device):
            outputs = self._forward(input_ids, use_cache=True)
        input_ids = input_ids.to(self.device)
        return (
            LMState(
                input_ids=input_ids,
                past_key_values=outputs.past_key_values,
                final_hidden=outputs.hidden_states[-1],
                logits=outputs.logits,
                position=input_ids.shape[1],
            ),
            timings,
        )

    @torch.inference_mode()
    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        state: LMState,
        *,
        use_cache: bool = True,
    ) -> tuple[LMState, TimingStats]:
        timings = TimingStats()
        with timed_section(timings, "forward_with_cache", self.device):
            outputs = self._forward(
                input_ids,
                past_key_values=state.past_key_values,
                use_cache=use_cache,
            )
        new_input_ids = torch.cat([state.input_ids, input_ids.to(self.device)], dim=1)
        final_hidden = torch.cat([state.final_hidden, outputs.hidden_states[-1]], dim=1)
        logits = torch.cat([state.logits, outputs.logits], dim=1)
        return (
            LMState(
                input_ids=new_input_ids,
                past_key_values=outputs.past_key_values,
                final_hidden=final_hidden,
                logits=logits,
                position=new_input_ids.shape[1],
            ),
            timings,
        )

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = getattr(self.model.lm_head, "weight", None)
        if weight is not None:
            hidden_states = hidden_states.to(dtype=weight.dtype)
        return self.model.lm_head(hidden_states)

    @staticmethod
    def latest_hidden(state: LMState) -> torch.Tensor:
        return state.final_hidden[:, -1, :]

    @torch.inference_mode()
    def next_token(self, state: LMState) -> tuple[torch.Tensor, TimingStats]:
        timings = TimingStats()
        with timed_section(timings, "next_token", self.device):
            token = state.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return token, timings
