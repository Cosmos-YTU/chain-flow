from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from chained_flow.frozen_lm import FrozenLMWrapper


class FakeCache:
    def __init__(self, length: int = 0):
        self.length = length

    def crop(self, max_length: int) -> None:
        self.length = min(self.length, max_length)

    def get_seq_length(self) -> int:
        return self.length


@dataclass
class FakeOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor]
    past_key_values: FakeCache | None = None


class ShiftLMHead(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        weight = torch.zeros(vocab_size, vocab_size)
        for token_id in range(vocab_size):
            weight[(token_id + 1) % vocab_size, token_id] = 10.0
        self.weight = nn.Parameter(weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden_states, self.weight.t())


class FakeConfig:
    hidden_size = 8
    vocab_size = 8
    eos_token_id = 7


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeConfig()
        self.lm_head = ShiftLMHead(self.config.hidden_size)
        # vocab == hidden_size for the fake shift model; embedding table for token-feedback drafters
        self.embed_tokens = nn.Embedding(self.config.hidden_size, self.config.hidden_size)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: FakeCache | None = None,
        use_cache: bool = True,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        **_: object,
    ) -> FakeOutput:
        hidden = torch.nn.functional.one_hot(
            input_ids % self.config.hidden_size,
            num_classes=self.config.hidden_size,
        ).float()
        logits = self.lm_head(hidden)
        cache = past_key_values if past_key_values is not None else FakeCache()
        if use_cache:
            cache.length += input_ids.shape[1]
        return FakeOutput(logits=logits, hidden_states=(hidden,), past_key_values=cache if use_cache else None)


class FakeTokenizer:
    eos_token_id = 7

    def __call__(self, text: str, return_tensors: str = "pt"):
        mapping = {"a": 1, "b": 2, "c": 3}
        ids = [mapping.get(part, 1) for part in text.split()]
        return type("Encoded", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})

    def decode(self, ids, **_: object) -> str:
        return " ".join(str(i) for i in ids)


@pytest.fixture
def fake_wrapper() -> FrozenLMWrapper:
    return FrozenLMWrapper(FakeModel(), FakeTokenizer(), model_id="fake")
