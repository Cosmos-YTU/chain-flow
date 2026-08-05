import torch

from chained_flow.training.collect_teacher import (
    TeacherCollectionConfig,
    _answer_row_from_dataset_row,
    _backbone,
    _effective_generation_batch_size,
    _effective_hidden_batch_size,
    _model_torch_dtype,
    _push_teacher_dataset_if_requested,
    _resolve_range,
    _sequence_spans,
    format_gsm8k_prompt,
    teacher_dataset_features,
)
from chained_flow.timing import TimingStats
from datasets import Dataset


def test_teacher_features_text_first_and_no_attention_mask():
    features = teacher_dataset_features()
    assert list(features.keys())[0] == "text"
    assert list(features.keys())[1] == "prompt_text"
    assert list(features.keys())[2] == "generated_text"
    assert "attention_mask" not in features
    assert "prompt_length" in features


def test_teacher_features_support_float16_hidden_storage():
    features = teacher_dataset_features("float16")
    assert str(features["final_hidden"].feature.feature.dtype) == "float16"


def test_answer_row_from_dataset_row_updates_hidden_dtype():
    row = _answer_row_from_dataset_row(
        {
            "text": "ab",
            "prompt_text": "a",
            "generated_text": "b",
            "input_ids": [1, 2],
            "example_id": "x",
            "source": "tmp",
            "split": "train",
            "format_name": "fmt",
            "model_id": "model",
            "hidden_dtype": "float32",
            "num_tokens": 2,
            "prompt_length": 1,
        },
        storage_dtype="float16",
    )

    assert row["input_ids"] == [1, 2]
    assert row["hidden_dtype"] == "float16"
    assert row["num_tokens"] == 2


def test_effective_batch_sizes_fallback_to_batch_size():
    config = TeacherCollectionConfig(batch_size=16)
    assert _effective_generation_batch_size(config) == 16
    assert _effective_hidden_batch_size(config) == 16

    config = TeacherCollectionConfig(batch_size=16, generation_batch_size=32, hidden_batch_size=4)
    assert _effective_generation_batch_size(config) == 32
    assert _effective_hidden_batch_size(config) == 4


def test_resolve_range_prefers_explicit_end_and_limit_fallback():
    assert _resolve_range(7473, start=0, end=1024, limit=None) == (0, 1024)
    assert _resolve_range(7473, start=1024, end=None, limit=1024) == (1024, 2048)
    assert _resolve_range(1500, start=1024, end=2048, limit=None) == (1024, 1500)


def test_gsm8k_format_uses_prompt_only_without_reference_answer(fake_wrapper):
    text = format_gsm8k_prompt(
        {"question": "What is 2+2?", "answer": "#### 4"},
        fake_wrapper.tokenizer,
    )
    assert "What is 2+2?" in text
    assert "#### 4" not in text


def test_sequence_spans_trim_left_padding_and_stop_at_eos():
    spans = _sequence_spans(
        torch.tensor(
            [
                [0, 0, 4, 5, 6],
                [0, 7, 8, 2, 9],
            ]
        ),
        pad_token_id=0,
        eos_token_id=2,
        prompt_lengths=[2, 2],
    )
    assert spans == [(2, 5), (1, 4)]


def test_sequence_spans_ignore_eos_inside_prompt():
    spans = _sequence_spans(
        torch.tensor([[0, 4, 2, 5, 6, 2, 9]]),
        pad_token_id=0,
        eos_token_id=2,
        prompt_lengths=[4],
    )
    assert spans == [(1, 6)]


def test_model_dtype_parser():
    assert _model_torch_dtype(None) is None
    assert _model_torch_dtype("float16") is torch.float16


def test_backbone_prefers_model_attr(fake_wrapper):
    class ModelWithBackbone:
        def __init__(self):
            self.model = object()

    model = ModelWithBackbone()
    assert _backbone(model) is model.model


def test_push_teacher_dataset_if_requested_calls_push_to_hub(monkeypatch):
    dataset = Dataset.from_list([{"x": 1}])
    calls = {}

    def fake_push_to_hub(repo_id, *, private):
        calls["repo_id"] = repo_id
        calls["private"] = private

    monkeypatch.setattr(dataset, "push_to_hub", fake_push_to_hub)
    _push_teacher_dataset_if_requested(
        dataset,
        TeacherCollectionConfig(push_to_hub="user/repo", private=True),
        TimingStats(),
        torch.device("cpu"),
    )

    assert calls == {"repo_id": "user/repo", "private": True}
