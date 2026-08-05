import importlib.util
from pathlib import Path

import pytest
from datasets import Dataset


def load_script_module():
    path = Path("scripts/merge_teacher_datasets.py").resolve()
    spec = importlib.util.spec_from_file_location("merge_teacher_datasets", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def teacher_row(example_id: str) -> dict:
    return {
        "text": "text",
        "prompt_text": "prompt",
        "generated_text": "answer",
        "input_ids": [1, 2],
        "final_hidden": [[0.0], [1.0]],
        "example_id": example_id,
        "source": "test",
        "split": "train",
        "format_name": "fmt",
        "model_id": "model",
        "hidden_dtype": "float32",
        "num_tokens": 2,
        "prompt_length": 1,
    }


def test_validate_teacher_dataset_rejects_missing_columns():
    module = load_script_module()
    dataset = Dataset.from_list([{"text": "x"}])

    with pytest.raises(ValueError, match="missing required teacher columns"):
        module.validate_teacher_dataset(dataset, name="bad")


def test_merge_teacher_datasets_concatenates_loaded_datasets(monkeypatch):
    module = load_script_module()
    datasets = {
        "a": Dataset.from_list([teacher_row("a0")]),
        "b": Dataset.from_list([teacher_row("b0"), teacher_row("b1")]),
    }

    def fake_load_teacher_dataset(path_or_repo, *, split):
        assert split == "train"
        return datasets[path_or_repo]

    monkeypatch.setattr(module, "load_teacher_dataset", fake_load_teacher_dataset)
    merged = module.merge_teacher_datasets(["a", "b"], split="train")

    assert len(merged) == 3
    assert merged["example_id"] == ["a0", "b0", "b1"]
