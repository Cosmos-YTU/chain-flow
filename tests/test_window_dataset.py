from datasets import Dataset
import torch

from chained_flow.training import window_dataset
from chained_flow.training.window_dataset import TeacherWindowDataset, build_flow_window_cache, is_flow_window_cache


def test_teacher_window_dataset_derives_dynamic_training_fields():
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [10, 11, 12, 13, 14],
                "final_hidden": [[float(i), float(i + 1)] for i in range(5)],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 5,
                "prompt_length": 2,
            }
        ]
    )
    windows = TeacherWindowDataset(dataset, context_size=3, draft_length=2, windows_per_epoch=1, seed=0)
    sample = windows[0]

    assert sample["context_hidden"].shape == (3, 2)
    assert sample["target_hidden"].shape == (2, 2)
    assert sample["future_tokens"].shape == (2,)
    assert windows.available_windows == 2
    assert windows.row_tensors is not None


def test_teacher_window_dataset_pads_short_left_context():
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [1, 2, 3],
                "final_hidden": [[1.0], [2.0], [3.0]],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 3,
                "prompt_length": 1,
            }
        ]
    )
    windows = TeacherWindowDataset(dataset, context_size=4, draft_length=1, windows_per_epoch=1, seed=0)
    sample = windows[0]

    assert torch.equal(sample["context_hidden"], torch.tensor([[1.0], [1.0], [1.0], [2.0]]))


def test_teacher_window_dataset_samples_response_side_only():
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [0, 1, 2, 3, 4, 5],
                "final_hidden": [[float(i)] for i in range(6)],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 6,
                "prompt_length": 4,
            }
        ]
    )
    windows = TeacherWindowDataset(dataset, context_size=2, draft_length=1, windows_per_epoch=16, seed=0)

    for index in range(len(windows)):
        sample = windows[index]
        assert int(sample["future_tokens"][0].item()) >= 4


def test_teacher_window_dataset_loads_hf_dataset_when_path_is_not_local(monkeypatch):
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [0, 1, 2],
                "final_hidden": [[0.0], [1.0], [2.0]],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 3,
                "prompt_length": 1,
            }
        ]
    )
    calls = {}

    def fake_load_dataset(path, *, split):
        calls["path"] = path
        calls["split"] = split
        return dataset

    monkeypatch.setattr(window_dataset, "load_dataset", fake_load_dataset)
    windows = TeacherWindowDataset.from_path(
        "user/repo",
        split="train",
        context_size=2,
        draft_length=1,
        windows_per_epoch=1,
    )

    assert calls == {"path": "user/repo", "split": "train"}
    assert len(windows) == 1


def test_teacher_window_dataset_can_skip_row_materialization():
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [0, 1, 2],
                "final_hidden": [[0.0], [1.0], [2.0]],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 3,
                "prompt_length": 1,
            }
        ]
    )
    windows = TeacherWindowDataset(
        dataset,
        context_size=2,
        draft_length=1,
        windows_per_epoch=1,
        materialize_rows=False,
    )

    sample = windows[0]

    assert windows.row_tensors is None
    assert sample["context_hidden"].shape == (2, 1)

def test_flow_window_cache_round_trips_teacher_windows(tmp_path):
    source_dir = tmp_path / "teacher"
    cache_dir = tmp_path / "flow-cache"
    dataset = Dataset.from_list(
        [
            {
                "text": "x",
                "input_ids": [10, 11, 12, 13, 14],
                "final_hidden": [[float(i), float(i + 1)] for i in range(5)],
                "example_id": "0",
                "source": "test",
                "split": "train",
                "format_name": "test",
                "model_id": "fake",
                "hidden_dtype": "float32",
                "num_tokens": 5,
                "prompt_length": 2,
            }
        ]
    )
    dataset.save_to_disk(str(source_dir))

    metadata = build_flow_window_cache(
        str(source_dir),
        cache_dir,
        draft_length=2,
        hidden_dtype="float32",
    )
    windows = TeacherWindowDataset.from_path(
        str(cache_dir),
        context_size=3,
        draft_length=2,
        windows_per_epoch=1,
        seed=0,
    )
    sample = windows[0]

    assert is_flow_window_cache(cache_dir)
    assert metadata["cache_type"] == "chained_flow.flow_window_cache"
    assert windows.available_windows == 2
    assert sample["context_hidden"].shape == (3, 2)
    assert sample["target_hidden"].shape == (2, 2)
    assert sample["future_tokens"].shape == (2,)

