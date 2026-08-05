from datasets import Dataset
import torch

from chained_flow.training import vae_dataset
from chained_flow.training.vae_dataset import TeacherHiddenTokenDataset, collate_hidden_tokens


def test_teacher_hidden_token_dataset_samples_response_tokens():
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                "num_tokens": 3,
                "prompt_length": 1,
            }
        ]
    )
    token_dataset = TeacherHiddenTokenDataset(dataset, tokens_per_epoch=4, seed=0, response_only=True)
    item = token_dataset[0]

    assert item["hidden"].shape == (2,)
    assert item["hidden"][0].item() in {1.0, 2.0}


def test_teacher_hidden_token_dataset_flattens_response_hidden_tokens():
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                "num_tokens": 3,
                "prompt_length": 1,
            },
            {
                "final_hidden": [[3.0, 3.0], [4.0, 4.0]],
                "num_tokens": 2,
                "prompt_length": 1,
            },
        ]
    )
    token_dataset = TeacherHiddenTokenDataset(dataset, seed=0, response_only=True)

    assert token_dataset.valid_rows == [(0, 1, 3), (1, 1, 2)]
    assert token_dataset.hidden_tokens.tolist() == [[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]
    assert len(token_dataset) == 3


def test_teacher_hidden_token_dataset_train_val_split_uses_held_out_tokens():
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[float(index), float(index)] for index in range(10)],
                "num_tokens": 10,
                "prompt_length": 0,
            }
        ]
    )
    token_dataset = TeacherHiddenTokenDataset(dataset, seed=0, response_only=True)
    train_dataset, val_dataset = token_dataset.train_val_split(val_fraction=0.2, seed=0)

    assert train_dataset.hidden_tokens.shape == (8, 2)
    assert val_dataset.hidden_tokens.shape == (2, 2)
    assert len(train_dataset) == 8
    assert len(val_dataset) == 2


def test_collate_hidden_tokens_stacks_batch():
    batch = [{"hidden": torch.zeros(2)}, {"hidden": torch.ones(2)}]
    collated = collate_hidden_tokens(batch)

    assert collated["hidden"].shape == (2, 2)


def test_teacher_hidden_token_dataset_loads_hf_dataset_when_path_is_not_local(monkeypatch):
    dataset = Dataset.from_list(
        [
            {
                "final_hidden": [[0.0, 0.0], [1.0, 1.0]],
                "num_tokens": 2,
                "prompt_length": 1,
            }
        ]
    )
    calls = {}

    def fake_load_dataset(path, *, split):
        calls["path"] = path
        calls["split"] = split
        return dataset

    monkeypatch.setattr(vae_dataset, "load_dataset", fake_load_dataset)
    token_dataset = TeacherHiddenTokenDataset.from_path("user/repo", split="train", tokens_per_epoch=1)

    assert calls == {"path": "user/repo", "split": "train"}
    assert len(token_dataset) == 1
