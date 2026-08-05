from pathlib import Path

from transformers import HfArgumentParser, TrainingArguments

from chained_flow.training.train_hidden_mlp import (
    HiddenMLPModelArguments,
    LossArguments,
    TeacherDataArguments,
)


def test_smoke_mlp_yaml_config_parses():
    parser = HfArgumentParser(
        (HiddenMLPModelArguments, TeacherDataArguments, LossArguments, TrainingArguments)
    )
    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(Path("train_configs/smoke_mlp.yaml").resolve())
    )

    assert data_args.dataset_path == "teacher_states/gsm8k-qwen35-08b-smoke"
    assert training_args.output_dir == "checkpoints/hidden-mlp-smoke"
    assert model_args.context_size == 4
    assert model_args.draft_length == 4
    assert model_args.device is None
    assert loss_args.lambda_mse == 1.0
