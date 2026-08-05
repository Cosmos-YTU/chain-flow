from pathlib import Path
import importlib.util

import torch

from transformers import HfArgumentParser, TrainingArguments
from transformers.trainer_utils import IntervalStrategy, SaveStrategy


def load_script_module():
    path = Path("scripts/train_vae.py").resolve()
    spec = importlib.util.spec_from_file_location("train_vae", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_vae_yaml_parses():
    module = load_script_module()
    parser = HfArgumentParser(
        (
            module.VAEModelArguments,
            module.VAEDataArguments,
            module.VAELossArguments,
            TrainingArguments,
        )
    )
    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(Path("train_configs/vae/smoke_vae.yaml").resolve())
    )

    assert model_args.vae_type == "residual_mlp"
    assert model_args.latent_size == 256
    assert model_args.device == "cuda"
    assert data_args.dataset_path == "sghosts/cf_gsm8k_1k_train"
    assert data_args.dataset_split == "train"
    assert data_args.tokens_per_epoch is None
    assert data_args.validation_fraction == 0.1
    assert loss_args.beta == 0.0001
    assert training_args.output_dir == "/content/drive/MyDrive/chained-flow/vae/ckpts/hidden-vae-smoke"
    assert training_args.per_device_eval_batch_size == training_args.per_device_train_batch_size
    assert training_args.per_device_eval_batch_size > 0
    assert training_args.eval_strategy == "epoch"
    assert training_args.save_strategy == "epoch"


def test_configure_epoch_eval_forces_eval_when_args_default_to_no_eval():
    module = load_script_module()
    training_args = TrainingArguments(output_dir="/tmp/chained-flow-test")

    module.configure_epoch_eval(training_args)

    assert training_args.eval_strategy == IntervalStrategy.EPOCH
    assert training_args.save_strategy == SaveStrategy.EPOCH
    assert training_args.do_eval is True


def test_transformer_hidden_seq8_sweep_configs_parse():
    module = load_script_module()
    parser = HfArgumentParser(
        (
            module.VAEModelArguments,
            module.VAEDataArguments,
            module.VAELossArguments,
            TrainingArguments,
        )
    )
    config_paths = sorted(Path("train_configs/vae/transformer_hidden/sweeps_seq8").glob("*.yaml"))

    assert len(config_paths) == 8

    output_dirs = []
    names = set()
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        names.add(config_path.stem.removeprefix("transformer_hidden_seq8_"))
        assert model_args.vae_type == "transformer_hidden"
        assert model_args.hidden_size == 1024
        assert model_args.num_heads == 4
        assert model_args.max_sequence_length == 16
        assert data_args.sequence_length == 8
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.tokens_per_epoch == 8192
        assert training_args.per_device_train_batch_size > 0
        assert training_args.per_device_eval_batch_size > 0
        output_dirs.append(training_args.output_dir)

    assert names == {"base", "deeper", "wider", "bigger_latent", "light_kl", "kl_1e4", "low_lr", "strong"}
    assert len(output_dirs) == len(set(output_dirs))


def test_vae_component_logging_trainer_prediction_step_logs_eval_components(tmp_path):
    from chained_flow.training.train_vae import (
        HiddenVAETrainingModule,
        VAEComponentLoggingTrainer,
        VAELossArguments,
        VAEModelArguments,
    )

    model = HiddenVAETrainingModule(
        VAEModelArguments(
            vae_type="mlp",
            hidden_size=8,
            latent_size=3,
            intermediate_size=5,
        ),
        VAELossArguments(),
    )
    trainer = VAEComponentLoggingTrainer(
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), report_to="none"),
    )
    logged = []
    trainer.log = lambda values, *args, **kwargs: logged.append(values)

    loss, logits, labels = trainer.prediction_step(
        model,
        {"hidden": torch.randn(2, 8)},
        prediction_loss_only=True,
    )

    assert torch.isfinite(loss)
    assert logits is None
    assert labels is None
    assert logged
    assert "eval_hidden.mse" in logged[-1]
    assert "eval_latent.kl" in logged[-1]


def test_transformer_hidden_seq8_kl_sweep_configs_parse():
    module = load_script_module()
    parser = HfArgumentParser(
        (
            module.VAEModelArguments,
            module.VAEDataArguments,
            module.VAELossArguments,
            TrainingArguments,
        )
    )
    config_paths = sorted(Path("train_configs/vae/transformer_hidden/kl_sweep_seq8").glob("*.yaml"))

    assert len(config_paths) == 10

    betas_by_arch = {"wider": set(), "strong": set()}
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        arch = "strong" if "strong" in config_path.name else "wider"
        assert model_args.vae_type == "transformer_hidden"
        assert model_args.hidden_size == 1024
        assert model_args.intermediate_size == 768
        assert model_args.max_sequence_length == 16
        if arch == "wider":
            assert model_args.latent_size == 256
            assert model_args.num_layers == 2
        else:
            assert model_args.latent_size == 384
            assert model_args.num_layers == 4
        assert data_args.sequence_length == 8
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.tokens_per_epoch == 8192
        assert training_args.per_device_train_batch_size == 8192
        assert training_args.per_device_eval_batch_size == 8192
        assert training_args.num_train_epochs == 50
        assert training_args.learning_rate == 0.0003
        betas_by_arch[arch].add(float(loss_args.beta))
        output_dirs.append(training_args.output_dir)

    expected_betas = {0.0, 0.000001, 0.000003, 0.00001, 0.00003}
    assert betas_by_arch == {"wider": expected_betas, "strong": expected_betas}
    assert len(output_dirs) == len(set(output_dirs))
