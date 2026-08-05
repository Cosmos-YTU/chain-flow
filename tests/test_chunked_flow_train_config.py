from pathlib import Path

from transformers import HfArgumentParser, TrainingArguments

from chained_flow.training.train_chunked_flow import (
    ChunkedFlowModelArguments,
    FlowLossArguments,
    TeacherDataArguments,
)


def test_chunked_flow_args_parse_minimal_cli():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )

    model_args, data_args, loss_args, training_args = parser.parse_args_into_dataclasses(
        [
            "--vae_dir",
            "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880",
            "--dataset_path",
            "teacher_states/gsm8k-qwen35-08b-smoke",
            "--output_dir",
            "out/flow/ckpts/smoke",
            "--per_device_train_batch_size",
            "2",
        ]
    )

    assert model_args.draft_length == 2
    assert model_args.chunk_size == 2
    assert model_args.vae_dir == "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880"
    assert data_args.dataset_path == "teacher_states/gsm8k-qwen35-08b-smoke"
    assert loss_args.gamma == 0.8
    assert training_args.output_dir == "out/flow/ckpts/smoke"


def test_chunked_flow_args_parse_yaml_config():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )

    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(Path("train_configs/chunked_flow/smoke_chunked_flow.yaml").resolve())
    )

    assert model_args.draft_length == 2
    assert model_args.chunk_size == 2
    assert model_args.vae_dir == "out/vae/ckpts/hidden-vae-lr3e3/checkpoint-1880"
    assert model_args.expert_dim == 64
    assert data_args.dataset_path == "teacher_states/gsm8k-qwen35-08b-smoke"
    assert data_args.dataset_split == "train"
    assert data_args.windows_per_epoch == 32
    assert data_args.materialize_rows is True
    assert loss_args.gamma == 0.8
    assert training_args.output_dir == "out/flow/ckpts/smoke-chunked-flow-k2"

def test_chunked_flow_sweep_yaml_configs_parse_and_use_unique_outputs():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps").glob("*/*.yaml"))

    assert len(config_paths) >= 15

    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.chunk_size == model_args.draft_length
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.materialize_rows is True
        assert loss_args.gamma == 0.8
        output_dirs.append(training_args.output_dir)

    assert len(output_dirs) == len(set(output_dirs))


def test_chunked_flow_k_sweep_yaml_configs_use_expected_grid():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/k_draft").glob("*.yaml"))

    assert len(config_paths) == 6

    grid = set()
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.context_size == 8
        assert model_args.chunk_size == model_args.draft_length
        assert model_args.draft_length in {4, 6, 8}
        assert model_args.expert_dim in {384, 512}
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 1024
        grid.add((model_args.draft_length, model_args.expert_dim))

    assert grid == {(4, 384), (4, 512), (6, 384), (6, 512), (8, 384), (8, 512)}

def test_chunked_flow_k16_c2_lambda_overfit_configs_use_expected_grid():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/lambda_k16_c2").glob("*.yaml"))

    assert len(config_paths) == 6

    output_dirs = []
    accept_weights = set()
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.context_size == 8
        assert model_args.draft_length == 16
        assert model_args.chunk_size == 2
        assert model_args.expert_dim == 512
        assert model_args.ffn_multiplier == 6
        assert model_args.train_vae is True
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.windows_per_epoch == 1024
        total = (
            loss_args.lambda_flow
            + loss_args.lambda_latent
            + loss_args.lambda_hidden
            + loss_args.lambda_cos
            + loss_args.lambda_ce
            + loss_args.lambda_accept
        )
        assert abs(total - 1.0) < 1e-9
        accept_weights.add(loss_args.lambda_accept)
        output_dirs.append(training_args.output_dir)

    assert accept_weights == {0.2, 0.3, 0.4, 0.5}
    assert len(output_dirs) == len(set(output_dirs))



def test_block_causal_wider_vae_overfit_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/wider_vae").glob("*.yaml"))

    assert len(config_paths) == 4

    windows = set()
    layers = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.context_size == 8
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.expert_dim == 512
        assert model_args.ffn_multiplier == 6
        assert model_args.train_vae is True
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert data_args.materialize_rows is True
        assert loss_args.gamma == 0.8
        assert training_args.per_device_train_batch_size == 256
        assert training_args.per_device_eval_batch_size == 256
        windows.add(data_args.windows_per_epoch)
        layers.add(model_args.num_drafter_layers)
        output_dirs.append(training_args.output_dir)

    assert windows == {1024}
    assert layers == {1, 2, 4, 8}
    assert len(output_dirs) == len(set(output_dirs))


def test_block_causal_wider_vae_flow_step_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/wider_vae/flow_steps").glob("*.yaml"))

    assert len(config_paths) == 3

    steps = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.num_drafter_layers == 2
        assert model_args.expert_dim == 512
        assert data_args.windows_per_epoch == 1024
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 256
        steps.add(model_args.num_flow_steps)
        output_dirs.append(training_args.output_dir)

    assert steps == {1, 2, 4}
    assert len(output_dirs) == len(set(output_dirs))


def test_block_causal_wider_vae_expert_dim_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/wider_vae/expert_dim").glob("*.yaml"))

    assert len(config_paths) == 3

    dims = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.num_drafter_layers == 2
        assert model_args.num_flow_steps == 1
        assert data_args.windows_per_epoch == 1024
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 256
        dims.add(model_args.expert_dim)
        output_dirs.append(training_args.output_dir)

    assert dims == {512, 768, 1024}
    assert len(output_dirs) == len(set(output_dirs))


def test_block_causal_wider_vae_k_ablation_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/wider_vae/k_ablation").glob("*.yaml"))

    assert len(config_paths) == 2

    grid = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.context_size == 8
        assert model_args.expert_dim == 512
        assert model_args.num_drafter_layers == 2
        assert model_args.num_flow_steps == 1
        assert data_args.windows_per_epoch == 1024
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 256
        grid.add((model_args.draft_length, model_args.chunk_size))
        output_dirs.append(training_args.output_dir)

    assert grid == {(2, 2), (4, 4)}
    assert len(output_dirs) == len(set(output_dirs))


def test_block_causal_wider_vae_train_ablation_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/wider_vae/vae_train").glob("*.yaml"))

    assert len(config_paths) == 2

    settings = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.context_size == 8
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.expert_dim == 512
        assert model_args.num_drafter_layers == 2
        assert model_args.num_flow_steps == 1
        assert data_args.windows_per_epoch == 1024
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 256
        settings.add((model_args.train_vae, model_args.vae_learning_rate_multiplier))
        output_dirs.append(training_args.output_dir)

    assert settings == {(True, 0.5), (False, 0.1)}
    assert len(output_dirs) == len(set(output_dirs))


def test_block_causal_frozen_vae_ablation_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = [
        path
        for path in Path("train_configs/chunked_flow/sweeps/overfit/block_causal/frozen_vae").rglob("*.yaml")
        if ".ipynb_checkpoints" not in path.parts
        and "lambda_sweep" not in path.parts
        and "multi_expert" not in path.parts
        and "init_mode" not in path.parts
        and "hidden_kv" not in path.parts
    ]

    assert len(config_paths) == 12

    output_dirs = []
    grid = set()
    for config_path in sorted(config_paths):
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.train_vae is False
        assert model_args.vae_learning_rate_multiplier == 0.1
        assert data_args.windows_per_epoch == 1024
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        assert training_args.per_device_train_batch_size == 256
        grid.add((model_args.draft_length, model_args.chunk_size))
        output_dirs.append(training_args.output_dir)

    assert {(2, 2), (4, 4), (8, 8)}.issubset(grid)
    assert len(output_dirs) == len(set(output_dirs))
    assert all("frozen" in output_dir for output_dir in output_dirs)


def test_block_causal_frozen_vae_init_mode_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(Path("train_configs/chunked_flow/sweeps/overfit/block_causal/frozen_vae/init_mode").glob("*.yaml"))

    assert len(config_paths) == 3

    modes = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.vae_dir == "out/vae/ckpts/transformer-hidden-seq8-wider-kl-beta3e5-full"
        assert model_args.architecture == "block_causal_experts"
        assert model_args.context_size == 8
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.expert_dim == 768
        assert model_args.num_drafter_layers == 4
        assert model_args.num_flow_steps == 2
        assert model_args.train_vae is False
        assert data_args.windows_per_epoch == 8192
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        total = (
            loss_args.lambda_flow
            + loss_args.lambda_latent
            + loss_args.lambda_hidden
            + loss_args.lambda_cos
            + loss_args.lambda_ce
            + loss_args.lambda_accept
        )
        assert abs(total - 1.0) < 1e-9
        modes.add(model_args.init_mode)
        output_dirs.append(training_args.output_dir)

    assert modes == {"noise", "repeat_last", "delta"}
    assert len(output_dirs) == len(set(output_dirs))



def test_hidden_kv_flow_baseline_config_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_path = Path(
        "train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/hidden_kv_flow_k8_c8_8192w_delta_layers4_ffn4.yaml"
    )

    model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
        yaml_file=str(config_path.resolve())
    )

    assert model_args.architecture == "hidden_kv_flow"
    assert model_args.vae_dir is None
    assert model_args.context_size == 8
    assert model_args.draft_length == 8
    assert model_args.chunk_size == 8
    assert model_args.expert_dim == 1024
    assert model_args.num_heads == 8
    assert model_args.ffn_multiplier == 4
    assert model_args.num_flow_steps == 2
    assert model_args.num_drafter_layers == 4
    assert model_args.init_mode == "delta"
    assert data_args.windows_per_epoch == 8192
    assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
    total = (
        loss_args.lambda_flow
        + loss_args.lambda_latent
        + loss_args.lambda_hidden
        + loss_args.lambda_cos
        + loss_args.lambda_ce
        + loss_args.lambda_accept
    )
    assert abs(total - 1.0) < 1e-9
    assert training_args.per_device_train_batch_size == 128



def test_hidden_kv_flow_4096w_sweep_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(
        Path("train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/sweep_4096w").glob("*.yaml")
    )

    assert len(config_paths) == 12

    labels = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.architecture == "hidden_kv_flow"
        assert model_args.vae_dir is None
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.expert_dim == 1024
        assert model_args.num_heads == 8
        assert model_args.init_mode == "delta"
        assert model_args.train_vae is False
        assert data_args.windows_per_epoch == 4096
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        total = (
            loss_args.lambda_flow
            + loss_args.lambda_latent
            + loss_args.lambda_hidden
            + loss_args.lambda_cos
            + loss_args.lambda_ce
            + loss_args.lambda_accept
        )
        assert abs(total - 1.0) < 1e-9
        assert training_args.per_device_train_batch_size == 128
        labels.add(config_path.stem.removeprefix("hidden_kv_flow_4096w_"))
        output_dirs.append(training_args.output_dir)

    assert labels == {
        "layers2",
        "layers4",
        "layers6",
        "ffn2",
        "ffn4",
        "ffn6",
        "steps1",
        "steps2",
        "context8",
        "context16",
        "lr3e4",
        "lr1e3",
    }
    assert len(output_dirs) == len(set(output_dirs))



def test_hidden_kv_flow_lr_scheduler_sweep_configs_parse():
    parser = HfArgumentParser(
        (ChunkedFlowModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments)
    )
    config_paths = sorted(
        Path("train_configs/chunked_flow/sweeps/overfit/block_causal/hidden_kv/lr_scheduler_4096w").glob("*.yaml")
    )

    assert len(config_paths) == 5

    settings = set()
    output_dirs = []
    for config_path in config_paths:
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(config_path.resolve())
        )
        assert model_args.architecture == "hidden_kv_flow"
        assert model_args.vae_dir is None
        assert model_args.context_size == 8
        assert model_args.draft_length == 8
        assert model_args.chunk_size == 8
        assert model_args.expert_dim == 1024
        assert model_args.num_heads == 8
        assert model_args.ffn_multiplier == 4
        assert model_args.num_flow_steps == 2
        assert model_args.num_drafter_layers == 6
        assert model_args.init_mode == "delta"
        assert data_args.windows_per_epoch == 4096
        assert data_args.dataset_path == "data/flow_cache/gsm8k_6.5k_train"
        total = (
            loss_args.lambda_flow
            + loss_args.lambda_latent
            + loss_args.lambda_hidden
            + loss_args.lambda_cos
            + loss_args.lambda_ce
            + loss_args.lambda_accept
        )
        assert abs(total - 1.0) < 1e-9
        settings.add((float(training_args.learning_rate), str(training_args.lr_scheduler_type), float(training_args.warmup_ratio)))
        output_dirs.append(training_args.output_dir)

    assert settings == {
        (0.001, "SchedulerType.CONSTANT", 0.0),
        (0.003, "SchedulerType.CONSTANT", 0.0),
        (0.01, "SchedulerType.CONSTANT", 0.0),
        (0.003, "SchedulerType.COSINE", 0.05),
        (0.01, "SchedulerType.COSINE", 0.05),
    }
    assert len(output_dirs) == len(set(output_dirs))
