import json

import pytest
import torch

from chained_flow.training.eval_chunked_flow import (
    ChunkedFlowEvalArguments,
    checkpoint_eval_output_path,
    dataset_eval_slug,
    discover_flow_checkpoints,
    find_flow_config_dir,
    load_flow_training_module,
    per_token_flow_metrics,
    select_checkpoint_stride,
    single_eval_output_path,
    summarize_metric,
    torch_dtype_from_string,
)
from chained_flow.training.train_chunked_flow import ChunkedFlowModelArguments, FlowLossArguments, SingleExpertFlowTrainingModule
from chained_flow.vae import HiddenVAEConfig, build_hidden_vae


def write_vae_checkpoint(path, *, hidden_size=8, latent_size=3, intermediate_size=5):
    path.mkdir(parents=True)
    config = {
        "model_args": {
            "vae_type": "mlp",
            "hidden_size": hidden_size,
            "latent_size": latent_size,
            "intermediate_size": intermediate_size,
            "device": None,
        },
        "loss_args": {},
        "data_args": {},
    }
    (path / "chained_flow_vae_config.json").write_text(json.dumps(config), encoding="utf-8")
    vae = build_hidden_vae(
        "mlp",
        HiddenVAEConfig(hidden_size=hidden_size, latent_size=latent_size, intermediate_size=intermediate_size),
    )
    torch.save({f"vae.{key}": value for key, value in vae.state_dict().items()}, path / "pytorch_model.bin")


def write_flow_checkpoint(path, fake_wrapper, vae_dir):
    path.mkdir(parents=True)
    model_args = ChunkedFlowModelArguments(
        model_id="fake",
        context_size=2,
        draft_length=2,
        chunk_size=2,
        vae_dir=str(vae_dir),
        expert_dim=8,
        num_heads=2,
        ffn_multiplier=2,
        local_files_only=True,
    )
    config = {
        "model_args": vars(model_args),
        "data_args": {},
        "loss_args": vars(FlowLossArguments()),
    }
    (path / "chained_flow_chunked_flow_config.json").write_text(json.dumps(config), encoding="utf-8")
    module = SingleExpertFlowTrainingModule(fake_wrapper, __import__(
        "chained_flow.training.train_chunked_flow",
        fromlist=["flow_config_from_args"],
    ).flow_config_from_args(model_args), FlowLossArguments())
    torch.save(module.state_dict(), path / "pytorch_model.bin")
    return config


def test_torch_dtype_from_string_accepts_bfloat16_alias():
    assert torch_dtype_from_string("bf16") is torch.bfloat16


def test_summarize_metric_reports_distribution_stats():
    summary = summarize_metric(torch.tensor([1.0, 2.0, 3.0]))

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(0.816496, rel=1e-5)
    assert set(summary) == {"mean", "std", "min", "max", "p50", "p90", "p95", "p99"}


def test_per_token_flow_metrics_reports_phase_one_metrics():
    pred_hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target_hidden = pred_hidden.clone()
    pred_latent = torch.tensor([[[1.0], [2.0]]])
    target_latent = pred_latent.clone()
    logits = torch.tensor([[[0.0, 5.0, 1.0], [0.0, 1.0, 5.0]]])
    future_tokens = torch.tensor([[1, 2]])

    metrics = per_token_flow_metrics(
        pred_hidden=pred_hidden,
        target_hidden=target_hidden,
        pred_latent=pred_latent,
        target_latent=target_latent,
        drafter_logits=logits,
        teacher_logits=logits,
        future_tokens=future_tokens,
    )

    assert {
        "token.top1_match@1",
        "token.top1_match@2",
        "token.sequence_match",
        "accept.greedy_prefix_len",
        "accept.rate@1",
        "accept.rate@2",
        "logit.ce",
        "logit.js_div_to_teacher",
        "logit.teacher_prob@1",
        "logit.teacher_prob@2",
        "hidden.rel_rmse",
        "hidden.cosine_similarity",
        "latent.mse",
    }.issubset(metrics)
    assert metrics["token.top1_match@1"].tolist() == [1.0]
    assert metrics["token.top1_match@2"].tolist() == [1.0]
    assert metrics["token.sequence_match"].tolist() == [1.0]
    assert metrics["accept.greedy_prefix_len"].tolist() == [2.0]
    assert metrics["accept.rate@1"].tolist() == [1.0]
    assert metrics["accept.rate@2"].tolist() == [1.0]
    assert metrics["hidden.rel_rmse"].tolist() == [0.0, 0.0]
    assert metrics["hidden.cosine_similarity"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["latent.mse"].tolist() == [0.0, 0.0]
    assert metrics["logit.js_div_to_teacher"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_find_flow_config_dir_uses_parent_for_trainer_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-20"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "chained_flow_chunked_flow_config.json").write_text("{}", encoding="utf-8")

    assert find_flow_config_dir(checkpoint_dir) == run_dir


def test_discover_flow_checkpoints_orders_numeric_checkpoints_with_weights(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-20").mkdir(parents=True)
    (run_dir / "checkpoint-20" / "model.safetensors").touch()
    (run_dir / "checkpoint-3").mkdir()
    (run_dir / "checkpoint-3" / "pytorch_model.bin").touch()
    (run_dir / "checkpoint-10").mkdir()

    checkpoints = discover_flow_checkpoints(run_dir)

    assert [path.name for path in checkpoints] == ["checkpoint-3", "checkpoint-20"]


def test_dataset_eval_slug_sanitizes_hub_dataset_names():
    assert dataset_eval_slug("sghosts/cf_gsm8k_1k_test") == "sghosts_cf_gsm8k_1k_test"


def test_checkpoint_eval_output_path_includes_dataset_and_checkpoint_name(tmp_path):
    args = ChunkedFlowEvalArguments(flow_dir=str(tmp_path / "run"), dataset_path="sghosts/cf_gsm8k_1k_test")
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-3"

    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        run_dir / "flow_eval_metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )

    args.output_path = str(tmp_path / "metrics.json")
    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        tmp_path / "metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )

    args.output_path = str(tmp_path / "evals")
    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        tmp_path / "evals" / "flow_eval_metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )


def test_single_eval_output_path_includes_dataset_name(tmp_path):
    args = ChunkedFlowEvalArguments(
        flow_dir=str(tmp_path / "run"),
        dataset_path="sghosts/cf_gsm8k_1k_test",
    )

    assert single_eval_output_path(args) == (
        tmp_path / "run" / "flow_eval_metrics_sghosts_cf_gsm8k_1k_test.json"
    )

    args.output_path = str(tmp_path / "metrics.json")
    assert single_eval_output_path(args) == tmp_path / "metrics_sghosts_cf_gsm8k_1k_test.json"


def test_load_flow_training_module_uses_parent_config_for_checkpoint(fake_wrapper, tmp_path):
    vae_dir = tmp_path / "vae"
    write_vae_checkpoint(vae_dir, hidden_size=fake_wrapper.model.config.hidden_size)
    run_dir = tmp_path / "run"
    write_flow_checkpoint(run_dir, fake_wrapper, vae_dir)
    checkpoint_dir = run_dir / "checkpoint-1"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "pytorch_model.bin").write_bytes((run_dir / "pytorch_model.bin").read_bytes())

    module, config = load_flow_training_module(checkpoint_dir, frozen_lm=fake_wrapper, device=torch.device("cpu"))

    assert isinstance(module, SingleExpertFlowTrainingModule)
    assert config["model_args"]["vae_dir"] == str(vae_dir)
    assert module.drafter.config.draft_length == 2

def test_select_checkpoint_stride_keeps_every_nth_checkpoint(tmp_path):
    checkpoints = [tmp_path / f"checkpoint-{step}" for step in [0, 5, 10, 15, 20]]

    selected = select_checkpoint_stride(checkpoints, stride=2)

    assert [path.name for path in selected] == ["checkpoint-0", "checkpoint-10", "checkpoint-20"]


def test_select_checkpoint_stride_rejects_non_positive_stride(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_stride"):
        select_checkpoint_stride([tmp_path / "checkpoint-0"], stride=0)

