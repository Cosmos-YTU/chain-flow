import torch
import pytest

from chain_flow.training.eval_vae import (
    VAEEvalArguments,
    checkpoint_eval_output_path,
    dataset_eval_slug,
    discover_vae_checkpoints,
    find_vae_config_dir,
    load_flow_cache_hidden_tokens,
    single_eval_output_path,
    logit_kl_divergence,
    per_token_logit_kl_divergence,
    per_token_vae_metrics,
    select_checkpoint_stride,
    summarize_metric,
    torch_dtype_from_string,
)


def test_logit_kl_divergence_is_zero_for_matching_logits():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    assert logit_kl_divergence(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_torch_dtype_from_string_accepts_float16_alias():
    assert torch_dtype_from_string("fp16") is torch.float16


def test_per_token_logit_kl_divergence_returns_one_value_per_token():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])

    kl = per_token_logit_kl_divergence(logits, logits)

    assert kl.shape == (2,)
    assert kl.tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_summarize_metric_reports_distribution_stats():
    summary = summarize_metric(torch.tensor([1.0, 2.0, 3.0]))

    assert summary["mean"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(0.816496, rel=1e-5)
    assert summary["min"] == 1.0
    assert summary["max"] == 3.0
    assert summary["p50"] == 2.0
    assert set(summary) == {"mean", "std", "min", "max", "p50", "p90", "p95", "p99"}


def test_per_token_vae_metrics_reports_all_eval_metrics():
    hidden = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    metrics = per_token_vae_metrics(
        hidden,
        hidden,
        mu=torch.zeros(2, 2),
        logvar=torch.zeros(2, 2),
        real_logits=logits,
        recon_logits=logits,
    )

    assert set(metrics) == {
        "hidden.mse",
        "hidden.rel_mse",
        "hidden.rel_rmse",
        "hidden.cos",
        "hidden.cosine_similarity",
        "hidden.norm",
        "latent.kl",
        "logit.kl",
        "logit.js_div",
        "logit.ce_delta",
        "token.match",
        "token.top1_match",
        "token.top5_match",
        "token.top10_match",
        "token.real_top1_rank_in_recon",
        "token.recon_prob_on_real_top1",
        "token.prob_ratio_on_real_top1",
    }
    assert metrics["hidden.mse"].tolist() == [0.0, 0.0]
    assert metrics["hidden.rel_mse"].tolist() == [0.0, 0.0]
    assert metrics["hidden.rel_rmse"].tolist() == [0.0, 0.0]
    assert metrics["hidden.cosine_similarity"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["logit.js_div"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert metrics["logit.ce_delta"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)
    assert metrics["token.match"].tolist() == [1.0, 1.0]
    assert metrics["token.top1_match"].tolist() == [1.0, 1.0]
    assert metrics["token.top5_match"].tolist() == [1.0, 1.0]
    assert metrics["token.top10_match"].tolist() == [1.0, 1.0]
    assert metrics["token.real_top1_rank_in_recon"].tolist() == [1.0, 1.0]
    assert metrics["token.prob_ratio_on_real_top1"].tolist() == pytest.approx([1.0, 1.0])


def test_find_vae_config_dir_uses_parent_for_trainer_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-20"
    checkpoint_dir.mkdir(parents=True)
    (run_dir / "chain_flow_vae_config.json").write_text("{}", encoding="utf-8")

    assert find_vae_config_dir(checkpoint_dir) == run_dir


def test_discover_vae_checkpoints_orders_numeric_checkpoints_with_weights(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoint-20").mkdir(parents=True)
    (run_dir / "checkpoint-20" / "model.safetensors").touch()
    (run_dir / "checkpoint-3").mkdir()
    (run_dir / "checkpoint-3" / "pytorch_model.bin").touch()
    (run_dir / "checkpoint-10").mkdir()

    checkpoints = discover_vae_checkpoints(run_dir)

    assert [path.name for path in checkpoints] == ["checkpoint-3", "checkpoint-20"]


def test_dataset_eval_slug_sanitizes_hub_dataset_names():
    assert dataset_eval_slug("sghosts/cf_gsm8k_1k_test") == "sghosts_cf_gsm8k_1k_test"


def test_checkpoint_eval_output_path_includes_dataset_and_checkpoint_name(tmp_path):
    args = VAEEvalArguments(vae_dir=str(tmp_path / "run"), dataset_path="sghosts/cf_gsm8k_1k_test")
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoint-3"

    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        run_dir / "vae_eval_metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )

    args.output_path = str(tmp_path / "metrics.json")
    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        tmp_path / "metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )

    args.output_path = str(tmp_path / "evals")
    assert checkpoint_eval_output_path(args, checkpoint_dir, run_dir=run_dir) == (
        tmp_path / "evals" / "vae_eval_metrics_sghosts_cf_gsm8k_1k_test_checkpoint-3.json"
    )


def test_single_eval_output_path_includes_dataset_name(tmp_path):
    args = VAEEvalArguments(
        vae_dir=str(tmp_path / "run"),
        dataset_path="sghosts/cf_gsm8k_1k_test",
    )

    assert single_eval_output_path(args) == (
        tmp_path / "run" / "vae_eval_metrics_sghosts_cf_gsm8k_1k_test.json"
    )

    args.output_path = str(tmp_path / "metrics.json")
    assert single_eval_output_path(args) == tmp_path / "metrics_sghosts_cf_gsm8k_1k_test.json"

def test_select_checkpoint_stride_keeps_every_nth_vae_checkpoint(tmp_path):
    checkpoints = [tmp_path / f"checkpoint-{step}" for step in [0, 5, 10, 15, 20]]

    selected = select_checkpoint_stride(checkpoints, stride=2)

    assert [path.name for path in selected] == ["checkpoint-0", "checkpoint-10", "checkpoint-20"]


def test_select_checkpoint_stride_rejects_non_positive_vae_stride(tmp_path):
    with pytest.raises(ValueError, match="checkpoint_stride"):
        select_checkpoint_stride([tmp_path / "checkpoint-0"], stride=0)



def test_load_flow_cache_hidden_tokens_respects_response_only(tmp_path):
    from chain_flow.training.window_dataset import FLOW_CACHE_FILES, FLOW_CACHE_METADATA
    import json

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    hidden = torch.arange(5 * 2, dtype=torch.float32).reshape(5, 2)
    torch.save(hidden, cache_dir / FLOW_CACHE_FILES["hidden"])
    torch.save(torch.arange(5), cache_dir / FLOW_CACHE_FILES["input_ids"])
    torch.save(torch.tensor([0, 3]), cache_dir / FLOW_CACHE_FILES["row_offsets"])
    torch.save(torch.tensor([3, 2]), cache_dir / FLOW_CACHE_FILES["row_lengths"])
    torch.save(torch.tensor([1, 1]), cache_dir / FLOW_CACHE_FILES["prompt_lengths"])
    torch.save(torch.tensor([0, 1]), cache_dir / FLOW_CACHE_FILES["original_row_indices"])
    (cache_dir / FLOW_CACHE_METADATA).write_text(json.dumps({"cache_type": "test"}), encoding="utf-8")

    response_tokens = load_flow_cache_hidden_tokens(cache_dir, response_only=True)
    all_tokens = load_flow_cache_hidden_tokens(cache_dir, response_only=False)

    assert response_tokens.shape == (3, 2)
    assert response_tokens.tolist() == hidden[[1, 2, 4]].tolist()
    assert all_tokens.shape == (5, 2)
