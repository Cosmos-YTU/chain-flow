import importlib.util
import json
from pathlib import Path


def load_script_module():
    path = Path("scripts/plot_vae_metrics.py").resolve()
    spec = importlib.util.spec_from_file_location("plot_vae_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metric_series_extracts_loss_metrics_only():
    module = load_script_module()
    series = module.metric_series(
        [
            {"step": 1, "loss": 3.0, "hidden.mse": 4.0, "token.match": 0.5, "epoch": 0.1},
            {"step": 2, "eval_loss": 2.0, "eval_samples_per_second": 10.0, "train_runtime": 10.0},
        ]
    )

    assert series == {"loss": ([1], [3.0]), "hidden.mse": ([1], [4.0]), "eval_loss": ([2], [2.0])}


def test_load_log_history_reads_trainer_state(tmp_path):
    module = load_script_module()
    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(json.dumps({"log_history": [{"step": 1, "loss": 1.0}]}), encoding="utf-8")

    assert module.load_log_history(tmp_path) == [{"step": 1, "loss": 1.0}]


def test_plot_metric_comparisons_writes_train_val_figures(tmp_path):
    module = load_script_module()
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_a.mkdir()
    run_b.mkdir()
    state = {
        "log_history": [
            {"step": 1, "loss": 2.0, "hidden.mse": 3.0},
            {"step": 2, "eval_loss": 1.5, "eval_hidden.mse": 2.5},
        ]
    }
    (run_a / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")
    (run_b / "trainer_state.json").write_text(json.dumps(state), encoding="utf-8")

    paths = module.plot_metric_comparisons(
        [run_a, run_b],
        labels=["a", "b"],
        output_dir=tmp_path / "plots",
        metrics=["loss", "hidden.mse"],
    )

    assert [path.name for path in paths] == ["loss.png", "hidden_mse.png"]
    assert all(path.exists() for path in paths)
