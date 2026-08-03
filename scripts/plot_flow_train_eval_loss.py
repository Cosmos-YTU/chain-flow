from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_log_history(output_dir: str | Path) -> list[dict[str, Any]]:
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return list(state.get("log_history", []))


def _mean_by_epoch(points: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    buckets: dict[float, list[float]] = {}
    for epoch, value in points:
        buckets.setdefault(epoch, []).append(value)
    epochs = sorted(buckets)
    values = [sum(buckets[epoch]) / len(buckets[epoch]) for epoch in epochs]
    return epochs, values


def metric_series(log_history: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    points = {
        "train_loss": [],
        "eval_loss": [],
        "train_neg_expected_accept": [],
        "eval_neg_expected_accept": [],
    }
    for entry in log_history:
        epoch = entry.get("epoch")
        if not isinstance(epoch, int | float):
            continue
        epoch = float(epoch)
        if isinstance(entry.get("loss"), int | float):
            points["train_loss"].append((epoch, float(entry["loss"])))
        if isinstance(entry.get("eval_loss"), int | float):
            points["eval_loss"].append((epoch, float(entry["eval_loss"])))
        if isinstance(entry.get("verifier.expected_accept"), int | float):
            points["train_neg_expected_accept"].append((epoch, -float(entry["verifier.expected_accept"])))
        if isinstance(entry.get("eval_verifier.expected_accept"), int | float):
            points["eval_neg_expected_accept"].append((epoch, -float(entry["eval_verifier.expected_accept"])))
    return {key: _mean_by_epoch(value) for key, value in points.items()}


# Backwards-compatible name for older callers/tests.
def loss_series(log_history: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    series = metric_series(log_history)
    return {"train": series["train_loss"], "eval": series["eval_loss"]}

def run_label(output_dir: Path) -> str:
    return output_dir.name


def _plot_2x2_runs(runs: list[tuple[str, dict[str, tuple[list[float], list[float]]]]], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    panels = (
        (axes[0][0], "train_loss", "Train Loss", "loss"),
        (axes[0][1], "eval_loss", "Validation Loss", "loss"),
        (axes[1][0], "train_neg_expected_accept", "Train -Expected Acceptance", "-expected_acceptance"),
        (axes[1][1], "eval_neg_expected_accept", "Validation -Expected Acceptance", "-expected_acceptance"),
    )
    for axis, key, title, ylabel in panels:
        plotted = False
        for label, series in runs:
            epochs, values = series[key]
            if not epochs:
                continue
            axis.plot(epochs, values, marker="o", linewidth=1.6, markersize=2.5, label=label)
            plotted = True
        if not plotted:
            axis.text(0.5, 0.5, f"no {title.lower()} logged", ha="center", va="center", transform=axis.transAxes)
        else:
            axis.legend(fontsize=8)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def plot_flow_train_eval_loss(
    output_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_path = Path(output_path) if output_path else output_dir / "flow_train_eval_loss.png"
    series = metric_series(load_log_history(output_dir))
    if not any(values[0] for values in series.values()):
        raise ValueError("no train/eval loss or expected-acceptance entries found in trainer_state.json")
    return _plot_2x2_runs([(run_label(output_dir), series)], output_path)

def plot_multi_run_loss(
    output_dirs: list[str | Path],
    *,
    labels: list[str] | None = None,
    output_path: str | Path = "flow_train_eval_loss_runs.png",
) -> Path:
    dirs = [Path(output_dir) for output_dir in output_dirs]
    if not dirs:
        raise ValueError("at least one output directory is required")
    if labels is not None and len(labels) != len(dirs):
        raise ValueError("--labels must provide exactly one label per output directory")
    labels = labels or [run_label(output_dir) for output_dir in dirs]

    runs = []
    for output_dir, label in zip(dirs, labels, strict=True):
        series = metric_series(load_log_history(output_dir))
        runs.append((label, series))
    if not any(any(values[0] for values in series.values()) for _, series in runs):
        raise ValueError("no train/eval loss or expected-acceptance entries found in any trainer_state.json")
    return _plot_2x2_runs(runs, output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flow train loss and eval loss by epoch from trainer_state.json.")
    parser.add_argument("output_dirs", nargs="+", help="Flow training output directories containing trainer_state.json.")
    parser.add_argument("--output-path", default=None, help="Single-run two-panel output path. Only valid with one output dir.")
    parser.add_argument("--train-output-path", default=None, help="Deprecated; use --output-path for multi-run plots.")
    parser.add_argument("--eval-output-path", default=None, help="Deprecated; use --output-path for multi-run plots.")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels for multi-run plots.")
    args = parser.parse_args()

    if len(args.output_dirs) == 1:
        if args.labels is not None and len(args.labels) != 1:
            raise ValueError("--labels must provide exactly one label with one output directory")
        path = plot_flow_train_eval_loss(args.output_dirs[0], output_path=args.output_path)
        print(f"flow train/eval loss plot saved: {path}")
        return

    if args.train_output_path is not None or args.eval_output_path is not None:
        raise ValueError("--train-output-path and --eval-output-path are deprecated; use --output-path")
    path = plot_multi_run_loss(
        args.output_dirs,
        labels=args.labels,
        output_path=args.output_path or "flow_train_eval_loss_runs.png",
    )
    print(f"flow train/eval loss plot saved: {path}")


if __name__ == "__main__":
    main()
