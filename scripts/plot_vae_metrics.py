from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IGNORED_KEYS = {
    "epoch",
    "step",
    "total_flos",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
    "eval_runtime",
    "eval_samples_per_second",
    "eval_steps_per_second",
}

LOSS_COMPONENT_NAMES = {
    "hidden.mse",
    "hidden.cos",
    "hidden.norm",
    "latent.kl",
    "logit.kl",
}

DEFAULT_COMPARISON_METRICS = [
    "loss",
    "hidden.mse",
    "hidden.cos",
    "hidden.norm",
    "latent.kl",
]


def is_loss_metric(name: str) -> bool:
    normalized = name.removeprefix("eval_")
    return (
        normalized == "loss"
        or normalized.endswith("_loss")
        or normalized in LOSS_COMPONENT_NAMES
        or normalized.startswith("loss_component/")
    )


def load_log_history(output_dir: str | Path) -> list[dict[str, Any]]:
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return list(state.get("log_history", []))


def metric_series(log_history: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for name, value in entry.items():
            if name in IGNORED_KEYS or not is_loss_metric(name) or not isinstance(value, int | float):
                continue
            steps, values = series.setdefault(name, ([], []))
            steps.append(int(step))
            values.append(float(value))
    return series


def plot_metrics(
    output_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    metrics: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_path = Path(output_path) if output_path else output_dir / "vae_metrics.png"
    series = metric_series(load_log_history(output_dir))
    selected = metrics or sorted(series)
    selected = [name for name in selected if name in series]
    if not selected:
        raise ValueError("no matching numeric metrics found in trainer_state.json")

    figure, axis = plt.subplots(figsize=(10, 6))
    for name in selected:
        steps, values = series[name]
        axis.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=name)
    axis.set_xlabel("step")
    axis.set_ylabel("metric")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _metric_filename(metric: str) -> str:
    return metric.replace("/", "_").replace(".", "_") + ".png"


def plot_metric_comparisons(
    output_dirs: list[str | Path],
    *,
    labels: list[str] | None = None,
    output_dir: str | Path,
    metrics: list[str] | None = None,
) -> list[Path]:
    import matplotlib.pyplot as plt

    if not output_dirs:
        raise ValueError("at least one output_dir is required")
    if labels is not None and len(labels) != len(output_dirs):
        raise ValueError("labels length must match output_dirs length")

    run_labels = labels or [Path(path).name for path in output_dirs]
    run_series = [metric_series(load_log_history(path)) for path in output_dirs]
    selected = metrics or DEFAULT_COMPARISON_METRICS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for metric in selected:
        eval_metric = f"eval_{metric}"
        has_any = any(metric in series or eval_metric in series for series in run_series)
        if not has_any:
            continue

        figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
        train_axis, eval_axis = axes
        for label, series in zip(run_labels, run_series, strict=True):
            if metric in series:
                steps, values = series[metric]
                train_axis.plot(steps, values, linewidth=1.5, marker="o", markersize=2.5, label=label)
            if eval_metric in series:
                steps, values = series[eval_metric]
                eval_axis.plot(steps, values, linewidth=1.5, marker="o", markersize=2.5, label=label)
        train_axis.set_title(f"train {metric}")
        eval_axis.set_title(f"validation {metric}")
        for axis in axes:
            axis.set_xlabel("step")
            axis.set_ylabel(metric)
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", fontsize="x-small")
        figure.tight_layout()
        path = output_dir / _metric_filename(metric)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        saved.append(path)

    if not saved:
        raise ValueError("no matching numeric metrics found in trainer_state.json files")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VAE Trainer metrics from trainer_state.json.")
    parser.add_argument("output_dirs", nargs="+")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    args = parser.parse_args()

    if len(args.output_dirs) == 1 and args.output_dir is None:
        path = plot_metrics(args.output_dirs[0], output_path=args.output_path, metrics=args.metrics)
        print(f"VAE metric plot saved: {path}")
        return

    if args.output_path is not None:
        raise ValueError("--output-path is only supported for single-run plots; use --output-dir for comparisons")
    paths = plot_metric_comparisons(
        args.output_dirs,
        labels=args.labels,
        output_dir=args.output_dir or "tmp/vae_metric_plots",
        metrics=args.metrics,
    )
    for path in paths:
        print(f"VAE comparison plot saved: {path}")


if __name__ == "__main__":
    main()
