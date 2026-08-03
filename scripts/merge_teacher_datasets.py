from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk


REQUIRED_COLUMNS = {
    "text",
    "prompt_text",
    "generated_text",
    "input_ids",
    "final_hidden",
    "example_id",
    "source",
    "split",
    "format_name",
    "model_id",
    "hidden_dtype",
    "num_tokens",
    "prompt_length",
}


def load_teacher_dataset(path_or_repo: str, *, split: str) -> Dataset:
    path = Path(path_or_repo)
    if path.exists():
        return load_from_disk(str(path))
    return load_dataset(path_or_repo, split=split)


def validate_teacher_dataset(dataset: Dataset, *, name: str) -> None:
    missing = REQUIRED_COLUMNS - set(dataset.column_names)
    if missing:
        raise ValueError(f"{name} is missing required teacher columns: {sorted(missing)}")


def merge_teacher_datasets(paths_or_repos: list[str], *, split: str = "train") -> Dataset:
    if not paths_or_repos:
        raise ValueError("at least one dataset must be provided")

    datasets: list[Dataset] = []
    for path_or_repo in paths_or_repos:
        print(f"loading teacher dataset: {path_or_repo} split={split}", flush=True)
        dataset = load_teacher_dataset(path_or_repo, split=split)
        validate_teacher_dataset(dataset, name=path_or_repo)
        print(f"loaded teacher dataset: {path_or_repo} rows={len(dataset)}", flush=True)
        datasets.append(dataset)

    if len(datasets) == 1:
        merged = datasets[0]
    else:
        print(f"merging teacher datasets: count={len(datasets)}", flush=True)
        merged = concatenate_datasets(datasets)
    print(f"merged teacher dataset: rows={len(merged)} columns={merged.column_names}", flush=True)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge teacher hidden-state datasets without preprocessing.")
    parser.add_argument("--datasets", nargs="+", required=True, help="HF repo ids or local save_to_disk directories.")
    parser.add_argument("--split", default="train", help="Split to load for HF datasets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional local save_to_disk destination.")
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id for the merged dataset.")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None and args.push_to_hub is None:
        raise ValueError("provide --output-dir, --push-to-hub, or both")

    merged = merge_teacher_datasets(args.datasets, split=args.split)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"saving merged teacher dataset: {args.output_dir}", flush=True)
        merged.save_to_disk(str(args.output_dir))
    if args.push_to_hub:
        print(f"pushing merged teacher dataset: {args.push_to_hub}", flush=True)
        merged.push_to_hub(args.push_to_hub, private=args.private)


if __name__ == "__main__":
    main()
