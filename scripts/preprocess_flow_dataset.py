from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.training.window_dataset import build_flow_window_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess teacher states into a reusable flow tensor cache.")
    parser.add_argument("--dataset-path", required=True, help="HF repo id or local save_to_disk teacher dataset.")
    parser.add_argument("--dataset-split", default="train", help="Split to load for HF datasets.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the flow tensor cache.")
    parser.add_argument("--draft-length", type=int, default=2, help="Draft length used to decide valid rows.")
    parser.add_argument(
        "--hidden-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        # NB float32 is the historical default but every Turkish/English flow cache in this repo
        # was built as float16, and concat_flow_caches refuses to merge mismatched dtypes. Callers
        # extending an existing cache must pass --hidden-dtype float16 explicitly.
        help="Hidden tensor dtype to store in the cache. Pass float16 to match the existing caches.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = build_flow_window_cache(
        args.dataset_path,
        args.output_dir,
        split=args.dataset_split,
        draft_length=args.draft_length,
        hidden_dtype=args.hidden_dtype,
        overwrite=args.overwrite,
    )
    print(f"saved={args.output_dir}")
    print(f"metadata={metadata}")


if __name__ == "__main__":
    main()
