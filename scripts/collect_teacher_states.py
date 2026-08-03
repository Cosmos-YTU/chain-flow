from __future__ import annotations

import argparse
from pathlib import Path
import sys

from dotenv import load_dotenv
import yaml

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chained_flow.frozen_lm import DEFAULT_MODEL_ID
from chained_flow.training.collect_teacher import TeacherCollectionConfig, collect_teacher_dataset


def tmp_output_dir(output_dir: Path) -> Path:
    return output_dir.parent / f"_tmp_{output_dir.name}"


def tmp_hub_id(repo_id: str | None) -> str | None:
    if not repo_id:
        return None
    if "/" not in repo_id:
        return f"_tmp_{repo_id}"
    namespace, name = repo_id.rsplit("/", 1)
    return f"{namespace}/_tmp_{name}"


def load_yaml_args(path: Path) -> argparse.Namespace:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("collection YAML config must contain a mapping at the top level")
    data["output_dir"] = Path(data.get("output_dir", "teacher_states/gsm8k-qwen35-08b-smoke"))
    return argparse.Namespace(
        model_id=data.get("model_id", DEFAULT_MODEL_ID),
        dataset_name=data.get("dataset_name", "gsm8k"),
        dataset_config=data.get("dataset_config", "main"),
        split=data.get("split", "train"),
        source=data.get("source", "gsm8k"),
        format_name=data.get("format_name", "qwen_chat_qa"),
        limit=data.get("limit"),
        dataset_start=data.get("dataset_start", 0),
        dataset_end=data.get("dataset_end"),
        streaming=bool(data.get("streaming", False)),
        stream_take=data.get("stream_take"),
        data_files=data.get("data_files"),
        target_layer_ids=data.get("target_layer_ids"),
        max_tokens=data.get("max_tokens"),
        generation_max_new_tokens=data.get("generation_max_new_tokens", 4096),
        batch_size=data.get("batch_size", 1),
        generation_batch_size=data.get("generation_batch_size"),
        hidden_batch_size=data.get("hidden_batch_size"),
        storage_dtype=data.get("storage_dtype", "float32"),
        local_files_only=bool(data.get("local_files_only", False)),
        device=data.get("device"),
        dtype=data.get("dtype"),
        seed=data.get("seed", 0),
        output_dir=data["output_dir"],
        tmp_output_dir=Path(data["tmp_output_dir"]) if data.get("tmp_output_dir") else tmp_output_dir(data["output_dir"]),
        push_to_hub=data.get("push_to_hub"),
        tmp_push_to_hub=data.get("tmp_push_to_hub") or tmp_hub_id(data.get("push_to_hub")),
        answer_dataset_path=data.get("answer_dataset_path"),
        answer_dataset_split=data.get("answer_dataset_split"),
        private=bool(data.get("private", False)),
    )


def parse_args() -> argparse.Namespace:
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        return load_yaml_args(Path(sys.argv[1]))

    parser = argparse.ArgumentParser(description="Collect K-independent frozen-LM teacher hidden states.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset-name", default="gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="train")
    parser.add_argument("--source", default="gsm8k")
    parser.add_argument("--format-name", default="qwen_chat_qa")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--dataset-end", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--stream-take", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=None)
    parser.add_argument("--hidden-batch-size", type=int, default=None)
    parser.add_argument(
        "--storage-dtype",
        choices=["float32", "float16"],
        default="float32",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default=None, help="Device for model loading, e.g. cuda, cuda:0, cpu, mps, or auto.")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("teacher_states/gsm8k-qwen35-08b-smoke"),
    )
    parser.add_argument("--tmp-output-dir", type=Path, default=None)
    parser.add_argument("--push-to-hub", default=None, help="Optional HF repo id, e.g. user/dataset-name.")
    parser.add_argument("--tmp-push-to-hub", default=None, help="Optional temp HF repo id for generated answers only.")
    parser.add_argument(
        "--answer-dataset-path",
        default=None,
        help="Optional local save_to_disk path or HF repo id for a temporary answer-only dataset.",
    )
    parser.add_argument("--answer-dataset-split", default=None, help="Split for HF answer datasets; defaults to train.")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    args.tmp_output_dir = args.tmp_output_dir or tmp_output_dir(args.output_dir)
    args.tmp_push_to_hub = args.tmp_push_to_hub or tmp_hub_id(args.push_to_hub)
    return args


def main() -> None:
    args = parse_args()
    config = TeacherCollectionConfig(
        model_id=args.model_id,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        source=args.source,
        format_name=args.format_name,
        limit=args.limit,
        dataset_start=args.dataset_start,
        dataset_end=args.dataset_end,
        streaming=getattr(args, "streaming", False),
        stream_take=getattr(args, "stream_take", None),
        data_files=getattr(args, "data_files", None),
        target_layer_ids=getattr(args, "target_layer_ids", None),
        max_tokens=args.max_tokens,
        generation_max_new_tokens=args.generation_max_new_tokens,
        batch_size=args.batch_size,
        generation_batch_size=args.generation_batch_size,
        hidden_batch_size=args.hidden_batch_size,
        storage_dtype=args.storage_dtype,
        local_files_only=args.local_files_only,
        device=args.device,
        dtype=args.dtype,
        seed=args.seed,
        tmp_output_dir=str(args.tmp_output_dir) if args.tmp_output_dir else None,
        tmp_push_to_hub=args.tmp_push_to_hub,
        push_to_hub=args.push_to_hub,
        answer_dataset_path=args.answer_dataset_path,
        answer_dataset_split=args.answer_dataset_split,
        private=args.private,
    )
    dataset, timings = collect_teacher_dataset(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(args.output_dir))

    print(f"saved={args.output_dir}")
    print(f"rows={len(dataset)}")
    print(f"columns={dataset.column_names}")
    print(f"timings={timings.scalar_components() if hasattr(timings, 'scalar_components') else timings.sections}")


if __name__ == "__main__":
    main()
