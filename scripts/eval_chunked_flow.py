from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import HfArgumentParser

from chained_flow.training.eval_chunked_flow import ChunkedFlowEvalArguments, evaluate_flow


def main() -> None:
    parser = HfArgumentParser(ChunkedFlowEvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (args,) = parser.parse_yaml_file(yaml_file=str(Path(sys.argv[1]).resolve()))
    else:
        (args,) = parser.parse_args_into_dataclasses()
    evaluate_flow(args)


if __name__ == "__main__":
    main()
