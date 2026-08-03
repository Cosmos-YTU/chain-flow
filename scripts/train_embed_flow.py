from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformers import HfArgumentParser, TrainingArguments
from chained_flow.training.train_chunked_flow import FlowLossArguments, TeacherDataArguments
from chained_flow.training.train_embed_flow import EmbedModelArguments, train_embed_with_trainer

def main() -> None:
    parser = HfArgumentParser((EmbedModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        margs, dargs, largs, targs = parser.parse_yaml_file(yaml_file=str(Path(sys.argv[1]).resolve()))
    else:
        margs, dargs, largs, targs = parser.parse_args_into_dataclasses()
    print(train_embed_with_trainer(margs, dargs, largs, targs))

if __name__ == "__main__":
    main()
