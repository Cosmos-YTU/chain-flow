from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import HfArgumentParser, TrainingArguments

from chained_flow.training.train_vae import (
    VAEDataArguments,
    VAELossArguments,
    VAEModelArguments,
    train_vae_with_trainer,
)


def main() -> None:
    parser = HfArgumentParser((VAEModelArguments, VAEDataArguments, VAELossArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        model_args, data_args, loss_args, training_args = parser.parse_yaml_file(
            yaml_file=str(Path(sys.argv[1]).resolve())
        )
    else:
        model_args, data_args, loss_args, training_args = parser.parse_args_into_dataclasses()

    result = train_vae_with_trainer(model_args, data_args, loss_args, training_args)
    print(result)


if __name__ == "__main__":
    main()
