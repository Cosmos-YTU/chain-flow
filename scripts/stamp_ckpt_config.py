"""Write `chained_flow_tree_config.json` into a mid-training checkpoint directory.

`train_tree_with_trainer` only writes that file when the whole run finishes, but every consumer
(`load_tree_module`, and `diff_plugin_vs_harness.build_proposer`, which does NOT fall back to the
parent directory) needs it to rebuild the architecture. Regenerating it from the SAME yaml the run
was launched with is exact: the file is just `asdict()` of the three arg dataclasses.

  python scripts/stamp_ckpt_config.py train_configs/recovered/joint_9btr.yaml <ckpt-dir>...
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import HfArgumentParser, TrainingArguments  # noqa: E402

from chained_flow.training.train_chunked_flow import FlowLossArguments, TeacherDataArguments  # noqa: E402
from chained_flow.training.train_tree_flow import TreeModelArguments  # noqa: E402


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    yaml_path, ckpt_dirs = sys.argv[1], sys.argv[2:]
    p = HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
    m, d, l, _ = p.parse_yaml_file(yaml_file=str(Path(yaml_path).resolve()))
    payload = {"model_args": asdict(m), "data_args": asdict(d), "loss_args": asdict(l)}
    for c in ckpt_dirs:
        out = Path(c) / "chained_flow_tree_config.json"
        if not (Path(c) / "model.safetensors").exists():
            raise SystemExit(f"{c}: no model.safetensors")
        out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
