from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from transformers import HfArgumentParser, TrainingArguments
from chain_flow.training.train_chunked_flow import FlowLossArguments, TeacherDataArguments
from chain_flow.training.train_markov2_flow import Markov2ModelArguments, train_markov2_with_trainer
def main():
    p = HfArgumentParser((Markov2ModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
    if len(sys.argv)==2 and sys.argv[1].endswith((".yaml",".yml")):
        m,d,l,t = p.parse_yaml_file(yaml_file=str(Path(sys.argv[1]).resolve()))
    else:
        m,d,l,t = p.parse_args_into_dataclasses()
    print(train_markov2_with_trainer(m,d,l,t))
if __name__=="__main__": main()
