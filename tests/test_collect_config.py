from pathlib import Path
import importlib.util


def load_script_module():
    path = Path("scripts/collect_teacher_states.py").resolve()
    spec = importlib.util.spec_from_file_location("collect_teacher_states", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_gsm8k_collection_yaml_parses():
    module = load_script_module()
    args = module.load_yaml_args(Path("collect_configs/smoke_gsm8k.yaml"))
    assert args.dataset_name == "openai/gsm8k"
    assert args.limit == 10
    assert args.dataset_start == 0
    assert args.dataset_end is None
    assert args.local_files_only is True
    assert args.device is None
    assert args.dtype == "float16"
    assert args.seed == 0
    assert args.output_dir == Path("teacher_states/gsm8k-qwen35-08b-smoke")
    assert args.tmp_output_dir == Path("teacher_states/_tmp_gsm8k-qwen35-08b-smoke")
    assert args.tmp_push_to_hub is None
    assert args.push_to_hub is None
    assert args.answer_dataset_path is None
    assert args.answer_dataset_split is None


def test_64_gsm8k_collection_yaml_parses_split_batch_sizes():
    module = load_script_module()
    args = module.load_yaml_args(Path("collect_configs/64_train_gsm8k.yaml"))
    assert args.dataset_start == 0
    assert args.dataset_end == 64
    assert args.limit is None
    assert args.batch_size == 64
    assert args.generation_batch_size == 64
    assert args.hidden_batch_size == 8


def test_resume_collection_yaml_keeps_final_push_target():
    module = load_script_module()
    args = module.load_yaml_args(Path("collect_configs/64_train_gsm8k_resume.yaml"))

    assert args.answer_dataset_path == "sghosts/_tmp_cf_gsm8k_64_train_v1"
    assert args.push_to_hub == "sghosts/cf_gsm8k_64_train_v1"
