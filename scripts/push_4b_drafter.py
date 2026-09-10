"""Wait for the 4B joint-VAE drafter to finish training, then package + push it to
selimaktas/Flow-Drafter-4B on the HF Hub. Best-effort accept measurement for the card.
"""
import json, shutil, subprocess, sys, time, os
from pathlib import Path

ROOT = Path("/home/shadeform/chain-flow")
CKD = ROOT / "out/flow/ckpts/tree-vae-joint-4b-640-k8-l8"
VAE = ROOT / "out/vae/ckpts/transformer-hidden-4b-2560-latent640-fp16"
CFG_YAML = ROOT / "train_configs/recovered/joint_4b.yaml"
REPO = "selimaktas/Flow-Drafter-4B"
STAGE = Path("/tmp/flow-drafter-4b-push")


def log(m): print(f"[push] {time.strftime('%H:%M:%S')} {m}", flush=True)


def wait_for_final():
    log("waiting for 4B final model.safetensors …")
    while not (CKD / "model.safetensors").exists():
        time.sleep(120)
    time.sleep(30)  # let the final save + config dump settle
    log("final model present.")


def ensure_config():
    cfgp = CKD / "chained_flow_tree_config.json"
    if cfgp.exists():
        return
    log("synthesizing config from joint_4b.yaml")
    sys.path.insert(0, str(ROOT / "src"))
    import yaml, dataclasses
    from dataclasses import asdict
    from pathlib import Path as P
    from transformers import HfArgumentParser, TrainingArguments
    from chain_flow.training.train_tree_flow import TreeModelArguments
    from chain_flow.training.train_chunked_flow import TeacherDataArguments, FlowLossArguments
    p = HfArgumentParser((TreeModelArguments, TeacherDataArguments, FlowLossArguments, TrainingArguments))
    m, d, l, t = p.parse_yaml_file(yaml_file=str(CFG_YAML.resolve()))
    json.dump({"model_args": asdict(m), "data_args": asdict(d), "loss_args": asdict(l)}, open(cfgp, "w"), indent=2)


def measure_accept():
    """Best-effort: real greedy tree accept on GPU 1 (free)."""
    try:
        ed = Path("/tmp/eval-4b-final"); ed.mkdir(exist_ok=True)
        shutil.copy(CKD / "chained_flow_tree_config.json", ed)
        (ed / "model.safetensors").unlink(missing_ok=True)
        os.symlink(CKD / "model.safetensors", ed / "model.safetensors")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="1", PYTHONPATH="src",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        out = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "scripts/eval_tree_flow.py",
             "--flow_dir", str(ed), "--model_id", "Qwen/Qwen3.5-4B",
             "--dataset_path", "data/flow_cache/stage1_4b_mix5_k4", "--dtype", "float16",
             "--batch_size", "256", "--max_batches", "3", "--measure_tree"],
            cwd=ROOT, capture_output=True, text=True, timeout=900)
        for line in out.stdout.splitlines():
            if "LIVE tree_accept" in line:
                log("accept: " + line.strip())
                return line.strip()
    except Exception as e:
        log(f"accept measure skipped: {type(e).__name__} {e}")
    return None


def readme(accept_line):
    acc = ""
    if accept_line:
        import re
        m = re.search(r"tree_accept=([0-9.]+)", accept_line)
        if m:
            acc = f"\n- **Measured live tree-accept (in-domain):** ~{m.group(1)} tokens/step\n"
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- speculative-decoding
- draft-model
- flow-matching
- chain-flow
---

# Flow-Drafter-4B

A **speculative-decoding draft model** for `Qwen/Qwen3.5-4B`, from the **Chained-Flow** project.

Instead of drafting tokens one-at-a-time (a chain), it predicts **all K future hidden states in a
single flow pass**, expands them into a **draft tree**, and the base model verifies the whole tree in
**one forward pass** — accepting the longest path it agrees with (lossless).

## Architecture — joint-VAE tree flow drafter
- A **transformer-hidden VAE** compresses the 2560-d backbone hidden into a **640-d latent**; the flow
  runs in that small latent (cheap, base-decoupled).
- A **flow-matching drafter** predicts K=8 future latents in 2 Euler steps, decoded back to hiddens.
- **Token feedback** at zero sequential cost: a low-rank **Markov head** (`bias = W2(W1[prev])`) plus
  path-conditioned residuals correct each branch by its chosen token.
- The VAE is **unfrozen and trained jointly with the flow for acceptance** (not reconstruction), with a
  small reconstruction anchor — the recipe that made the latent flow work at scale.
{acc}
## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, and the jointly-trained VAE weights)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint (needed to construct the VAE at load time)

## Loading
Load with the `chain-flow` project's `load_tree_module` (set `vae_dir` to the bundled `vae/` folder).
This drafter is meant to be driven as a tree proposer + lossless tree-verify over the Qwen3.5-4B backbone.

Trained on a diverse mix (gsm8k, nemotron-math, nemotron-stem, alpaca-code, dolly-chat).
"""


def main():
    wait_for_final()
    ensure_config()
    acc = measure_accept()

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD / "model.safetensors", STAGE)
    shutil.copy(CKD / "chained_flow_tree_config.json", STAGE)
    (STAGE / "vae").mkdir()
    shutil.copy(VAE / "model.safetensors", STAGE / "vae")
    shutil.copy(VAE / "chain_flow_vae_config.json", STAGE / "vae")
    (STAGE / "README.md").write_text(readme(acc))
    log(f"staged: {[p.name for p in STAGE.iterdir()]}")

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    log(f"uploading to {REPO} …")
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Flow-Drafter-4B — joint-VAE tree drafter for Qwen3.5-4B")
    log(f"DONE → https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
