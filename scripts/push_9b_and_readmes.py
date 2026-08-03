"""Push the 9B joint-VAE drafter to selimaktas/Flow-Drafter-9B, and refresh BOTH the 4B and 9B
model cards so each one carries the acceptance table for both models (measured, lossless).
"""
import json, shutil, time
from pathlib import Path

ROOT = Path("/home/shadeform/chained-flow")
CKD9 = ROOT / "out/flow/ckpts/tree-vae-joint-9b-1024-k8-l8"
VAE9 = ROOT / "out/vae/ckpts/transformer-hidden-9b-4096-latent1024-fp16"
REPO_4B = "selimaktas/Flow-Drafter-4B"
REPO_9B = "selimaktas/Flow-Drafter-9B"
STAGE = Path("/tmp/flow-drafter-9b-push")


def log(m): print(f"[push] {time.strftime('%H:%M:%S')} {m}", flush=True)


# Measured live tree-accept (tokens accepted per verify pass), from docs/deployment_results.md.
# RedHatAI/speculator_benchmarks domains + held-out gsm8k. All lossless (spec output == base).
ACCEPT_TABLE = """## Acceptance — 4B & 9B (RedHatAI/speculator_benchmarks + held-out gsm8k)

Live tree-accept = real greedy tokens accepted per verify pass, measured against the backbone.
All numbers are **lossless** (speculative output is identical to the base model).

| domain | Flow-Drafter-4B | Flow-Drafter-9B |
|---|---|---|
| gsm8k held-out (in-distribution) | 5.25 | 5.70 |
| math_reasoning (structured) | 5.44 | 5.25 |
| HumanEval (code) | 4.02 | 4.27 |
| writing (prose) | 3.28 | 3.55 |
| qa (short free-form) | 2.38 | 2.69 |
| summarization (prose) | 2.05 | 2.30 |

Accept tracks **output predictability**, not strict in/out-of-distribution: structured math/code stay
high; free-form prose/QA/summarization dip. In vLLM (compiled-forward + full-cudagraph verify +
cudagraph draft) the 9B drafter reaches **1.27× mean** decode speedup (1.77× math, 1.57× code), and the
win grows with base-model size because the latent-flow draft is base-decoupled.
"""


def card(model_tag, base_model, hidden, latent, repo_self):
    return f"""---
license: apache-2.0
base_model: {base_model}
tags:
- speculative-decoding
- draft-model
- flow-matching
- chained-flow
---

# Flow-Drafter-{model_tag}

A **speculative-decoding draft model** for `{base_model}`, from the **Chained-Flow** project.

Instead of drafting tokens one-at-a-time (a chain), it predicts **all K future hidden states in a
single flow pass**, expands them into a **draft tree**, and the base model verifies the whole tree in
**one forward pass** — accepting the longest path it agrees with (lossless).

## Architecture — joint-VAE tree flow drafter
- A **transformer-hidden VAE** compresses the {hidden}-d backbone hidden into a **{latent}-d latent**; the
  flow runs in that small latent (cheap, base-decoupled).
- A **flow-matching drafter** predicts K=8 future latents in 2 Euler steps, decoded back to hiddens.
- **Token feedback** at zero sequential cost: a low-rank **Markov head** (`bias = W2(W1[prev])`) plus
  path-conditioned residuals correct each branch by its chosen token.
- The VAE is **unfrozen and trained jointly with the flow for acceptance** (not reconstruction), with a
  small reconstruction anchor — the recipe that made the latent flow work at scale.

{ACCEPT_TABLE}
## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, and the jointly-trained VAE weights)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint (needed to construct the VAE at load time)

## Loading
Load with the `chained-flow` project's `load_tree_module` (set `vae_dir` to the bundled `vae/` folder).
This drafter is meant to be driven as a tree proposer + lossless tree-verify over the {base_model} backbone.

Trained on a diverse mix (gsm8k, nemotron-math, nemotron-stem, alpaca-code, dolly-chat).

_Companion model: {"[Flow-Drafter-4B](https://huggingface.co/selimaktas/Flow-Drafter-4B)" if model_tag == "9B" else "[Flow-Drafter-9B](https://huggingface.co/selimaktas/Flow-Drafter-9B)"})._
"""


def main():
    from huggingface_hub import HfApi
    api = HfApi()

    # --- 1. stage + push the 9B model ---
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD9 / "model.safetensors", STAGE)
    shutil.copy(CKD9 / "chained_flow_tree_config.json", STAGE)
    (STAGE / "vae").mkdir()
    shutil.copy(VAE9 / "model.safetensors", STAGE / "vae")
    shutil.copy(VAE9 / "chained_flow_vae_config.json", STAGE / "vae")
    (STAGE / "README.md").write_text(card("9B", "Qwen/Qwen3.5-9B", 4096, 1024, REPO_9B))
    log(f"staged 9B: {[p.name for p in STAGE.iterdir()]}")

    api.create_repo(REPO_9B, repo_type="model", exist_ok=True)
    log(f"uploading 9B model → {REPO_9B} …")
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO_9B, repo_type="model",
                      commit_message="Flow-Drafter-9B — joint-VAE tree drafter for Qwen3.5-9B (+ 4B/9B accept table)")
    log(f"9B DONE → https://huggingface.co/{REPO_9B}")

    # --- 2. refresh the 4B card so it also carries the combined accept table ---
    card4 = card("4B", "Qwen/Qwen3.5-4B", 2560, 640, REPO_4B)
    p4 = Path("/tmp/README-4b.md"); p4.write_text(card4)
    log(f"updating 4B card → {REPO_4B} …")
    api.upload_file(path_or_fileobj=str(p4), path_in_repo="README.md", repo_id=REPO_4B,
                    repo_type="model", commit_message="Add 4B+9B acceptance table to card")
    log(f"4B card updated → https://huggingface.co/{REPO_4B}")


if __name__ == "__main__":
    main()
