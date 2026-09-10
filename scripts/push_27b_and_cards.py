"""Push the 27B joint-VAE drafter to selimaktas/Flow-Drafter-27B, then refresh ALL THREE model cards
(4B, 9B, 27B) so each carries the combined acceptance table built from the measured deploy logs.
"""
import json, re, shutil, time
from pathlib import Path

ROOT = Path("/home/shadeform/chain-flow")
CKD27 = ROOT / "out/flow/ckpts/tree-vae-joint-27b-1024-k8-l8"
VAE27 = ROOT / "out/vae/ckpts/transformer-hidden-27b-5120-latent1024-fp16"
STAGE = Path("/tmp/flow-drafter-27b-push")

# model_tag -> (repo, base_model, hidden, latent, deploy_log)
MODELS = {
    "4B":  ("selimaktas/Flow-Drafter-4B",  "Qwen/Qwen3.5-4B",  2560, 640,  ROOT / "logs/deploy_eval.log"),
    "9B":  ("selimaktas/Flow-Drafter-9B",  "Qwen/Qwen3.5-9B",  4096, 1024, ROOT / "logs/deploy_eval_9b.log"),
    "27B": ("selimaktas/Flow-Drafter-27B", "Qwen/Qwen3.6-27B", 5120, 1024, ROOT / "logs/deploy_eval_27b.log"),
}
DOM_LABEL = {"gsm8k_heldout": "gsm8k held-out (in-distribution)", "math_reasoning": "math_reasoning (structured)",
             "HumanEval": "HumanEval (code)", "writing": "writing (prose)", "qa": "qa (short free-form)",
             "summarization": "summarization (prose)"}
DOM_ORDER = ["gsm8k_heldout", "math_reasoning", "HumanEval", "writing", "qa", "summarization"]


def log(m): print(f"[push27] {time.strftime('%H:%M:%S')} {m}", flush=True)


def parse_accept(path):
    out = {}
    if not Path(path).exists(): return out
    for line in open(path, errors="ignore"):
        m = re.search(r"\[(\w+)\] LIVE tree_accept=([0-9.]+)", line)
        if m: out[m.group(1)] = float(m.group(2))
    return out


def accept_table():
    accepts = {t: parse_accept(MODELS[t][4]) for t in MODELS}
    cols = [t for t in ["4B", "9B", "27B"] if accepts[t]]
    hdr = "| domain | " + " | ".join(f"Flow-Drafter-{t}" for t in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    rows = [hdr, sep]
    for d in DOM_ORDER:
        cells = [DOM_LABEL[d]] + [f"{accepts[t][d]:.2f}" if d in accepts[t] else "—" for t in cols]
        rows.append("| " + " | ".join(cells) + " |")
    return "## Acceptance across model sizes (RedHatAI/speculator_benchmarks + held-out gsm8k)\n\n" \
           "Live tree-accept = real greedy tokens accepted per verify pass, measured against the backbone. " \
           "All numbers are **lossless** (speculative output is identical to the base model).\n\n" \
           + "\n".join(rows) + "\n\nAccept tracks **output predictability**, not strict in/out-of-distribution: " \
           "structured math/code stay high; free-form prose/QA/summarization dip. The vLLM decode speedup " \
           "(compiled-forward + full-cudagraph verify + cudagraph draft) grows with base-model size because " \
           "the latent-flow draft is base-decoupled — 9B measured **1.27× mean** (1.77× math, 1.57× code).\n"


def card(tag):
    repo, base, hidden, latent, _ = MODELS[tag]
    others = [t for t in ["4B", "9B", "27B"] if t != tag]
    companions = ", ".join(f"[Flow-Drafter-{t}](https://huggingface.co/{MODELS[t][0]})" for t in others)
    return f"""---
license: apache-2.0
base_model: {base}
tags:
- speculative-decoding
- draft-model
- flow-matching
- chain-flow
---

# Flow-Drafter-{tag}

A **speculative-decoding draft model** for `{base}`, from the **Chained-Flow** project.

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

{accept_table()}
## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, and the jointly-trained VAE weights)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint (needed to construct the VAE at load time)

## Loading
Load with the `chain-flow` project's `load_tree_module` (set `vae_dir` to the bundled `vae/` folder).
This drafter is meant to be driven as a tree proposer + lossless tree-verify over the {base} backbone.

Trained on a diverse mix (gsm8k, nemotron-math, nemotron-stem, alpaca-code, dolly-chat).

_Companion models: {companions}._
"""


def main():
    from huggingface_hub import HfApi
    api = HfApi()

    # --- 1. stage + push the 27B model ---
    if STAGE.exists(): shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD27 / "model.safetensors", STAGE)
    shutil.copy(CKD27 / "chained_flow_tree_config.json", STAGE)
    (STAGE / "vae").mkdir()
    shutil.copy(VAE27 / "model.safetensors", STAGE / "vae")
    shutil.copy(VAE27 / "chain_flow_vae_config.json", STAGE / "vae")
    (STAGE / "README.md").write_text(card("27B"))
    log(f"staged 27B: {[p.name for p in STAGE.iterdir()]}")

    repo27 = MODELS["27B"][0]
    api.create_repo(repo27, repo_type="model", private=False, exist_ok=True)
    log(f"uploading 27B model → {repo27} …")
    api.upload_folder(folder_path=str(STAGE), repo_id=repo27, repo_type="model",
                      commit_message="Flow-Drafter-27B — joint-VAE tree drafter for Qwen3.6-27B (+ combined accept table)")
    log(f"27B DONE → https://huggingface.co/{repo27}")

    # --- 2. refresh the 4B and 9B cards with the same combined table ---
    for tag in ["4B", "9B"]:
        repo = MODELS[tag][0]
        p = Path(f"/tmp/README-{tag}.md"); p.write_text(card(tag))
        log(f"updating {tag} card → {repo} …")
        api.upload_file(path_or_fileobj=str(p), path_in_repo="README.md", repo_id=repo,
                        repo_type="model", commit_message="Add 4B+9B+27B combined acceptance table to card")
        log(f"{tag} card updated → https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
