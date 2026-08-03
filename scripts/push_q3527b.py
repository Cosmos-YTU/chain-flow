"""Push the Qwen3.5-27B joint-VAE drafter to selimaktas/Flow-Drafter-Qwen3.5-27B with a card
carrying its measured accept table (parsed from logs/deploy_eval_q3527b.log)."""
import json, re, shutil, time
from pathlib import Path

ROOT = Path("/home/shadeform/chained-flow")
CKD = ROOT / "out/flow/ckpts/tree-vae-joint-q3527b-1024-k8-l8"
VAE = ROOT / "out/vae/ckpts/transformer-hidden-q3527b-5120-latent1024-fp16"
DEPLOY_LOG = ROOT / "logs/deploy_eval_q3527b.log"
REPO = "selimaktas/Flow-Drafter-Qwen3.5-27B"
STAGE = Path("/tmp/flow-drafter-q3527b-push")
DOM_LABEL = {"gsm8k_heldout": "gsm8k held-out (in-distribution)", "math_reasoning": "math_reasoning (structured)",
             "HumanEval": "HumanEval (code)", "writing": "writing (prose)", "qa": "qa (short free-form)",
             "summarization": "summarization (prose)"}
DOM_ORDER = ["gsm8k_heldout", "math_reasoning", "HumanEval", "writing", "qa", "summarization"]


def log(m): print(f"[push-q35] {time.strftime('%H:%M:%S')} {m}", flush=True)


def accept_table():
    acc = {}
    if DEPLOY_LOG.exists():
        for line in open(DEPLOY_LOG, errors="ignore"):
            m = re.search(r"\[(\w+)\] LIVE tree_accept=([0-9.]+)", line)
            if m: acc[m.group(1)] = float(m.group(2))
    if not acc:
        return "_(accept table pending)_\n"
    rows = ["| domain | live tree-accept |", "|---|---|"]
    for d in DOM_ORDER:
        if d in acc: rows.append(f"| {DOM_LABEL[d]} | {acc[d]:.2f} |")
    return ("## Acceptance (RedHatAI/speculator_benchmarks + held-out gsm8k)\n\n"
            "Live tree-accept = real greedy tokens accepted per verify pass vs the backbone; **lossless**.\n\n"
            + "\n".join(rows) + "\n")


def card():
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-27B
tags:
- speculative-decoding
- draft-model
- flow-matching
- chained-flow
---

# Flow-Drafter-Qwen3.5-27B

A **speculative-decoding draft model** for `Qwen/Qwen3.5-27B`, from the **Chained-Flow** project —
the Qwen3.5 sibling of [Flow-Drafter-27B](https://huggingface.co/selimaktas/Flow-Drafter-27B)
(which targets the post-trained Qwen3.6-27B). Same joint-VAE tree-flow recipe.

Predicts **all K future hidden states in a single flow pass**, expands them into a **draft tree**,
and the base model verifies the whole tree in **one forward pass** — accepting the longest agreed path (lossless).

## Architecture — joint-VAE tree flow drafter
- A **transformer-hidden VAE** compresses the 5120-d backbone hidden into a **1024-d latent**; the flow
  runs in that small latent (cheap, base-decoupled).
- A **flow-matching drafter** predicts K=8 future latents in 2 Euler steps, decoded back to hiddens.
- **Token feedback** at zero sequential cost: a low-rank **Markov head** plus path-conditioned residuals.
- The VAE is **unfrozen and trained jointly with the flow for acceptance** (not reconstruction), with a
  small reconstruction anchor.

{accept_table()}
## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, jointly-trained VAE weights)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint (needed to construct the VAE at load time)

Trained on a diverse mix (gsm8k, nemotron-math, nemotron-stem, alpaca-code, dolly-chat), 2-GPU DDP.
"""


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    if STAGE.exists(): shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD / "model.safetensors", STAGE)
    shutil.copy(CKD / "chained_flow_tree_config.json", STAGE)
    (STAGE / "vae").mkdir()
    shutil.copy(VAE / "model.safetensors", STAGE / "vae")
    shutil.copy(VAE / "chained_flow_vae_config.json", STAGE / "vae")
    (STAGE / "README.md").write_text(card())
    log(f"staged: {[p.name for p in STAGE.iterdir()]}")
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    log(f"uploading → {REPO} …")
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Flow-Drafter-Qwen3.5-27B — joint-VAE tree drafter (2-GPU DDP)")
    log(f"DONE → https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
