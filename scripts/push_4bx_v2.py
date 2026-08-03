"""Push the expanded-data Qwen3.5-4B drafter to selimaktas/Flow-Drafter-4B-v2, with a card carrying
its accept table and the v1->v2 comparison. Only invoked by the bench arm when the result is 'good'."""
import json, re, shutil, time
from pathlib import Path

ROOT = Path("/home/shadeform/chained-flow")
CKD = ROOT / "out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8"
VAE = ROOT / "out/vae/ckpts/transformer-hidden-4bx-2560-latent640-fp16"
REPO = "selimaktas/Flow-Drafter-4B-v2"
STAGE = Path("/tmp/flow-drafter-4b-v2-push")
DOM_LABEL = {"gsm8k_heldout": "gsm8k held-out", "math_reasoning": "math_reasoning",
             "HumanEval": "HumanEval (code)", "writing": "writing (prose)", "qa": "qa (short free-form)",
             "summarization": "summarization (prose)"}
DOM_ORDER = ["gsm8k_heldout", "math_reasoning", "HumanEval", "writing", "qa", "summarization"]


def log(m): print(f"[push-v2] {time.strftime('%H:%M:%S')} {m}", flush=True)


def acc(p):
    d = {}
    if Path(p).exists():
        for l in open(p, errors="ignore"):
            m = re.search(r"\[(\w+)\] LIVE tree_accept=([0-9.]+)", l)
            if m: d[m.group(1)] = float(m.group(2))
    return d


def table():
    new = acc(ROOT / "logs/deploy_eval_4bx.log")
    old = acc(ROOT / "logs/deploy_eval.log")
    rows = ["| domain | v1 (5 tech sources) | v2 (10 diverse sources) | Δ |", "|---|---|---|---|"]
    for d in DOM_ORDER:
        if d in new:
            o = old.get(d)
            dd = f"{new[d]-o:+.2f}" if o is not None else "—"
            ov = f"{o:.2f}" if o is not None else "—"
            rows.append(f"| {DOM_LABEL[d]} | {ov} | {new[d]:.2f} | {dd} |")
    return ("## Acceptance — v2 (expanded diverse data) vs v1\n\n"
            "Live tree-accept (tokens/verify pass), **lossless**, on RedHatAI/speculator_benchmarks + held-out gsm8k.\n\n"
            + "\n".join(rows) + "\n")


def card():
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- speculative-decoding
- draft-model
- flow-matching
- chained-flow
---

# Flow-Drafter-4B-v2

Expanded-data revision of [Flow-Drafter-4B](https://huggingface.co/selimaktas/Flow-Drafter-4B) — a
speculative-decoding draft model for `Qwen/Qwen3.5-4B` (Chained-Flow, joint-VAE tree drafter).

**What changed in v2:** trained on a broader, diversity-weighted mix (~46k rows, 10 sources) that adds
multi-turn chat (UltraChat), diverse instructions (No-Robots), creative prose (WritingPrompts),
summarization (CNN/DailyMail) and translation (OPUS) on top of the original math/code/STEM core — to lift
acceptance on the low-predictability domains (prose / QA / summarization) toward uniform speedup.

{table()}
## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, jointly-trained VAE)
- `chained_flow_tree_config.json` — architecture + loss config
- `vae/` — the base VAE checkpoint

Drafts all K future hidden states in one flow pass, expands to a tree, base model verifies the tree in one
forward pass (lossless). Trained 2-GPU DDP.
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
                      commit_message="Flow-Drafter-4B-v2 — expanded diverse-data drafter")
    log(f"DONE → https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
