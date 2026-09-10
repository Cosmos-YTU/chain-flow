"""Push the diverse-data 9B drafter as selimaktas/Flow-Drafter-9B-v2.

Card numbers are v1 and v2 measured THE SAME WAY (scripts/diff_plugin_vs_harness.py, 5 bench domains,
200 windows/domain, generated tokens only, top_b=8/max_nodes=8/max_depth=5, K=8). They are NOT the
canonical RedHatAI speculator-benchmark figures on the v1 card, which use a 64-node/depth-8 tree and
therefore read higher — do not mix the two.
"""
import shutil
from pathlib import Path

CKD = Path("out/flow/ckpts/tree-vae-joint-9bx-1024-k8-l8")
VAE = Path("out/vae/ckpts/transformer-hidden-9bx-4096-latent1024-fp16")
STAGE = Path("/tmp/flow-drafter-9bx-v2-push")
REPO = "selimaktas/Flow-Drafter-9B-v2"

ROWS = [  # domain, v1 tree, v2 tree
    ("HumanEval", 5.12, 5.10), ("math_reasoning", 5.77, 5.54), ("qa", 3.18, 3.31),
    ("summarization", 2.78, 3.13), ("writing", 2.87, 3.07),
]


def card() -> str:
    tbl = "\n".join(
        f"| {d} | {a:.2f} | {b:.2f} | {b - a:+.2f} |" for d, a, b in ROWS
    )
    m1 = sum(r[1] for r in ROWS) / len(ROWS)
    m2 = sum(r[2] for r in ROWS) / len(ROWS)
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags: [speculative-decoding, flow-matching, draft-model, vllm]
---

# Flow-Drafter-9B-v2

Joint-VAE **tree** drafter for `Qwen/Qwen3.5-9B`. One flow pass produces K future hidden states, which
are expanded into a draft **tree** and verified in a single forward pass — accepting the longest path
the target model agrees with, so decoding stays **lossless** (bit-exact at temperature 0).

**v2 = v1 architecture retrained on an expanded, more diverse data mix** (10 sources instead of 5:
the original technical set plus ultrachat, opus_translation, no_robots, writingprompts, cnn_dailymail).
No architecture change; drop-in replacement for [Flow-Drafter-9B](https://huggingface.co/selimaktas/Flow-Drafter-9B).

## What changed

Tree-accept (tokens accepted per verify pass), **v1 and v2 measured identically**:

| domain | v1 | v2 | Δ |
|---|---|---|---|
{tbl}
| **mean** | **{m1:.2f}** | **{m2:.2f}** | **{m2 - m1:+.2f}** |

The gain is concentrated on the **free-form** domains (summarization +0.35, writing +0.20, qa +0.13),
which is what the diverse mix targets; math/code are flat-to-slightly-down as capacity reallocates,
while staying strong in absolute terms. The same pattern was observed at 4B and 27B.

**Read the numbers carefully.** These come from an offline differential
(`scripts/diff_plugin_vs_harness.py`): 5 bench domains, 200 windows each, generated tokens only,
tree `top_b=8, max_nodes=8, max_depth=5`, K=8. They are **not** comparable to the canonical RedHatAI
speculator-benchmark figures on the v1 card, which use a 64-node/depth-8 tree and read higher. The
v1 row here is re-measured under this configuration so the comparison is apples-to-apples.

## Deployment note

Under vLLM the achievable accept is lower than the offline numbers above, for two structural reasons:
the served tree is smaller (16 nodes), and vLLM cannot supply the hidden state of the just-committed
token at draft time, costing ~0.7 accept. Measured end-to-end at 9B, batch 1, 7 domains: **~1.1x**
over stock vLLM, lossless. Speculative decoding pays off in proportion to base decode cost — 27B sees
~1.4x, 4B does not clear break-even.

## Files

- `model.safetensors` — drafter weights
- `chained_flow_tree_config.json` — drafter config
- `vae/` — the jointly-trained hidden VAE (unfrozen during drafter training, optimised for acceptance
  rather than reconstruction, with a reconstruction anchor keeping the latent decodable)
"""


def main():
    from huggingface_hub import HfApi
    api = HfApi()
    for p in (CKD / "model.safetensors", CKD / "chained_flow_tree_config.json",
              VAE / "model.safetensors"):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD / "model.safetensors", STAGE)
    shutil.copy(CKD / "chained_flow_tree_config.json", STAGE)
    (STAGE / "vae").mkdir()
    shutil.copy(VAE / "model.safetensors", STAGE / "vae")
    vcfg = VAE / "chain_flow_vae_config.json"
    if vcfg.exists():
        shutil.copy(vcfg, STAGE / "vae")
    (STAGE / "README.md").write_text(card())
    print("staged:", sorted(p.name for p in STAGE.iterdir()))
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Flow-Drafter-9B-v2 — diverse-data retrain of the joint-VAE tree drafter")
    print("pushed →", f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
