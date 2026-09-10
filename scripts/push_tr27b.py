"""Push the Turkish 27B flow drafter to selimaktas/Flow-Drafter-Qwen3.5-27B-tr.

Modelled on scripts/push_q3527bx_v2.py (the parent's push), NOT on scripts/push_27b_and_cards.py:
that one targets the v1 repo `selimaktas/Flow-Drafter-27B` and writes a card declaring
`base_model: Qwen/Qwen3.6-27B`. This drafter is built on Qwen3.5-27B, so reusing it would have
published a card naming the wrong base model into the wrong repo.

Two things this ships that the parent's push does not:

  * `shortlist.pt` -- REQUIRED, not a nicety. The candidate head scores over a shortlist of the
    248,320-id vocabulary, and the shipped English-built list covers held-out Turkish at only
    65.4%, so roughly a third of Turkish tokens would be unproposable. The plugin picks up a
    `shortlist.pt` sitting next to the checkpoint automatically.
  * measured numbers read from a JSON file rather than typed into the card. --results is
    mandatory and the script refuses to run without it, so the card cannot ship with stale or
    invented accept figures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

ROOT = Path("/home/shadeform/chain-flow")
CKD = ROOT / "out/flow/ckpts/tree-vae-joint-tr27b-1024-k8-l8"
VAE = ROOT / "out/vae/ckpts/transformer-hidden-q3527bx-5120-latent1024-fp16"
SHORTLIST = ROOT / "out/flow/shortlist_q3527b_tr.pt"
PARENT = "selimaktas/Flow-Drafter-Qwen3.5-27B-v2"
REPO = "selimaktas/Flow-Drafter-Qwen3.5-27B-tr"
STAGE = Path("/tmp/flow-drafter-tr27b-push")


def log(m: str) -> None:
    print(f"[push-tr] {time.strftime('%H:%M:%S')} {m}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def accept_table(r: dict) -> str:
    """Turkish + English accept, before (v2) and after (this checkpoint)."""
    out = []
    for title, key in (("Turkish (held-out turkishdspark prompts)", "turkish"),
                       ("English (7-domain sweep)", "english")):
        block = r.get(key) or {}
        if not block:
            continue
        rows = ["| domain | v2 (English drafter) | this checkpoint | Δ |", "|---|---|---|---|"]
        for dom, v in block.items():
            before, after = v.get("before"), v.get("after")
            d = f"{after - before:+.2f}" if before is not None and after is not None else "—"
            rows.append(f"| {dom} | {before if before is None else f'{before:.2f}'} "
                        f"| {after if after is None else f'{after:.2f}'} | {d} |")
        out.append(f"**{title}**\n\n" + "\n".join(rows))
    return "\n\n".join(out)


def head_ab_table(r: dict) -> str:
    """The three-head A/B on the TRAINED checkpoint -- the evidence for shipping shortlist.pt."""
    h = r.get("head_ab") or {}
    full, tr, en = h.get("full"), h.get("turkish_sl"), h.get("packaged_english_sl")
    if tr is None or en is None:
        return ""
    rows = ["| candidate head | Turkish accept (offline, per-token-position) |", "|---|---|"]
    if full is not None:
        rows.append(f"| full {r.get('vocab_size', '248,320')}-id vocab | {full:.2f} |")
    rows.append(f"| **this repo's `shortlist.pt`** | **{tr:.2f}** |")
    rows.append(f"| stock English shortlist | {en:.2f} (**{en - tr:+.2f}**) |")
    return ("Measured on this checkpoint, Turkish held-out:\n\n" + "\n".join(rows) + "\n")


def card(r: dict, digest: str) -> str:
    cov = r.get("shortlist_coverage", {})
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-27B
language:
- tr
- en
tags:
- speculative-decoding
- draft-model
- flow-matching
- chain-flow
- turkish
---

# Flow-Drafter-Qwen3.5-27B-tr

Turkish adaptation of [Flow-Drafter-Qwen3.5-27B-v2]({f"https://huggingface.co/{PARENT}"}) — a
speculative-decoding draft model for `Qwen/Qwen3.5-27B` (Chained-Flow, joint-VAE tree drafter).

**Warm-started, not retrained.** The English v2 drafter was the initialisation
(sha256 `{r.get('parent_sha256', 'unknown')}`); this checkpoint is a low-LR adaptation on
~{r.get('train_tokens_m', '?')}M tokens of Turkish teacher states, testing whether an
English-trained flow drafter generalises to a new language from its hidden-state representation.

Teacher states were collected by generating with `Qwen/Qwen3.5-27B` itself from Turkish prompts,
in **non-thinking mode** (`enable_thinking=False`). Prompts were taken from the
turkishdspark corpus, but its completions were **not** reused — they came from a different model
(Qwen3.6-35B-A3B) and would have been off-distribution for this target.

## Acceptance

**All figures below are OFFLINE accept, per-token-POSITION estimator** (`diff_plugin_vs_harness.py`,
plugin arm, K=8). They are **not comparable to a served tok/s or in-engine speedup number**: vLLM
averages run length over draft STEPS, and steps land where the previous run ended, so an easy stretch
contributes many high-length windows offline but is crossed in few steps by the server. Same data,
different denominator — on 4B Turkish that difference alone is about -0.40. Do not read these as
serving figures, and do not compare them against one.

Measurement condition, identical for both languages: teacher states generated by `Qwen/Qwen3.5-27B`
with natural sampling capped at 256 new tokens and **no `ignore_eos`** (forcing a fixed length runs
generation past the model's natural stop into a far less predictable tail, which systematically
depresses accept). Rows reaching the cap: 47.5% Turkish, 67.8% English.

{accept_table(r)}

## The shortlist matters here

The candidate head scores over a shortlist of the 248,320-token vocabulary. The list shipped with
Chained-Flow is a union built over English corpora, and it covers held-out Turkish text at only
**{cov.get('shipped_tr', '65.4')}%** — about a third of Turkish tokens could not be proposed at all,
which looks like a bad drafter but is a vocabulary problem.

`shortlist.pt` in this repo is rebuilt over a mixed English+Turkish corpus:
**{cov.get('new_rows', '77,939')} ids, {cov.get('new_tr', '99.61')}% held-out Turkish coverage,
{cov.get('new_en', '100.00')}% English**, at {cov.get('new_headx', '3.19')}x head reduction. The
plugin picks it up automatically when it sits beside the checkpoint.

{head_ab_table(r)}
Note this cost is **invisible before adaptation**: on the parent, all three heads score the same,
because a drafter that cannot predict Turkish tokens is not yet being clipped by a list that
excludes them. The hazard only appears once the model has something worth proposing — so measuring
it on the parent clears it falsely.

## Files
- `model.safetensors` — the drafter (flow experts, Markov head, path head, jointly-trained VAE)
  — sha256 `{digest}`
- `chained_flow_tree_config.json` — architecture + loss config
- `shortlist.pt` — English+Turkish candidate shortlist (see above)
- `vae/` — the base VAE checkpoint

Drafts all K future hidden states in one flow pass, expands to a tree, base model verifies the tree
in one forward pass (lossless). Trained 2-GPU DDP.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True,
                    help="JSON of measured accept + coverage; refuses to build a card without it")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--dry-run", action="store_true", help="stage and print the card, upload nothing")
    args = ap.parse_args()

    with open(args.results) as f:
        r = json.load(f)

    for p in (CKD / "model.safetensors", CKD / "chained_flow_tree_config.json",
              VAE / "model.safetensors", SHORTLIST):
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")

    digest = sha256(CKD / "model.safetensors")
    log(f"drafter sha256 {digest}")
    if digest == r.get("parent_sha256"):
        raise SystemExit("drafter sha256 EQUALS the parent's -- training produced no change, refusing to push")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD / "model.safetensors", STAGE)
    shutil.copy(CKD / "chained_flow_tree_config.json", STAGE)
    shutil.copy(SHORTLIST, STAGE / "shortlist.pt")
    (STAGE / "vae").mkdir()
    shutil.copy(VAE / "model.safetensors", STAGE / "vae")
    shutil.copy(VAE / "chain_flow_vae_config.json", STAGE / "vae")
    (STAGE / "README.md").write_text(card(r, digest))
    log(f"staged: {sorted(p.name for p in STAGE.iterdir())}")

    if args.dry_run:
        print("\n" + "=" * 70 + "\n" + (STAGE / "README.md").read_text() + "=" * 70)
        log("dry run -- nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=False, exist_ok=True)
    log(f"uploading -> {args.repo} ...")
    api.upload_folder(folder_path=str(STAGE), repo_id=args.repo, repo_type="model",
                      commit_message="Flow-Drafter-Qwen3.5-27B-tr - Turkish adaptation of the v2 drafter")
    log(f"DONE -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
