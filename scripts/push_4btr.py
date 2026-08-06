"""Push the Turkish 4B flow drafter as selimaktas/Flow-Drafter-4B-tr.

Numbers come from `out/flow/4btr_results.json`, written by scripts/parse_4btr_eval.py from the two
eval-driver logs, so the card cannot drift from what was measured. Everything is the offline
differential (scripts/diff_plugin_vs_harness.py), K=8, tree keep=8/depth=5/topb=8, generated tokens
only -- the same instrument the v2 card uses, so the English rows are comparable to v2's table.

  python scripts/push_4btr.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

CKD = Path("out/flow/ckpts/tree-vae-joint-4btr-640-k8-l8")
VAE = Path("out/vae/ckpts/transformer-hidden-4bx-2560-latent640-fp16")
# Built and validated by the 27B Turkish agent; vocab-level, so one list serves 4B/9B/27B.
# See out/flow/SHORTLIST_REBUILD_CLAIM.txt. Deliberately NOT the packaged English list.
SHORTLIST = Path("out/flow/shortlist_q3527b_tr.pt")
RESULTS = Path("out/flow/4btr_results.json")
STAGE = Path("/tmp/flow-drafter-4b-tr-push")
REPO = "selimaktas/Flow-Drafter-4B-tr"
PARENT = "selimaktas/Flow-Drafter-4B-v2"
PARENT_SHA = "9c3962cc7a3b5e6129053ba5da235a0296b6c795"
# sha256 of the parent's model.safetensors -- the bytes this run was actually warm-started from,
# verified identical between the HF snapshot and out/flow/ckpts/tree-vae-joint-4bx-640-k8-l8.
PARENT_WEIGHTS_SHA256 = "7a06e317d4a6c8648dd836b8ae92be462d8d97a4bcaf83739a644cf2121f3779"


def table(rows: dict[str, dict[str, float]]) -> str:
    out = [f"| {d} | {v['before']:.2f} | {v['after']:.2f} | {v['after'] - v['before']:+.2f} |"
           for d, v in rows.items()]
    n = max(len(rows), 1)
    mb = sum(v["before"] for v in rows.values()) / n
    ma = sum(v["after"] for v in rows.values()) / n
    out.append(f"| **mean** | **{mb:.2f}** | **{ma:.2f}** | **{ma - mb:+.2f}** |")
    return "\n".join(out)


def mean(rows: dict, k: str) -> float:
    return sum(v[k] for v in rows.values()) / max(len(rows), 1)


def card(r: dict) -> str:
    sl = r["shortlist"]
    d = r["data"]
    en_delta = mean(r["english"], "after") - mean(r["english"], "before")
    tr_delta = mean(r["turkish"], "after") - mean(r["turkish"], "before")
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags: [speculative-decoding, flow-matching, draft-model, vllm, turkish]
language: [tr, en]
---

# Flow-Drafter-4B-tr

Joint-VAE **tree** drafter for `Qwen/Qwen3.5-4B`, adapted to **Turkish**. One flow pass produces K
future hidden states, which are expanded into a draft tree and verified in a single forward pass --
accepting the longest path the target agrees with, so decoding stays **lossless** (bit-exact at
temperature 0).

Warm-started from [{PARENT}](https://huggingface.co/{PARENT}) (commit `{PARENT_SHA}`,
`model.safetensors` sha256 `{PARENT_WEIGHTS_SHA256[:16]}...`) rather than trained from scratch: the
English drafter's flow field is most of the answer, and the fine-tune only has to move it onto
Turkish token statistics.

## Results

Tree-accept (tokens accepted per verify pass) on the **shipping path** -- the plugin's real lagged
context and branching, `keep=8 depth=5 topb=8`, K=8, generated tokens only. Parent vs this
checkpoint, measured with the same instrument and the same Turkish shortlist on both arms.

**Turkish** ({r['tr_windows']} windows/domain, conversation-disjoint holdout):

| domain | v2 (English) | **4B-tr** | Δ |
|---|---|---|---|
{table(r['turkish'])}

**English** -- the regression check ({r['en_windows']} windows/domain):

| domain | v2 (English) | **4B-tr** | Δ |
|---|---|---|---|
{table(r['english'])}

Turkish {tr_delta:+.2f}, English {en_delta:+.2f}.

## The shortlist matters more than the weights here

The drafter's candidate head scores a **shortlist** of the 248,320-token vocabulary instead of the
full `lm_head`. The list shipped with `chained-flow` was built from English-weighted corpora and
covers only **{r['cov_tr_before']:.1f}%** of held-out Turkish tokens against **{r['cov_en']:.1f}%** of English --
about **1 in 3 Turkish target tokens is unproposable at any quality of training**.

This repo therefore ships its own **`shortlist.pt`** ({sl['n']:,} ids, {100 * sl['n'] / 248320:.1f}% of the
head, {sl['reduction']:.2f}x less head traffic than the full `lm_head`), rebuilt as the union over the
English corpora *and* the Turkish corpus. It covers **{r['cov_tr_after']:.1f}%** of held-out Turkish tokens
and still {r['cov_en']:.1f}% of English. `chained-flow` picks up a `shortlist.pt` sitting next to the
checkpoint automatically -- no env var -- so this is handled as long as `CF_DRAFTER_DIR` points here.

The list is **vocabulary-level**, so it is the same file the 9B and 27B Turkish drafters ship.

## Training data

Turkish teacher states over prompts from the **turkishdspark** corpus mix --
{d['tr_rows']:,} windows across tr-instructurca, tr-function-calling, tr-multiturn and
tr-tool-calling -- with the **continuations generated by `Qwen/Qwen3.5-4B` itself**, not taken from
the corpus. Only the chat-templated prompt *prefix* of each corpus row is reused; the corpus's own
answers were written by a different model (`Qwen3.6-35B-A3B-FP8`) and are the wrong target
distribution for a drafter, which has to predict what *this* target actually says.

**Thinking is disabled**: every prompt carries the `<think>\\n\\n</think>\\n\\n` non-thinking prefix,
because the target tends to think in English. Serve it the same way, or the drafter is off
distribution -- with vLLM's OpenAI server that means
`chat_template_kwargs: {{"enable_thinking": false}}` on the request.

Mixed with an **English replay slice** ({d['en_rows']:,} rows, {d['en_frac']:.0f}% of tokens) drawn from the
cache v2 itself trained on, which is what keeps the English table above where it is.

Total: {d['tokens']:.2f}M tokens, {d['epochs']} epochs at lr {d['lr']}, single GPU.

## Files

- `model.safetensors` -- drafter weights (includes the jointly-trained hidden VAE)
- `chained_flow_tree_config.json` -- drafter config
- `shortlist.pt` -- **English+Turkish** candidate-head shortlist; picked up automatically
- `vae/` -- the hidden VAE used to construct the latent module

## Caveats

These are **offline** differential numbers. Under vLLM the achievable accept is lower for structural
reasons documented on the v2 card (smaller served tree; no hidden state for the just-committed
token). Turkish and English rows are measured with the same instrument and are comparable to each
other; they are **not** comparable to the RedHatAI speculator-benchmark figures, which use a
64-node/depth-8 tree and read higher.

The English replay slice comes from the five **technical** domains of v2's mix (the free-form half
of that cache no longer existed on the training box), so free-form English is replayed by nothing.
The English table above measures free-form domains directly -- read those rows, not just the mean.
"""


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    r = json.loads(RESULTS.read_text())
    need = [CKD / "model.safetensors", CKD / "chained_flow_tree_config.json",
            VAE / "model.safetensors", SHORTLIST]
    for p in need:
        if not p.exists():
            raise SystemExit(f"missing {p}")

    # A push of the PARENT's bytes would be a silent no-op disguised as a deliverable.
    if sha256(CKD / "model.safetensors") == PARENT_WEIGHTS_SHA256:
        raise SystemExit("REFUSING: the checkpoint to push is byte-identical to the parent v2 -- "
                         "training did not write new weights")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy(CKD / "model.safetensors", STAGE)
    shutil.copy(CKD / "chained_flow_tree_config.json", STAGE)
    shutil.copy(SHORTLIST, STAGE / "shortlist.pt")
    (STAGE / "vae").mkdir()
    shutil.copy(VAE / "model.safetensors", STAGE / "vae")
    vcfg = VAE / "chained_flow_vae_config.json"
    if vcfg.exists():
        shutil.copy(vcfg, STAGE / "vae")
    (STAGE / "README.md").write_text(card(r))
    print("staged:", sorted(p.name for p in STAGE.iterdir()))

    if args.dry_run:
        print(card(r))
        print("\n--dry-run: nothing uploaded")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(STAGE), repo_id=REPO, repo_type="model",
                      commit_message="Flow-Drafter-4B-tr — Turkish adaptation of the 4B joint-VAE "
                                     "tree drafter, warm-started from v2, with a Turkish shortlist")
    print("pushed →", f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
