"""Push the Turkish 9B flow drafter as selimaktas/Flow-Drafter-9B-tr.

Numbers come from `out/flow/9btr_results.json`, so the card cannot drift from what was measured.
Every figure there is the offline differential (scripts/diff_plugin_vs_harness.py) at K=8, tree
keep=8/depth=5/topb=8, generated tokens only, 600 windows/domain -- the same instrument the v2 card
uses. At the card's own 200 windows this reproduced v2's published English table to +/-0.01, which is
how the instrument was validated before any of these numbers were trusted.

  python scripts/push_9btr.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

CKD = Path("out/flow/ckpts/tree-vae-joint-9btr-1024-k8-l8")
VAE = Path("out/vae/ckpts/transformer-hidden-9bx-4096-latent1024-fp16")
# Built and validated by the 27B Turkish agent; vocab-level, so one list serves 4B/9B/27B.
# See out/flow/SHORTLIST_REBUILD_CLAIM.txt. Deliberately NOT the packaged English list.
SHORTLIST = Path("out/flow/shortlist_q3527b_tr.pt")
RESULTS = Path("out/flow/9btr_results.json")
STAGE = Path("/tmp/flow-drafter-9b-tr-push")
REPO = "selimaktas/Flow-Drafter-9B-tr"
PARENT = "selimaktas/Flow-Drafter-9B-v2"
PARENT_SHA = "ed77e698e501423858effa9a596908a700876be7"


def table(rows: dict[str, dict[str, float]]) -> str:
    """rows: {domain: {"before": x, "after": y}} -> markdown table with a mean row."""
    out = [f"| {d} | {v['before']:.2f} | {v['after']:.2f} | {v['after'] - v['before']:+.2f} |"
           for d, v in rows.items()]
    n = max(len(rows), 1)
    mb = sum(v["before"] for v in rows.values()) / n
    ma = sum(v["after"] for v in rows.values()) / n
    out.append(f"| **mean** | **{mb:.2f}** | **{ma:.2f}** | **{ma - mb:+.2f}** |")
    return "\n".join(out)


def card(r: dict) -> str:
    sl = r["shortlist"]
    tj = r["trajectory"]
    d = r["data"]
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
tags: [speculative-decoding, flow-matching, draft-model, vllm, turkish]
language: [tr, en]
---

# Flow-Drafter-9B-tr

Joint-VAE **tree** drafter for `Qwen/Qwen3.5-9B`, adapted to **Turkish**. One flow pass produces K
future hidden states, expanded into a draft tree and verified in a single forward pass -- accepting
the longest path the target agrees with, so decoding stays **lossless** (bit-exact at temperature 0).

Warm-started from [{PARENT}](https://huggingface.co/{PARENT}) (weights
`sha256:{r['parent']['weights_sha256'][:16]}...`) rather than trained from scratch. That was the whole
hypothesis, and it held: **Turkish accept nearly doubles, and it gets there in about 1.3 epochs.**

## Results

Tree accept on the **shipping path** (lagged context, branching draft, tree acceptance) --
tokens accepted per verify pass, {r['windows_per_domain']} windows/domain, parent vs this checkpoint,
measured identically with `scripts/diff_plugin_vs_harness.py` (K=8, keep=8, depth=5, topb=8).

**Turkish**:

| domain | v2 (English drafter) | **9B-tr** | delta |
|---|---|---|---|
{table(r['turkish'])}

**English** -- the regression check:

| domain | v2 | **9B-tr** | delta |
|---|---|---|---|
{table(r['english'])}

Turkish **{r['turkish_mean']['before']:.2f} -> {r['turkish_mean']['after']:.2f}** (+{100*(r['turkish_mean']['after']/r['turkish_mean']['before']-1):.0f}%)
for **{r['english_mean']['after'] - r['english_mean']['before']:+.2f}** on English ({100*(r['english_mean']['after']/r['english_mean']['before']-1):.1f}%).
The English cost is bought with a 25%-of-tokens English replay slice in the training mix.

## The shortlist is half the result -- do not drop it

The drafter's candidate head scores a **shortlist** of the 248,320-token vocabulary rather than the
full `lm_head`. The list that ships inside `chained-flow` was built from English-weighted corpora and
covers only **{sl['tr_coverage_shipped_list']:.1f}%** of held-out Turkish tokens against
**{sl['en_coverage']:.0f}%** of English -- the most frequent Turkish morphemes (`'ın'`, `' için'`,
`' veya'`, `'ş'`) are simply **not in it**, so the drafter cannot propose them however well it is trained.

Measured on this checkpoint, Turkish accept:

| candidate head | Turkish accept |
|---|---|
| full 248,320-row `lm_head` (ceiling) | {sl['turkish_plugin_tree']['full_head']:.2f} |
| **this repo's `shortlist.pt`** ({sl['n']:,} ids) | **{sl['turkish_plugin_tree']['turkish_list']:.2f}** |
| the packaged English shortlist (62,642 ids) | {sl['turkish_plugin_tree']['english_list']:.2f} |

Shipping the English list would cost **{sl['cost_of_shipping_english_list']:+.2f}** accept -- **half the
entire training gain**, and it would look like a bad drafter rather than a bad shortlist. Note this is
invisible before adaptation: the parent scores 1.41 either way, because a drafter that cannot predict
Turkish is not being clipped by the shortlist yet.

`shortlist.pt` here is the English+Turkish union ({sl['n']:,} ids, {100*sl['n']/sl['vocab']:.1f}% of the
head, {sl['reduction']:.2f}x less head traffic), covering {sl['tr_coverage_this_list']:.2f}% of Turkish and
{sl['en_coverage']:.0f}% of English -- it costs English nothing
({sl['english_plugin_tree']['turkish_list']:.2f} with it vs {sl['english_plugin_tree']['full_head']:.2f} on the
full head). `chained-flow` picks up a `shortlist.pt` next to the checkpoint automatically, no env var.

## Training

Turkish teacher states over {d['tr_rows']:,} prompts from the turkishdspark corpus mix (tr-instructurca,
tr-multiturn, tr-tool-calling, tr-function-calling), with the **continuations generated by
`Qwen/Qwen3.5-9B` itself** -- the corpus's own answers come from a different model and are the wrong
target distribution for a drafter, which must predict what *this* target actually says. Prompts are
replayed verbatim with `enable_thinking=False`. Benchmark-held-out ids were excluded.

Mixed with {d['en_rows']:,} rows of English replay ({d['en_frac']:.0f}% of tokens) sampled across the 10
sources of v2's own training mix. Total {d['tokens']:.2f}M tokens, {d['epochs']:.1f} epochs, lr {d['lr']}, single GPU.

**It converges early.** At 1.26 epochs Turkish was already {tj['ckpt400_epoch1.26']['tr']:.2f}; at
{d['epochs']:.2f} epochs it is {tj['ckpt800_epoch2.52']['tr']:.2f}, while English drifts
{tj['ckpt800_epoch2.52']['en'] - tj['ckpt400_epoch1.26']['en']:+.2f}. Training longer buys Turkish almost
nothing and slowly costs English.

## Files

- `model.safetensors` -- drafter weights (includes the jointly-trained hidden VAE)
- `chained_flow_tree_config.json` -- drafter config
- `shortlist.pt` -- **English+Turkish** candidate head; picked up automatically. See above.
- `vae/` -- the hidden VAE used to construct the latent module

## Caveats

These are **offline** differential numbers. Under vLLM the achievable accept is lower for the
structural reasons documented on the v2 card (smaller served tree; no hidden state for the
just-committed token). Turkish and English rows are measured with the same instrument and are
comparable to each other; they are **not** comparable to the RedHatAI speculator-benchmark figures,
which use a 64-node/depth-8 tree and read higher.
"""


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
                      commit_message="Flow-Drafter-9B-tr — Turkish adaptation of the 9B joint-VAE "
                                     "tree drafter, warm-started from v2, with a Turkish shortlist")
    print("pushed →", f"https://huggingface.co/{REPO}")


if __name__ == "__main__":
    main()
