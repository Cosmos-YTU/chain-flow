"""Publish the released drafters and prompt datasets to the `ytu-ce-cosmos` org.

THIS FILE IS THE AUTHORITATIVE NAME MAP. The public naming scheme is

    Flow-Drafter-Qwen3.5-<SIZE>        -- English
    Flow-Drafter-Qwen3.5-<SIZE>-tr     -- Turkish

with **v2 as the base**: the v2 checkpoints are strictly better, so they take the unsuffixed
public name and the `-v2` suffix disappears. Two consequences worth stating plainly, because
both are easy to get wrong:

  * `selimaktas/Flow-Drafter-Qwen3.5-27B` and `ytu-ce-cosmos/Flow-Drafter-Qwen3.5-27B` are
    DIFFERENT MODELS. The former is v1; the latter is v2. The name collides across orgs, so
    never resolve a bare name without its org.
  * The `selimaktas/` repos are deliberately NOT renamed. They stay as they are; this script
    only copies out of them.

There is no server-side copy for model repos, so each one is snapshot-downloaded and re-uploaded.
Card text is rewritten as it goes, so a published card refers to its siblings by their published
names rather than pointing back at `selimaktas/`.

  python scripts/publish_to_cosmos.py                 # dry run: plan only, touches nothing
  python scripts/publish_to_cosmos.py --apply         # do it (needs write access to the org)
  python scripts/publish_to_cosmos.py --apply --only Flow-Drafter-Qwen3.5-27B-tr
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

ORG = "ytu-ce-cosmos"          # the Hugging Face org (NOT a GitHub org)
GITHUB = "Cosmos-YTU/chain-flow"   # the public code repo the cards should point at

# The project renamed chained-flow -> chain-flow, and the public GitHub is the chain-flow one, so
# a card published from here must tell users `pip install chain-flow` and the chain_flow module
# path. EXCEPT for one token: `chained_flow_tree_config.json` is a REAL FILE inside every
# published checkpoint (the loader opens it by that name), so it survives the rename. Every card
# names it in its Files section -- rewriting it would document a file that does not exist.
# Immutable identifiers that must survive the chained-flow -> chain-flow rename:
#   chained_flow_tree_config  - a REAL FILE inside every checkpoint; the loader opens it by name
#   chained-flow-drafters-... - the Hugging Face COLLECTION slug; renaming it 404s the link
_KEEP = ["chained_flow_tree_config", "chained-flow-drafters"]
_PKG_RENAMES = [("chained_flow", "chain_flow"), ("chained-flow", "chain-flow")]

# (source repo on selimaktas/, published name, repo_type)
PLAN: list[tuple[str, str, str]] = [
    # English -- v2 becomes the base name
    ("selimaktas/Flow-Drafter-4B-v2",            "Flow-Drafter-Qwen3.5-4B",     "model"),
    ("selimaktas/Flow-Drafter-9B-v2",            "Flow-Drafter-Qwen3.5-9B",     "model"),
    ("selimaktas/Flow-Drafter-Qwen3.5-27B-v2",   "Flow-Drafter-Qwen3.5-27B",    "model"),
    # Turkish
    ("selimaktas/Flow-Drafter-4B-tr",            "Flow-Drafter-Qwen3.5-4B-tr",  "model"),
    ("selimaktas/Flow-Drafter-9B-tr",            "Flow-Drafter-Qwen3.5-9B-tr",  "model"),
    ("selimaktas/Flow-Drafter-Qwen3.5-27B-tr",   "Flow-Drafter-Qwen3.5-27B-tr", "model"),
    # Prompt datasets
    ("selimaktas/turkish-flow-drafter-prompts",  "turkish-flow-drafter-prompts", "dataset"),
    ("selimaktas/english-flow-drafter-prompts",  "english-flow-drafter-prompts", "dataset"),
]

# Applied to README.md/*.md inside each published repo. ORDER MATTERS: the longest source name
# has to be rewritten before any name that is a prefix of it, or `...-27B-v2` would first match
# the `...-27B` rule and be left as `...-27B-v2` pointing at the wrong model.
def _rewrites() -> list[tuple[str, str]]:
    pairs = [(src, f"{ORG}/{dst}") for src, dst, _ in PLAN]
    pairs.sort(key=lambda p: -len(p[0]))
    return pairs


def dangling_refs(text: str) -> set[str]:
    """`selimaktas/...` repos a card still points at after rewriting.

    Not an error: a card may legitimately cite something we are not publishing (the v1 27B, the
    ablations repo). It IS worth surfacing, because `selimaktas/Flow-Drafter-Qwen3.5-27B` is the
    v1 model while `ytu-ce-cosmos/Flow-Drafter-Qwen3.5-27B` is v2 -- same bare name, different
    weights -- so a reader who drops the org gets the wrong model silently."""
    return set(re.findall(r"selimaktas/[A-Za-z0-9._-]+", text))


def rewrite_text(text: str) -> str:
    """Repo renames + the project rename, with the checkpoint config filename protected."""
    for i, k in enumerate(_KEEP):
        text = text.replace(k, f"\x00KEEP{i}\x00")
    for src, dst in _rewrites():
        text = text.replace(src, dst)
    # Bare model names in LINK TEXT and prose, e.g. "warm-started from Flow-Drafter-4B-v2".
    # Without this the URL says .../Flow-Drafter-Qwen3.5-4B while the visible text still reads
    # "-v2", naming a repo that does not exist under this org. Longest-first for the same
    # prefix reason as _rewrites(); only names we actually publish are rewritten, so the
    # unpublished v1 drafters keep their real names.
    for src, dst, _k in sorted(PLAN, key=lambda t: -len(t[0])):
        bare = src.split("/", 1)[1]
        if bare != dst:
            text = text.replace(bare, dst)
    for src, dst in _PKG_RENAMES:
        text = text.replace(src, dst)
    # Point cards at the public code repo rather than whatever private/renamed remote they cite.
    text = re.sub(r"github\.com/[A-Za-z0-9._-]+/chain-flow", f"github.com/{GITHUB}", text)
    for i, k in enumerate(_KEEP):
        text = text.replace(f"\x00KEEP{i}\x00", k)
    return text


def rewrite_cards(root: str, verbose: bool = True) -> int:
    n = 0
    dangling: set[str] = set()
    for dirpath, _, files in os.walk(root):
        for f in files:
            # MARKDOWN ONLY. Rewriting is a DOCUMENTATION edit, and the config/JSON files in a
            # checkpoint are DATA. A previous version also walked .json and rewrote
            # chained_flow_tree_config.json's `vae_dir` / `init_from` -- local training paths
            # that record where the weights actually came from. Editing those does not help a
            # user (the paths are on our box either way) and falsifies the provenance.
            if not f.endswith(".md"):
                continue
            p = os.path.join(dirpath, f)
            try:
                s = open(p, encoding="utf-8").read()
            except (UnicodeDecodeError, OSError):
                continue
            out = rewrite_text(s)
            if out != s:
                open(p, "w", encoding="utf-8").write(out)
                n += 1
                if verbose:
                    print(f"      rewrote refs in {os.path.relpath(p, root)}")
            dangling |= dangling_refs(out)
    if dangling and verbose:
        print(f"      NOTE: still points at {len(dangling)} unpublished selimaktas repo(s): "
              f"{', '.join(sorted(dangling))}")
        if any(d.endswith("Flow-Drafter-Qwen3.5-27B") for d in dangling):
            print("      ^ that one is the V1 27B. The published v2 takes the same bare name "
                  "under this org, so keep the org prefix wherever it appears.")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually create and upload")
    ap.add_argument("--only", default=None, help="publish just this target name")
    ap.add_argument("--private", action="store_true", help="create the target repos private")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        who = api.whoami()
    except Exception as e:
        print(f"not logged in to Hugging Face ({type(e).__name__}). Run `hf auth login` first.")
        return 2
    orgs = [o["name"] for o in who.get("orgs", [])]
    print(f"  logged in as {who.get('name')}   orgs: {orgs or '(none)'}")
    if ORG not in orgs:
        print(f"  NOTE: {ORG} is not in your orgs. Publishing will fail until it is; "
              f"the dry run below still works.")

    plan = [t for t in PLAN if args.only in (None, t[1])]
    if not plan:
        print(f"  --only {args.only!r} matched nothing; targets: "
              f"{', '.join(t[1] for t in PLAN)}")
        return 2

    print(f"\n  cards will be rewritten to say:  pip install chain-flow  |  "
          f"chain_flow.vllm_plugin...  |  github.com/{GITHUB}")
    print(f"  preserved verbatim (real file in every checkpoint): chained_flow_tree_config.json")
    print(f"\n  {'source':<44}{'->':^4}{ORG + '/...':<32}{'type':>8}")
    for src, dst, kind in plan:
        exists = ""
        try:
            (api.dataset_info if kind == "dataset" else api.model_info)(f"{ORG}/{dst}")
            exists = "  [target exists, will update]"
        except Exception:
            pass
        print(f"  {src:<44}{'->':^4}{dst:<32}{kind:>8}{exists}")

    if not args.apply:
        print("\n  DRY RUN -- nothing was created, downloaded or uploaded. Re-run with --apply.")
        return 0

    from huggingface_hub import snapshot_download

    for src, dst, kind in plan:
        target = f"{ORG}/{dst}"
        print(f"\n  === {src} -> {target}")
        with tempfile.TemporaryDirectory(prefix="cf-publish-") as tmp:
            local = snapshot_download(src, repo_type=kind, local_dir=os.path.join(tmp, "r"))
            # Never publish the cache bookkeeping directory.
            cache = os.path.join(local, ".cache")
            if os.path.isdir(cache):
                import shutil
                shutil.rmtree(cache, ignore_errors=True)
            rewrite_cards(local)
            api.create_repo(target, repo_type=kind, exist_ok=True, private=args.private)
            api.upload_folder(
                folder_path=local, repo_id=target, repo_type=kind,
                commit_message=f"Publish {dst} (from {src})",
            )
            print(f"      uploaded https://huggingface.co/"
                  f"{'datasets/' if kind == 'dataset' else ''}{target}")
    print("\n  done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
