"""``chain-flow`` -- the "what am I actually running?" command.

Every number this project produces depends on facts that are invisible from the outside: which
vLLM is installed, whether the async-scheduling guard got relaxed, whether the fused CUDA kernel
compiles on this box, and which of ten capability-gated flags survived their gates.  Answering
those from a 200-line engine log is how two silent fallbacks went unnoticed for days.  So they
get one command.

    chain-flow info              # resolved flags, vLLM build, async guard, kernel, drafter
    chain-flow build-kernel      # precompile the fused-block extension (else it JITs on first use)
    chain-flow build-shortlist   # build a CF_SHORTLIST for a vocabulary we do not ship one for
    chain-flow tree-patch        # apply/revert/verify the OPTIONAL forked-vLLM tree patch
    chain-flow docs             # print docs/BENCHMARKING.md (it ships in the wheel)
    chain-flow env               # `eval "$(chain-flow env)"`: the flag table as exports

``info`` is deliberately runnable WITHOUT a GPU or a drafter checkpoint: it reports what it
cannot determine instead of failing.
"""
from __future__ import annotations

import argparse
import os
import sys


def _fork_line() -> str:
    from chain_flow import defaults

    fk = defaults.fork()
    if fk["present"]:
        return f"vLLM build : FORKED (chain-flow tree ops present) at {fk['root']}"
    miss = ", ".join(fk["missing"][:4]) or "-"
    return (f"vLLM build : stock / unpatched (chain path only; tree needs the fork)\n"
            f"             missing: {miss}")


def _vllm_version() -> str:
    try:
        import vllm

        return getattr(vllm, "__version__", "?")
    except Exception as e:                                  # noqa: BLE001 - reporting only
        return f"NOT IMPORTABLE ({type(e).__name__}: {e})"


def _plugin_line() -> str:
    """Is the entry point REGISTERED, and would it fire?

    Registration and firing are different failures with the same symptom (a synchronous engine),
    so they are reported separately: a missing entry point means the install is wrong, while a
    registered-but-inactive one means ``CF_ASYNC_SPEC`` is off.
    """
    from importlib.metadata import entry_points

    eps = [e for e in entry_points(group="vllm.general_plugins")
           if "chain_flow" in (e.value or "")]
    if not eps:
        return ("async guard: ENTRY POINT NOT REGISTERED. The package is importable but not "
                "installed (a bare PYTHONPATH does not create entry points), so vLLM will "
                "never call it and a spec run will be SYNCHRONOUS -- about 10% slower at 4B, "
                "with nothing in the log to say so. Fix: pip install -e .")
    from chain_flow.vllm_plugin import async_guard

    reg = ", ".join(f"{e.name} -> {e.value}" for e in eps)
    return (f"async guard: entry point registered ({reg})\n"
            f"             in THIS process: {async_guard.status()} "
            f"(it only fires inside vLLM's EngineArgs.__post_init__)")


def _kernel_line() -> str:
    try:
        import torch
    except Exception as e:                                  # noqa: BLE001
        return f"cuda kernel: torch not importable ({e!r})"
    if not torch.cuda.is_available():
        return "cuda kernel: no CUDA device visible -- cannot probe (the PyTorch path is exact)"
    from chain_flow import cuda_block

    ok, err = cuda_block.available()
    if ok:
        return f"cuda kernel: BUILT and cached in {cuda_block.build_dir()}"
    return (f"cuda kernel: NOT AVAILABLE -- falling back to the bit-identical PyTorch block "
            f"stack (~2x slower draft)\n             {err.splitlines()[0][:300]}")


def _jit_cache_line() -> str:
    """flashinfer's sampling kernels: prebuilt, or compiled on YOUR first run?

    `pip install chain-flow` pulls `vllm`, which pulls `flashinfer-python` -- the Python
    frontend only. The compiled kernels live in a SEPARATE distribution, `flashinfer-jit-cache`,
    which is not on PyPI (it is per-CUDA-version and is published from flashinfer's own index),
    so a plain pip install leaves flashinfer to JIT-compile them on the first sampling call. That
    needs a full CUDA toolchain and costs minutes, and it happens BEFORE any chain-flow code
    runs -- which makes it look like our stall. Reported here rather than left to be discovered.
    """
    import importlib.util

    if importlib.util.find_spec("flashinfer_jit_cache") is not None:
        return "flashinfer : jit-cache present (sampling kernels prebuilt)"
    if importlib.util.find_spec("flashinfer") is None:
        return "flashinfer : not installed (vLLM will use its own sampler)"
    return ("flashinfer : jit-cache NOT installed -- flashinfer will COMPILE its sampling "
            "kernels on the first run\n"
            "             (minutes, needs nvcc; unrelated to chain-flow's own kernel). Fix:\n"
            "             pip install flashinfer-jit-cache "
            "--extra-index-url https://flashinfer.ai/whl/cu130/   # match your CUDA")


def _shortlist_line() -> str:
    """Which shortlist WOULD be used, and how big it is.

    The vocab guard needs a loaded model, so this is the candidate list, not the verdict -- the
    verdict is on the `[cf-defaults]` line of a real run. What it does answer is the question
    that cost pip users 1.13x -> 1.04x: is there a shortlist here at all?
    """
    from chain_flow import defaults

    cands = defaults.shortlist_candidates()
    if not cands:
        return ("shortlist  : NONE FOUND -- the drafter would score the full lm_head at every "
                "depth (~43-47% of the draft).\n"
                "             build one: chain-flow build-shortlist")
    head = f"shortlist  : {cands[0][0]}  [{cands[0][1]}]"
    try:
        from chain_flow import shortlist as sl

        ids, meta = sl.load(cands[0][0])
        head += (f"\n             {ids.numel()} rows, built for vocab_size="
                 f"{meta.get('vocab_size', '? (legacy format, max-id check only)')}")
    except Exception as e:                                  # noqa: BLE001 - reporting only
        head += f"\n             UNREADABLE: {e!r}"
    if len(cands) > 1:
        head += "\n             fallbacks: " + ", ".join(f"{p} [{w}]" for p, w in cands[1:])
    return head


def _drafter_line() -> str:
    d = os.environ.get("CF_DRAFTER_DIR")
    if not d:
        return ("drafter    : CF_DRAFTER_DIR unset -- set it to a checkpoint dir or an HF repo "
                "id (e.g. ytu-ce-cosmos/Flow-Drafter-Qwen3.5-4B)")
    return f"drafter    : {d}" + ("" if os.path.isdir(d) else "  (not a local dir: treated as an HF repo id)")


def cmd_info(_args) -> int:
    from chain_flow import defaults

    defaults.apply()
    print(f"chain-flow, python {sys.version.split()[0]}")
    print(f"vLLM       : {_vllm_version()}")
    print(_fork_line())
    print(_plugin_line())
    print(_kernel_line())
    print(_jit_cache_line())
    print(_drafter_line())
    print(_shortlist_line())
    print()
    print(defaults.summary())
    print("(gates that need the loaded drafter or the engine config are only resolved inside a "
          "real run; the flags above are the import-time proposal.)")
    for f, why in defaults.RETIRED.items():
        print(f"[cf-defaults] RETIRED {f}: {why}")
    return 0


def cmd_build_kernel(_args) -> int:
    """Precompile the fused-block extension so the first engine start does not stall ~60 s.

    Never a hard failure by itself -- a box without nvcc runs the PyTorch path correctly -- but
    it DOES exit nonzero, because someone who typed this asked for the kernel and a silent 0
    would let a CI image ship without it.
    """
    from chain_flow import cuda_block

    ok, err = cuda_block.available()
    if ok:
        print(f"[cf] fused-block CUDA extension ready in {cuda_block.build_dir()}")
        return 0
    print(f"[cf] could not build the fused-block CUDA extension:\n{err}", file=sys.stderr)
    print("[cf] this is NOT fatal for running: chain-flow falls back to the bit-identical "
          "PyTorch block stack (~2x slower draft). Set CF_CUDA_BLOCK=0 to silence the probe.",
          file=sys.stderr)
    return 1


def cmd_build_shortlist(_args, rest=None) -> int:
    from chain_flow.shortlist import build

    return build(rest)


DOCS = {"benchmarking": "BENCHMARKING.md"}


def cmd_docs(args) -> int:
    """Print a shipped doc.  ``docs/BENCHMARKING.md`` is package data precisely so that the
    README's "read this before quoting any speedup" is reachable by someone who has a wheel and
    no checkout -- a relative link in a PyPI README goes nowhere."""
    from pathlib import Path

    name = DOCS.get(getattr(args, "doc", "benchmarking"))
    if name is None:
        print(f"[cf] unknown doc {args.doc!r}; have: {', '.join(DOCS)}", file=sys.stderr)
        return 2
    p = Path(__file__).resolve().parent / "docs" / name
    if not p.is_file():
        print(f"[cf] {name} is missing from this install (expected {p}).", file=sys.stderr)
        return 1
    print(p.read_text())
    return 0


TREE_PATCH = "vllm-0.25.1-chain-flow-tree.patch"
# The vLLM the patch is generated against, taken FROM the patch's own filename so the pin has one
# source. Spelled out here (rather than imported from chain_flow.tree_patch) because building
# the argument parser must not import the package: `chain-flow --help` would then pay for torch.
_PIN = TREE_PATCH.split("-")[1]


def cmd_tree_patch(args) -> int:
    """Apply / revert / verify the optional forked-vLLM tree path.

    The whole implementation lives in ``chain_flow.tree_patch`` and is stdlib-only, so this
    command works on a box that is only PREPARING an environment -- no torch, no GPU, no drafter.
    Everything it can refuse, it refuses before writing a byte: see that module's docstring for
    why a half-applied fork is worse than an unpatched one.
    """
    from chain_flow import tree_patch

    return tree_patch.main(args)


def cmd_env(_args) -> int:
    from chain_flow import defaults

    print(defaults._sh())
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="chain-flow", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("info", help="resolved flags, vLLM build, async guard, kernel, drafter")
    sub.add_parser("build-kernel", help="precompile the fused-block CUDA extension")
    # Its own arguments are parsed by chain_flow.shortlist.build, so everything after the
    # subcommand is passed through verbatim rather than duplicated here and left to drift.
    sub.add_parser("build-shortlist", add_help=False,
                   help="build a CF_SHORTLIST token-id list (--help for its options)")
    # One command, three verbs, and no verb at all is the SAFE one (report, do not write).
    # `--apply` is the explicit token the safety rules ask for; `--dry-run` pairs with either
    # mutating verb and stages the whole change in memory without touching the install.
    tp = sub.add_parser("tree-patch",
                        help="apply / revert / verify the optional forked-vLLM tree patch")
    g = tp.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true",
                   help=f"patch the vLLM of THIS interpreter (needs vllm=={_PIN}); "
                        "verified against the wheel's RECORD before and after")
    g.add_argument("--revert", action="store_true",
                   help="restore the vLLM byte-identically and delete the added files")
    g.add_argument("--status", action="store_true",
                   help="is it applied, and is every other file still the wheel's? "
                        "(exit 0 applied / 1 not applied / 3 inconsistent)")
    g.add_argument("--show", action="store_true",
                   help="print the patch file's path and the manual `patch -p1` route")
    tp.add_argument("--dry-run", action="store_true",
                    help="with --apply/--revert: stage and verify everything, write nothing")
    d = sub.add_parser("docs", help="print a shipped doc (default: the benchmarking protocol)")
    d.add_argument("doc", nargs="?", default="benchmarking", choices=sorted(DOCS))
    sub.add_parser("env", help='shell exports for the flag table: eval "$(chain-flow env)"')
    args, rest = p.parse_known_args(argv)
    if args.cmd == "build-shortlist":
        return cmd_build_shortlist(args, rest)
    if rest:
        p.error(f"unrecognized arguments: {' '.join(rest)}")
    fn = {"info": cmd_info, "build-kernel": cmd_build_kernel, "tree-patch": cmd_tree_patch,
          "docs": cmd_docs, "env": cmd_env}.get(args.cmd)
    if fn is None:
        p.print_help()
        return 2
    return fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
