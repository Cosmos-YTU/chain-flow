"""Capability-gated DEFAULTS for the Chained-Flow flag set — one table, one summary line.

WHY THIS EXISTS
---------------
Ten optimisation flags were developed and verified one at a time, all defaulting OFF.  A flag
that is off by default is a flag nobody runs, and a flag that is blindly turned on is worse:
``CF_CUDA_BLOCK`` spent hours looking enabled at 9B/27B while a ``hidden_size == 640`` gate
returned ``None`` and it did exactly nothing, producing confident, meaningless A/Bs.

So the rule here is: a default is a PROPOSAL, a gate decides, and the decision is PRINTED with
the state that was actually used — never the env var that was requested.  Three properties:

1. **Capability-gated.**  Every default has a gate that is evaluated against the real process:
   does the CUDA extension build, is the forked vLLM installed, does the drafter's shape fit the
   kernel, did the engine actually enable async scheduling.  A gate that fails downgrades the
   flag to OFF and records WHY.
2. **Escape hatch.**  Everything here uses ``os.environ.setdefault`` semantics, so an explicitly
   set ``CF_X=0`` always wins over the default and is recorded as ``off (explicit)``.  An
   explicit ``CF_X=1`` that then FAILS its gate is an error the caller asked for, so it is
   reported as ``forced-off`` in the loudest terms rather than silently honoured.
3. **Loud once.**  ``summary()`` returns ONE line naming every flag, its resolved state, and the
   reason for anything that is not on.  ``flow_proposer._build`` prints it.

WHERE THE DEFAULTS ARE APPLIED (ordering matters, and it is not obvious)
-----------------------------------------------------------------------
Different consumers read these env vars at very different times:

* ``CF_ASYNC_SPEC`` is read while ``LLM(...)`` is still being constructed, inside
  ``VllmConfig.__post_init__``, long before anything instantiates the proposer.  On STOCK vLLM
  that reader is our own ``vllm.general_plugins`` entry point
  (``chained_flow.vllm_plugin.async_guard``), which runs in ``EngineArgs.__post_init__`` and
  calls ``apply()`` itself — so there the default works in-process.  On the FORK the reader is
  the patched ``config/vllm.py``, which we do not control and which runs before any import of
  ours, which is why ``bench_cf.sh`` also evaluates the ``--sh`` emitter below.
  Either way the proposer re-gates it against the engine's real
  ``scheduler_config.async_scheduling``, so a stale ``CF_ASYNC_SPEC=1`` in an environment whose
  engine ended up synchronous is turned back OFF instead of silently corrupting the draft path.
* the tree flags (``CF_GDN_DEFER`` / ``CF_GDN_BV`` / ``CF_TREE_FUSED_ATTN`` / ``CF_TREE_FULLCG``)
  are read by the fork at ``GPUModelRunner.__init__`` (``CF_TREE_FULLCG``, line ~854) or lazily at
  first kernel launch.  The custom_class proposer is instantiated at line ~585 of that SAME
  ``__init__``, which imports this package — so ``apply()`` at import time is early enough.
* the drafter flags are read by ``FlowDrafterProposer.__init__`` / ``_build``, later still.

``apply()`` is therefore called from ``chained_flow/__init__.py`` and is idempotent.

KEEPING THE DRAFTER SEPARABLE FROM THE FORK
-------------------------------------------
A chain-only build on STOCK vLLM is THE DEFAULT shipping target, so the two groups are strictly
separate: ``DRAFTER`` defaults never look at the fork, and ``TREE`` defaults are not merely
"tried and failed" without it — they are NOT OFFERED at all (``fork_missing`` in the summary).

``CF_ASYNC_SPEC`` sits in ``DRAFTER`` even though it once lived in ``TREE``, and the placement
is the point: vLLM's async contract (GPU cnt/k0, a GPU draft TENSOR, a published
``valid_sampled_token_count``) is generic, and only the tree SHAPE hand-off needs the fork.  So
the flag is offered on stock vLLM, refused in tree mode without the fork, and gated everywhere
on the engine's resolved ``async_scheduling`` — see ``finalize_async``.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# ------------------------------------------------------------------ truthiness / raw env

_FALSE = ("0", "", "false", "False", "FALSE", "no", "off", "OFF")


def truthy(v) -> bool:
    return v is not None and str(v) not in _FALSE


def resolved(flag: str) -> str | None:
    """The value in force RIGHT NOW -- after defaults and after any gate wrote one down.  This is
    what the rest of the stack (including the un-editable fork) reads; ``was_explicit`` below is
    the separate question of who put it there."""
    return os.environ.get(flag)


# ------------------------------------------------------------------ the table
#
# side:    "drafter" works on stock vLLM;  "tree" needs the fork.
# default: the value proposed when the gate passes.
# gate:    where the gate is EVALUATED -- "import" (process-level, in apply()) or
#          "build" (needs the loaded drafter / engine config, in finalize()).
#
# The `why` strings are the ones that reach the summary line, so they are written to be read at
# 2am by someone wondering why a flag they set did nothing.

DRAFTER: dict[str, str] = {
    # the shortlist head.  Not a boolean: a PATH.  Unset means the drafter silently scores the
    # full 248k-row lm_head at every depth, ~43-47% of the draft for nothing -- which is exactly
    # what happened for a week, and then again for every pip user until the list shipped inside
    # the wheel.  Defaulted by `_shortlist_default()`; the VOCAB GUARD that decides whether the
    # default may actually be used lives in `chained_flow.shortlist.check`, because it needs the
    # loaded model's head size.
    "CF_SHORTLIST": "",              # filled in by _shortlist_default()
    # torch.compile the flow net (predict_hidden): ~16 tiny layer-passes -> fused kernels,
    # measured 9.79 -> 5.26 ms (1.86x) at 27B.  It was hard-coded to 1 in bench_cf.sh's chain and
    # tree arms and defaulted to 0 in the proposer, so every published number was taken on a path
    # a pip user was not on and no line anywhere said so.  Gate: inductor must have a working
    # backend (triton) -- without one `torch.compile` either falls back to eager silently or
    # raises inside the first draft, and a raise inside a cudagraph capture is not recoverable.
    "CF_COMPILE": "1",
    # hand-written fused block-stack kernel.  Gate: the extension must BUILD and the drafter's
    # shape must be one the kernel is instantiated for.  Accept-neutral.
    "CF_CUDA_BLOCK": "1",
    # the two chunk-experts on two streams.  Gate: CF_CUDA_BLOCK engaged + exactly 2 experts.
    # Accept-IDENTICAL (it only changes the issue order of two independent kernels).
    "CF_CUDA_PAIR": "1",
    # skip PathHead offsets that provably have no ancestor.  BIT-EXACT, no gate.
    "CF_PATH_TRIM": "1",
    # narrow the context-ring append to the slots a TREE step can fill.  BIT-EXACT.
    # Gate: branching tree mode, and refuses under CF_NONGREEDY_CHAIN (a chain step can commit
    # more than tree_depth+1 tokens and a too-small Kmax would SILENTLY drop accepted hiddens).
    "CF_RING_TRIM": "1",
    # two-pass candidate head.  CHANGES CANDIDATES (deliberately -- see _candidates); the
    # in-engine 7-domain sweep put M=8192+shared at accept 2.623 vs 2.619 one-pass, i.e. up.
    # Gate: M must be smaller than the head it is narrowing.
    "CF_TWOPASS_M": "8192",
    "CF_TWOPASS_SHARED": "1",
    # issue the draft on a side stream right after _sample.  Accept-identical (cnt/k0 come off
    # the verify's own GPU tensors, so an off-by-one would move accept immediately).
    "CF_DRAFT_EARLY": "1",
    # 8 offset_proj Linears -> one baddbmm.  MEASURED EXACTLY NEUTRAL end-to-end under the
    # draft cudagraph (38.6 vs 38.7 tok/s), because the win it targets is launch count and
    # cudagraphs already removed launch cost.  Kept ON, see RETIRED note at the bottom.
    "CF_FUSE_PATH": "1",
    # GPU-token-native proposer, so vLLM's async scheduling can be enabled for the spec arm.
    # Bit-exact, and worth +10.5% / +5.5% / +1.6% at 4B / 9B / 27B -- the single largest
    # remaining item at 4B (docs/BENCHMARKING.md).
    #
    # THIS IS A DRAFTER FLAG, NOT A TREE FLAG, and that placement is the fork-free shipping
    # target: the async CONTRACT (GPU cnt/k0, a GPU draft tensor, published
    # valid_sampled_token_count) is generic, and only the TREE-SHAPE hand-off needs the fork.
    # `chained_flow.vllm_plugin.async_guard` -- a `vllm.general_plugins` entry point -- relaxes
    # stock vLLM's guard for us, so a chain build on unmodified vLLM gets async too.
    # Gated twice: the engine must actually have resolved `async_scheduling` True, and TREE mode
    # additionally needs the fork (see `finalize_async`).
    "CF_ASYNC_SPEC": "1",
}

TREE: dict[str, str] = {
    # don't write the per-node GDN recurrent state during verify; finalize the accepted leaf
    # afterwards.  BIT-EXACT.
    "CF_GDN_DEFER": "1",
    # v-rows per Triton program for the deferred GDN.  8 beat 16/32 at all three sizes.
    "CF_GDN_BV": "8",
    # the whole tree-attention suffix half + LSE merge in one CUDA kernel.  Exact at all three
    # sizes; the kernel REFUSES N >= 128 itself and falls back, which is why there is no
    # shape gate here -- the fallback is inside the fork and is silent-but-correct.
    "CF_TREE_FUSED_ATTN": "1",
    # dispatch a tree step to vLLM's FULL decode cudagraph instead of PIECEWISE.  BIT-EXACT,
    # and the PIECEWISE cliff is worth 6-10 ms/step.  (Already the de-facto default: it was
    # hard-coded to 1 in bench_cf.sh.  Folded into the table so it has a gate and a reason.)
    "CF_TREE_FULLCG": "1",
}

ALL = {**DRAFTER, **TREE}

# Flags that were verified, measured, and are NOT being promoted.  Kept in the table so the
# summary can say "considered and rejected" instead of leaving a silent hole.
RETIRED: dict[str, str] = {
    # CF_DRAFT_DEFER: +0.03 ms of host window, measured 128.9 vs 130.2 tok/s (WORSE).  Its value
    # is the null result, not the speed -- it proves the remaining GPU idle is downstream of
    # take_draft_token_ids.  Never default it on.
    "CF_DRAFT_DEFER": "measured NEGATIVE (128.9 vs 130.2 tok/s); kept only as the null result",
    # CF_TWOPASS_SEED: tree accept -0.02..-0.03 at every M, and slower.
    "CF_TWOPASS_SEED": "measured WORSE at every M (accept -0.02..-0.03) and slower",
    # CF_COMPILE_BEAM: inductor's AOT pass dies on the beam under inference_mode.
    "CF_COMPILE_BEAM": "inductor AOT pass fails under inference_mode; the beam is not the cost",
}


# ------------------------------------------------------------------ capability probes


def repo_root() -> Path:
    # src/chained_flow/defaults.py -> src/chained_flow -> src -> <repo>
    return Path(__file__).resolve().parents[2]


def packaged_shortlist() -> Path:
    """The 250 KB int32 list that ships INSIDE the wheel.  Duplicated from
    ``chained_flow.shortlist.packaged_path`` on purpose: this module must stay importable as a
    bare FILE with no ``chained_flow`` package and no torch (see ``_sh``)."""
    return Path(__file__).resolve().parent / "data" / "shortlist_qwen3_5.pt"


def shortlist_candidates(drafter_dir: str | None = None) -> list[tuple[str, str]]:
    """Every shortlist this process could use, best first, as ``(path, provenance)``.

    Deliberately a list rather than one answer, and deliberately FILE CHECKS rather than bare
    paths: the caller (``FlowDrafterProposer._build``) has the model's vocab size and can reject
    a candidate that was built for a different vocabulary, at which point it needs the next one
    rather than a hard failure.  ``CF_SHORTLIST`` still raises on a path that does not exist,
    so a default pointing at a missing file would turn every stock run into a crash.

    Order, and why:

    1. ``<drafter dir>/shortlist.pt``.  A drafter checkpoint may ship a list tuned to itself, and
       it must win over ours.  ``drafter_dir`` is passed in by ``_build`` AFTER any HF snapshot
       download, which is the whole reason this takes an argument: at import time
       ``CF_DRAFTER_DIR`` is usually a repo id, ``os.path.isdir`` is False, and the documented
       "drop shortlist.pt next to the checkpoint" instruction could never fire.
    2. the repo's own ``out/flow/shortlist_q3527b.pt``, for a source checkout.  ``repo_root()``
       is meaningless once the package lives in ``site-packages``; the ``is_file()`` check keeps
       that harmless rather than a bogus path in the log.
    3. the PACKAGED list.  This is the one that survives ``pip install`` -- it is keyed by token
       id, so it is a property of the TOKENIZER and one file serves 4B / 9B / 27B.
    """
    out: list[tuple[str, str]] = []
    d = drafter_dir if drafter_dir is not None else (os.environ.get("CF_DRAFTER_DIR") or "")
    if d and os.path.isdir(d):
        p = Path(d) / "shortlist.pt"
        if p.is_file():
            out.append((str(p), "drafter checkpoint"))
    p = repo_root() / "out" / "flow" / "shortlist_q3527b.pt"
    if p.is_file():
        out.append((str(p), "source checkout"))
    p = packaged_shortlist()
    if p.is_file():
        out.append((str(p), "packaged"))
    return out


def _shortlist_default() -> str:
    """The best candidate available at IMPORT time.  ``_build`` re-resolves with the downloaded
    checkpoint dir and the real vocab size; this value is what reaches ``chained-flow env`` and
    any process that only reads the environment."""
    c = shortlist_candidates()
    return c[0][0] if c else ""


_FORK_MARKERS = (
    # (file relative to the vllm package root, a token that only the fork contains)
    ("v1/spec_decode/tree_state.py", "publish_gpu_tree"),
    ("v1/spec_decode/tree_gdn.py", "CF_GDN_DEFER"),
    ("v1/spec_decode/tree_attn_fused.py", "CF_TREE_FUSED_ATTN"),
    ("config/vllm.py", "CF_ASYNC_SPEC"),
)

_FORK: dict | None = None


def fork() -> dict:
    """Is the Chained-Flow vLLM fork installed?  ``{"present": bool, "root": str, "missing": [..]}``

    Probed by READING FILES, not by importing: ``import vllm`` costs seconds and drags in torch,
    and this has to be cheap enough to run from a shell prompt (``--sh``) as well as in-process.
    ``find_spec("vllm")`` on a top-level package does not execute it.
    """
    global _FORK
    if _FORK is not None:
        return _FORK
    root, missing = None, []
    try:
        spec = importlib.util.find_spec("vllm")
        locs = list(spec.submodule_search_locations) if spec is not None else []
        root = Path(locs[0]) if locs else None
    except Exception as e:                      # a broken/absent vllm is "no fork", not a crash
        missing.append(f"find_spec failed: {e!r}")
    if root is None:
        missing.append("vllm not importable")
    else:
        for rel, token in _FORK_MARKERS:
            f = root / rel
            try:
                if not f.is_file() or token not in f.read_text(errors="ignore"):
                    missing.append(rel)
            except Exception:
                missing.append(rel)
    _FORK = {"present": not missing, "root": str(root) if root else "",
             "missing": missing}
    return _FORK


# ------------------------------------------------------------------ decision registry

_STATE: dict[str, tuple[bool, str]] = {}
_ORDER: list[str] = []
_PRINTED = False


def note(flag: str, on: bool, why: str = "") -> None:
    """Record the state a flag ACTUALLY ended up in.  Last writer wins, so a `build`-time gate
    overrides the `import`-time proposal."""
    if flag not in _STATE:
        _ORDER.append(flag)
    _STATE[flag] = (on, why)


def state(flag: str) -> tuple[bool, str]:
    return _STATE.get(flag, (truthy(os.environ.get(flag)), "unmanaged"))


def _set(flag: str, value: str, why: str) -> None:
    """Apply a default.  An explicit setting always wins and is recorded as such.

    "Explicit" means the CALLER set it -- not that it is present in the environment.  The shell
    emitter (``--sh``) puts this very table into the environment before the process starts, and
    treating its output as a user request made every bench log claim `explicit CF_X=1` and made
    the "you asked for this and it cannot engage" banner fire on every chain run."""
    if was_explicit(flag):
        cur = _EXPLICIT[flag]
        note(flag, truthy(cur), f"explicit {flag}={cur}")
        return
    if value in ("", "0"):
        os.environ[flag] = value or "0"
        note(flag, False, why)
        return
    os.environ[flag] = value
    note(flag, True, why)


def disable(flag: str, why: str) -> None:
    """A gate failed at build time.  Write the env DOWN to 0 so every later reader (including
    the un-editable fork, which reads os.environ lazily) sees the real decision, and shout if
    the caller had asked for it explicitly."""
    asked = was_requested(flag)
    os.environ[flag] = "0"
    note(flag, False, ("FORCED-OFF although requested: " if asked else "") + why)
    if asked:
        # ONLY when the caller asked for it.  A default that a gate turns off is normal
        # operation and belongs in the summary line; a REQUEST that cannot be honoured means
        # somebody is about to draw a conclusion from an A/B that never happened.
        print(f"[cf-defaults] {flag} was requested but CANNOT ENGAGE: {why}. "
              f"Any A/B against it is meaningless.", flush=True)


# ------------------------------------------------------------------ apply (import time)

_APPLIED = False
_EXPLICIT: dict[str, str] = {}
# Set by the ``--sh`` emitter to the flag=value pairs IT defaulted, so the child process can tell
# its own defaults apart from the caller's requests.  Without it every bench run reports
# `explicit CF_X=1` for the whole table and cannot distinguish "you asked for this" from "we
# proposed this".
#
# It carries the VALUES, not just the names, and that is load-bearing: a launcher legitimately
# overwrites an emitted default AFTER eval-ing the emitter (bench_cf.sh's chain arm does exactly
# that with CF_ASYNC_SPEC=0, since the chain path has no tree to run it on).  Name-only
# provenance made `apply()` treat that deliberate 0 as its own default and write it back to 1.
#
# `apply()` writes it too, for the same reason one process further down: vLLM's engine core is
# a SPAWNED subprocess and inherits this environment.
_SHELL_MARK = "CF_DEFAULTS_FROM_SHELL"


def was_explicit(flag: str) -> bool:
    """True if the CALLER set this flag, as opposed to us defaulting it.  Snapshotted at
    ``apply()`` because everything after that point sees our own writes in ``os.environ`` and
    can no longer tell the two apart -- which is the difference between "refuse to start" and
    "quietly gate off", and between a loud "you asked for this and it cannot happen" and noise."""
    return flag in _EXPLICIT


def was_requested(flag: str) -> bool:
    """The caller explicitly asked for it to be ON."""
    return truthy(_EXPLICIT.get(flag))


def apply(force: bool = False) -> None:
    """Set the process-level defaults.  Idempotent; safe to call from anywhere."""
    global _APPLIED
    if _APPLIED and not force:
        return
    _APPLIED = True
    # Flags that OUR OWN shell emitter put in the environment AND that still hold the value it
    # wrote are defaults, not requests.  Anything the launcher changed afterwards is a request.
    from_shell = dict(kv.split("=", 1) for kv in
                      filter(None, os.environ.get(_SHELL_MARK, "").split(",")) if "=" in kv)
    _EXPLICIT.update({f: os.environ[f] for f in ALL
                      if f in os.environ and from_shell.get(f) != os.environ[f]})

    sl = _EXPLICIT.get("CF_SHORTLIST")
    if sl is None:
        sl = _shortlist_default()
        if sl:
            os.environ["CF_SHORTLIST"] = sl
            note("CF_SHORTLIST", True, f"default {os.path.basename(sl)}")
        else:
            note("CF_SHORTLIST", False,
                 "NO SHORTLIST FOUND -- the drafter will score the FULL lm_head at every "
                 "depth (~43-47% of the draft wasted). Build one: "
                 "`chained-flow build-shortlist`")
            # Only shout at someone who is actually about to run a drafter.  `apply()` now runs
            # inside the `vllm.general_plugins` entry point, i.e. in EVERY vLLM process --
            # including a plain `vllm serve` that has chained-flow installed and is not using
            # it.  A warning about our draft head in an unrelated engine's log is noise, and
            # noise is how real warnings stop being read.
            if os.environ.get("CF_DRAFTER_DIR"):
                print("[cf-defaults] WARNING: CF_SHORTLIST is unset and no shortlist was found "
                      "(looked for $CF_DRAFTER_DIR/shortlist.pt, out/flow/shortlist_q3527b.pt "
                      f"in a source checkout, and the packaged {packaged_shortlist()}) -- "
                      "running the FULL vocab head. This is the silent ~45%-of-the-draft waste; "
                      "build one with `chained-flow build-shortlist`.", flush=True)
    else:
        note("CF_SHORTLIST", bool(sl), f"explicit {sl or '<empty: full head>'}")

    for flag, val in DRAFTER.items():
        if flag == "CF_SHORTLIST":
            continue
        _set(flag, val, "default on")

    fk = fork()
    for flag, val in TREE.items():
        if fk["present"]:
            _set(flag, val, "default on (fork)")
        elif was_requested(flag):
            # the fork is absent: the flag is not merely off, it is NOT OFFERED.  Say so.
            note(flag, False, "fork_missing (not offered on stock vLLM)")
            os.environ[flag] = "0"
            print(f"[cf-defaults] {flag} was set but the Chained-Flow vLLM fork is NOT "
                  f"installed (missing: {', '.join(fk['missing'][:3])}) -- ignoring it. "
                  f"The DRAFTER flags are unaffected; they run on stock vLLM.", flush=True)
        else:
            os.environ[flag] = "0"
            note(flag, False, "fork_missing")

    # ---- provenance ACROSS A PROCESS BOUNDARY --------------------------------------------
    # vLLM starts its engine core in a SPAWNED subprocess.  By the time the child runs, every
    # value this function wrote is an ordinary environment variable, indistinguishable from a
    # caller's request -- so the child treated the whole table as explicit and shouted
    # "CF_CUDA_BLOCK was requested but CANNOT ENGAGE" about a default it had proposed itself,
    # which is precisely the false alarm this module exists to prevent.  Worse, an inherited
    # CF_SHORTLIST read as explicit skips the candidate search, so the drafter checkpoint's own
    # shortlist.pt could never win in the process that actually loads the drafter.
    #
    # Same problem the `--sh` emitter already solved for the shell, so it reuses the same
    # marker and the same rule: record flag=value for everything WE set, so a value the caller
    # changes afterwards still reads as a request.
    mine = dict(from_shell)
    mine.update({f: os.environ[f] for f in ALL
                 if f in os.environ and not was_explicit(f) and "," not in os.environ[f]})
    os.environ[_SHELL_MARK] = ",".join(f"{f}={v}" for f, v in mine.items())


# ------------------------------------------------------------------ finalize (build time)


def finalize_cuda_block(drafter) -> None:
    """Resolve CF_CUDA_BLOCK / CF_CUDA_PAIR against the REAL drafter: shape gate first (cheap,
    pure python) and only then the extension build (expensive, and pointless if the shape is
    already rejected).

    This runs in ``_build``, i.e. before the first forward and before any torch.compile /
    cudagraph capture, so the resolved env value is what ``HiddenKVFlowExpert._cf_fused_off``
    latches onto.  Doing the build probe on the compiled path instead would put a 60 s ninja
    invocation inside a traced region.
    """
    from chained_flow import cuda_block

    if not truthy(os.environ.get("CF_CUDA_BLOCK")):
        note("CF_CUDA_BLOCK", False, state("CF_CUDA_BLOCK")[1])
        disable("CF_CUDA_PAIR", "needs CF_CUDA_BLOCK")
        return

    try:
        experts = list(drafter._chunk_experts())
    except Exception as e:
        disable("CF_CUDA_BLOCK", f"no chunk experts ({e!r})")
        disable("CF_CUDA_PAIR", "needs CF_CUDA_BLOCK")
        return

    # --- shape gate, mirroring HiddenKVFlowExpert._cf_fused_runner exactly -------------
    bad = []
    for i, ex in enumerate(experts):
        b0 = ex.blocks[0]
        D = ex.hidden_size
        fm = b0.ffn[1].out_features // D
        # the kernel sees the CONCATENATED rows, not the chunk: chunk i is preceded by
        # i * chunk_size previous-hidden rows, so S grows 4, 8, ... across the experts.
        S = (i + 1) * ex.chunk_size
        if D not in cuda_block.SUPPORTED_D:
            bad.append(f"expert{i} D={D} not in {cuda_block.SUPPORTED_D}")
        if fm not in cuda_block.SUPPORTED_FM:
            bad.append(f"expert{i} ffn=x{fm} not in {cuda_block.SUPPORTED_FM}")
        if b0.self_attn.num_heads != cuda_block.NUM_HEADS:
            bad.append(f"expert{i} heads={b0.self_attn.num_heads} != {cuda_block.NUM_HEADS}")
        if S not in cuda_block.SUPPORTED_S:
            bad.append(f"expert{i} S={S} not in {cuda_block.SUPPORTED_S}")
    if bad:
        disable("CF_CUDA_BLOCK", "shape unsupported: " + "; ".join(bad))
        disable("CF_CUDA_PAIR", "needs CF_CUDA_BLOCK")
        return

    # --- build gate: does the extension actually compile on this box? -----------------
    ok, err = cuda_block.available()
    if not ok:
        disable("CF_CUDA_BLOCK",
                f"CUDA extension failed to build ({err}); falling back to PyTorch")
        disable("CF_CUDA_PAIR", "needs CF_CUDA_BLOCK")
        return

    ex0 = experts[0]
    D = ex0.hidden_size
    Ss = ",".join(str((i + 1) * ex.chunk_size) for i, ex in enumerate(experts))
    note("CF_CUDA_BLOCK", True, f"D={D} ffn=x{ex0.blocks[0].ffn[1].out_features // D} "
                                f"S={Ss} L={len(ex0.blocks)} ext=built")

    if not truthy(os.environ.get("CF_CUDA_PAIR")):
        note("CF_CUDA_PAIR", False, state("CF_CUDA_PAIR")[1])
    elif len(experts) != 2:
        disable("CF_CUDA_PAIR", f"{len(experts)} chunk experts, run_pair needs exactly 2")
    else:
        note("CF_CUDA_PAIR", True, f"2 experts S={Ss}")


def finalize_compile(p) -> bool:
    """CF_COMPILE: torch.compile the flow net.  Gated on inductor having a backend at all.

    Called BEFORE ``fused.compile_flow`` so a failed gate means the wrap never happens -- the
    alternative (wrap and hope) defers the failure into the first draft, which on the shipping
    path is inside a cudagraph capture, where an exception is not something the engine survives.

    ``torch.compile`` is lazy, so "the gate passed" means "inductor can be asked", not "it
    compiled".  That is the honest claim, and the compiled callable is installed on the drafter
    object, so ``_build`` prints ``compile=<mode>`` off the object rather than off the env var.
    """
    if not truthy(os.environ.get("CF_COMPILE")):
        note("CF_COMPILE", False, state("CF_COMPILE")[1])
        return False
    if truthy(os.environ.get("TORCHDYNAMO_DISABLE")):
        disable("CF_COMPILE", "TORCHDYNAMO_DISABLE is set: torch.compile is a no-op here")
        return False
    if importlib.util.find_spec("triton") is None:
        disable("CF_COMPILE",
                "no inductor backend (triton is not importable) -- the flow net stays eager, "
                "which measured 9.79 vs 5.26 ms per draft at 27B")
        return False
    note("CF_COMPILE", True, f"flow net, mode={p.compile_mode}")
    return True


def finalize_proposer(p) -> None:
    """Resolve the flags whose gate is a property of the PROPOSER / ENGINE, not the drafter."""
    # --- CF_RING_TRIM: tree-only, and refuses under CF_NONGREEDY_CHAIN ----------------
    if not truthy(os.environ.get("CF_RING_TRIM")):
        note("CF_RING_TRIM", False, state("CF_RING_TRIM")[1])
    elif not p.branching:
        disable("CF_RING_TRIM", "chain mode: a chain step can commit more than tree_depth+1")
    elif truthy(os.environ.get("CF_NONGREEDY_CHAIN")):
        disable("CF_RING_TRIM", "CF_NONGREEDY_CHAIN: a chain step can overflow the trimmed ring")
    else:
        note("CF_RING_TRIM", True, f"{p._wmax()} of {p.K + 1} slots")

    # --- CF_PATH_TRIM ------------------------------------------------------------------
    if p.path_trim:
        note("CF_PATH_TRIM", True, f"nlive<=depth {p.tree_depth} of order {p.order}")

    # --- two-pass candidate head -------------------------------------------------------
    rows = int(p._hw.shape[0])
    if not p.twopass_m:
        # _build already zeroed it if the default did not fit the head; distinguish the cases.
        if truthy(os.environ.get("CF_TWOPASS_M")) and not was_explicit("CF_TWOPASS_M"):
            disable("CF_TWOPASS_M", f"default M={os.environ.get('CF_TWOPASS_M')} "
                                    f">= head rows {rows}")
        else:
            note("CF_TWOPASS_M", False, state("CF_TWOPASS_M")[1])
        note("CF_TWOPASS_SHARED", False, "needs CF_TWOPASS_M")
    else:
        note("CF_TWOPASS_M", True, f"M={p.twopass_m} of {rows} rows")
        note("CF_TWOPASS_SHARED", bool(p.twopass_shared),
             "one candidate set for all depths" if p.twopass_shared else state("CF_TWOPASS_SHARED")[1])

    # CF_DRAFT_EARLY is the one flag that can be forced ON against an explicit request to turn
    # it OFF: CF_ASYNC_SPEC needs the `_sample` hook and sets `early` itself.  Silently honouring
    # neither the request nor the summary is exactly the failure mode this module exists to stop,
    # so the override is named in the line and shouted once.
    if p.early and _EXPLICIT.get("CF_DRAFT_EARLY") is not None \
            and not truthy(_EXPLICIT["CF_DRAFT_EARLY"]):
        note("CF_DRAFT_EARLY", True, "FORCED ON by CF_ASYNC_SPEC (which needs the _sample hook) "
                                     "despite CF_DRAFT_EARLY=0")
        print("[cf-defaults] CF_DRAFT_EARLY=0 was requested but CF_ASYNC_SPEC=1 REQUIRES the "
              "side-stream draft, so it stays ON. Set CF_ASYNC_SPEC=0 too if you meant to "
              "measure the draft-early A/B.", flush=True)
    else:
        note("CF_DRAFT_EARLY", bool(p.early),
             "side-stream draft" if p.early else state("CF_DRAFT_EARLY")[1])
    note("CF_FUSE_PATH", bool(p.fuse_path),
         "path head -> one baddbmm" if p.fuse_path else state("CF_FUSE_PATH")[1])
    # Read off the BUILT head, not off CF_SHORTLIST: the env var is a path, and a path that
    # loaded, failed the vocab guard and was dropped looks identical to one that engaged.
    note("CF_SHORTLIST", p._sl is not None,
         f"{p._sl.numel()} rows, {p._sl_src}" if p._sl is not None else p._sl_why)


def finalize_async(p, vllm_config) -> bool:
    """CF_ASYNC_SPEC: gated on the ENGINE, not on the env var.

    Async scheduling for a ``custom_class`` proposer has to be requested BEFORE the engine
    config is built, by whoever relaxes ``vllm/config/vllm.py``'s guard:

    * on STOCK vLLM, ``chained_flow.vllm_plugin.async_guard`` (a ``vllm.general_plugins`` entry
      point) rebinds ``NgramGPUTypes`` when ``CF_ASYNC_SPEC`` is on, which relaxes both the
      explicit-request and the auto-decide branch, and then asserts the resolved value;
    * on the FORK, ``config/vllm.py`` has the same conditional inline, but only on the
      explicit-request branch -- so there the caller must ALSO pass ``async_scheduling=True``.

    Either way the engine's own resolved value is the gate here, checked BEFORE any async hook
    is installed: handing a GPU draft tensor to a synchronous ``_prepare_inputs`` is not a lost
    speedup, it is a draft that is silently ignored.

    TREE mode needs one thing more.  The tree SHAPE (parents -> ancestor mask + GDN chain) is
    consumed by ``vllm.v1.spec_decode.tree_state``, which exists only in the fork, so
    ``CF_ASYNC_SPEC`` in tree mode without the fork is refused rather than half-installed.
    CHAIN mode has no shape to hand over and runs on unmodified vLLM -- that is the
    fork-free shipping path.
    """
    want = truthy(os.environ.get("CF_ASYNC_SPEC"))
    if not want:
        note("CF_ASYNC_SPEC", False, state("CF_ASYNC_SPEC")[1])
        return False
    if p.branching and not fork()["present"]:
        disable("CF_ASYNC_SPEC",
                "tree mode needs the fork's tree_state hand-off (chain mode does not: run "
                "without VLLM_SPEC_TREE for the fork-free async path)")
        return False
    if not getattr(vllm_config.scheduler_config, "async_scheduling", False):
        disable("CF_ASYNC_SPEC",
                "engine async_scheduling is OFF -- it must be requested BEFORE the engine is "
                "built. On stock vLLM the `vllm.general_plugins` entry point does that "
                "(pip install chained-flow, and check for the '[cf-plugin] ... guard relaxed' "
                "line); on the fork, pass async_scheduling=True to LLM() "
                "(bench_cf.sh: CF_ASYNC_SCHED=1)")
        return False
    note("CF_ASYNC_SPEC", True,
         "engine async ON, GPU-resident tree" if p.branching else "engine async ON, chain")
    return True


def finalize_tree(p) -> None:
    """The fork-side flags.  Nothing to gate beyond fork presence and tree mode: each kernel
    carries its own internal fallback (``CF_TREE_FUSED_ATTN`` refuses N>=128 and drops to the
    Triton path; ``CF_GDN_DEFER`` is a no-op outside a tree step).  What we CAN do is stop
    offering them when the run is not a tree run at all."""
    if not fork()["present"]:
        return
    if p.branching:
        note("CF_GDN_DEFER", truthy(os.environ.get("CF_GDN_DEFER")),
             f"bv={os.environ.get('CF_GDN_BV', '32')}"
             if truthy(os.environ.get("CF_GDN_DEFER")) else state("CF_GDN_DEFER")[1])
        # BV is a TUNABLE of the deferred path, not an independent switch: with the defer off
        # the fork falls back to its stock 32 and the value here is inert.
        note("CF_GDN_BV", truthy(os.environ.get("CF_GDN_DEFER")),
             f"{os.environ.get('CF_GDN_BV') or 32} v-rows/program"
             if truthy(os.environ.get("CF_GDN_DEFER")) else "needs CF_GDN_DEFER (stock BV=32)")
        note("CF_TREE_FUSED_ATTN", truthy(os.environ.get("CF_TREE_FUSED_ATTN")),
             f"N={p.N} (kernel refuses N>=128)"
             if truthy(os.environ.get("CF_TREE_FUSED_ATTN"))
             else state("CF_TREE_FUSED_ATTN")[1])
        note("CF_TREE_FULLCG", truthy(os.environ.get("CF_TREE_FULLCG")),
             f"FULL cudagraph, ntok={p.N + 1}"
             if truthy(os.environ.get("CF_TREE_FULLCG")) else state("CF_TREE_FULLCG")[1])
    else:
        for f in ("CF_GDN_DEFER", "CF_GDN_BV", "CF_TREE_FUSED_ATTN", "CF_TREE_FULLCG"):
            if truthy(os.environ.get(f)):
                disable(f, "chain mode: there is no tree to apply it to")
            else:
                note(f, False, "chain mode")


# ------------------------------------------------------------------ the summary line


_SHORT = {
    "CF_SHORTLIST": "shortlist", "CF_COMPILE": "compile",
    "CF_CUDA_BLOCK": "cuda_block", "CF_CUDA_PAIR": "cuda_pair",
    "CF_PATH_TRIM": "path_trim", "CF_RING_TRIM": "ring_trim", "CF_TWOPASS_M": "twopass",
    "CF_TWOPASS_SHARED": "twopass_shared", "CF_DRAFT_EARLY": "draft_early",
    "CF_FUSE_PATH": "fuse_path", "CF_GDN_DEFER": "gdn_defer", "CF_GDN_BV": "gdn_bv",
    "CF_TREE_FUSED_ATTN": "tree_fused_attn", "CF_TREE_FULLCG": "tree_fullcg",
    "CF_ASYNC_SPEC": "async_spec",
}
_KEYS = list(DRAFTER) + list(TREE)


def summary() -> str:
    """ONE line: every flag, the state it is ACTUALLY in, and the reason for anything off.

    Deliberately built from the recorded decisions and not from ``os.environ``: the whole point
    is that the env var is the request and this is the outcome."""
    on, off = [], []
    for flag in _KEYS:
        s, why = _STATE.get(flag, (None, "not reached"))
        name = _SHORT.get(flag, flag)
        if s:
            on.append(f"{name}({why})" if why and not why.startswith("default") else name)
        else:
            off.append(f"{name}({why})")
    return ("[cf-defaults] ON: " + (" ".join(on) or "-")
            + "  |  OFF: " + (" ".join(off) or "-")
            + "  |  " + build())


def build() -> str:
    """Which BUILD this is: forked vLLM or stock + the entry-point plugin.

    Printed next to the flags because "which vLLM am I on" is the first question asked of any
    number this repo produces, and inferring it from the flag list is exactly the kind of
    guesswork that produced the async-scheduling mess in the first place.
    """
    fk = fork()
    if fk["present"]:
        return "build=forked-vllm (tree available)"
    try:
        from chained_flow.vllm_plugin import async_guard

        return f"build=stock-vllm (chain only), async-guard {async_guard.status()}"
    except Exception as e:                                  # noqa: BLE001 - reporting only
        return f"build=stock-vllm (chain only), async-guard unknown ({e!r})"


def print_summary() -> None:
    global _PRINTED
    if _PRINTED:
        return
    _PRINTED = True
    print(summary(), flush=True)


# ------------------------------------------------------------------ shell emitter


def _sh() -> str:
    """``eval "$(python .../defaults.py --sh)"`` -- the ONLY way to default ``CF_ASYNC_SPEC``,
    which the fork reads before anything imports us (see the module docstring).

    Emits nothing for a flag the caller already set, so ``CF_X=0 ./bench_cf.sh ...`` still wins.
    Runs as a FILE, not ``-m``: executing the file directly skips ``chained_flow/__init__``
    and therefore skips importing torch, so this costs ~40 ms rather than ~5 s per bench run.

    "The caller already set it" is ``was_explicit``, NOT "it is in os.environ right now".  Those
    two are the same thing only when ``apply()`` has not run yet -- which is true for the FILE
    invocation and false for ``chained-flow env``, because importing the package runs ``apply()``
    (see ``chained_flow/__init__``) and therefore puts the whole table into ``os.environ`` before
    this function is ever called.  Reading the live environment there made the emitter skip every
    flag and print an empty export list: a command documented as ``eval "$(chained-flow env)"``
    that silently exported nothing.  ``_EXPLICIT`` is snapshotted inside ``apply()``, i.e. at the
    one moment the two are still distinguishable, so it is correct for both entry points.
    """
    out, mine = [], []
    fk = fork()
    apply()
    for flag in _KEYS:
        if was_explicit(flag):
            continue
        v = os.environ.get(flag)
        if v:
            out.append(f"export {flag}={v}")
            # A value containing the separator would corrupt the marker; omit it rather than
            # mis-parse.  The only value that could (a shortlist PATH with a comma in it) then
            # degrades to being reported as "explicit", which is cosmetic and not a behaviour
            # change -- it is already the value we would have chosen.
            if "," not in v:
                mine.append(f"{flag}={v}")
    # Provenance: tell the child which of these it is looking at are OUR defaults rather than
    # the caller's requests, so it can report them honestly and not shout about them.  Values
    # included so that a launcher overwriting one AFTER this eval still reads as a request.
    out.append(f"export {_SHELL_MARK}='{','.join(mine)}'")
    # Which vLLM the launcher is about to run against.  A shell script cannot probe this
    # cheaply itself (it would have to import vllm), and the two builds want different engine
    # flags: the fork only relaxes the EXPLICIT async-scheduling branch, so it needs
    # `async_scheduling=True` passed in, while stock + our entry point relaxes the AUTO-DECIDE
    # branch and must be left on auto so the plugin's post-resolution assert can fire.
    out.append(f"export CF_VLLM_BUILD={'fork' if fk['present'] else 'stock'}")
    out.append(f"# fork={'yes' if fk['present'] else 'NO -- tree flags not offered'}")
    if not fk["present"]:
        out.append(f"# fork missing: {', '.join(fk['missing'][:4])}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys

    if "--sh" in sys.argv:
        print(_sh())
    else:
        apply()
        print(summary())
        fk = fork()
        print(f"[cf-defaults] fork: present={fk['present']} root={fk['root']} "
              f"missing={fk['missing']}")
        for f, why in RETIRED.items():
            print(f"[cf-defaults] RETIRED {f}: {why}")
