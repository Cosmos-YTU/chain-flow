"""The flag table's own invariants.

`defaults.py` exists because a flag that is on by default, off by default, or silently gated is
indistinguishable from the outside unless something PRINTS the resolved state. These tests are
about that promise rather than about any one flag: every proposed flag must reach the summary
line, and every flag the benchmark scripts turn on must be in the table (`CF_COMPILE` was not,
for months -- it defaulted to 0 in the proposer while `bench_cf.sh` hard-coded 1, so every
published number came from a path a pip user was not on).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from chain_flow import defaults


REPO = defaults.repo_root()


def test_every_flag_in_the_table_has_a_short_name_for_the_summary():
    missing = [f for f in defaults.ALL if f not in defaults._SHORT]
    assert not missing, f"{missing} would print as raw env names on the [cf-defaults] line"


def test_cf_compile_is_a_gated_default_not_a_hidden_hard_code():
    assert defaults.DRAFTER["CF_COMPILE"] == "1"
    assert defaults._SHORT["CF_COMPILE"] == "compile"


def test_every_cf_flag_the_bench_scripts_export_is_in_the_table_or_documented():
    """The specific drift this catches: a bench arm exporting `CF_X=1` that the table does not
    know about, i.e. a published number taken on an undocumented path."""
    known = set(defaults.ALL) | set(defaults.RETIRED) | {
        # engine / harness plumbing, not drafter behaviour -- these belong to the scripts.
        "CF_TAG", "CF_BATCH", "CF_POFF", "CF_PROMPTS", "CF_MAXTOK", "CF_ACCEPT", "CF_MODEL",
        "CF_DRAFTER_DIR", "CF_GMU", "CF_MODE", "CF_K", "CF_TREE_KEEP", "CF_TREE_DEPTH",
        "CF_ASYNC_SCHED", "CF_PY", "CF_GPU", "CF_OUT", "CF_VLLM_BUILD", "CF_DEFAULTS_FROM_SHELL",
        "CF_PLENS", "CF_OUT_JSON",
        # set to the value the proposer already defaults it to (flow_proposer: CF_CUDAGRAPH
        # defaults "1"), i.e. redundant rather than a hidden difference. Unlike CF_COMPILE was.
        "CF_CUDAGRAPH",
    }
    sh = (REPO / "vllm" / "bench_cf.sh").read_text()
    # `export CF_A=1 CF_B=1` and `export CF_A` both appear in the arms
    exported = set(re.findall(r"\bexport ((?:CF_\w+[= ]?\S*\s*)+)", sh))
    names = {n for grp in exported for n in re.findall(r"CF_\w+", grp)}
    unknown = names - known
    assert not unknown, (f"{sorted(unknown)} are set by bench_cf.sh but are not in the defaults "
                         f"table -- add them with a gate and a reason, or they are an invisible "
                         f"difference between the benchmark and a pip install")


def test_summary_names_every_flag_exactly_once():
    defaults.apply(force=True)
    line = defaults.summary()
    for flag in defaults.ALL:
        short = defaults._SHORT[flag]
        # word-boundary match: `twopass` must not be satisfied by `twopass_shared`
        assert re.search(rf"(?<![\w_]){re.escape(short)}(?![\w_])", line), \
            f"{flag} ({short}) is missing from the [cf-defaults] line: {line}"


def test_the_shell_emitter_runs_as_a_bare_file_without_torch():
    """`bench_cf.sh` runs `python .../defaults.py --sh` as a FILE precisely so it does not import
    torch (~40 ms instead of ~5 s). Importing anything from the package here would break that,
    which is why the packaged-shortlist path is duplicated rather than imported."""
    out = subprocess.run([sys.executable, str(Path(defaults.__file__)), "--sh"],
                         capture_output=True, text=True, timeout=120,
                         env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"})
    assert out.returncode == 0, out.stderr
    assert "export CF_COMPILE=1" in out.stdout
    assert "torch" not in out.stderr
    # CF_SHORTLIST must NOT be exported, for the reason the next test spells out: an inherited
    # CF_SHORTLIST reads as EXPLICIT in the child and skips the candidate search in the one
    # process that actually loads the drafter -- overriding a checkpoint's own `shortlist.pt`,
    # which is documented as winning, and pinning every benchmark in this repo to the source
    # checkout's copy so the PACKAGED list a pip user gets is never exercised.
    assert "export CF_SHORTLIST=" not in out.stdout, (
        "the shell emitter is exporting CF_SHORTLIST again; its resolution is deliberately "
        "deferred to FlowDrafterProposer._build(), which is the only place that knows the "
        "downloaded drafter dir and the target's vocab size")


def test_apply_marks_its_own_defaults_so_a_spawned_child_does_not_read_them_as_requests(
        monkeypatch):
    """vLLM's engine core is a SPAWNED subprocess and inherits this environment.

    Without a provenance marker every default the parent applied arrives in the child as an
    ordinary env var, so the child reports the whole table as `explicit` and shouts
    "CF_CUDA_BLOCK was requested but CANNOT ENGAGE" about a default it proposed itself -- and,
    worse, an inherited CF_SHORTLIST reads as explicit and skips the candidate search in the
    one process that actually loads the drafter.
    """
    for f in list(defaults.ALL) + [defaults._SHELL_MARK]:
        monkeypatch.delenv(f, raising=False)
    monkeypatch.setattr(defaults, "_EXPLICIT", {})
    defaults.apply(force=True)
    parent_env = {f: os.environ[f] for f in defaults.ALL if f in os.environ}
    assert defaults._EXPLICIT == {}, "nothing was explicit in the parent either"

    # the child: same environment, fresh module state
    monkeypatch.setattr(defaults, "_EXPLICIT", {})
    defaults.apply(force=True)
    assert defaults._EXPLICIT == {}, (
        f"the child read its parent's defaults as caller requests: "
        f"{sorted(defaults._EXPLICIT)}")

    # ...but a value the caller CHANGES after the parent applied it is still a request
    monkeypatch.setenv("CF_CUDA_BLOCK", "0")
    monkeypatch.setattr(defaults, "_EXPLICIT", {})
    defaults.apply(force=True)
    assert defaults._EXPLICIT == {"CF_CUDA_BLOCK": "0"}
    assert parent_env["CF_CUDA_BLOCK"] == "1"


@pytest.mark.parametrize("flag", sorted(defaults.RETIRED))
def test_retired_flags_are_not_also_proposed(flag):
    """A flag cannot be both 'measured negative, never default it on' and a default."""
    assert flag not in defaults.ALL
