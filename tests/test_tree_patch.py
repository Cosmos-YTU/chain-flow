"""``chained-flow tree-patch``: the invariants that keep a half-applied fork impossible.

The failure this guards against is not a crash.  A vLLM with five of the six patched files is a
tree verifier with a stale ancestor mask, which produces MALFORMED DRAFTS at full speed -- wrong
text, no traceback, plausible throughput.  So the tests here are about refusals and about
byte-exactness, not about happy-path output.

Nothing here needs vLLM, a GPU, or a drafter: a synthetic "install" (a package dir plus a
hand-written RECORD) exercises the same code the real command runs, and one test parses the
SHIPPED patch to pin its file list.
"""
from __future__ import annotations

import base64
import csv
import hashlib
from pathlib import Path

import pytest

from chained_flow import tree_patch as tp


# --------------------------------------------------------------------------- shipped patch


def test_shipped_patch_touches_exactly_the_documented_files():
    """6 modified + 8 added, and every path inside the vllm package.

    The count is asserted rather than merely listed because the patch shipped for a while with
    the three ``.cu`` sources MISSING while their ``.py`` JIT loaders were present: applying it
    produced an install that passed every marker check and then raised inside
    ``torch.utils.cpp_extension.load`` at model-load time, with ``CF_TREE_FUSED_ATTN`` on by
    default because the fork "was present".
    """
    fps = tp.load_patch()
    mod = sorted(f.rel for f in fps if not f.is_new)
    new = sorted(f.rel for f in fps if f.is_new)
    assert mod == [
        "config/vllm.py",
        "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py",
        "v1/attention/backends/flash_attn.py",
        "v1/sample/rejection_sampler.py",
        "v1/spec_decode/metadata.py",
        "v1/worker/gpu_model_runner.py",
    ]
    assert new == [
        "v1/spec_decode/tree_attn_fused.cu",
        "v1/spec_decode/tree_attn_fused.py",
        "v1/spec_decode/tree_gdn.py",
        "v1/spec_decode/tree_gdn_factor.cu",
        "v1/spec_decode/tree_gdn_factor.py",
        "v1/spec_decode/tree_gdn_verify.cu",
        "v1/spec_decode/tree_gdn_verify.py",
        "v1/spec_decode/tree_state.py",
    ]


def test_every_py_loader_gets_its_cu_source():
    """Any added ``.py`` that JIT-compiles a sibling ``.cu`` must ship that ``.cu`` too."""
    fps = tp.load_patch()
    added = {f.rel for f in fps if f.is_new}
    pairs = 0
    for fp in fps:
        if not fp.is_new or not fp.rel.endswith(".py"):
            continue
        cu = fp.rel[:-3] + ".cu"
        if Path(cu).name in fp.new_content():
            pairs += 1
            assert cu in added, (
                f"{fp.rel} loads {Path(cu).name} at runtime but the patch does not add it")
    # Otherwise a refactor that renamed the sources would make this test vacuously green.
    assert pairs == 3, f"expected 3 .py/.cu loader pairs, found {pairs}"


def test_shipped_patch_version_matches_the_pin():
    assert tp.PIN in tp.PATCH_NAME
    assert tp.patch_path().is_file()


# --------------------------------------------------------------------------- the diff engine


SIMPLE = """--- a/vllm/x.py
+++ b/vllm/x.py
@@ -1,4 +1,5 @@
 one
-two
+TWO
+two-and-a-half
 three
 four
--- a/vllm/new.cu
+++ b/vllm/new.cu
@@ -0,0 +1,3 @@
+// a .cu body may contain lines that look like diff headers:
+--- not a header
++++ also not a header
"""


def test_hunk_bodies_are_consumed_by_count_not_by_scanning():
    """The three added ``.cu`` files contain ``---`` and ``+++`` lines in their own comments.  A
    parser that looks for the next ``---`` to end a section truncates them silently, which is the
    exact shape of bug this module exists to make impossible."""
    fps = tp.parse_patch(SIMPLE)
    assert [f.rel for f in fps] == ["x.py", "new.cu"]
    assert fps[1].is_new
    assert fps[1].new_content() == (
        "// a .cu body may contain lines that look like diff headers:\n"
        "--- not a header\n"
        "+++ also not a header\n")


def test_apply_and_reverse_round_trip_is_byte_exact():
    orig = "one\ntwo\nthree\nfour\n"
    fp = tp.parse_patch(SIMPLE)[0]
    patched = tp.transform(orig, fp, reverse=False)
    assert patched == "one\nTWO\ntwo-and-a-half\nthree\nfour\n"
    assert tp.transform(patched, fp, reverse=True) == orig


def test_a_single_changed_context_line_refuses_instead_of_fuzzing():
    """No fuzz, no offset search.  ``patch(1)`` would happily apply this with an offset and
    report success; a near-miss vLLM is precisely where that is fatal."""
    fp = tp.parse_patch(SIMPLE)[0]
    with pytest.raises(tp.PatchError) as e:
        tp.transform("one\ntwo\nTHREE\nfour\n", fp, reverse=False)
    assert "three" in str(e.value)


def test_trailing_newline_is_preserved_exactly():
    fp = tp.parse_patch(SIMPLE)[0]
    assert tp.transform("one\ntwo\nthree\nfour\n", fp, reverse=False).endswith("four\n")


def test_patch_outside_the_vllm_package_is_refused():
    with pytest.raises(tp.PatchError):
        tp.parse_patch("--- a/etc/passwd\n+++ b/etc/passwd\n@@ -0,0 +1,1 @@\n+root\n")


# --------------------------------------------------------------------------- a fake install


def _b64(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


@pytest.fixture
def install(tmp_path):
    """A minimal site-packages: ``vllm/x.py`` plus a RECORD that hashes it."""
    site = tmp_path / "site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "vllm" / "x.py").write_text("one\ntwo\nthree\nfour\n")
    (site / "vllm" / "_version.py").write_text(f"__version__ = {tp.PIN!r}\n")
    di = site / f"vllm-{tp.PIN}.dist-info"
    di.mkdir()
    with open(di / "RECORD", "w", newline="") as f:
        w = csv.writer(f)
        for rel in ("vllm/x.py", "vllm/_version.py"):
            b = (site / rel).read_bytes()
            w.writerow([rel, _b64(b), len(b)])
        w.writerow([f"vllm-{tp.PIN}.dist-info/RECORD", "", ""])
    return tp.Target(site / "vllm", site, di, tp.PIN, tp.PIN)


def test_record_scan_sees_a_clean_install(install):
    s = tp.scan(install)
    assert (s.modified, s.missing, s.added) == ([], [], [])


def test_record_scan_names_the_file_that_differs(install):
    (install.pkg / "x.py").write_text("one\nCHANGED\nthree\nfour\n")
    (install.pkg / "stowaway.py").write_text("hi\n")
    s = tp.scan(install)
    assert s.modified == ["vllm/x.py"]
    assert s.added == ["vllm/stowaway.py"]


def test_pycache_is_not_reported_as_an_added_file(install):
    c = install.pkg / "__pycache__"
    c.mkdir()
    (c / "x.cpython-312.pyc").write_bytes(b"\x00")
    assert tp.scan(install).added == []


def test_state_machine_pristine_patched_unknown(install):
    fps = tp.parse_patch(SIMPLE)
    ents = tp.record_entries(install)

    assert tp.overall(tp.classify(install, fps, ents), fps) == tp.CLEAN

    # apply by hand, exactly as cmd_apply would
    (install.pkg / "x.py").write_text(tp.transform(tp.read(install.pkg / "x.py"), fps[0], False))
    (install.pkg / "new.cu").write_text(fps[1].new_content())
    assert tp.overall(tp.classify(install, fps, ents), fps) == tp.APPLIED

    # a stray local edit on top of the patch is neither state
    (install.pkg / "x.py").write_text(tp.read(install.pkg / "x.py") + "# stray\n")
    v = tp.classify(install, fps, ents)
    assert v["x.py"] == tp.UNKNOWN
    assert tp.overall(v, fps) == tp.MIXED


def test_added_file_with_the_right_name_but_wrong_content_is_unknown(install):
    """"The file exists" is not the test.  An older or hand-edited ``tree_state.py`` is exactly
    how a fork ends up half-applied while every marker check passes."""
    fps = tp.parse_patch(SIMPLE)
    (install.pkg / "new.cu").write_text("// something else entirely\n")
    assert tp.classify(install, fps, tp.record_entries(install))["new.cu"] == tp.UNKNOWN


def test_version_gate_refuses_a_near_miss_and_a_disagreement(install):
    assert tp.check_version(install) == []
    bad = tp.Target(install.pkg, install.site, install.dist_info, "0.26.0", "0.26.0")
    assert any("0.26.0" in m for m in tp.check_version(bad))
    swapped = tp.Target(install.pkg, install.site, install.dist_info, tp.PIN, "0.25.2")
    assert any("disagrees" in m for m in tp.check_version(swapped))
    no_meta = tp.Target(install.pkg, install.site, None, None, tp.PIN)
    assert any("dist-info" in m for m in tp.check_version(no_meta))


def test_apply_then_revert_restores_the_record_hashes(install, capsys):
    fps = tp.parse_patch(SIMPLE)
    before = (install.pkg / "x.py").read_bytes()
    assert tp.cmd_apply(install, fps, dry=False) == 0
    assert (install.pkg / "new.cu").is_file()
    assert (install.pkg / "x.py").read_bytes() != before

    assert tp.cmd_apply(install, fps, dry=False) == 0          # idempotent
    assert "ALREADY APPLIED" in capsys.readouterr().out

    assert tp.cmd_revert(install, fps, dry=False) == 0
    assert (install.pkg / "x.py").read_bytes() == before        # byte-identical
    assert not (install.pkg / "new.cu").exists()                # added file removed
    s = tp.scan(install)
    assert (s.modified, s.missing, s.added) == ([], [], [])
    assert tp.cmd_revert(install, fps, dry=False) == 0          # idempotent


def test_dry_run_writes_nothing(install):
    fps = tp.parse_patch(SIMPLE)
    before = (install.pkg / "x.py").read_bytes()
    assert tp.cmd_apply(install, fps, dry=True) == 0
    assert (install.pkg / "x.py").read_bytes() == before
    assert not (install.pkg / "new.cu").exists()


def test_apply_and_revert_refuse_a_half_applied_install(install):
    fps = tp.parse_patch(SIMPLE)
    (install.pkg / "new.cu").write_text("// an older version of this file\n")
    assert tp.cmd_apply(install, fps, dry=False) == 2
    assert tp.cmd_revert(install, fps, dry=False) == 2
    assert (install.pkg / "new.cu").read_text() == "// an older version of this file\n"


def test_apply_refuses_a_wrong_version_before_touching_anything(install):
    fps = tp.parse_patch(SIMPLE)
    wrong = tp.Target(install.pkg, install.site, install.dist_info, "0.26.0", "0.26.0")
    assert tp.cmd_apply(wrong, fps, dry=False) == 2
    assert not (install.pkg / "new.cu").exists()
    assert tp.scan(install).modified == []
