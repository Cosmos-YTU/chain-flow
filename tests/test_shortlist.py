"""The shipped shortlist and the vocab guard that decides whether it may be used.

Two failures are worth pinning down here, and neither of them raises anything on its own:

1. **The list not being there.** `CF_SHORTLIST` used to default to a path inside this repo's
   `out/` directory, so every `pip install` silently scored the full 248,320-row `lm_head` at
   every draft depth -- 145.7 tok/s (1.04x) instead of 157.8 (1.13x) at 4B, with nothing in the
   log to say which of the two you were getting.
2. **The list being there and being WRONG.** A shortlist is a list of integers; against a
   different vocabulary every id names a different token. The old loader clamped ids into range
   (`sl[sl < V]`) and carried on, which turns a nonsense list into a quietly lower accept. The
   guard has to REFUSE, and it has to refuse the packaged default too.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from chained_flow import defaults, shortlist


# ------------------------------------------------------------------ the shipped file


def test_packaged_shortlist_is_present_and_is_the_qwen35_vocab():
    p = shortlist.packaged_path()
    assert p.is_file(), (
        f"{p} is missing -- a wheel without it silently runs the full head. It is declared in "
        f"pyproject.toml under [tool.setuptools.package-data] 'chained_flow.data'."
    )
    ids, meta = shortlist.load(str(p))
    assert meta["vocab_size"] == shortlist.PACKAGED_VOCAB
    assert ids.numel() == shortlist.PACKAGED_ROWS
    assert int(ids.max()) < shortlist.PACKAGED_VOCAB and int(ids.min()) >= 0
    # sorted + unique is what lets `lm_w[sl]` be a plain gather with no dedup pass
    assert torch.equal(ids, torch.unique(ids))


def test_packaged_shortlist_is_int32_and_small():
    """int32 halves it to ~250 KB, and max id 248,076 fits with three orders of magnitude to
    spare. Size is the whole reason it can ship at all."""
    raw = torch.load(shortlist.packaged_path(), map_location="cpu")
    assert raw["ids"].dtype == torch.int32
    assert shortlist.packaged_path().stat().st_size < 400_000


def test_packaged_shortlist_matches_the_checkout_copy():
    """The wheel's copy and `out/flow/shortlist_q3527b.pt` must be the same ids, or a source
    checkout and a pip install measure different things and neither run says so."""
    repo = defaults.repo_root() / "out" / "flow" / "shortlist_q3527b.pt"
    if not repo.is_file():
        pytest.skip("no source-checkout shortlist to compare against")
    a, _ = shortlist.load(str(shortlist.packaged_path()))
    b, _ = shortlist.load(str(repo))
    assert torch.equal(a, torch.unique(b))


# ------------------------------------------------------------------ the vocab guard


def test_guard_accepts_the_vocabulary_it_was_built_for():
    ids, meta = shortlist.load(str(shortlist.packaged_path()))
    assert shortlist.check(ids, meta, shortlist.PACKAGED_VOCAB) is None


@pytest.mark.parametrize("v", [151936, 248319, 248321, 32000])
def test_guard_refuses_a_different_vocabulary(v):
    """Including a vocab LARGER than the list's max id -- that is the case a max-id check
    cannot catch, and it is the realistic one (a model with a superset tokenizer)."""
    ids, meta = shortlist.load(str(shortlist.packaged_path()))
    why = shortlist.check(ids, meta, v)
    assert why and str(v) in why and "248320" in why


def test_guard_falls_back_to_a_max_id_check_for_a_legacy_bare_tensor(tmp_path):
    """Every shortlist built before the metadata existed is a bare tensor. It still has to be
    usable, and it still has to be refused when it is too wide for the head."""
    p = tmp_path / "legacy.pt"
    torch.save(torch.tensor([0, 5, 999], dtype=torch.int64), p)
    ids, meta = shortlist.load(str(p))
    assert meta == {}
    assert shortlist.check(ids, meta, 1000) is None
    why = shortlist.check(ids, meta, 500)
    assert why and "999" in why and "no vocab_size metadata" in why


# ------------------------------------------------------------------ resolution order


def test_candidates_prefer_the_drafter_checkpoint_then_fall_back_to_the_packaged_list(tmp_path):
    """The order is the contract: a drafter that ships its own list wins, and the packaged list
    is always the last resort so there is never NO shortlist on a fresh install."""
    (tmp_path / "shortlist.pt").write_bytes(b"")
    got = defaults.shortlist_candidates(str(tmp_path))
    assert got[0] == (str(tmp_path / "shortlist.pt"), "drafter checkpoint")
    assert got[-1] == (str(defaults.packaged_shortlist()), "packaged")


def test_candidates_ignore_a_drafter_dir_that_has_no_shortlist(tmp_path):
    got = defaults.shortlist_candidates(str(tmp_path))
    assert all(w != "drafter checkpoint" for _, w in got)
    assert got[-1][1] == "packaged"


def test_candidates_ignore_an_hf_repo_id(monkeypatch):
    """`CF_DRAFTER_DIR` is a repo id in the documented usage, so `os.path.isdir` is False and
    the checkpoint candidate cannot be offered until `_build` re-resolves with the downloaded
    snapshot path. What must NOT happen is a bogus `selimaktas/Flow-Drafter-4B-v2/shortlist.pt`
    reaching the loader."""
    monkeypatch.setenv("CF_DRAFTER_DIR", "selimaktas/Flow-Drafter-4B-v2")
    got = defaults.shortlist_candidates()
    assert all(not p.startswith("selimaktas/") for p, _ in got)


def test_default_is_the_best_candidate_and_exists():
    d = defaults._shortlist_default()
    assert d and os.path.isfile(d)
    assert d == defaults.shortlist_candidates()[0][0]


def test_packaged_path_agrees_between_the_two_modules():
    """`defaults.py` re-derives the path so it stays importable as a bare file with no torch and
    no `chained_flow` package (it is run that way by `bench_cf.sh --sh`). Duplication is the
    price; drift is what the test is for."""
    assert Path(defaults.packaged_shortlist()) == shortlist.packaged_path()
