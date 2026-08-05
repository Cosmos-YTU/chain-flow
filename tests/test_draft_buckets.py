"""The draft cudagraph's batch-bucket ladder.

The ladder used to stop at 32 and the consequence was measured, not theorised: profiling a 4B
chain server at a decode batch of 64, **1901 of 2000 steps ran the draft with no cudagraph at
all**, and every distinct batch size in 33..64 was its own `dynamic=False` max-autotune compile
(the ~850-autotune-event storm, ~70-80 s of engine stall apiece, inside live traffic).

These tests do not need a GPU: the ladder is a pure function of `max_num_seqs` and one env var,
and the property that matters is that it REACHES the largest batch the engine can schedule.
"""
import pytest

torch = pytest.importorskip("torch")
from chained_flow.vllm_plugin.flow_proposer import FlowDrafterProposer  # noqa: E402


def ladder(max_reqs, env=None, monkeypatch=None):
    if monkeypatch is not None:
        if env is None:
            monkeypatch.delenv("CF_DRAFT_BUCKETS", raising=False)
        else:
            monkeypatch.setenv("CF_DRAFT_BUCKETS", env)

    class _Fake:
        pass
    f = _Fake()
    f.max_reqs = max_reqs
    return FlowDrafterProposer._bucket_ladder(f)


@pytest.mark.parametrize("max_reqs", [1, 2, 5, 8, 33, 64, 100, 256])
def test_the_ladder_reaches_max_num_seqs(monkeypatch, max_reqs):
    """THE regression this file exists for. Above the last bucket `_ctx_gpu` falls back to
    `bucket = B`, which is no cudagraph and a fresh compile per batch size."""
    b = ladder(max_reqs, None, monkeypatch)
    assert b[-1] == max_reqs, b
    assert b[0] == 1 or max_reqs < 1


def test_batch_1_is_still_the_first_rung(monkeypatch):
    """Batch 1 is the measurement of record (4B chain 1.26x / tree 1.44x, 27B 1.75x / 2.00x) and
    bucket 1 is the ONLY bucket on which the fused CUDA block kernel fires
    (`chunked_flow._cf_fused_runner` takes `x.shape[0] == 1`). Nothing about extending the top of
    the ladder may move the bottom of it."""
    for m in (1, 8, 64, 256):
        assert ladder(m, None, monkeypatch)[0] == 1


def test_the_old_ladder_is_a_prefix_of_the_new_one(monkeypatch):
    """Every batch that had a bucket before still replays the SAME bucket, so no step below 32
    changes shape, kernel or output."""
    assert ladder(64, None, monkeypatch)[:6] == [1, 2, 4, 8, 16, 32]


def test_it_never_offers_a_bucket_the_engine_cannot_reach(monkeypatch):
    """A bucket above `max_num_seqs` is pure capture time and pool memory for a shape that cannot
    occur -- and `CF_WARM_BUCKETS` would pay for it at startup."""
    for m in (3, 8, 40):
        assert max(ladder(m, None, monkeypatch)) <= m


def test_env_override(monkeypatch):
    assert ladder(64, "1,2,4,8,16,32", monkeypatch) == [1, 2, 4, 8, 16, 32]
    assert ladder(64, "1 8 64", monkeypatch) == [1, 8, 64]
    assert ladder(64, "1,2,4,8,16,32,40,48,56,64", monkeypatch)[-1] == 64
    # duplicates and order are the caller's problem, not the engine's
    assert ladder(64, "8,1,8", monkeypatch) == [1, 8]
    # ... but a rung the engine cannot reach is still dropped
    assert ladder(16, "1,32,64", monkeypatch) == [1]
