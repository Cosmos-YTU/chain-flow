"""Fast state snapshot/restore for the hybrid Qwen3.5 backbone cache.

The speculative verifier currently rolls the SSM state back with ``copy.deepcopy(past_key_values)``,
which clones the whole cache object (~8 ms/verify, measured — the single biggest fixed per-step cost).
But the entire per-layer state is a few small explicit tensors:

  * linear-attention (Gated DeltaNet) layers:  conv_states [1, 6144, 4]  +  recurrent_states [1, H, Dk, Dv]
  * full-attention layers:                      keys / values  (grow by the draft length each forward)

Snapshotting = cloning just those tensors (~10 MB total → microseconds). Restoring = writing them
back, which also crops the grown KV. This is token-for-token identical to the deepcopy path, and it
is the exact per-branch state we will fork for tree verification.
"""
from __future__ import annotations

from typing import Any

import torch

# state attributes a cache layer may hold; conv/recurrent are fixed-shape (overwritten in place),
# keys/values grow by the forwarded length and must be cropped back on restore.
_STATE_ATTRS = ("conv_states", "recurrent_states", "keys", "values")


def _layers(pkv: Any) -> list:
    layers = getattr(pkv, "layers", None)
    if layers is None:
        raise TypeError(f"cache {type(pkv).__name__} has no .layers; unsupported cache type")
    return layers


def clone_state_tensors(pkv: Any) -> list[dict[str, torch.Tensor]]:
    """Clone every per-layer state tensor. Returns a snapshot usable with :func:`restore_state`."""
    snap: list[dict[str, torch.Tensor]] = []
    for layer in _layers(pkv):
        saved: dict[str, torch.Tensor] = {}
        for name in _STATE_ATTRS:
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                saved[name] = t.detach().clone()
        snap.append(saved)
    return snap


# snapshot is just the clone; named separately for intent at call sites.
snapshot_state = clone_state_tensors


def tile_state(pkv: Any, n: int) -> Any:
    """Return a copy of the (batch-1) cache expanded to batch ``n``, with independent per-row state.

    Used to verify ``n`` root-to-leaf tree paths as independent sequences in one batched forward:
    every row starts from the same committed prefix but diverges as its path tokens are appended.
    Batched inference is bit-identical to running each row separately, so this is lossless.
    """
    import copy

    if n < 1:
        raise ValueError("n must be >= 1")
    c = copy.deepcopy(pkv)
    for layer in _layers(c):
        for name in _STATE_ATTRS:
            t = getattr(layer, name, None)
            if torch.is_tensor(t):
                if t.shape[0] != 1:
                    raise ValueError(f"tile_state expects batch-1 cache; {name} has batch {t.shape[0]}")
                setattr(layer, name, t.repeat(n, *([1] * (t.dim() - 1))).contiguous())
    return c


def restore_state(pkv: Any, snapshot: list[dict[str, torch.Tensor]]) -> None:
    """Undo any forward since ``snapshot`` by writing the saved state tensors back into the cache.

    Re-clones on write so the snapshot stays pristine and reusable (a partial-accept step may restore
    then re-forward). Fixed-shape states (conv/recurrent) are overwritten; grown KV is replaced by the
    shorter saved tensor, cropping the draft tokens the verify pass appended.
    """
    layers = _layers(pkv)
    if len(layers) != len(snapshot):
        raise ValueError(f"snapshot layer count {len(snapshot)} != cache {len(layers)}")
    for layer, saved in zip(layers, snapshot):
        for name, t in saved.items():
            setattr(layer, name, t.clone())
