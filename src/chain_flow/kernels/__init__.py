"""Systems/kernels for turning the flow-tree drafter's accepted-length into wall-clock.

cache_fork: fast state snapshot/restore for the hybrid Gated-DeltaNet backbone cache, replacing the
copy.deepcopy rollback in speculative verify (the ~8ms/verify tax) and providing the per-branch
state-fork primitive that tree verification is built on.
"""
from chain_flow.kernels.cache_fork import snapshot_state, restore_state, clone_state_tensors, tile_state
from chain_flow.kernels.tree_verify import generate_with_tree, tree_spec_step

__all__ = ["snapshot_state", "restore_state", "clone_state_tensors", "tile_state",
           "generate_with_tree", "tree_spec_step"]
