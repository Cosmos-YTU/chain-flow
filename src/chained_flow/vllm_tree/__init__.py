"""vLLM tree spec-decode extension for the V9 flow-tree drafter.

The SSM tree-verify reuses vLLM's OWN kernels (no new Triton) via a parent-routed state trick,
validated bit-exact against vLLM's linear per-path recurrence. See tree_ssm.py.
"""
from chained_flow.vllm_tree.tree_ssm import tree_gated_delta_recurrence, tree_causal_conv1d

__all__ = ["tree_gated_delta_recurrence", "tree_causal_conv1d"]
