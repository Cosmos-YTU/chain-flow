"""Lossless draft-tree speculative decoding.

Correctness-first design: verify each root-to-leaf path of the draft tree as an INDEPENDENT batched
sequence (prefix + path tokens). Batched inference is bit-identical to running each path separately,
so there is no tree-mask / SSM-state-fork subtlety to get wrong. We accept a draft token only if it
equals the backbone's own greedy token given the accepted prefix, and always emit the backbone's
greedy "bonus" token. Every emitted token is therefore exactly what greedy decoding would produce —
the output is token-for-token identical to plain greedy generation, just (eventually) faster.

This is the reference-correct verify. Speed optimizations (fold commit, vectorize build, per-branch
state-fork instead of per-path recompute, Triton) layer on top WITHOUT changing the accepted tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from chain_flow.frozen_lm import FrozenLMWrapper, LMState
from chain_flow.kernels.cache_fork import snapshot_state, restore_state, tile_state


@dataclass
class TreeGenResult:
    tokens: torch.Tensor            # [1, prompt+generated]
    prompt_len: int
    accepted_per_step: list[int] = field(default_factory=list)   # draft tokens accepted (excl. bonus)
    steps: int = 0

    @property
    def generated(self) -> int:
        return self.tokens.shape[1] - self.prompt_len

    @property
    def mean_accept(self) -> float:
        return (sum(self.accepted_per_step) / len(self.accepted_per_step)) if self.accepted_per_step else 0.0


def _greedy_next(state: LMState) -> int:
    return int(state.logits[:, -1, :].argmax(dim=-1).item())


@torch.inference_mode()
def tree_spec_step(fw: FrozenLMWrapper, drafter, committed: LMState, *, top_b: int, max_nodes: int,
                   sequential_commit: bool = False):
    """One speculative step. Returns (emitted_tokens[list[int]], accepted_draft_len, new_committed).

    sequential_commit=True verifies the proposed path one token at a time with the single-token decode
    kernel (bit-exact vs greedy, but no speedup — used to prove the accept logic is correct and isolate
    the multi-token-kernel gap). Default False uses the fast single multi-token commit forward.
    """
    m = fw.model
    tree = drafter.build_tree_fast(committed, top_b=top_b, max_nodes=max_nodes)
    paths = tree.root_to_leaf_paths()                       # list of node-index lists (root..leaf)

    # children index + a representative (path, pos) for reading each node's logits
    children: dict[int, list[int]] = {}
    for i, par in enumerate(tree.parents):
        children.setdefault(par, []).append(i)
    node_pos: dict[int, tuple[int, int]] = {}
    tok_seqs: list[list[int]] = []
    for pi, path in enumerate(paths):
        tok_seqs.append([tree.tokens[n] for n in path])
        for pos, n in enumerate(path):
            node_pos.setdefault(n, (pi, pos))

    P = len(paths)
    Lmax = max(len(s) for s in tok_seqs)
    dev = fw.device
    padded = torch.zeros((P, Lmax), dtype=torch.long, device=dev)
    for pi, s in enumerate(tok_seqs):
        padded[pi, : len(s)] = torch.tensor(s, dtype=torch.long, device=dev)

    # verify all paths in one batched forward from a tiled copy of the committed prefix
    tiled = tile_state(committed.past_key_values, P)
    logits = m(input_ids=padded, past_key_values=tiled, use_cache=True).logits   # [P, Lmax, V]
    # logits[p, i] = backbone distribution AFTER path token i (i.e. greedy for path position i+1)

    # PROPOSAL: walk the batched-verify greedy through the tree while a child matches it. This is a
    # fast (possibly fp-imperfect at batch P) guess of which path the backbone greedily follows.
    proposed: list[int] = []
    cur = -1                                                # virtual root = committed prefix
    g = _greedy_next(committed)                             # depth-0 greedy is from the committed prefix (batch-1, exact)
    while True:
        match = next((c for c in children.get(cur, []) if tree.tokens[c] == g), None)
        if match is None:
            break
        proposed.append(g)
        cur = match
        pi, pos = node_pos[cur]
        g = int(logits[pi, pos].argmax().item())           # proposed next (from batch-P verify)
    proposed.append(g)                                      # trailing bonus candidate

    if sequential_commit:
        # verify one token at a time with the SAME single-token kernel greedy uses -> bit-exact
        final: list[int] = []
        state = committed
        gt = _greedy_next(committed)
        for t in proposed:
            keep = t if t == gt else gt
            final.append(keep)
            state, _ = fw.forward_with_cache(torch.tensor([[keep]], dtype=torch.long, device=dev), state, use_cache=True)
            if keep != t:
                break
            gt = _greedy_next(state)
        return final, max(0, len(final) - 1), state

    # TRUTH: commit the proposal with a batch-1 forward and verify against ITS greedy. Truncate to the
    # longest prefix that matches, replacing the first divergence with the batch-1 token. Every kept
    # token is then exactly batch-1 greedy → lossless. Rollback via snapshot if we over-forwarded.
    prop_t = torch.tensor([proposed], dtype=torch.long, device=dev)
    snap = snapshot_state(committed.past_key_values)
    tentative, _ = fw.forward_with_cache(prop_t, committed, use_cache=True)
    M = len(proposed)
    new_logits = tentative.logits[:, -M:, :]               # new_logits[0,i] = greedy for position i+1
    truth = [_greedy_next(committed)] + [int(new_logits[0, i].argmax().item()) for i in range(M - 1)]
    final: list[int] = []
    diverged = False
    for i in range(M):
        if proposed[i] == truth[i]:
            final.append(proposed[i])
        else:
            final.append(truth[i])                         # correct batch-1 token, then stop
            diverged = True
            break
    if diverged and len(final) != M:
        restore_state(committed.past_key_values, snap)      # undo the over-forward
        new_committed, _ = fw.forward_with_cache(torch.tensor([final], dtype=torch.long, device=dev),
                                                 committed, use_cache=True)
    else:
        new_committed = tentative
    return final, max(0, len(final) - 1), new_committed


@torch.inference_mode()
def generate_with_tree(fw: FrozenLMWrapper, drafter, prompt: torch.Tensor, *, max_new_tokens: int,
                       top_b: int | None = None, max_nodes: int | None = None,
                       eos_token_id: int | None = None, sequential_commit: bool = False) -> TreeGenResult:
    top_b = drafter.config.tree_top_b if top_b is None else top_b
    max_nodes = drafter.config.tree_max_nodes if max_nodes is None else max_nodes
    eos = fw.eos_token_id if eos_token_id is None else eos_token_id
    prompt = prompt.to(fw.device)
    state, _ = fw.prefill(prompt)
    res = TreeGenResult(tokens=prompt, prompt_len=prompt.shape[1])

    while res.generated < max_new_tokens:
        emitted, acc, state = tree_spec_step(fw, drafter, state, top_b=top_b, max_nodes=max_nodes,
                                             sequential_commit=sequential_commit)
        room = max_new_tokens - res.generated
        keep = emitted[:room]
        res.tokens = torch.cat([res.tokens, torch.tensor([keep], dtype=torch.long, device=fw.device)], dim=1)
        res.accepted_per_step.append(min(acc, len(keep)))
        res.steps += 1
        if eos is not None and eos in keep:
            break
    return res


@torch.inference_mode()
def greedy_generate(fw: FrozenLMWrapper, prompt: torch.Tensor, *, max_new_tokens: int,
                    eos_token_id: int | None = None) -> torch.Tensor:
    """Plain greedy decoding — the reference the tree loop must reproduce token-for-token."""
    eos = fw.eos_token_id if eos_token_id is None else eos_token_id
    prompt = prompt.to(fw.device)
    state, _ = fw.prefill(prompt)
    out = prompt
    for _ in range(max_new_tokens):
        g = _greedy_next(state)
        out = torch.cat([out, torch.tensor([[g]], dtype=torch.long, device=fw.device)], dim=1)
        if eos is not None and g == eos:
            break
        state, _ = fw.forward_with_cache(torch.tensor([[g]], dtype=torch.long, device=fw.device), state, use_cache=True)
    return out
