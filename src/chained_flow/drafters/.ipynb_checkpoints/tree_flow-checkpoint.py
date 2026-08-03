"""Tree-flow drafter (V9): the path-conditioned, tree-native successor to V6 (markov_flow).

Motivation (grounded in V6's own eval). A parallel flow emits every per-position distribution in
ONE pass, which is exactly the input a draft TREE needs — EAGLE must run K sequential passes to grow
the same tree. V6's measured top-b coverage says a tree over its distributions would lift accepted
length from ~2.4-3.0 (chain) toward ~3.3-3.5 (tree, b=10), past EAGLE's chain (3.21). But V6's
distributions are MARGINALS: it was trained fully parallel, so position i's distribution ignores
which token was chosen at i-1. Expanding a tree from marginals gives incoherent deep branches.

V9 fixes exactly that while keeping the single-flow-pass advantage:
  1. Parallel flow ONCE -> K marginal hiddens (unchanged from V6; reuses HiddenKVFlowExpert).
  2. A cheap per-branch conditioner re-forecasts each node given its PARENT PATH, with NO flow re-run:
       - MarkovHead (V6): order-1 token->logit bias.  W2(W1[prev_token]).
       - PathHead (new): order-p token->hidden residual using the frozen LM token embeddings, so the
         hidden itself becomes path-conditioned (strictly generalises the logit-only markov head).
  3. Trained FOR the tree: a top-b COVERAGE loss puts the true token inside the drafter's top-b at
     each (path-conditioned) position, so the tree actually has the right candidate to branch to.

Trained teacher-forced on the correct trajectory, the conditioner learns E[token_i | parent_path];
at inference on an off-greedy tree branch it emits that learned conditional for the branch's parent
token — the same mechanism that lets EAGLE generalise to tree nodes, no off-trajectory data needed.

propose() is a V6-identical chain (drop-in for the existing generation/verifier path). build_tree()
is the new capability: one flow pass -> breadth-first tree with per-branch markov+path rescoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.drafters.chunked_flow import HiddenKVFlowExpert
from chained_flow.drafters.markov_flow import MarkovHead
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class TreeFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4
    expert_dim: int = 1024  # must equal LM hidden size
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8
    num_flow_steps: int = 2
    init_mode: str = "delta"
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    noise_scale: float = 1.0
    markov_rank: int = 256
    # --- path conditioning (the "full" tree upgrade over V6's logit-only markov head) ---
    path_order: int = 4         # how many parent tokens condition each node's hidden (<= draft_length)
    path_ffn_multiplier: int = 2
    # --- tree-training objective ---
    lambda_cov: float = 0.3     # weight of the top-b coverage loss (0 disables -> ~V6 + path head)
    cov_b: int = 8              # tree width the coverage loss optimises for
    cov_margin: float = 1.0     # hinge margin pushing the true token above the (b+1)-th logit
    # --- default tree shape used by build_tree / eval ---
    tree_top_b: int = 8         # branch factor per node
    tree_max_nodes: int = 48    # global node budget (DDTree-style prune by cumulative logprob)
    architecture: str = "tree_flow"

    def __post_init__(self) -> None:
        if self.draft_length % self.chunk_size != 0:
            raise ValueError("draft_length must be divisible by chunk_size")
        if self.markov_rank < 1:
            raise ValueError("markov_rank must be >= 1")
        if not 1 <= self.path_order <= self.draft_length:
            raise ValueError("path_order must be in [1, draft_length]")
        if self.init_mode not in {"noise", "repeat_last", "delta"}:
            raise ValueError("init_mode must be 'noise', 'repeat_last', or 'delta'")
        if self.architecture != "tree_flow":
            raise ValueError("TreeFlowConfig.architecture must be 'tree_flow'")


class PathHead(nn.Module):
    """Order-p token->hidden residual. For position i it consumes the frozen LM embeddings of parent
    tokens t_{i-1}..t_{i-p} (per-offset projected, masked when a parent does not exist) and produces a
    residual added to the marginal flow hidden. Zero-initialised output -> starts as a no-op, so at
    init V9 reproduces V6's marginal hidden exactly."""

    def __init__(self, hidden_size: int, order: int, ffn_multiplier: int, dropout: float = 0.0):
        super().__init__()
        self.order = order
        self.offset_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(order)])
        ffn = hidden_size * ffn_multiplier
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, hidden_size)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, parent_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # parent_embeds [B, K, order, D]  mask [B, K, order] (1 where the parent exists)
        h = None
        for j in range(self.order):
            term = self.offset_proj[j](parent_embeds[:, :, j, :]) * mask[:, :, j : j + 1]
            h = term if h is None else h + term
        return self.mlp(self.norm(h))


class TreeFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: TreeFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("tree_flow requires expert_dim to equal the target LM hidden size")
        self.num_chunks = config.draft_length // config.chunk_size

        def make_expert():
            return HiddenKVFlowExpert(
                hidden_size=self.hidden_size, context_size=config.context_size,
                draft_length=config.draft_length, chunk_size=config.chunk_size,
                expert_dim=config.expert_dim, num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier, num_layers=config.num_drafter_layers,
                dropout=config.drafter_dropout,
            )
        self.expert = make_expert()
        self.extra_experts = nn.ModuleList([make_expert() for _ in range(self.num_chunks - 1)])

        vocab = frozen_lm.model.config.vocab_size
        self.markov = MarkovHead(vocab, config.markov_rank)
        self.path_head = PathHead(self.hidden_size, config.path_order, config.path_ffn_multiplier,
                                  dropout=config.drafter_dropout)
        self._dtype = next(self.expert.parameters()).dtype

    # ---- flow machinery (identical to V6) --------------------------------

    def _chunk_experts(self):
        return [self.expert, *self.extra_experts]

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad = hidden[:, :1, :].expand(-1, self.config.context_size - hidden.shape[1], -1)
        return torch.cat([pad, hidden], dim=1)

    def init_latents(self, context: torch.Tensor) -> torch.Tensor:
        context = context.to(dtype=self._dtype)
        b, _, d = context.shape
        k = self.config.draft_length
        if self.config.init_mode == "noise":
            return torch.randn(b, k, d, device=context.device, dtype=self._dtype) * self.config.noise_scale
        last = context[:, -1:, :]
        if self.config.init_mode == "repeat_last":
            return last.expand(b, k, d).clone()
        delta = (context[:, -1:, :] - context[:, -2:-1, :]) if context.shape[1] > 1 else torch.zeros_like(last)
        steps = torch.arange(1, k + 1, device=context.device, dtype=self._dtype)
        return last + steps.view(1, k, 1) * delta

    def flow_velocity(self, z_tau, tau, context, previous=None) -> torch.Tensor:
        vels = []
        for ci, expert in enumerate(self._chunk_experts()):
            start = ci * self.config.chunk_size
            end = start + self.config.chunk_size
            prev = (previous if previous is not None else z_tau)[:, :start, :]
            if previous is not None and self.config.detach_previous_chunks:
                prev = prev.detach()
            vels.append(expert(context_hidden=context, previous_hidden=prev,
                               current_h_tau=z_tau[:, start:end, :], tau=tau, chunk_start=start))
        return torch.cat(vels, dim=1)

    def integrate(self, context: torch.Tensor, z0: torch.Tensor) -> torch.Tensor:
        z = z0
        dt = 1.0 / self.config.num_flow_steps
        for s in range(self.config.num_flow_steps):
            tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=self._dtype)
            z = z + dt * self.flow_velocity(z, tau, context)
        return z

    def predict_hidden(self, context: torch.Tensor) -> torch.Tensor:
        return self.integrate(context.to(self._dtype), self.init_latents(context))

    def lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.frozen_lm.lm_head(hidden)

    # ---- token feedback helpers ------------------------------------------

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        """Frozen LM input embeddings for token ids -> [..., D] in the expert dtype (backbone frozen,
        so no gradient reaches the table; PathHead's projections carry all trainable capacity)."""
        emb = self.frozen_lm.model.get_input_embeddings()(ids)
        return emb.to(self._dtype)

    def _parent_stack(self, tokens: torch.Tensor):
        """tokens [B, K] -> (parent_ids [B, K, order], mask [B, K, order]).
        parent_ids[:, i, j] = token at position i-(j+1); mask 0 where that position < 0."""
        b, k = tokens.shape
        order = self.config.path_order
        ids = tokens.new_zeros(b, k, order)
        mask = torch.zeros(b, k, order, device=tokens.device, dtype=self._dtype)
        for j in range(order):
            s = j + 1
            if s < k:
                ids[:, s:, j] = tokens[:, : k - s]
                mask[:, s:, j] = 1.0
        return ids, mask

    def _path_residual(self, tokens: torch.Tensor) -> torch.Tensor:
        """Path-conditioned hidden residual for every position given a committed token path [B, K]."""
        ids, mask = self._parent_stack(tokens)
        return self.path_head(self._embed(ids), mask)

    # ---- training entry --------------------------------------------------

    def forward_teacher(self, context: torch.Tensor, target_hidden: torch.Tensor, future_tokens: torch.Tensor):
        """Flow matching (marginal) + path-conditioned logits, teacher-forced on the correct path.
        Returns marginal + conditioned hiddens/logits so the loss can supervise both."""
        context = context.to(self._dtype)
        z0 = self.init_latents(context)
        b = z0.shape[0]
        tau = torch.rand(b, 1, 1, device=z0.device, dtype=self._dtype)
        z_tau = (1.0 - tau) * z0 + tau * target_hidden.to(self._dtype)
        v_star = target_hidden.to(self._dtype) - z0
        v_pred = self.flow_velocity(z_tau, tau, context, previous=target_hidden.to(self._dtype))
        pred_hidden = self.integrate(context, z0)
        base_logits = self.lm_head(pred_hidden)

        # path-conditioned hidden + logits, using the REAL parent tokens (teacher forcing).
        residual = self._path_residual(future_tokens)          # [B, K, D]; position 0 -> 0 (no parents)
        cond_hidden = pred_hidden + residual
        cond_logits = self.lm_head(cond_hidden)
        prev = torch.zeros_like(future_tokens)
        prev[:, 1:] = future_tokens[:, :-1]
        bias = self.markov.bias(prev)
        bias[:, 0, :] = 0.0                                    # position 0 has no previous draft token
        cond_logits = cond_logits + bias
        return {"v_pred": v_pred, "v_star": v_star, "pred_hidden": pred_hidden, "cond_hidden": cond_hidden,
                "base_logits": base_logits, "cond_logits": cond_logits}

    # ---- inference: V6-identical chain (drop-in for generation/verifier) --

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)
        with timed_section(timings, "drafter_tree_flow", self.frozen_lm.device):
            context = self._context(state).to(self._dtype)
            pred_hidden = self.predict_hidden(context)             # one parallel flow pass
            base_logits = self.lm_head(pred_hidden)
            tokens = []
            path = []                                              # committed tokens so far (for path head)
            for i in range(draft_len):
                if not path:
                    logit_i = base_logits[:, i, :]                 # position 0: no parent conditioning
                else:
                    committed = torch.stack(path, dim=1)           # [B, i]
                    res_i = self._position_residual(committed, i)
                    logit_i = self.lm_head(pred_hidden[:, i, :] + res_i) + self.markov.bias(path[-1])
                tok_i = logit_i.argmax(dim=-1)
                tokens.append(tok_i)
                path.append(tok_i)
            draft_tokens = torch.stack(tokens, dim=1)
        return DraftResult(tokens=draft_tokens, hidden_states=pred_hidden[:, :draft_len, :],
                           logits=base_logits[:, :draft_len, :], timings=timings)

    def _position_residual(self, committed: torch.Tensor, position: int) -> torch.Tensor:
        """Path residual for a single decode position given the committed tokens [B, position]."""
        order = self.config.path_order
        b = committed.shape[0]
        ids = committed.new_zeros(b, order)
        mask = torch.zeros(b, order, device=committed.device, dtype=self._dtype)
        for j in range(order):
            src = position - (j + 1)
            if src >= 0:
                ids[:, j] = committed[:, src]
                mask[:, j] = 1.0
        emb = self._embed(ids)                                    # [B, order, D]
        # reuse PathHead reduction for a single position
        h = None
        for j in range(order):
            term = self.path_head.offset_proj[j](emb[:, j, :]) * mask[:, j : j + 1]
            h = term if h is None else h + term
        return self.path_head.mlp(self.path_head.norm(h))

    # ---- inference: draft TREE (the new capability) ----------------------

    @torch.inference_mode()
    def build_tree(self, state: LMState, *, top_b: int | None = None, max_nodes: int | None = None):
        """One flow pass -> breadth-first draft tree. Each node re-forecasts its position conditioned
        on its parent path via the (cheap) markov + path heads; NO flow re-runs. Assumes batch size 1.

        Returns a DraftTree: parallel arrays over nodes (token, parent, depth, cum_logprob) plus the
        marginal hiddens, ready for backbone tree-verification."""
        cfg = self.config
        top_b = cfg.tree_top_b if top_b is None else top_b
        max_nodes = cfg.tree_max_nodes if max_nodes is None else max_nodes
        context = self._context(state).to(self._dtype)
        if context.shape[0] != 1:
            raise ValueError("build_tree expects batch size 1")
        pred_hidden = self.predict_hidden(context)[0]              # [K, D]
        base_logits = self.lm_head(pred_hidden[:1])[0]             # position-0 logits [V]
        top_b = min(top_b, base_logits.shape[-1])
        logp0 = torch.log_softmax(base_logits.float(), dim=-1)
        vals, idx = logp0.topk(top_b)

        # node arrays; node 0..b-1 are depth-0 roots
        tokens: list[int] = [int(t) for t in idx.tolist()]
        parents: list[int] = [-1] * len(tokens)
        depths: list[int] = [0] * len(tokens)
        cum: list[float] = [float(v) for v in vals.tolist()]
        frontier = list(range(len(tokens)))                       # node indices at the current depth

        for d in range(1, cfg.draft_length):
            if not frontier:
                break
            # gather each frontier node's committed parent path (length d), batched
            paths = torch.tensor([self._path_to(n, tokens, parents, depths) for n in frontier],
                                 device=context.device)           # [F, d]
            res = self._position_residual(paths, d)               # [F, D]
            hid = pred_hidden[d].unsqueeze(0) + res               # [F, D]
            logits = self.lm_head(hid)                            # [F, V]
            prev_tok = paths[:, -1]
            logits = logits + self.markov.bias(prev_tok)
            logp = torch.log_softmax(logits.float(), dim=-1)
            vals, idx = logp.topk(top_b, dim=-1)                  # [F, b]
            # DDTree-style global prune: rank ALL candidate children at this depth by cumulative
            # log-prob and keep only the top max_nodes, so total tree size stays bounded.
            cand = [(cum[node] + float(vals[fi, bi]), node, int(idx[fi, bi]))
                    for fi, node in enumerate(frontier) for bi in range(top_b)]
            cand.sort(key=lambda c: c[0], reverse=True)
            new_frontier = []
            for cprob, node, tok in cand[:max_nodes]:
                tokens.append(tok)
                parents.append(node)
                depths.append(d)
                cum.append(cprob)
                new_frontier.append(len(tokens) - 1)
            frontier = new_frontier

        return DraftTree(tokens=tokens, parents=parents, depths=depths, cum_logprob=cum,
                         marginal_hidden=pred_hidden)

    def _residual_from_lastp(self, lastp: torch.Tensor) -> torch.Tensor:
        """Path residual for a batch of nodes given their last-`order` path tokens [W, order]
        (lastp[:, j] = token at offset j+1 back; -1 = no such ancestor). Vectorised twin of
        _position_residual used by build_tree_fast."""
        mask = (lastp >= 0).to(self._dtype)
        emb = self._embed(lastp.clamp_min(0))                 # [W, order, D]
        h = None
        for j in range(self.config.path_order):
            term = self.path_head.offset_proj[j](emb[:, j, :]) * mask[:, j : j + 1]
            h = term if h is None else h + term
        return self.path_head.mlp(self.path_head.norm(h))

    @torch.inference_mode()
    def build_tree_fast(self, state: LMState, *, top_b: int | None = None, max_nodes: int | None = None,
                        max_depth: int | None = None):
        """GPU-vectorised beam-search equivalent of build_tree: same one flow pass + per-branch
        markov/path rescoring, but the depth loop is tensor ops (topk/gather) with a single host
        transfer at the end — no per-node Python or .item() syncs. Assumes batch size 1."""
        cfg = self.config
        top_b = cfg.tree_top_b if top_b is None else top_b
        max_nodes = cfg.tree_max_nodes if max_nodes is None else max_nodes
        order, K = cfg.path_order, cfg.draft_length
        context = self._context(state).to(self._dtype)
        if context.shape[0] != 1:
            raise ValueError("build_tree_fast expects batch size 1")
        pred_hidden = self.predict_hidden(context)[0]          # [K, D]
        dev = context.device

        base0 = self.lm_head(pred_hidden[:1])[0]               # [V]
        b0 = min(top_b, base0.shape[-1])
        vals, idx = torch.log_softmax(base0.float(), dim=-1).topk(b0)
        tokens = idx.clone()
        parents = torch.full((b0,), -1, dtype=torch.long, device=dev)
        depths = torch.zeros(b0, dtype=torch.long, device=dev)
        cum = vals.clone()
        frontier = torch.arange(b0, device=dev)
        lastp = torch.full((b0, order), -1, dtype=torch.long, device=dev)
        lastp[:, 0] = idx

        kmax = K if max_depth is None else min(K, max_depth + 1)
        for d in range(1, kmax):
            W = frontier.shape[0]
            if W == 0:
                break
            res = self._residual_from_lastp(lastp)             # [W, D]
            logits = self.lm_head(pred_hidden[d].unsqueeze(0) + res) + self.markov.bias(lastp[:, 0])
            cvals, cidx = torch.log_softmax(logits.float(), dim=-1).topk(top_b, dim=-1)   # [W, top_b]
            cand_cum = (cum[frontier].unsqueeze(1) + cvals).reshape(-1)
            keep = min(max_nodes, cand_cum.shape[0])
            topv, topi = cand_cum.topk(keep)
            src_row = topi // top_b                             # which frontier node
            new_tokens = cidx.reshape(-1)[topi]
            new_parents = frontier[src_row]
            start = tokens.shape[0]
            tokens = torch.cat([tokens, new_tokens])
            parents = torch.cat([parents, new_parents])
            depths = torch.cat([depths, torch.full((keep,), d, dtype=torch.long, device=dev)])
            cum = torch.cat([cum, topv])
            lastp = torch.cat([new_tokens.unsqueeze(1), lastp[src_row][:, :-1]], dim=1)
            frontier = torch.arange(start, start + keep, device=dev)

        t = tokens.tolist(); p = parents.tolist(); dp = depths.tolist(); c = cum.tolist()
        return DraftTree(tokens=t, parents=p, depths=dp, cum_logprob=c, marginal_hidden=pred_hidden)

    @staticmethod
    def _path_to(node: int, tokens, parents, depths) -> list[int]:
        seq = []
        n = node
        while n != -1:
            seq.append(tokens[n])
            n = parents[n]
        seq.reverse()
        return seq


@dataclass
class DraftTree:
    tokens: list[int]
    parents: list[int]
    depths: list[int]
    cum_logprob: list[float]
    marginal_hidden: torch.Tensor
    _paths: list[list[int]] = field(default=None, repr=False)  # type: ignore[assignment]

    def num_nodes(self) -> int:
        return len(self.tokens)

    def root_to_leaf_paths(self) -> list[list[int]]:
        """Every maximal path's node indices (root..leaf), leaf = node that is nobody's parent."""
        children: dict[int, list[int]] = {}
        for i, p in enumerate(self.parents):
            children.setdefault(p, []).append(i)
        leaves = [i for i in range(len(self.tokens)) if i not in children]
        paths = []
        for leaf in leaves:
            seq = []
            n = leaf
            while n != -1:
                seq.append(n)
                n = self.parents[n]
            seq.reverse()
            paths.append(seq)
        return paths
