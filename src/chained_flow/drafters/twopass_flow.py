"""Two-pass causal token-conditioned flow drafter (V8).

Motivation (from our own measurements): the parallel hidden-flow (stage4/V6) predicts all K hidden
states from context only, so deep positions are blind to the earlier draft tokens -> acc@4 plateaus
(base-flow CE ~1.1). V6's Markov head corrects LOGITS with a bigram bias but the hiddens stay blind.
EAGLE fixes it fully-causally at the hidden level but pays K sequential passes and isn't a flow.

V8 fixes the hidden at the flow level, in TWO parallel passes (not K):
  pass 1: parallel flow (noised-delta init) -> ĥ¹ -> lm_head -> provisional tokens t¹   (V6's pass)
  pass 2: a second velocity net, INITIALIZED at ĥ¹, where position i is fused with the embedding of
          the previous draft token t¹_{i-1} (strictly causal, shifted) and self-attends causally.
          -> refined ĥ² -> lm_head -> final tokens.

Why not V3 (which failed): V3 was bidirectional (position i saw its own/future estimates -> leak,
ill-posed) and transported from noise over N steps. V8 is strictly causal (position i sees only
t¹_{<i}, teacher-forceable, leak-impossible) and does a short residual transport from ĥ¹.

Training is on-policy for free: pass 1 is parallel, so pass 2 conditions on pass 1's OWN argmax
tokens (detached) -> the model learns to correct its actual mistakes at zero sequential cost.

noised-delta init also makes pass 1 a sampler (best-of-N drafting) -- exposed via noise_scale>0.
Reuses HiddenKVFlowExpert (pass 1) + HiddenKVFlowBlock (pass 2). Drop-in propose()->DraftResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from chained_flow.drafters.base import DraftResult
from chained_flow.drafters.chunked_flow import HiddenKVFlowBlock, HiddenKVFlowExpert
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class TwoPassFlowConfig:
    context_size: int = 8
    draft_length: int = 4
    chunk_size: int = 4          # pass-1 chunking (chunk_size==draft_length => fully parallel pass 1)
    expert_dim: int = 1024       # must equal LM hidden size
    num_heads: int = 8
    ffn_multiplier: int = 6
    num_drafter_layers: int = 8      # pass-1 velocity net depth
    num_refine_layers: int = 4       # pass-2 velocity net depth
    num_flow_steps: int = 2          # pass-1 Euler steps
    num_refine_steps: int = 1        # pass-2 Euler steps (short residual correction)
    init_mode: str = "delta"
    noise_scale: float = 0.0         # >0 => noised-delta init (enables best-of-N sampling)
    detach_previous_chunks: bool = True
    drafter_dropout: float = 0.0
    architecture: str = "twopass_flow"

    def __post_init__(self) -> None:
        if self.draft_length % self.chunk_size != 0:
            raise ValueError("draft_length must be divisible by chunk_size")
        if self.num_refine_steps < 1 or self.num_flow_steps < 1:
            raise ValueError("flow/refine steps must be >= 1")
        if self.init_mode not in {"noise", "repeat_last", "delta"}:
            raise ValueError("init_mode must be 'noise', 'repeat_last', or 'delta'")
        if self.architecture != "twopass_flow":
            raise ValueError("TwoPassFlowConfig.architecture must be 'twopass_flow'")


class _RefineNet(nn.Module):
    """Pass-2 velocity net: fuse [hidden ; emb(prev_token)] -> width, causal self-attn over K,
    cross-attn to context, predict a velocity. Position i's prev_token is the draft token at i-1
    (position 0 gets a zero token-embedding). Strictly causal => no leak."""

    def __init__(self, hidden_size: int, num_heads: int, ffn_multiplier: int, num_layers: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(2 * hidden_size, hidden_size)
        self.time_mlp = nn.Sequential(nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.blocks = nn.ModuleList([
            HiddenKVFlowBlock(hidden_size=hidden_size, num_heads=num_heads, ffn_multiplier=ffn_multiplier, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, z: torch.Tensor, tau, context: torch.Tensor, prev_tok_emb: torch.Tensor) -> torch.Tensor:
        # z [B,K,D]; prev_tok_emb [B,K,D] (emb of token i-1, zeros at i=0); context [B,m,D]
        b, k, _ = z.shape
        x = self.input_proj(torch.cat([z, prev_tok_emb], dim=-1))
        tau_t = (tau.reshape(b, 1) if isinstance(tau, torch.Tensor) else torch.full((b, 1), float(tau), device=z.device, dtype=z.dtype))
        x = x + self.time_mlp(tau_t.to(x.dtype)).unsqueeze(1)
        mask = torch.triu(torch.full((k, k), float("-inf"), device=z.device, dtype=x.dtype), diagonal=1)  # causal over K
        for blk in self.blocks:
            x = blk(x, context, attn_mask=mask)
        return self.out_proj(self.out_norm(x))


class TwoPassFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: TwoPassFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        if config.expert_dim != self.hidden_size:
            raise ValueError("twopass_flow requires expert_dim to equal the target LM hidden size")
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
        self.refine = _RefineNet(self.hidden_size, config.num_heads, config.ffn_multiplier,
                                 config.num_refine_layers, config.drafter_dropout)
        embed = frozen_lm.model.get_input_embeddings()
        self.register_buffer("token_embedding_weight", embed.weight.detach().clone(), persistent=False)
        self._dtype = next(self.expert.parameters()).dtype

    # ---- shared helpers -------------------------------------------------

    def _chunk_experts(self):
        return [self.expert, *self.extra_experts]

    def _context(self, state: LMState) -> torch.Tensor:
        h = state.final_hidden
        if h.shape[1] >= self.config.context_size:
            return h[:, -self.config.context_size :, :]
        pad = h[:, :1, :].expand(-1, self.config.context_size - h.shape[1], -1)
        return torch.cat([pad, h], dim=1)

    def _embed(self, tok: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(tok.to(self.token_embedding_weight.device), self.token_embedding_weight).to(self._dtype)

    def lm_head(self, h):
        return self.frozen_lm.lm_head(h)

    def init_latents(self, context: torch.Tensor, *, generator=None) -> torch.Tensor:
        context = context.to(self._dtype)
        b, _, d = context.shape
        k = self.config.draft_length
        if self.config.init_mode == "noise":
            return torch.randn(b, k, d, device=context.device, dtype=self._dtype, generator=generator) * max(self.config.noise_scale, 1.0)
        last = context[:, -1:, :]
        if self.config.init_mode == "repeat_last":
            base = last.expand(b, k, d).clone()
        else:
            delta = (context[:, -1:, :] - context[:, -2:-1, :]) if context.shape[1] > 1 else torch.zeros_like(last)
            steps = torch.arange(1, k + 1, device=context.device, dtype=self._dtype)
            base = last + steps.view(1, k, 1) * delta
        if self.config.noise_scale > 0:
            base = base + torch.randn(b, k, d, device=context.device, dtype=self._dtype, generator=generator) * self.config.noise_scale
        return base

    # ---- pass 1: parallel flow (identical to hidden_kv/V6) --------------

    def _pass1_velocity(self, z_tau, tau, context, previous=None):
        vels = []
        for ci, expert in enumerate(self._chunk_experts()):
            s = ci * self.config.chunk_size; e = s + self.config.chunk_size
            prev = (previous if previous is not None else z_tau)[:, :s, :]
            if previous is not None and self.config.detach_previous_chunks:
                prev = prev.detach()
            vels.append(expert(context_hidden=context, previous_hidden=prev,
                               current_h_tau=z_tau[:, s:e, :], tau=tau, chunk_start=s))
        return torch.cat(vels, dim=1)

    def _integrate_pass1(self, context, z0):
        z = z0
        dt = 1.0 / self.config.num_flow_steps
        for s in range(self.config.num_flow_steps):
            tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=self._dtype)
            z = z + dt * self._pass1_velocity(z, tau, context)
        return z

    # ---- pass 2: causal token-conditioned residual flow ----------------

    def _shift_emb(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B,K] -> [B,K,D] embedding of token i-1 (zeros at i=0). Strictly causal."""
        b, k = tokens.shape
        emb = self._embed(tokens)                       # [B,K,D] = emb of token i
        shifted = torch.zeros_like(emb)
        shifted[:, 1:, :] = emb[:, :-1, :]              # position i gets emb of token i-1
        return shifted

    def _integrate_pass2(self, context, z0, prev_tok_emb):
        z = z0
        dt = 1.0 / self.config.num_refine_steps
        for s in range(self.config.num_refine_steps):
            tau = torch.full((z.shape[0],), s * dt, device=z.device, dtype=self._dtype)
            z = z + dt * self.refine(z, tau, context, prev_tok_emb)
        return z

    # ---- training entry (on-policy: pass2 conditions on pass1 argmax) ---

    def forward_teacher(self, context, target_hidden, future_tokens):
        context = context.to(self._dtype)
        th = target_hidden.to(self._dtype)
        b = context.shape[0]
        # pass 1 flow-matching term
        z0 = self.init_latents(context)
        tau = torch.rand(b, 1, 1, device=context.device, dtype=self._dtype)
        z_tau = (1.0 - tau) * z0 + tau * th
        v1_star = th - z0
        v1_pred = self._pass1_velocity(z_tau, tau, context, previous=th)
        h1 = self._integrate_pass1(context, z0)         # pass-1 hiddens
        logits1 = self.lm_head(h1)
        # on-policy conditioning: pass 2 sees pass-1's OWN decoded tokens (detached), shifted causal
        with torch.no_grad():
            t1 = logits1.argmax(dim=-1)
        prev_emb = self._shift_emb(t1)
        # pass 2 flow-matching term: init at h1 (detached so pass-2 learns a correction), target th
        z0b = h1.detach()
        taub = torch.rand(b, 1, 1, device=context.device, dtype=self._dtype)
        z_taub = (1.0 - taub) * z0b + taub * th
        v2_star = th - z0b
        v2_pred = self.refine(z_taub, taub, context, prev_emb)
        h2 = self._integrate_pass2(context, z0b, prev_emb)
        logits2 = self.lm_head(h2)
        return {"v1_pred": v1_pred, "v1_star": v1_star, "v2_pred": v2_pred, "v2_star": v2_star,
                "h1": h1, "h2": h2, "logits1": logits1, "logits2": logits2}

    # ---- inference: 2 parallel passes -----------------------------------

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)
        with timed_section(timings, "drafter_twopass_flow", self.frozen_lm.device):
            context = self._context(state).to(self._dtype)
            z0 = self.init_latents(context)
            h1 = self._integrate_pass1(context, z0)
            t1 = self.frozen_lm.lm_head(h1).argmax(dim=-1)        # provisional tokens
            prev_emb = self._shift_emb(t1)
            h2 = self._integrate_pass2(context, h1, prev_emb)     # causal token-conditioned refine
            logits2 = self.frozen_lm.lm_head(h2)
            tokens = logits2.argmax(dim=-1)[:, :draft_len]
        return DraftResult(tokens=tokens, hidden_states=h2[:, :draft_len, :],
                           logits=logits2[:, :draft_len, :], timings=timings)
