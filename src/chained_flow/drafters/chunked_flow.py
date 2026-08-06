from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from chained_flow.drafters.base import DraftResult
from chained_flow.frozen_lm import FrozenLMWrapper, LMState
from chained_flow.timing import TimingStats, timed_section


@dataclass
class SingleExpertFlowConfig:
    context_size: int = 4
    draft_length: int = 2
    chunk_size: int = 2
    vae_dir: str | None = None
    expert_dim: int = 128
    num_heads: int = 4
    ffn_multiplier: int = 4
    num_flow_steps: int = 1
    noise_scale: float = 1.0
    init_mode: str = "noise"
    train_vae: bool = False
    architecture: str = "independent_experts"
    num_drafter_layers: int = 2
    drafter_dropout: float = 0.0
    detach_previous_chunks: bool = True

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError("draft_length must be >= 1")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.draft_length % self.chunk_size != 0:
            raise ValueError("draft_length must be divisible by chunk_size")
        if self.num_flow_steps < 1:
            raise ValueError("num_flow_steps must be >= 1")
        if self.vae_dir is None and self.architecture != "hidden_kv_flow":
            raise ValueError("SingleExpertFlowDrafter requires vae_dir unless architecture='hidden_kv_flow'")
        if self.architecture not in {"independent_experts", "block_causal_experts", "hidden_kv_flow"}:
            raise ValueError("architecture must be 'independent_experts', 'block_causal_experts', or 'hidden_kv_flow'")
        if self.init_mode not in {"noise", "repeat_last", "delta"}:
            raise ValueError("init_mode must be 'noise', 'repeat_last', or 'delta'")
        if self.num_drafter_layers < 1:
            raise ValueError("num_drafter_layers must be >= 1")


class FusedBlockCausalFlowExpert(nn.Module):
    def __init__(
        self,
        *,
        latent_size: int,
        context_size: int,
        draft_length: int,
        chunk_size: int,
        expert_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if expert_dim % num_heads != 0:
            raise ValueError("expert_dim must be divisible by num_heads")
        self.latent_size = latent_size
        self.context_size = context_size
        self.draft_length = draft_length
        self.chunk_size = chunk_size
        self.latent_proj = nn.Linear(latent_size, expert_dim)
        self.position_embedding = nn.Embedding(context_size + draft_length, expert_dim)
        self.segment_embedding = nn.Embedding(3, expert_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=expert_dim,
            nhead=num_heads,
            dim_feedforward=expert_dim * ffn_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(expert_dim)
        self.out_proj = nn.Linear(expert_dim, latent_size)

    def _tau_embedding(self, tau: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        elif tau.ndim == 3:
            tau = tau.reshape(batch, 1)
        elif tau.ndim != 2:
            raise ValueError("tau must have shape [B], [B, 1], or [B, 1, 1]")
        return self.time_mlp(tau.to(dtype=dtype))

    def _attention_mask(self, *, context_len: int, prev_len: int, current_len: int, device: torch.device) -> torch.Tensor:
        seq_len = context_len + prev_len + current_len
        mask = torch.zeros((seq_len, seq_len), device=device)
        current_start = context_len + prev_len
        for row in range(current_len):
            row_idx = current_start + row
            first_future = current_start + row + 1
            if first_future < seq_len:
                mask[row_idx, first_future:] = float("-inf")
        return mask

    def forward(
        self,
        *,
        context_latents: torch.Tensor,
        previous_latents: torch.Tensor,
        current_z_tau: torch.Tensor,
        tau: torch.Tensor,
        chunk_start: int,
    ) -> torch.Tensor:
        if context_latents.ndim != 3 or previous_latents.ndim != 3 or current_z_tau.ndim != 3:
            raise ValueError("all latent inputs must have shape [B, L, Z]")
        batch = current_z_tau.shape[0]
        if current_z_tau.shape[1] != self.chunk_size:
            raise ValueError(f"current_z_tau chunk length must be {self.chunk_size}")
        if chunk_start + self.chunk_size > self.draft_length:
            raise ValueError("current chunk exceeds draft_length")

        previous_latents = previous_latents.to(device=current_z_tau.device, dtype=current_z_tau.dtype)
        context_latents = context_latents.to(device=current_z_tau.device, dtype=current_z_tau.dtype)
        tokens = torch.cat([context_latents, previous_latents, current_z_tau], dim=1)
        x = self.latent_proj(tokens)
        context_len = context_latents.shape[1]
        prev_len = previous_latents.shape[1]
        current_len = current_z_tau.shape[1]

        context_pos = torch.arange(context_len, device=x.device)
        prev_pos = torch.arange(prev_len, device=x.device) + self.context_size
        current_pos = torch.arange(current_len, device=x.device) + self.context_size + chunk_start
        pos = torch.cat([context_pos, prev_pos, current_pos], dim=0)
        if int(pos.max().item()) >= self.context_size + self.draft_length:
            raise ValueError("fused sequence position exceeds configured embedding table")
        x = x + self.position_embedding(pos).unsqueeze(0)

        segment_ids = torch.cat([
            torch.zeros(context_len, dtype=torch.long, device=x.device),
            torch.full((prev_len,), 2, dtype=torch.long, device=x.device),
            torch.ones(current_len, dtype=torch.long, device=x.device),
        ])
        x = x + self.segment_embedding(segment_ids).unsqueeze(0)
        if current_len > 0:
            x[:, -current_len:, :] = x[:, -current_len:, :] + self._tau_embedding(tau, batch, x.dtype).unsqueeze(1)

        mask = self._attention_mask(
            context_len=context_len,
            prev_len=prev_len,
            current_len=current_len,
            device=x.device,
        )
        x = self.blocks(x, mask=mask)
        current_out = x[:, -current_len:, :]
        return self.out_proj(self.out_norm(current_out))




class HiddenKVFlowBlock(nn.Module):
    def __init__(self, *, hidden_size: int, num_heads: int, ffn_multiplier: int, dropout: float = 0.0):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        ffn_dim = hidden_size * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor, context_hidden: torch.Tensor, *, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        self_in = self.self_norm(x)
        # bf16-safe: additive masks are built in fp32; match the compute dtype, and force a
        # non-cuDNN SDPA backend (nn.MultiheadAttention's cuDNN path has no bf16 plan for these shapes).
        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(self_in.dtype)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            self_out, _ = self.self_attn(self_in, self_in, self_in, attn_mask=attn_mask, need_weights=False)
            x = x + self_out
            cross_out, _ = self.cross_attn(
                self.cross_norm(x),
                self.context_norm(context_hidden),
                self.context_norm(context_hidden),
                need_weights=False,
            )
        x = x + cross_out
        x = x + self.ffn(x)
        return x


class HiddenKVFlowExpert(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        context_size: int,
        draft_length: int,
        chunk_size: int,
        expert_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        anchor: bool = False,
    ):
        super().__init__()
        if expert_dim != hidden_size:
            raise ValueError("hidden_kv_flow requires expert_dim to equal the target LM hidden size")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.draft_length = draft_length
        self.chunk_size = chunk_size
        self.position_embedding = nn.Embedding(draft_length, hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.blocks = nn.ModuleList(
            [
                HiddenKVFlowBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        # ANCHOR conditioning: emb(last committed token) injected into the residual stream the same
        # way tau is, so EVERY block sees it and it conditions the VELOCITY FIELD itself. This is
        # what EAGLE ([embedding || feature]) and DeepSeek-V3 MTP (M_k[RMSNorm(h); RMSNorm(Emb(t))])
        # do, and is distinct from our two failed attempts, which put a SYNTHESISED HIDDEN in the
        # context (lag_proj) or the token in the RESCORER (path_head/markov).
        # Zero-init the output so an anchored model starts identical to the unanchored one.
        # Gated on `anchor`: creating it unconditionally makes EVERY pre-anchor checkpoint report
        # missing keys on load, which destroys the value of that warning.
        self.anchor_mlp = None
        if anchor:
            self.anchor_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
            )
            nn.init.zeros_(self.anchor_mlp[-1].weight)
            nn.init.zeros_(self.anchor_mlp[-1].bias)

    def _tau_embedding(self, tau: torch.Tensor, batch: int, dtype: torch.dtype) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        elif tau.ndim == 3:
            tau = tau.reshape(batch, 1)
        elif tau.ndim != 2:
            raise ValueError("tau must have shape [B], [B, 1], or [B, 1, 1]")
        return self.time_mlp(tau.to(dtype=dtype))

    def _draft_attention_mask(self, *, prev_len: int, current_len: int, device: torch.device) -> torch.Tensor:
        seq_len = prev_len + current_len
        return torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1)

    @property
    def _cf_fused_off(self) -> bool:
        off = self.__dict__.get("_cf_fused_off_cached")
        if off is None:
            from chained_flow import cuda_block
            off = not cuda_block.enabled()
            self.__dict__["_cf_fused_off_cached"] = off
        return off

    @property
    def _cf_batch_off(self) -> bool:
        off = self.__dict__.get("_cf_batch_off_cached")
        if off is None:
            from chained_flow import cuda_block
            off = not cuda_block.batch_enabled()
            self.__dict__["_cf_batch_off_cached"] = off
        return off

    def _cf_fused_runner(self, x: torch.Tensor, context_hidden: torch.Tensor):
        """Fused-CUDA block runner, or None to stay on the PyTorch path.

        Only engages for the shape family the kernel is instantiated for (fp16,
        D in {640, 1024}, 8 heads, ffn x6, S in {4,8}, C<=16); anything else silently falls
        back.  ``can_run`` also rejects a (S, C, B) whose shared memory does not fit or whose
        batch would need more co-resident blocks than the machine has.

        BATCH > 1 IS GATED ON ``CF_CUDA_BLOCK_BATCH`` (default off).  The gate used to be a flat
        ``x.shape[0] == 1``, so at any decode concurrency the whole block stack silently reverted
        to the PyTorch/cutlass path -- which is where ~48% of a bucket-64 draft went."""
        from chained_flow import cuda_block

        if not cuda_block.enabled():
            return None
        if not (x.is_cuda and x.dtype == torch.float16
                and x.shape[1] in cuda_block.SUPPORTED_S):
            return None
        if x.shape[0] != 1 and self._cf_batch_off:
            return None
        static_ok = getattr(self, "_cf_static_ok", None)
        if static_ok is None:
            b0 = self.blocks[0]
            static_ok = (
                self.hidden_size in cuda_block.SUPPORTED_D
                and b0.ffn[1].out_features // self.hidden_size in cuda_block.SUPPORTED_FM
                and b0.ffn[1].out_features % self.hidden_size == 0
                and b0.self_attn.num_heads == cuda_block.NUM_HEADS
                and b0.self_attn._qkv_same_embed_dim
            )
            self._cf_static_ok = static_ok
        if not static_ok:
            return None
        fb = cuda_block.attach(self)            # None if the extension would not build
        if fb is None:
            return None
        return fb if fb.can_run(x.shape[1], context_hidden.shape[1], x.shape[0]) else None

    def pre_blocks(
        self,
        *,
        context_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        current_h_tau: torch.Tensor,
        tau: torch.Tensor,
        chunk_start: int,
        anchor: torch.Tensor | None = None,
    ):
        """Everything before the block stack -> (x, attn_mask, current_len).

        Split out of ``forward`` (which is still the only caller on the default path) so
        ``CF_CUDA_PAIR`` can build BOTH chunks' block-stack inputs before issuing either
        stack -- see cuda_block.run_pair."""
        if context_hidden.ndim != 3 or previous_hidden.ndim != 3 or current_h_tau.ndim != 3:
            raise ValueError("all hidden inputs must have shape [B, L, D]")
        batch = current_h_tau.shape[0]
        if current_h_tau.shape[1] != self.chunk_size:
            raise ValueError(f"current_h_tau chunk length must be {self.chunk_size}")
        if chunk_start + self.chunk_size > self.draft_length:
            raise ValueError("current chunk exceeds draft_length")
        if current_h_tau.shape[-1] != self.hidden_size or context_hidden.shape[-1] != self.hidden_size:
            raise ValueError("hidden_kv_flow inputs must use the target LM hidden size")

        previous_hidden = previous_hidden.to(device=current_h_tau.device, dtype=current_h_tau.dtype)
        context_hidden = context_hidden.to(device=current_h_tau.device, dtype=current_h_tau.dtype)
        x = torch.cat([previous_hidden, current_h_tau], dim=1)
        prev_len = previous_hidden.shape[1]
        current_len = current_h_tau.shape[1]
        current_pos = torch.arange(current_len, device=x.device) + chunk_start
        if prev_len > 0:
            prev_pos = torch.arange(prev_len, device=x.device)
            pos = torch.cat([prev_pos, current_pos], dim=0)
        else:
            pos = current_pos
        # pos is a static arange over Python ints; compute its max without a .item() CUDA sync
        # (that assertion was a capture blocker for CUDA graphs and a per-call sync in eager).
        if max(prev_len - 1, chunk_start + current_len - 1) >= self.draft_length:
            raise ValueError("hidden_kv_flow position exceeds configured draft length")
        x = x + self.position_embedding(pos).unsqueeze(0)
        tau_add = self._tau_embedding(tau, batch, x.dtype).unsqueeze(1)
        x = torch.cat([x[:, :-current_len, :], x[:, -current_len:, :] + tau_add], dim=1)
        if anchor is not None and self.anchor_mlp is not None:
            x = x + self.anchor_mlp(anchor.to(x.dtype)).unsqueeze(1)   # conditions every block

        attn_mask = self._draft_attention_mask(prev_len=prev_len, current_len=current_len, device=x.device)
        return x, attn_mask, current_len

    def post_blocks(self, x: torch.Tensor, current_len: int) -> torch.Tensor:
        """Everything after the block stack: keep the current chunk's rows and project out."""
        return self.out_proj(self.out_norm(x[:, -current_len:, :]))

    def forward(
        self,
        *,
        context_hidden: torch.Tensor,
        previous_hidden: torch.Tensor,
        current_h_tau: torch.Tensor,
        tau: torch.Tensor,
        chunk_start: int,
        anchor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, attn_mask, current_len = self.pre_blocks(
            context_hidden=context_hidden, previous_hidden=previous_hidden,
            current_h_tau=current_h_tau, tau=tau, chunk_start=chunk_start, anchor=anchor)
        # CF_CUDA_BLOCK=1 runs the whole block stack as ONE hand-written CUDA kernel instead of the
        # ~24 tiny kernels per block PyTorch emits (see chained_flow.cuda_block). Off by default.
        # The enabled() check is resolved ONCE at construction, not here: a local import plus
        # env lookups inside this torch.compile'd forward graph-breaks on EVERY call, which
        # costs the compiled drafter even when the kernel is off.
        fused = None if self._cf_fused_off else self._cf_fused_runner(x, context_hidden)
        if fused is not None:
            # Dynamo must NOT trace into the extension: under vLLM the drafter runs inside
            # torch.inference_mode(), and AOT functionalization tries to version-track the
            # kernel's out-tensors -- "Inference tensors do not track version counter".
            # Tracing it also perturbed Inductor's autotune on the surrounding graph.
            x = torch._dynamo.disable(fused.forward)(x, context_hidden, attn_mask)
        else:
            for block in self.blocks:
                x = block(x, context_hidden, attn_mask=attn_mask)
        return self.post_blocks(x, current_len)

class CrossAttentionFlowExpert(nn.Module):
    def __init__(
        self,
        *,
        latent_size: int,
        chunk_size: int,
        expert_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
    ):
        super().__init__()
        if expert_dim % num_heads != 0:
            raise ValueError("expert_dim must be divisible by num_heads")
        self.latent_size = latent_size
        self.chunk_size = chunk_size
        self.query_proj = nn.Linear(latent_size, expert_dim)
        self.key_proj = nn.Linear(latent_size, expert_dim)
        self.value_proj = nn.Linear(latent_size, expert_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, expert_dim),
            nn.SiLU(),
            nn.Linear(expert_dim, expert_dim),
        )
        self.slot_embedding = nn.Embedding(chunk_size, expert_dim)
        self.query_norm = nn.LayerNorm(expert_dim)
        self.context_norm = nn.LayerNorm(expert_dim)
        self.self_attn = nn.MultiheadAttention(expert_dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(expert_dim, num_heads, batch_first=True)
        ffn_dim = expert_dim * ffn_multiplier
        self.ffn = nn.Sequential(
            nn.LayerNorm(expert_dim),
            nn.Linear(expert_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, expert_dim),
        )
        self.out_norm = nn.LayerNorm(expert_dim)
        self.out_proj = nn.Linear(expert_dim, latent_size)

    def forward(self, z_tau: torch.Tensor, tau: torch.Tensor, context_latents: torch.Tensor) -> torch.Tensor:
        if z_tau.ndim != 3:
            raise ValueError("z_tau must have shape [B, C, Z]")
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, m, Z]")
        if z_tau.shape[1] != self.chunk_size:
            raise ValueError(f"z_tau chunk length must be {self.chunk_size}, got {z_tau.shape[1]}")
        if z_tau.shape[-1] != self.latent_size or context_latents.shape[-1] != self.latent_size:
            raise ValueError("z_tau and context_latents must use the configured latent size")

        batch = z_tau.shape[0]
        if tau.ndim == 1:
            tau = tau[:, None]
        elif tau.ndim == 3:
            tau = tau.reshape(batch, 1)
        elif tau.ndim != 2:
            raise ValueError("tau must have shape [B], [B, 1], or [B, 1, 1]")
        tau = tau.to(device=z_tau.device, dtype=z_tau.dtype)

        slot_ids = torch.arange(self.chunk_size, device=z_tau.device)
        q = self.query_proj(z_tau)
        q = q + self.time_mlp(tau).unsqueeze(1)
        q = q + self.slot_embedding(slot_ids).unsqueeze(0)
        q = self.query_norm(q)

        k = self.context_norm(self.key_proj(context_latents))
        v = self.context_norm(self.value_proj(context_latents))

        self_out, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + self_out
        cross_out, _ = self.cross_attn(q, k, v, need_weights=False)
        q = q + cross_out
        q = q + self.ffn(q)
        return self.out_proj(self.out_norm(q))


class SingleExpertFlowDrafter(nn.Module):
    def __init__(self, frozen_lm: FrozenLMWrapper, config: SingleExpertFlowConfig):
        super().__init__()
        self.frozen_lm = frozen_lm
        self.config = config
        self.hidden_size = frozen_lm.model.config.hidden_size
        self.num_chunks = config.draft_length // config.chunk_size
        self.vae = None
        self.include_vae_in_state_dict = config.train_vae and config.architecture != "hidden_kv_flow"

        if config.architecture == "hidden_kv_flow":
            self.latent_size = self.hidden_size
            self.expert = HiddenKVFlowExpert(
                hidden_size=self.hidden_size,
                context_size=config.context_size,
                draft_length=config.draft_length,
                chunk_size=config.chunk_size,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier,
                num_layers=config.num_drafter_layers,
                dropout=config.drafter_dropout,
            )
            self.extra_experts = nn.ModuleList(
                [
                    HiddenKVFlowExpert(
                        hidden_size=self.hidden_size,
                        context_size=config.context_size,
                        draft_length=config.draft_length,
                        chunk_size=config.chunk_size,
                        expert_dim=config.expert_dim,
                        num_heads=config.num_heads,
                        ffn_multiplier=config.ffn_multiplier,
                        num_layers=config.num_drafter_layers,
                        dropout=config.drafter_dropout,
                    )
                    for _ in range(self.num_chunks - 1)
                ]
            )
            self._cached_vae_dtype = torch.float32
            self._cached_expert_dtype = next(self.expert.parameters()).dtype
            return

        from chained_flow.vae import load_hidden_vae_from_dir

        self.vae = load_hidden_vae_from_dir(config.vae_dir, device=frozen_lm.device, freeze=not config.train_vae)
        self._cached_vae_dtype = next(self.vae.parameters()).dtype
        self.latent_size = self.vae.config.latent_size
        expert_cls = FusedBlockCausalFlowExpert if config.architecture == "block_causal_experts" else CrossAttentionFlowExpert
        if config.architecture == "block_causal_experts":
            self.expert = expert_cls(
                latent_size=self.latent_size,
                context_size=config.context_size,
                draft_length=config.draft_length,
                chunk_size=config.chunk_size,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier,
                num_layers=config.num_drafter_layers,
                dropout=config.drafter_dropout,
            )
            self.extra_experts = nn.ModuleList(
                [
                    expert_cls(
                        latent_size=self.latent_size,
                        context_size=config.context_size,
                        draft_length=config.draft_length,
                        chunk_size=config.chunk_size,
                        expert_dim=config.expert_dim,
                        num_heads=config.num_heads,
                        ffn_multiplier=config.ffn_multiplier,
                        num_layers=config.num_drafter_layers,
                        dropout=config.drafter_dropout,
                    )
                    for _ in range(self.num_chunks - 1)
                ]
            )
        else:
            self.expert = expert_cls(
                latent_size=self.latent_size,
                chunk_size=config.chunk_size,
                expert_dim=config.expert_dim,
                num_heads=config.num_heads,
                ffn_multiplier=config.ffn_multiplier,
            )
            self.extra_experts = nn.ModuleList(
                [
                    expert_cls(
                        latent_size=self.latent_size,
                        chunk_size=config.chunk_size,
                        expert_dim=config.expert_dim,
                        num_heads=config.num_heads,
                        ffn_multiplier=config.ffn_multiplier,
                    )
                    for _ in range(self.num_chunks - 1)
                ]
            )
        self._cached_expert_dtype = next(self.expert.parameters()).dtype

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        if self.include_vae_in_state_dict:
            return state
        return {key: value for key, value in state.items() if not key.startswith("vae.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        full_state = super().state_dict()
        full_state.update(state_dict)
        return super().load_state_dict(full_state, strict=strict, assign=assign)

    def _chunk_experts(self) -> list[nn.Module]:
        return [self.expert, *self.extra_experts]

    def flow_velocity(
        self,
        z_tau: torch.Tensor,
        tau: torch.Tensor,
        context_latents: torch.Tensor,
        *,
        previous_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if z_tau.ndim != 3:
            raise ValueError("z_tau must have shape [B, K, Z]")
        if z_tau.shape[1] != self.config.draft_length:
            raise ValueError(f"z_tau must have draft length {self.config.draft_length}, got {z_tau.shape[1]}")
        velocities = []
        for chunk_idx, expert in enumerate(self._chunk_experts()):
            start = chunk_idx * self.config.chunk_size
            end = start + self.config.chunk_size
            if self.config.architecture == "hidden_kv_flow":
                if previous_latents is None:
                    prev = z_tau[:, :start, :]
                else:
                    prev = previous_latents[:, :start, :]
                    if self.config.detach_previous_chunks:
                        prev = prev.detach()
                velocities.append(
                    expert(
                        context_hidden=context_latents,
                        previous_hidden=prev,
                        current_h_tau=z_tau[:, start:end, :],
                        tau=tau,
                        chunk_start=start,
                    )
                )
            elif self.config.architecture == "block_causal_experts":
                if previous_latents is None:
                    prev = z_tau[:, :start, :]
                else:
                    prev = previous_latents[:, :start, :]
                    if self.config.detach_previous_chunks:
                        prev = prev.detach()
                velocities.append(
                    expert(
                        context_latents=context_latents,
                        previous_latents=prev,
                        current_z_tau=z_tau[:, start:end, :],
                        tau=tau,
                        chunk_start=start,
                    )
                )
            else:
                velocities.append(expert(z_tau[:, start:end, :], tau, context_latents))
        return torch.cat(velocities, dim=1)

    def _context(self, state: LMState) -> torch.Tensor:
        hidden = state.final_hidden
        if hidden.shape[1] >= self.config.context_size:
            return hidden[:, -self.config.context_size :, :]
        pad_len = self.config.context_size - hidden.shape[1]
        pad = hidden[:, :1, :].expand(-1, pad_len, -1)
        return torch.cat([pad, hidden], dim=1)

    def _vae_dtype(self) -> torch.dtype:
        return self._cached_vae_dtype

    def _expert_dtype(self) -> torch.dtype:
        return self._cached_expert_dtype

    def encode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.config.architecture == "hidden_kv_flow":
            return hidden.to(device=hidden.device, dtype=self._expert_dtype())
        original_shape = hidden.shape[:-1]
        hidden = hidden.to(device=hidden.device, dtype=self._vae_dtype())
        if hidden.ndim == 3:
            if self.config.train_vae:
                latent = self.vae.encode_sequence(hidden).mu
            else:
                with torch.no_grad():
                    latent = self.vae.encode_sequence(hidden).mu
            return latent
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        if self.config.train_vae:
            latent = self.vae.encode(flat_hidden).mu
        else:
            with torch.no_grad():
                latent = self.vae.encode(flat_hidden).mu
        return latent.reshape(*original_shape, -1)

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.config.architecture == "hidden_kv_flow":
            return latent
        original_shape = latent.shape[:-1]
        latent = latent.to(device=latent.device, dtype=self._vae_dtype())
        if latent.ndim == 3:
            decoded = self.vae.decode_sequence(latent)
            return decoded
        flat_latent = latent.reshape(-1, latent.shape[-1])
        decoded = self.vae.decode(flat_latent)
        return decoded.reshape(*original_shape, -1)

    def init_latents(
        self,
        context_latents: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, m, Z]")
        context_latents = context_latents.to(device=context_latents.device, dtype=self._expert_dtype())
        batch = context_latents.shape[0]
        draft_length = self.config.draft_length
        if self.config.init_mode == "noise":
            return torch.randn(
                batch,
                draft_length,
                self.latent_size,
                generator=generator,
                device=context_latents.device,
                dtype=context_latents.dtype,
            ) * self.config.noise_scale

        last = context_latents[:, -1:, :]
        if self.config.init_mode == "repeat_last":
            return last.expand(batch, draft_length, self.latent_size).clone()

        if context_latents.shape[1] > 1:
            delta = context_latents[:, -1:, :] - context_latents[:, -2:-1, :]
        else:
            delta = torch.zeros_like(last)
        steps = torch.arange(1, draft_length + 1, device=context_latents.device, dtype=context_latents.dtype)
        return last + steps.view(1, draft_length, 1) * delta

    def integrate_latents(
        self,
        context_latents: torch.Tensor,
        *,
        z0: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError("context_latents must have shape [B, m, Z]")
        context_latents = context_latents.to(device=context_latents.device, dtype=self._expert_dtype())
        batch = context_latents.shape[0]
        steps = self.config.num_flow_steps if num_steps is None else num_steps
        if steps < 1:
            raise ValueError("num_steps must be >= 1")
        if z0 is None:
            z = self.init_latents(context_latents)
        else:
            z = z0.to(device=context_latents.device, dtype=context_latents.dtype)
            if z.shape[1] != self.config.draft_length:
                raise ValueError(f"z0 must have draft length {self.config.draft_length}, got {z.shape[1]}")
        dt = 1.0 / steps
        for step in range(steps):
            tau = torch.full((batch,), step * dt, device=context_latents.device, dtype=context_latents.dtype)
            z = z + dt * self.flow_velocity(z, tau, context_latents)
        return z

    def predict_latent_from_context(
        self,
        context_hidden: torch.Tensor,
        max_tokens: int | None = None,
        *,
        z0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_hidden.ndim != 3:
            raise ValueError("context_hidden must have shape [B, m, D]")
        if context_hidden.shape[1] != self.config.context_size:
            raise ValueError(
                f"context_hidden must have context size {self.config.context_size}, got {context_hidden.shape[1]}"
            )
        draft_len = self.config.draft_length if max_tokens is None else min(max_tokens, self.config.draft_length)
        context_latents = self.encode_hidden(context_hidden)
        pred_latents = self.integrate_latents(context_latents, z0=z0)
        return pred_latents[:, :draft_len, :]

    def predict_hidden(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        latent = self.predict_latent_from_context(self._context(state), max_tokens=max_tokens)
        return self.decode_latent(latent)

    def forward(self, state: LMState, max_tokens: int | None = None) -> torch.Tensor:
        return self.predict_hidden(state, max_tokens=max_tokens)

    @torch.inference_mode()
    def propose(self, state: LMState, max_tokens: int) -> DraftResult:
        timings = TimingStats()
        draft_len = min(max_tokens, self.config.draft_length)
        if draft_len <= 0:
            empty = torch.empty((state.input_ids.shape[0], 0), dtype=torch.long, device=self.frozen_lm.device)
            return DraftResult(tokens=empty, timings=timings)

        with timed_section(timings, "drafter_single_expert_flow", self.frozen_lm.device):
            future_latent = self.predict_latent_from_context(self._context(state), max_tokens=draft_len)
            future_hidden = self.decode_latent(future_latent)
            logits = self.frozen_lm.lm_head(future_hidden)
            tokens = logits.argmax(dim=-1)
        return DraftResult(
            tokens=tokens,
            hidden_states=future_hidden,
            latent_states=future_latent,
            logits=logits,
            timings=timings,
        )
