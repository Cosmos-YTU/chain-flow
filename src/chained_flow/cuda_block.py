"""Hand-written CUDA fusion for the drafter's HiddenKVFlowBlock stack.

**DEFAULT ON**, capability-gated by ``chained_flow.defaults.finalize_cuda_block``: the shape gate
below is checked against the loaded drafter and the extension build is attempted BEFORE the first
forward, and ``CF_CUDA_BLOCK`` is written down to ``0`` if either fails.  ``CF_CUDA_BLOCK=0``
always wins and restores the byte-identical PyTorch path.  See ``csrc/fused_block.cu``.

What it replaces: PyTorch emits ~24 kernels per HiddenKVFlowBlock.forward and the drafter runs 32
block-passes per draft.  This module packs each expert's per-block weights into one contiguous
fp16 buffer and runs the whole 8-block stack as ONE persistent grid-synchronising kernel, plus one
small kernel per expert-forward that hoists the cross-attention K/V (constant across all
block-passes of a draft) out of the inner loop.

Covered shapes (anything else silently falls back to PyTorch -- see ``supported()``):
    batch 1, fp16, heads 8, ffn multiplier 6, S in {4, 8}, C <= 16, and
    D = 640  (Flow-Drafter-4B-v2)
    D = 1024 (Flow-Drafter-9B, Flow-Drafter-Qwen3.5-27B-v2)

Tunables (env): ``CF_CUDA_BLOCK_G`` grid size, ``CF_CUDA_BLOCK_T`` threads/block,
``CF_CUDA_BLOCK_LB`` launch-bound variant (0 = (512,1), 1 = (512,2), 2 = (1024,1); the
default follows the thread count),
``CF_CUDA_BLOCK_STAGES`` profiling hook, ``100*onebar + 10*skip + n``: run only the first n of the
6 stages per block; skip the attention (skip=1) or attention plus its staging (skip=2); drop 5 of
the 6 grid barriers while keeping all the work (onebar=1, a stopwatch -- the result is wrong).

``CF_CUDA_PAIR`` (DEFAULT ON behind ``CF_CUDA_BLOCK``) additionally runs the draft's TWO
CHUNK-EXPERTS CONCURRENTLY on two streams -- see ``run_pair``.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_MOD = None
_SRC = Path(__file__).resolve().parent / "csrc" / "fused_block.cu"

MAXS = 8
SUPPORTED_S = (4, 8)     # the kernel is instantiated for these two row counts only
SUPPORTED_D = (640, 1024)
SUPPORTED_FM = (6,)
NUM_HEADS = 8
MAX_C = 16


class Layout:
    """Offsets (in halves) into the packed per-pass weight buffer.  Mirrors ``WL``/``PL`` in
    fused_block.cu exactly -- both are the same running sum over the block's parameter list."""

    __slots__ = ("D", "FM", "FD", "off", "wstride", "poff", "pstride")

    def __init__(self, D: int, FM: int):
        self.D, self.FM = D, FM
        self.FD = FD = D * FM
        o, sizes = 0, [
            ("SN_W", D), ("SN_B", D),
            ("SIN_W", 3 * D * D), ("SIN_B", 3 * D),
            ("SOUT_W", D * D), ("SOUT_B", D),
            ("XN_W", D), ("XN_B", D),
            ("CQ_W", D * D), ("CQ_B", D),
            ("COUT_W", D * D), ("COUT_B", D),
            ("FN_W", D), ("FN_B", D),
            ("FU_W", FD * D), ("FU_B", FD),
            ("FD_W", D * FD), ("FD_B", D),
        ]
        self.off = {}
        for name, n in sizes:
            self.off[name] = o
            o += n
        self.wstride = o
        o, self.poff = 0, {}
        for name, n in (("CN_W", D), ("CN_B", D), ("KV_W", 2 * D * D), ("KV_B", 2 * D)):
            self.poff[name] = o
            o += n
        self.pstride = o


_PLD = 17  # score-table row stride, must match PLD in fused_block.cu


def smem_bytes(D: int, S: int, C: int) -> int:
    """Dynamic shared memory the kernel needs.  Mirrors ``smem_halves`` in fused_block.cu."""
    stage = max(S * (3 * D + 8), (S + 2 * C) * (D + 8), S * D)
    return (NUM_HEADS * S * _PLD * 2 + S * D + stage) * 2


def supported(D: int, ffn_mult: int, num_heads: int, S: int, C: int, device=None) -> bool:
    """Pure-Python shape gate.  Kept free of any extension call so it can run inside a
    torch.compile'd forward without a graph break."""
    if not (D in SUPPORTED_D and ffn_mult in SUPPORTED_FM and num_heads == NUM_HEADS
            and S in SUPPORTED_S and 0 < C <= MAX_C):
        return False
    limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return smem_bytes(D, S, C) <= limit


def _ver(t) -> int:
    try:  # inference tensors do not track a version counter
        return t._version
    except RuntimeError:
        return -1


def enabled() -> bool:
    # DEFAULT ON, gated: `chained_flow.defaults` writes this env var DOWN to "0" in
    # `finalize_cuda_block` when the shape is unsupported or the extension will not build, so by
    # the time this is read the answer is already the resolved one.  Kept as a bare env read on
    # purpose: it is evaluated inside a torch.compile'd forward and must not import or probe.
    return os.environ.get("CF_CUDA_BLOCK", "1") not in ("0", "", "false", "False")


_LOAD_ERR: str | None = None


def available() -> tuple[bool, str]:
    """(built_ok, error).  Attempts the JIT build ONCE and remembers the failure.

    The whole point of a default-ON kernel flag is that a box without nvcc / with a bad
    TORCH_CUDA_ARCH_LIST must fall back to PyTorch rather than crash the engine -- but SILENTLY
    falling back is how `CF_CUDA_BLOCK` produced meaningless A/Bs for hours, so the failure is
    printed once, in full, and recorded in the startup summary.
    """
    global _LOAD_ERR
    if _MOD is not None:
        return True, ""
    if _LOAD_ERR is not None:
        return False, _LOAD_ERR
    if not _SRC.is_file():
        # A wheel built without `package-data` would land here: importable package, no .cu.
        # Named explicitly because "ninja: no such file" would send the reader after a
        # toolchain problem that does not exist.
        _LOAD_ERR = (f"the CUDA source {_SRC} is missing from the installed package "
                     f"(pyproject's [tool.setuptools.package-data] ships csrc/*.cu)")
        print(f"[cf] CF_CUDA_BLOCK: {_LOAD_ERR} -- falling back to the PyTorch block stack "
              f"(bit-identical, ~2x slower draft).", flush=True)
        return False, _LOAD_ERR
    try:
        _load()
        # OFFLINE HAZARD, now that the flag defaults ON: this kernel is nondeterministic
        # OUTSIDE cudagraph capture (2.70% of tree nodes flip run to run), and offline scripts
        # do not capture anything.  In-engine the draft IS captured, so the note would be a
        # false alarm there -- the proposer module's presence is the discriminator.
        import sys
        if "chained_flow.vllm_plugin.flow_proposer" not in sys.modules:
            print("[cf] CF_CUDA_BLOCK is ON BY DEFAULT and this process is not the vLLM "
                  "proposer (no cudagraph capture): the kernel is nondeterministic outside "
                  "capture. Any offline bit-exactness check must set CF_CUDA_BLOCK=0.",
                  flush=True)
        return True, ""
    except Exception as e:                                  # noqa: BLE001 - any build failure
        _LOAD_ERR = f"{type(e).__name__}: {e}"
        print(f"[cf] CF_CUDA_BLOCK: the fused-block CUDA extension DID NOT BUILD -- falling "
              f"back to the PyTorch block stack (bit-identical, ~2x slower draft).\n"
              f"[cf]   {_LOAD_ERR.splitlines()[0][:400]}", flush=True)
        return False, _LOAD_ERR


_NAME = "cf_fused_block"


def build_dir() -> str:
    """Where the JIT build lands.

    torch's default is ``$TORCH_EXTENSIONS_DIR`` or ``~/.cache/torch_extensions/<pyver_cuver>``,
    keyed by the extension NAME, and it reuses an existing ``.so`` whenever the source
    timestamps/hashes are unchanged -- so the ~60 s compile is a once-per-machine cost, not a
    once-per-run one.  ``CF_KERNEL_DIR`` overrides it, for read-only or shared HOMEs (a
    container image can precompile into an image-local directory and point every run at it).
    """
    d = os.environ.get("CF_KERNEL_DIR")
    if d:
        os.environ.setdefault("TORCH_EXTENSIONS_DIR", d)
    from torch.utils.cpp_extension import _get_build_directory

    return _get_build_directory(_NAME, verbose=False)


def _arch_list() -> str:
    """``TORCH_CUDA_ARCH_LIST`` for THIS box, not for the one it was developed on.

    It used to be hard-coded to ``12.0`` (the RTX PRO 6000 this was written on).  That is the
    exact shape of a silent portability failure: on any other GPU the extension either fails to
    build or builds code the device cannot run, and the only symptom is the PyTorch fallback and
    a slower draft.  Derived from the device instead; an explicit env var still wins, because
    cross-compiling for a target box is a legitimate thing to want.
    """
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _load():
    """Compile-or-reuse the extension.  Prints ONE line, and only when it actually compiles."""
    global _MOD
    if _MOD is not None:
        return _MOD
    import time

    from torch.utils.cpp_extension import load

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _arch_list())
    bd = build_dir()
    # A cache HIT must be silent (it is ~10 ms and happens on every engine start); a cache MISS
    # takes ~60 s of ninja and MUST say so, because an unexplained one-minute stall at model
    # load looks exactly like a hang.
    cached = os.path.exists(os.path.join(bd, _NAME + ".so"))
    if not cached:
        print(f"[cf] compiling the fused-block CUDA extension for sm_"
              f"{os.environ['TORCH_CUDA_ARCH_LIST'].replace('.', '')} -- this takes ~60 s the "
              f"FIRST time on this machine and is then cached in {bd}. "
              f"(CF_CUDA_BLOCK=0 skips it and uses the bit-identical PyTorch path.)",
              flush=True)
    t0 = time.time()
    _MOD = load(
        name=_NAME,
        sources=[str(_SRC)],
        extra_cuda_cflags=["-O3", "-lineinfo", "--expt-relaxed-constexpr"],
        verbose=False,
    )
    if not cached:
        print(f"[cf] fused-block CUDA extension built in {time.time() - t0:.0f} s", flush=True)
    return _MOD


def _scratch(device, D, FD):
    """Statically allocated cross-stage buffers.  Allocated once per expert, outside any graph
    capture, so the kernel path does no dynamic allocation and no host sync.

    PER EXPERT, not shared: under ``CF_CUDA_PAIR`` the two chunk-experts run CONCURRENTLY, so a
    shared residual/qkv/hf scratch would have them overwrite each other.  The whole set is
    ~130 KB, so privacy is free."""
    h = dict(device=device, dtype=torch.float16)
    return dict(
        res=torch.zeros(MAXS * D, device=device, dtype=torch.float32),
        qkv=torch.zeros(MAXS * 3 * D, **h),
        t1=torch.zeros(MAXS * D, **h),
        hf=torch.zeros(MAXS * FD, **h),
        xout=torch.zeros(MAXS * D, **h),
    )


class FusedExpertBlocks:
    """Packed weights + launch wrapper for one HiddenKVFlowExpert's block stack."""

    def __init__(self, blocks, device, dtype=torch.float16):
        mod = _load()
        L = len(blocks)
        b0 = blocks[0]
        D = b0.self_norm.normalized_shape[0]
        FM = b0.ffn[1].out_features // D
        lay = Layout(D, FM)
        self.L, self.D, self.FM, self.lay = L, D, FM, lay
        self.device = device
        w = torch.zeros(L, lay.wstride, device=device, dtype=dtype)
        p = torch.zeros(L, lay.pstride, device=device, dtype=dtype)
        for i, b in enumerate(blocks):
            def put(dst, off, t):
                t = t.detach().to(device=device, dtype=dtype).reshape(-1)
                dst[i, off : off + t.numel()] = t

            o, po = lay.off, lay.poff
            put(w, o["SN_W"], b.self_norm.weight)
            put(w, o["SN_B"], b.self_norm.bias)
            put(w, o["SIN_W"], b.self_attn.in_proj_weight)
            put(w, o["SIN_B"], b.self_attn.in_proj_bias)
            put(w, o["SOUT_W"], b.self_attn.out_proj.weight)
            put(w, o["SOUT_B"], b.self_attn.out_proj.bias)
            put(w, o["XN_W"], b.cross_norm.weight)
            put(w, o["XN_B"], b.cross_norm.bias)
            put(w, o["CQ_W"], b.cross_attn.in_proj_weight[:D])
            put(w, o["CQ_B"], b.cross_attn.in_proj_bias[:D])
            put(w, o["COUT_W"], b.cross_attn.out_proj.weight)
            put(w, o["COUT_B"], b.cross_attn.out_proj.bias)
            put(w, o["FN_W"], b.ffn[0].weight)
            put(w, o["FN_B"], b.ffn[0].bias)
            put(w, o["FU_W"], b.ffn[1].weight)
            put(w, o["FU_B"], b.ffn[1].bias)
            put(w, o["FD_W"], b.ffn[4].weight)
            put(w, o["FD_B"], b.ffn[4].bias)
            put(p, po["CN_W"], b.context_norm.weight)
            put(p, po["CN_B"], b.context_norm.bias)
            put(p, po["KV_W"], b.cross_attn.in_proj_weight[D:])
            put(p, po["KV_B"], b.cross_attn.in_proj_bias[D:])
        self.w, self.p = w, p
        self.mod = mod
        self._kv_cache = None
        self._scratch = _scratch(device, D, lay.FD)
        self.threads = int(os.environ.get("CF_CUDA_BLOCK_T", "512"))
        # launch-bound variant: 0 = (512,1) 128 regs, 1 = (512,2) 64 regs, 2 = (1024,1) 64 regs
        self.tb = int(os.environ.get("CF_CUDA_BLOCK_LB", "2" if self.threads > 512 else "0"))
        self._gcfg = int(os.environ.get("CF_CUDA_BLOCK_G", "0"))
        self._stages = int(os.environ.get("CF_CUDA_BLOCK_STAGES", "6"))
        self._grid = {}

    def can_run(self, S: int, C: int) -> bool:
        return supported(self.D, self.FM, NUM_HEADS, S, C, self.device)

    def cap(self, S, C) -> int:
        """Resident-block cap: the largest grid whose blocks are ALL guaranteed co-resident.
        The grid barrier spins, so a grid larger than this deadlocks."""
        return int(self.mod.max_grid(self.threads, self.D, self.FM, S, C, self.tb))

    def _cfg(self, S, C, g=None):
        cfg = self._grid.get((S, C, g))
        if cfg is None:
            cap = self.cap(S, C)
            if cap <= 0:
                raise RuntimeError(
                    f"fused block kernel has no instantiation that fits D={self.D} ffn=x{self.FM} "
                    f"S={S} C={C} threads={self.threads}; can_run() should have rejected it")
            # At most one threadblock per SM, and in fact FEWER than the 188 available: swept on
            # an RTX PRO 6000 (188 SM), 128 blocks won at D=640 and 96 at D=1024, both by ~10%
            # over a full 188-block grid.  Past that point the grid barrier (6 per block-pass)
            # costs more than the extra streaming warps return -- the wide stages already have
            # more columns than warps.  Two blocks per SM was measured and lost at both widths.
            sm = torch.cuda.get_device_properties(self.device).multi_processor_count
            default_g = min(sm, 128 if self.D <= 640 else 96)
            want = g if g else (self._gcfg if self._gcfg > 0 else default_g)
            G = min(want, cap)
            # One barrier counter per grid size: the counter is monotonic (never reset, so it is
            # CUDA-graph safe) and its bucket arithmetic is only valid while G stays fixed.
            cfg = (G, torch.zeros(1, device=self.device, dtype=torch.int64))
            self._grid[(S, C, g)] = cfg
        return cfg

    def cross_kv(self, context_hidden):
        """[1,C,D] -> [L,2,C,D] fp16.  Cached on tensor identity so the two Euler steps of a
        draft share one evaluation; recomputed during graph capture so the work is baked in."""
        D = self.D
        ctx = context_hidden.reshape(-1, D).to(torch.float16).contiguous()
        capturing = torch.cuda.is_current_stream_capturing()
        c = self._kv_cache
        if not capturing and c is not None and c[0] is context_hidden and c[1] == _ver(context_hidden):
            return c[2]
        C = ctx.shape[0]
        out = torch.empty(self.L, 2, C, D, device=self.device, dtype=torch.float16)
        self.mod.cross_kv(self.p, ctx, out, D, C, self.L, 40, 256)
        if not capturing:
            self._kv_cache = (context_hidden, _ver(context_hidden), out)
        return out

    def prepare(self, x, context_hidden, attn_mask=None, g=None):
        """Everything the launch needs, done on the CALLER's stream: cross K/V, the contiguous
        input copy, the mask cast, and the grid config.  Split out of ``forward`` so ``run_pair``
        can keep every allocation on the main stream and put only the kernel on the side one."""
        S = x.shape[1]
        kv = self.cross_kv(context_hidden)
        C = kv.shape[2]
        xin = x.reshape(-1, self.D).to(torch.float16).contiguous()
        m = attn_mask.to(torch.float16).contiguous() if attn_mask is not None else None
        return (kv, xin, m, S, C) + self._cfg(S, C, g)

    def launch(self, kv, xin, m, S, C, G, bar):
        """The kernel launch alone -- no allocation, no host sync, safe on any stream."""
        s = self._scratch
        xout = s["xout"][: S * self.D]
        self.mod.expert(self.w, kv, xin, xout, m, s["res"], s["qkv"], s["t1"], s["hf"], bar,
                        self.D, self.FM, S, C, self.L, G, self.threads, self._stages, self.tb)
        return xout

    def forward(self, x, context_hidden, attn_mask=None):
        """x [1,S,D] fp16 -> [1,S,D] fp16, running the whole L-block stack in one kernel."""
        kv, xin, m, S, C, G, bar = self.prepare(x, context_hidden, attn_mask)
        xout = self.launch(kv, xin, m, S, C, G, bar)
        # clone: xout is a reused static scratch buffer, so the caller must not alias it
        return xout.view(1, S, self.D).to(x.dtype).clone()


def attach(expert) -> "FusedExpertBlocks | None":
    """Build (and memoise on the module) the packed fused-block runner for a HiddenKVFlowExpert.

    Returns None -- i.e. "use the PyTorch stack" -- if the extension cannot be built.  Now that
    CF_CUDA_BLOCK defaults ON, a box without a working nvcc must degrade, not crash; `available()`
    has already printed the reason once.  Callers must handle None (they all did already, for the
    shape gate)."""
    fb = getattr(expert, "_cf_fused", None)
    if fb is None:
        ok, _ = available()
        if not ok:
            return None
        p = next(expert.parameters())
        fb = FusedExpertBlocks(list(expert.blocks), p.device, torch.float16)
        expert._cf_fused = fb
    return fb


# ================== CF_CUDA_PAIR: run the two chunk-experts CONCURRENTLY ==================
#
# THE DEPENDENCE THE DRAFT ACTUALLY HAS.  A draft is 32 serial block-passes only because we
# ISSUE it that way.  Of the three nested loops:
#   * the 2 Euler steps are a sequential ODE          -- genuinely serial
#   * the 8 blocks are transformer depth              -- genuinely serial
#   * the 2 CHUNKS are NOT.  ``flow_velocity`` feeds chunk 1 ``z_tau[:, :chunk_size]`` -- the
#     CURRENT Euler iterate, which is an INPUT to the step -- not chunk 0's velocity.  Chunk 1
#     attends to chunk 1's own copy of positions 0..3, so nothing chunk 0 computes reaches it.
# The two chunk-experts also hold DIFFERENT weights (``expert`` and ``extra_experts[0]``), so
# their weight streams are independent too.  The real critical path is 16 block-passes, not 32.
#
# Exploiting it needs no kernel change: launch the two persistent kernels on two streams.
# MEASURED (4B, D=640, cudagraphed): one Euler step 617 -> 370 us, 1.67x.
#
# GRID BUDGET.  Each kernel spins on a grid barrier, so ALL of ITS OWN blocks must be resident
# or it deadlocks.  Two kernels whose grids BOTH exceed the free SM count can deadlock each
# other (each holds half the machine waiting for blocks that cannot launch), so the pair is
# only issued when G0 + G1 <= the resident-block cap.  Measured at D=640 (cap 188): the split
# matters more than the total -- 64/124 and 60/128 both hit 370 us, an even 94/94 only 409,
# because chunk 0 is the S=4 expert and needs less.  128/128 (over the cap) fell back to
# serial execution at 609 us, which is the benign failure but also proof the budget is real.
def pair_enabled() -> bool:
    # DEFAULT ON behind CF_CUDA_BLOCK; `defaults.finalize_cuda_block` writes it down to "0"
    # whenever the block kernel itself is off or there are not exactly 2 chunk experts.
    return os.environ.get("CF_CUDA_PAIR", "1") not in ("0", "", "false", "False")


_PAIR_ANNOUNCED = False


def _announce(fbs, prepped):
    """Print the config the pair path ACTUALLY launched with, once.  A flag that silently fails
    a shape gate is how CF_CUDA_BLOCK once looked enabled while doing nothing."""
    global _PAIR_ANNOUNCED
    _PAIR_ANNOUNCED = True
    d = " ".join(f"chunk{i}: S={p[3]} G={p[5]}/{fb.cap(p[3], p[4])}"
                 for i, (fb, p) in enumerate(zip(fbs, prepped)))
    print(f"[cf] CF_CUDA_PAIR engaged: D={fbs[0].D} C={prepped[0][4]} threads={fbs[0].threads} "
          f"lb={fbs[0].tb}  {d}", flush=True)


_SIDE_STREAM: dict = {}


def _side_stream(device):
    """One long-lived side stream per device.  Created lazily on the first (warm-up) call --
    a stream cannot be created during CUDA-graph capture."""
    key = str(device)
    s = _SIDE_STREAM.get(key)
    if s is None:
        s = torch.cuda.Stream(device=device)
        _SIDE_STREAM[key] = s
    return s


def pair_grids(fbs, Ss, C):
    """(G0, G1) for a concurrent pair, or None if the pair does not fit.

    Both grids occupy the machine at once, so the budget is the FRACTIONAL one:
    G0/cap0 + G1/cap1 <= 1, where cap_i is that shape's own resident-block cap (the two shapes
    can differ -- a narrower S fits more blocks per SM).  ``CF_CUDA_PAIR_G="g0,g1"`` overrides.

    The split is NOT 1:2 even though the row counts are 4 and 8: both experts stream exactly the
    same weight bytes and only the row count differs, so measured alone at G=128 they are 224 and
    311 us, i.e. 0.42:0.58.  Swept on integrate() (RTX PRO 6000, 188 SM, cap 188):
        D= 640: (62,126) 0.808 ms  < (56,132) 0.848 < (75,113) 0.901 < (94,94) 0.961
        D=1024: (75,113) 1.377 ms  < (56,132) 1.467 < (62,126) 1.479 < (94,94) 1.638
    The wider model wants MORE blocks on the small chunk for the same reason its single-expert
    default grid is smaller (96 vs 128): past that point chunk 1's own barrier costs more than
    its extra streaming warps return.

    Memoised: ``cap`` is an extension call, and this runs inside the compiled flow_velocity."""
    key = (id(fbs[0]), id(fbs[1]), tuple(Ss), C)
    if key in _PAIR_GRIDS:
        return _PAIR_GRIDS[key]
    g = _pair_grids(fbs, Ss, C)
    _PAIR_GRIDS[key] = g
    return g


_PAIR_GRIDS: dict = {}


def _pair_grids(fbs, Ss, C):
    caps = [fb.cap(S, C) for fb, S in zip(fbs, Ss)]
    if min(caps) <= 0:
        return None
    env = os.environ.get("CF_CUDA_PAIR_G", "")
    if env:
        g = tuple(int(v) for v in env.split(","))
        if len(g) != 2 or g[0] / caps[0] + g[1] / caps[1] > 1.0 + 1e-9:
            return None
        return g
    f0 = 0.33 if fbs[0].D <= 640 else 0.40   # measured; see pair_grids
    g0 = max(16, int(caps[0] * f0))
    g1 = max(16, int(caps[1] * (1.0 - g0 / caps[0])))
    return (g0, g1) if g0 / caps[0] + g1 / caps[1] <= 1.0 + 1e-9 else None


def run_pair(fbs, xs, context_hidden, masks, grids):
    """Run two independent expert block-stacks concurrently; returns their [1,S,D] outputs.

    Every allocation (cross K/V, the input copies, the output clones) happens on the CALLER's
    stream; the side stream carries only fbs[1]'s kernel, so nothing is freed on a stream other
    than the one that allocated it and no ``record_stream`` bookkeeping is needed."""
    main = torch.cuda.current_stream()
    side = _side_stream(fbs[0].device)
    prepped = [fb.prepare(x, context_hidden, m, g)
               for fb, x, m, g in zip(fbs, xs, masks, grids)]
    if not _PAIR_ANNOUNCED:
        _announce(fbs, prepped)
    side.wait_stream(main)
    with torch.cuda.stream(side):
        o1 = fbs[1].launch(*prepped[1])
    o0 = fbs[0].launch(*prepped[0])
    main.wait_stream(side)
    return [o.view(1, p[3], fb.D).to(x.dtype).clone()
            for fb, o, p, x in zip(fbs, (o0, o1), prepped, xs)]
