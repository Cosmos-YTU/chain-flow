// Fused CUDA kernel for the chain-flow drafter's HiddenKVFlowBlock stack.
//
// One persistent, grid-synchronising kernel replaces the ~24 tiny kernels PyTorch emits per
// transformer block, and loops over all L blocks of an expert internally, so an entire
// HiddenKVFlowExpert block stack becomes ONE launch.
//
// A BATCH of drafts is B independent copies of that stack, carried on gridDim.y: only the
// pointers move, so shared memory and every template parameter are unchanged (`S` is a draft's
// ROW COUNT, never the batch).  The grid barrier is per-slice; the caller must therefore keep
// G*B within the resident-block cap.  Templated over the expert width D so all three shipped
// drafters are covered:
//     4B  : D= 640, heads 8 (head_dim  80), ffn x6 (3840)
//     9B  : D=1024, heads 8 (head_dim 128), ffn x6 (6144)
//     27B : D=1024, heads 8 (head_dim 128), ffn x6 (6144)
// with S = 4 or 8 query rows and C <= 16 context rows.  Requirements: D % 128 == 0, heads == 8,
// head_dim even, S <= 8, and (nwarps | D) so the ffn-down k-split stays threadblock-uniform.
//
// Structural wins over the PyTorch path:
//   * cross-attention K/V are hoisted out entirely (context_hidden is constant across all 32
//     block-passes of a draft) and precomputed once by cf_cross_kv; that also collapses the
//     duplicated context_norm().
//   * the residual stream lives in a small fp32 scratch that stays L2-resident; no intermediate
//     ever round-trips HBM. Weights are the only thing streamed.
//   * LayerNorms and residual adds are folded into GEMM prologues/epilogues.
//   * attention over S<=8 with 8 heads runs in registers/smem: no SDPA, no transposes.
//
// Everything is bandwidth-bound on the weights (14.1 MiB per block-pass at D=640, 36.0 MiB at
// D=1024), so the GEMMs use plain FMA (M=8 underfills any tensor-core tile) and prioritise
// streaming: one warp per output column, a chain of 64-bit vector loads so several are in flight
// per warp, and fp32 accumulation.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#define NH 8  // all shipped drafters use 8 heads

#ifndef CF_NCOL
#define CF_NCOL 2  // output columns per warp in the wide projections; see cf_fused_expert
#endif

// ---- packed per-pass weight layout (offsets in halves), derived from D and the ffn multiplier.
// Must match the identical arithmetic in cuda_block.py.
template <int D, int FM>
struct WL {
  static constexpr long FD = (long)D * FM;
  static constexpr long SN_W = 0;                        // self_norm.weight            [D]
  static constexpr long SN_B = SN_W + D;                 // self_norm.bias              [D]
  static constexpr long SIN_W = SN_B + D;                // self_attn.in_proj_weight    [3D,D]
  static constexpr long SIN_B = SIN_W + 3L * D * D;      // self_attn.in_proj_bias      [3D]
  static constexpr long SOUT_W = SIN_B + 3L * D;         // self_attn.out_proj.weight   [D,D]
  static constexpr long SOUT_B = SOUT_W + (long)D * D;
  static constexpr long XN_W = SOUT_B + D;               // cross_norm (applied to x)
  static constexpr long XN_B = XN_W + D;
  static constexpr long CQ_W = XN_B + D;                 // cross_attn.in_proj_weight[0:D]
  static constexpr long CQ_B = CQ_W + (long)D * D;
  static constexpr long COUT_W = CQ_B + D;               // cross_attn.out_proj.weight  [D,D]
  static constexpr long COUT_B = COUT_W + (long)D * D;
  static constexpr long FN_W = COUT_B + D;               // ffn[0] LayerNorm
  static constexpr long FN_B = FN_W + D;
  static constexpr long FU_W = FN_B + D;                 // ffn[1].weight               [FD,D]
  static constexpr long FU_B = FU_W + FD * D;
  static constexpr long FD_W = FU_B + FD;                // ffn[4].weight               [D,FD]
  static constexpr long FD_B = FD_W + (long)D * FD;
  static constexpr long STRIDE = FD_B + D;
};

// ---- precompute (cross K/V) packed layout --------------------------------------------------
template <int D>
struct PL {
  static constexpr long CN_W = 0;                    // context_norm.weight
  static constexpr long CN_B = D;
  static constexpr long KV_W = 2L * D;               // cross_attn.in_proj_weight[D:3D] -> [2D,D]
  static constexpr long KV_B = KV_W + 2L * D * D;    // cross_attn.in_proj_bias[D:3D]   -> [2D]
  static constexpr long STRIDE = KV_B + 2L * D;
};

// ------------------------------------------------------------------------------------------
// Reset-free grid barrier. The counter increases monotonically forever (64-bit, never wraps in
// practice), so no host-side reset / memset node is needed: the kernel is safe to replay inside
// a CUDA graph with no host sync and no dynamic allocation.
//
// The arrival uses atom.add.release.gpu and the spin ld.acquire.gpu rather than a pair of
// membar.gl (__threadfence()) + volatile loads: release/acquire on the counter is cumulative in
// the PTX memory model, so it publishes the whole CTA's prior global writes (ordered by the
// __syncthreads() below) at a fraction of the cost of two full-system fences.  With 6 barriers
// per block-pass and 32 block-passes per draft this is on the critical path 192 times.
__device__ __forceinline__ void grid_bar(unsigned long long* ctr, unsigned int G) {
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned long long v;
    asm volatile("atom.add.release.gpu.u64 %0, [%1], %2;"
                 : "=l"(v)
                 : "l"(ctr), "l"((unsigned long long)1)
                 : "memory");
    const unsigned long long target = (v / G + 1ULL) * G;
    unsigned long long cur;
    do {
      asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(cur) : "l"(ctr) : "memory");
    } while (cur < target);
  }
  __syncthreads();
}

// CHEAPER BARRIERS WERE TRIED AND REFUTED -- do not reopen without new evidence.
//
// The barrier is real and it is 21% of the block-pass, MEASURED IN SITU rather than with the
// contention-maximising bare skeleton: CF_CUDA_BLOCK_STAGES=106 keeps every stage's work and drops
// 5 of the 6 barriers, so (full - onebar)/5 is the true per-barrier cost.  At D=640, S=8 that is
// 0.88 us at G=64, 1.17 at G=128 and 1.47 at G=188 -- linear in the grid, and the bare skeleton
// (1.21 us/barrier at G=128) turns out to have been ACCURATE, not an overstatement.
//
// Linear-in-G looks like the G same-address atom.add's serialising in one L2 slice, so two
// standard alternatives were built and measured (us per barrier, D=640 S=8):
//                                        G=64   G=128   G=188
//   shared atom.add counter (this one)   0.881   1.165   1.469
//   per-block counters, all blocks poll  1.188   1.980   3.217
//   fan-in to block 0, fan-out on 1 word 1.793   2.797   4.226
// Both lose, and lose HARDER as G grows.  Arrival was never the bottleneck: the atomic RMWs are
// pipelined against each other, while any scheme where a block observes O(G) state multiplies the
// poll traffic by G, and a two-level scheme just adds a second dependent round trip to the
// critical path.  One uncontended round trip on one address is the floor here; cutting barrier
// cost needs FEWER barriers (merging stages), not a cheaper one.
__device__ __forceinline__ float warp_all_reduce(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// acc[s] += sum_{k in [0,NCH*128)} A[s][k] * W[k]   for one output column.
// `sA` points at A[0][0] of the slice, rows strided by ldA.
//
// `#pragma unroll 2` is load-bearing: letting ptxas fully unroll the NCH x S body at the
// register cap set by __launch_bounds__ spilled the accumulator to local memory (6.7M local
// requests) and cost ~40% of the kernel.
template <int S>
struct Acc {
  float v[S];
};

// Select acc.v[lane] without a dynamic index -- indexing a local array by lane would push the
// whole accumulator into local memory and destroy the inner loop.
template <int S>
__device__ __forceinline__ float pick(const Acc<S>& a, int lane) {
  float r = 0.f;
#pragma unroll
  for (int s = 0; s < S; ++s)
    if (lane == s) r = a.v[s];
  return r;
}

// NCOL output columns at once.  The A-fragment loads (LDS) and their half->float conversions are
// shared across the columns, so the instruction count per streamed weight byte drops ~NCOL-fold
// on everything except the FFMAs.  The kernel is latency/issue-bound (Nsight: DRAM 28%, compute
// 22%, IPC 0.30 on the active SMs), so instructions per byte is what actually sets the rate.
template <int S, int NCH, int NCOL>
__device__ __forceinline__ void warp_dot_n(const __half* __restrict__ W, long ws,
                                           const __half* __restrict__ sA, int ldA,
                                           Acc<S>* __restrict__ acc) {
  const int base = (threadIdx.x & 31) * 4;
#pragma unroll 2
  for (int c = 0; c < NCH; ++c) {
    float2 w0[NCOL], w1[NCOL];
#pragma unroll
    for (int j = 0; j < NCOL; ++j) {
      const uint2 wv = *(const uint2*)(W + j * ws + c * 128 + base);
      w0[j] = __half22float2(*(const __half2*)&wv.x);
      w1[j] = __half22float2(*(const __half2*)&wv.y);
    }
#pragma unroll
    for (int s = 0; s < S; ++s) {
      const uint2 av = *(const uint2*)(sA + s * ldA + c * 128 + base);
      const float2 a0 = __half22float2(*(const __half2*)&av.x);
      const float2 a1 = __half22float2(*(const __half2*)&av.y);
#pragma unroll
      for (int j = 0; j < NCOL; ++j)
        acc[j].v[s] += a0.x * w0[j].x + a0.y * w0[j].y + a1.x * w1[j].x + a1.y * w1[j].y;
    }
  }
}

template <int S, int NCH>
__device__ __forceinline__ void warp_dot(const __half* __restrict__ W, const __half* __restrict__ sA,
                                         int ldA, Acc<S>& acc) {
  warp_dot_n<S, NCH, 1>(W, 0, sA, ldA, &acc);
}

// THE BATCH FALLOFF IS MATH THROUGHPUT, NOT MEMORY.  Four independent memory-side fixes were
// built and measured, and all four are dead ends.  Read this before attempting a fifth.
//
// The kernel's projection rate is FLAT in batch while cuBLAS's grows, and that flat line IS the
// falloff.  Same FLOPs, same shapes, same batch (D=640, S=8, projections only via STAGES=26):
//     rows   this kernel   cuBLAS (tensor cores)   headroom
//       64      12.0 TF/s          22.9 TF/s          1.9x
//      256      14.9 TF/s          74.9 TF/s          5.0x
//      512      15.9 TF/s         127.6 TF/s          8.0x
// Scalar fp32 FFMA cannot follow tensor cores, so every batch the ratio gets worse.  It lines up
// with the measured crossover on the real 4B drafter (integrate(), compiled+cudagraphed, ms):
//     B          1      4      8     32     64
//     PyTorch  2.02   2.19   2.50   3.77   4.41     <- grows 2.2x over a 64x batch
//     fused    1.26   1.63   2.57   7.85  18.28     <- grows 14.5x
// PyTorch is launch-bound at B=1 (which is why fusion wins 2.5x there) and absorbs batch almost
// for free after that.  ``batch_limit`` already hands off at B=8, which is exactly the crossover.
//
// REFUTED 1 -- packing the batch into the ROW dimension so one weight walk serves BS slices.
// Rows are NOT nearly free: at G=128 the FULL kernel costs 147.5 / 197.3 / 342.3 us at S=4/8/16,
// so the marginal cost of a row RISES (12.5 -> 18.1 us/row).  Fitting T = W + r*S + q*S^2 gives
// W=112.8 us, r=6.77, q=0.473 -- the q term is self-attention, and it is what makes packing get
// worse the more you pack.  smem is a second wall: 88.8 KB at (D=640,S=16) fits the ~99 KB optin
// cap, 176.5 KB at S=32 does not.
//
// REFUTED 2 -- packing the PROJECTIONS only, keeping attention per-slice.  This one survives the
// row-scaling test (STAGES=26, G=128: 114.7 / 139.7 / 217.9 us at S=4/8/16, so T(2S) < 2*T(S))
// and it halves the weight traffic, so it looked right.  It was then priced WITHOUT building it,
// by holding total rows fixed and varying the split -- (S=16,B=N/2) has exactly a packed BS=2
// weight-per-row ratio, and its timing is faithful even though its cross-attention is not:
//     total rows    64      128      256      512
//     S=8,  B=N/8  522.4    859.3   1522.9   2878.3 us
//     S=16, B=N/16 597.2    939.1   1494.6   2725.3 us   <- HALF the L2 weight traffic
//     speedup      0.87x    0.92x    1.02x    1.06x
// Halving the weight traffic buys ~nothing, which is the cleanest possible proof that this kernel
// is not bandwidth-bound.  Do not build the packed kernel on the strength of the row-scaling test
// alone; that test holds bandwidth constant and so cannot see this.
//
// REFUTED 3 -- a LOCKSTEP grid barrier (one counter for all G*B blocks instead of one per slice)
// so the batch marches the weight stream together and slices 2..B hit L2 instead of DRAM.  It was
// built, shown faithful, and measured at 0.99-1.02x everywhere.  ncu says why -- the hardware
// already does this reuse with no help, L2 MISSES ARE FLAT IN BATCH while hits scale:
//     B=1   miss 2,775,279 sectors (~89 MB)   hit   3.9M
//     B=8   miss 2,816,114        (~90 MB)    hit  28.4M
//     B=32  miss 2,955,600        (~95 MB)    hit 109.2M
// The weights come off DRAM ONCE at every batch.  Wall-clock alone suggests the opposite (16 x
// 88.5 MB / 833 us "=" 1670 GB/s, ~93% of peak) -- that traffic is L2, not DRAM.  Do not infer
// DRAM bandwidth from wall-clock here; measure dram__bytes / lts__t_sectors.
//
// REFUTED 4 -- see the two blocks below (k-splitting the narrow stages; a wider NCOL).
//
// THE DESIGN THAT WOULD ACTUALLY WORK, and the only one left: mma.sync on the projections.  The
// two refuted row-packing schemes are not wasted -- MMA needs M >= 16 and one draft has only
// S=8 rows, so a packed BS=2 supplies the M that makes m16n8k16 efficient.  Packing alone buys
// nothing (refuted 2) and tensor cores alone have no M to work with; they only pay off together.
// Note the prize is NOT tensor cores per se -- the PyTorch fallback already has them, and already
// scales.  It is fusion AND tensor cores at once, i.e. moving the B=8 crossover up rather than
// making high batch possible at all.
//
// NUMERICS NOTE for anyone A/B-ing a change here: stages 2 and 4 reduce into `res` with a float
// atomicAdd, so the kernel does NOT reproduce itself bit-for-bit -- measured run-to-run delta is
// 9.77e-04, one fp16 ulp.  Bit-equality is the wrong oracle; the right one is that a change stays
// inside that noise floor.
//
// K-SPLITTING THE NARROW STAGES WAS TRIED AND REFUTED -- do not reopen without new evidence.
//
// The four D-column stages run at 15-28% of their weight roofline while the FFN runs at 69-92%,
// and the obvious cause is tile starvation: the loop hands ONE output column to ONE warp, so at
// D=640/threads=512 stages 2/3/4 have only D/16 = 40 tiles and 88 of a 128-block grid park at the
// barrier.  Splitting the D-deep reduction KS ways (stages 2 and 4 finish into `res` with an
// atomicAdd, so partial sums are free there) multiplies the tile count by KS.  MEASURED, us per
// block-pass, S=8, with the attention and its staging stripped (CF_CUDA_BLOCK_STAGES=26) so the
// projections are isolated:
//     D= 640 (tiles 40 -> 200):  ksplit 1: 27.38   ksplit 5: 27.10
//     D=1024 (tiles 64 -> 512):  ksplit 1: 55.07   ksplit 2: 56.55   4: 56.39   8: 60.86
// and worse at EVERY smaller grid (D=640, G=64: 31.59 -> 35.13).  So the narrow stages are not
// tile-starved: their cost is the PER-OUTPUT-COLUMN EPILOGUE -- warp_all_reduce is 5 butterfly
// shuffles x S accumulators (40 shuffle ops at S=8) plus S atomics, per column -- and a k-split
// multiplies exactly that fixed cost by KS while dividing only the streaming, which is not the
// bottleneck.  It cancels.  Reaching FFN efficiency here needs a cheaper reduction (fewer lanes
// per column, more columns per warp), not more blocks.

template <int S>
__device__ __forceinline__ void warp_reduce_acc(Acc<S>& acc) {
#pragma unroll
  for (int s = 0; s < S; ++s) acc.v[s] = warp_all_reduce(acc.v[s]);
}

// LayerNorm(fp32 residual stream) -> sA (half), one warp per row, single pass over global memory
// (the raw values are parked in smem and re-read from there to normalise).
template <int S, int D>
__device__ __forceinline__ void ln_prologue(const float* __restrict__ res,
                                            const __half* __restrict__ g,
                                            const __half* __restrict__ b, __half* sA, int nwarps) {
  constexpr int NV = D / 128;  // D floats / 32 lanes / 4 per float4
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  for (int s = w; s < S; s += nwarps) {
    const float4* r = (const float4*)(res + s * D);
    __half* o = sA + s * D;
    float sum = 0.f, sq = 0.f;
    // Pass 1: park the row in smem as fp16 (exactly the dtype LayerNorm sees on the PyTorch
    // path) and accumulate the moments. Keeping the raw floats in registers instead spills.
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const float4 t = r[j * 32 + lane];
      __half h4[4] = {__float2half(t.x), __float2half(t.y), __float2half(t.z), __float2half(t.w)};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const float f = __half2float(h4[q]);
        sum += f;
        sq += f * f;
      }
      *(uint2*)(o + (j * 32 + lane) * 4) = *(const uint2*)h4;
    }
    sum = warp_all_reduce(sum);
    sq = warp_all_reduce(sq);
    const float mean = sum / D;
    const float rstd = rsqrtf(fmaxf(sq / D - mean * mean, 0.f) + 1e-5f);
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      const int i = (j * 32 + lane) * 4;
      const uint2 xx = *(const uint2*)(o + i);
      const uint2 gg = *(const uint2*)(g + i);
      const uint2 bb = *(const uint2*)(b + i);
      __half out[4];
#pragma unroll
      for (int q = 0; q < 4; ++q)
        out[q] = __float2half((__half2float(((const __half*)&xx)[q]) - mean) * rstd *
                                  __half2float(((const __half*)&gg)[q]) +
                              __half2float(((const __half*)&bb)[q]));
      *(uint2*)(o + i) = *(const uint2*)out;
    }
  }
  __syncthreads();
}

#define PLD 17  // score-table row stride: odd (conflict-free column reads) and caps L at 16

// Attention for S<=8 queries, L<PLD keys, NH heads of width D/NH.
//
// Split into two block-parallel phases with an intervening __syncthreads() rather than the
// obvious "one warp per head, loop the S rows".  That shape left half the warps idle (NH=8 <
// nwarps=16) and serialised S row-softmaxes plus an S*L chain of __shfl broadcasts behind one
// warp; the two attention stages were 43% of the whole kernel.
//   phase 1: one warp per (head, row)  -> NH*S = 64 independent softmaxes
//   phase 2: one warp per 32 output dims -> D/32 tasks, no shuffles at all, S accumulators per
//            lane and the probabilities broadcast out of shared memory
// sP is a [NH][S][PLD] fp32 scratch for the probabilities (~4 KB).
template <int S, int D>
__device__ __forceinline__ void small_attn(const __half* __restrict__ q, int ldq,
                                           const __half* __restrict__ k, int ldk,
                                           const __half* __restrict__ v, int ldv, int L,
                                           const __half* __restrict__ mask, __half* out, int ldo,
                                           int nwarps, float* __restrict__ sP) {
  constexpr int HD = D / NH;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float scale = rsqrtf((float)HD);
  // Lanes are split into `lpk` groups of `Lp`, so the HD-element q.k dot for all L keys is
  // computed cooperatively by the whole warp (~HD/lpk FMAs) instead of one key per lane running
  // all HD serially.
  int Lp = 4, lsh = 2;  // >=4 keeps HD/lpk even so the dot can use __half2
  while (Lp < L) {
    Lp <<= 1;
    ++lsh;
  }
  const int lpk = 32 >> lsh;    // lanes per key
  const int dchunk = HD / lpk;  // contiguous head dims per lane group
  const int t = lane & (Lp - 1);
  const int d0 = (lane >> lsh) * dchunk;

  for (int task = w; task < NH * S; task += nwarps) {
    const int h = task / S, s = task - h * S;  // S is 4 or 8, so this is a shift
    const int ho = h * HD;
    float part = 0.f;
    if (t < L) {
      const __half2* qq = (const __half2*)(q + s * ldq + ho + d0);
      const __half2* kk = (const __half2*)(k + t * ldk + ho + d0);
#pragma unroll 4
      for (int i = 0; i < dchunk / 2; ++i) {
        const float2 a = __half22float2(qq[i]), b = __half22float2(kk[i]);
        part += a.x * b.x + a.y * b.y;
      }
    }
    for (int o = Lp; o < 32; o <<= 1) part += __shfl_xor_sync(0xffffffffu, part, o);
    float sc = -INFINITY;
    if (t < L) {
      sc = part * scale;
      if (mask) sc += __half2float(mask[s * L + t]);
    }
    // every lane group now holds the same L scores, so reducing over the low lsh bits is enough
    float m = sc;
    for (int o = 1; o < Lp; o <<= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    const float e = (t < L && sc > -INFINITY) ? __expf(sc - m) : 0.f;
    float den = e;
    for (int o = 1; o < Lp; o <<= 1) den += __shfl_xor_sync(0xffffffffu, den, o);
    if (lane < L) sP[(h * S + s) * PLD + lane] = e / den;  // lane<Lp => d0==0 and t==lane
  }
  __syncthreads();

  for (int base = w * 32; base < D; base += nwarps * 32) {
    const int d = base + lane;
    const int h = d / HD;  // HD is a compile-time constant -> magic-number division
    const float* pp = sP + h * S * PLD;
    float acc[S];
#pragma unroll
    for (int s = 0; s < S; ++s) acc[s] = 0.f;
    for (int j = 0; j < L; ++j) {
      const float vv = __half2float(v[j * ldv + d]);
#pragma unroll
      for (int s = 0; s < S; ++s) acc[s] += pp[s * PLD + j] * vv;
    }
#pragma unroll
    for (int s = 0; s < S; ++s) out[s * ldo + d] = __float2half(acc[s]);
  }
  __syncthreads();
}

// ------------------------------------------------------------------------------------------
// BATCH lives on gridDim.y (see the slicing block at the top of cf_fused_expert).  Every buffer
// below except `w` and `mask` is per-slice and carries a leading batch dimension; `bar` carries
// one counter PER SLICE, because each slice's blocks synchronise only among themselves.
struct Args {
  const __half* __restrict__ w;   // packed per-pass weights, L blocks x WL::STRIDE halves
  const __half* __restrict__ kv;  // precomputed cross K/V: [B][L][2][C][D]
  const __half* __restrict__ xin;   // [B][S][D]
  __half* __restrict__ xout;        // [B][S][D]
  const __half* __restrict__ mask;  // [S][S] additive fp16, or null -- SHARED across the batch
  float* __restrict__ res;          // [B][S][D] fp32 residual stream
  __half* __restrict__ qkv;         // [B][S][3*D]
  __half* __restrict__ t1;          // [B][S][D]
  __half* __restrict__ hf;          // [B][S][FD]
  unsigned long long* bar;          // [B]
  int C, L, G;
  // Profiling hook, `100*onebar + 10*skip + n`: run only the first n of the 6 stages (6 = a full
  // block); skip attention (skip=1) or attention plus its smem staging (skip=2); n=0 runs the bare
  // barrier skeleton.  `onebar=1` keeps ALL the work but drops 5 of the 6 grid barriers, which
  // measures the barrier IN SITU -- the bare skeleton has every block arriving at once and so
  // maximises atomic contention on the counter.  onebar produces WRONG numbers (the stages race);
  // it is a stopwatch, not a mode.  Default 6.
  int stages;
};

// smem: sA [S][D] then a staging area.  Every warp k-slice is exactly D wide (NCH = D/128) --
// see warp_dot; the ffn-down k-split is therefore FM-way.
// Launch-bound variants.  The kernel is latency-bound (Nsight: 74% of scheduler cycles have no
// eligible warp), so warps-per-SM is the dominant knob, and shared memory pins us to one block
// per SM at D=1024 -- the only way to more warps there is a wider threadblock.
//   0 = (512, 1)  128 registers, 16 warps/block
//   1 = (512, 2)   64 registers, lets two blocks share an SM where smem allows
//   2 = (1024, 1)  64 registers, 32 warps/block
//   3 = (256, 1)  255 registers, 8 warps/block -- the only variant with register headroom for
//                 a wider NCOL (128 regs at 512 threads is already the whole 64K file).
// SPILLS (ptxas -v, sm_120): variants 1 and 2 are CLEAN at D=640 (4 B) but spill ~590 B/thread
// at D=1024 -- do not use them at the wide width without re-measuring.
template <int TB>
struct LB {
  static constexpr int T = (TB == 2) ? 1024 : (TB == 3) ? 256 : 512;
  static constexpr int B = (TB == 1) ? 2 : 1;
};

template <int D, int FM, int S, int TB>
__global__ __launch_bounds__(LB<TB>::T, LB<TB>::B) void cf_fused_expert(Args a) {
  constexpr int NCH = D / 128;
  constexpr int FD = D * FM;
  // Column tiling for the three wide projections (qkv, ffn-up, ffn-down).  The narrow D-column
  // stages keep NCOL=1: they already have fewer columns than resident warps, so tiling them
  // would only shrink the parallelism.
  //
  // NCOL is what amortises the SHARED-MEMORY side of the inner loop: per 128-wide k-chunk a warp
  // does S smem loads and S*NCOL*4 FFMAs, so smem bytes per FMA go as 1/NCOL.  It is compile-time
  // tunable (``CF_CUDA_BLOCK_NCOL`` -> -DCF_NCOL) because raising it costs S more accumulator
  // registers per column and variant 0 is already pinned at the 128-register cap.
  //
  // NCOL=4 WAS MEASURED AT BATCH AND REFUTED (D=640, us/block-pass per slice, best config per B):
  //     B=      1      4      8     16     32     64
  //     NCOL=2  38.25  12.90  9.48  7.49   6.92   6.90
  //     NCOL=4  41.49  14.06  9.32  7.62   6.58   6.52
  // i.e. -8% at the batches that matter and +5% at batches where the kernel loses to cutlass
  // anyway.  ptxas keeps variant 0 spill-free at NCOL 4 and 6, so this is not a spill effect:
  // smem bandwidth simply is not the binding constraint.  NCOL must also divide into D, 3D and
  // FM*D after multiplying by nwarps, which rules out 6 at D=640.
  constexpr int NCOL = CF_NCOL;
  constexpr int LDQ = 3 * D + 8;  // padded smem row stride for staged qkv (bank-conflict free)
  constexpr int LDS = D + 8;      // padded smem row stride for staged q / k / v
  extern __shared__ __half smem[];
  float* sP = (float*)smem;                      // [NH][S][PLD] attention probabilities
  __half* sA = smem + NH * S * PLD * 2;          // [S][D] normalized activation / attn output
  __half* sB = sA + S * D;                       // padded staging for qkv / cross q,k,v / h slice

  const int nwarps = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int C = a.C, G = a.G;
  const unsigned int Gu = (unsigned int)G;
  const int stages = a.stages % 10;        // 1..6 = run only the first N stages
  const int skip = (a.stages / 10) % 10;   // profiling: 1 = no attention, 2 = no attention/staging
  const bool onebar = a.stages >= 100;     // profiling: drop 5 of the 6 barriers (see Args)

  // ---- BATCH SLICING ------------------------------------------------------------------
  // The kernel is written for ONE draft; a batch is B independent copies of it, so slice `a`
  // and let gridDim.y carry the batch.  Nothing here depends on B: shared memory is a function
  // of (D, S, C) only, and `S` is a draft's ROW COUNT, not the batch -- so no new template
  // instantiation.  The two shared buffers (`w`, `mask`) are deliberately NOT offset: every
  // slice streams the same weights, which is the whole point (128 MB of L2 turns B-1 of the B
  // weight reads into L2 hits).
  //
  // THE BARRIER IS PER SLICE.  A slice's blocks synchronise only among themselves, so each
  // gets its own counter and `Gu` stays the PER-SLICE grid.  A shared counter would need all
  // G*B blocks to arrive and would serialise the slices for no reason.  The caller must keep
  // G*B <= the resident-block cap or every slice deadlocks -- see FusedExpertBlocks._cfg.
  if (blockIdx.y) {
    const long b = blockIdx.y;
    a.xin += b * S * D;
    a.xout += b * S * D;
    a.res += b * S * D;
    a.qkv += b * S * 3 * D;
    a.t1 += b * S * D;
    a.hf += b * S * FD;
    a.kv += b * a.L * 2 * (long)C * D;
    a.bar += b;
  }
#define GBAR() grid_bar(a.bar, Gu)

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S * D; i += G * blockDim.x)
    a.res[i] = __half2float(a.xin[i]);
  GBAR();

  for (int l = 0; l < a.L; ++l) {
    const __half* W = a.w + (long)l * WL<D, FM>::STRIDE;

    if (stages == 0) {  // profiling hook: barrier skeleton only, no arithmetic
#pragma unroll
      for (int i = 0; i < 6; ++i) GBAR();
      continue;
    }

    // ---- 1. self_norm + fused qkv projection -------------------------------------------
    ln_prologue<S, D>(a.res, W + WL<D, FM>::SN_W, W + WL<D, FM>::SN_B, sA, nwarps);
    for (int tb = blockIdx.x; tb * nwarps * NCOL < 3 * D; tb += G) {
      const int t = (tb * nwarps + warp) * NCOL;
      Acc<S> acc[NCOL] = {};
      warp_dot_n<S, NCH, NCOL>(W + WL<D, FM>::SIN_W + (long)t * D, D, sA, D, acc);
#pragma unroll
      for (int j = 0; j < NCOL; ++j) {
        warp_reduce_acc<S>(acc[j]);
        if (lane < S)
          a.qkv[lane * (3 * D) + t + j] =
              __float2half(pick<S>(acc[j], lane) + __half2float(W[WL<D, FM>::SIN_B + t + j]));
      }
    }
    if (!onebar) GBAR();
    if (stages < 2) continue;

    // ---- 2. self attention + out_proj + residual ---------------------------------------
    if (skip < 2) {  // vectorised, row-padded stage of qkv into smem (padding kills the 8-way
       // bank conflict the key-major dot product would otherwise hit)
      uint4* d = (uint4*)sB;
      const uint4* s4 = (const uint4*)a.qkv;
      for (int i = threadIdx.x; i < S * (3 * D / 8); i += blockDim.x) {
        const int r = i / (3 * D / 8), c = i - r * (3 * D / 8);
        d[r * (LDQ / 8) + c] = s4[i];
      }
    }
    __syncthreads();
    if (skip < 1)
      small_attn<S, D>(sB, LDQ, sB + D, LDQ, sB + 2 * D, LDQ, S, a.mask, sA, D, nwarps, sP);
    for (int tb = blockIdx.x; tb * nwarps < D; tb += G) {
      const int t = tb * nwarps + warp;
      Acc<S> acc = {};
      warp_dot<S, NCH>(W + WL<D, FM>::SOUT_W + (long)t * D, sA, D, acc);
      warp_reduce_acc<S>(acc);
      if (lane < S)
        atomicAdd(&a.res[lane * D + t],
                  pick<S>(acc, lane) + __half2float(W[WL<D, FM>::SOUT_B + t]));
    }
    if (!onebar) GBAR();
    if (stages < 3) continue;

    // ---- 3. cross_norm + cross query projection ----------------------------------------
    ln_prologue<S, D>(a.res, W + WL<D, FM>::XN_W, W + WL<D, FM>::XN_B, sA, nwarps);
    for (int tb = blockIdx.x; tb * nwarps < D; tb += G) {
      const int t = tb * nwarps + warp;
      Acc<S> acc = {};
      warp_dot<S, NCH>(W + WL<D, FM>::CQ_W + (long)t * D, sA, D, acc);
      warp_reduce_acc<S>(acc);
      if (lane < S)
        a.t1[lane * D + t] = __float2half(pick<S>(acc, lane) + __half2float(W[WL<D, FM>::CQ_B + t]));
    }
    if (!onebar) GBAR();
    if (stages < 4) continue;

    // ---- 4. cross attention (K/V hoisted) + out_proj + residual ------------------------
    if (skip < 2) {
      const uint4* kvb = (const uint4*)(a.kv + (long)l * 2 * C * D);
      uint4* d = (uint4*)sB;
      const uint4* q4 = (const uint4*)a.t1;
      for (int i = threadIdx.x; i < S * (D / 8); i += blockDim.x) {
        const int r = i / (D / 8), c = i - r * (D / 8);
        d[r * (LDS / 8) + c] = q4[i];
      }
      uint4* dk = (uint4*)(sB + S * LDS);
      for (int i = threadIdx.x; i < 2 * C * (D / 8); i += blockDim.x) {
        const int r = i / (D / 8), c = i - r * (D / 8);
        dk[r * (LDS / 8) + c] = kvb[i];
      }
    }
    __syncthreads();
    if (skip < 1)
      small_attn<S, D>(sB, LDS, sB + S * LDS, LDS, sB + (S + C) * LDS, LDS, C, nullptr, sA, D,
                       nwarps, sP);
    for (int tb = blockIdx.x; tb * nwarps < D; tb += G) {
      const int t = tb * nwarps + warp;
      Acc<S> acc = {};
      warp_dot<S, NCH>(W + WL<D, FM>::COUT_W + (long)t * D, sA, D, acc);
      warp_reduce_acc<S>(acc);
      if (lane < S)
        atomicAdd(&a.res[lane * D + t],
                  pick<S>(acc, lane) + __half2float(W[WL<D, FM>::COUT_B + t]));
    }
    if (!onebar) GBAR();
    if (stages < 5) continue;

    // ---- 5. ffn norm + up projection + GELU --------------------------------------------
    ln_prologue<S, D>(a.res, W + WL<D, FM>::FN_W, W + WL<D, FM>::FN_B, sA, nwarps);
    for (int tb = blockIdx.x; tb * nwarps * NCOL < FD; tb += G) {
      const int t = (tb * nwarps + warp) * NCOL;
      Acc<S> acc[NCOL] = {};
      warp_dot_n<S, NCH, NCOL>(W + WL<D, FM>::FU_W + (long)t * D, D, sA, D, acc);
#pragma unroll
      for (int j = 0; j < NCOL; ++j) {
        warp_reduce_acc<S>(acc[j]);
        if (lane < S) {
          const float v = pick<S>(acc[j], lane) + __half2float(W[WL<D, FM>::FU_B + t + j]);
          a.hf[lane * FD + t + j] =
              __float2half(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
        }
      }
    }
    if (!onebar) GBAR();
    if (stages < 6) continue;

    // ---- 6. ffn down projection + residual ---------------------------------------------
    // k-split FM => each warp reduces a D-wide slice of the FD hidden units. nwarps divides
    // D, so every warp of a threadblock shares one slice and the staging __syncthreads() stay
    // threadblock-uniform.
    for (int tb = blockIdx.x; tb * nwarps * NCOL < D * FM; tb += G) {
      const int t = (tb * nwarps + warp) * NCOL;
      const int ksi = (tb * nwarps * NCOL) / D, col = t - ksi * D;
      __syncthreads();
      {
        uint4* d = (uint4*)sB;
        const uint4* s4 = (const uint4*)(a.hf + ksi * D);
        for (int i = threadIdx.x; i < S * (D / 8); i += blockDim.x) {
          const int r = i / (D / 8), c = i - r * (D / 8);
          d[r * (D / 8) + c] = s4[r * (FD / 8) + c];
        }
      }
      __syncthreads();
      Acc<S> acc[NCOL] = {};
      warp_dot_n<S, NCH, NCOL>(W + WL<D, FM>::FD_W + (long)col * FD + ksi * D, FD, sB, D, acc);
#pragma unroll
      for (int j = 0; j < NCOL; ++j) {
        warp_reduce_acc<S>(acc[j]);
        if (lane < S) {
          float v = pick<S>(acc[j], lane);
          if (ksi == 0) v += __half2float(W[WL<D, FM>::FD_B + col + j]);
          atomicAdd(&a.res[lane * D + col + j], v);
        }
      }
    }
    GBAR();
  }
#undef GBAR

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < S * D; i += G * blockDim.x)
    a.xout[i] = __float2half(__ldcg(a.res + i));
}

// ------------------------------------------------------------------------------------------
// Precompute cross-attention K/V for every block of an expert: out[b][l][0]=K, out[b][l][1]=V.
// Runs once per draft (context_hidden is constant across all 32 block-passes) and evaluates
// context_norm ONCE instead of the two identical calls in the PyTorch block.
template <int D>
__global__ void cf_cross_kv(const __half* __restrict__ pw, const __half* __restrict__ ctx,
                            __half* __restrict__ out, int C, int L) {
  constexpr int NCH = D / 128;
  extern __shared__ __half smem[];
  __half* sA = smem;  // [C][D]
  const int nwarps = blockDim.x >> 5;
  const int lane = threadIdx.x & 31;
  const int l = blockIdx.y;
  // blockIdx.z is the batch slice.  There is no grid barrier in this kernel, so the batch
  // dimension costs nothing but pointer arithmetic and needs no co-residency budget.
  ctx += (long)blockIdx.z * C * D;
  out += (long)blockIdx.z * L * 2 * (long)C * D;
  const __half* P = pw + (long)l * PL<D>::STRIDE;

  for (int s = (threadIdx.x >> 5); s < C; s += nwarps) {
    const __half* r = ctx + (long)s * D;
    float sum = 0.f, sq = 0.f;
    for (int i = lane; i < D; i += 32) {
      const float v = __half2float(r[i]);
      sum += v;
      sq += v * v;
    }
    sum = warp_all_reduce(sum);
    sq = warp_all_reduce(sq);
    const float mean = sum / D;
    const float rstd = rsqrtf(fmaxf(sq / D - mean * mean, 0.f) + 1e-5f);
    for (int i = lane; i < D; i += 32)
      sA[s * D + i] = __float2half((__half2float(r[i]) - mean) * rstd *
                                       __half2float(P[PL<D>::CN_W + i]) +
                                   __half2float(P[PL<D>::CN_B + i]));
  }
  __syncthreads();

  const int gwarp = blockIdx.x * nwarps + (threadIdx.x >> 5);
  const int nw = gridDim.x * nwarps;
  for (int t = gwarp; t < 2 * D; t += nw) {
    for (int s0 = 0; s0 < C; s0 += 8) {
      Acc<8> acc = {};
      const int rows = min(8, C - s0);
      // reuse the 8-row kernel; rows beyond C simply read padding (their results are dropped)
      warp_dot<8, NCH>(P + PL<D>::KV_W + (long)t * D, sA + (long)s0 * D, D, acc);
      warp_reduce_acc<8>(acc);
      if (lane < rows)
        out[((long)l * 2 * C + (t / D) * C + s0 + lane) * D + (t % D)] =
            __float2half(pick<8>(acc, lane) + __half2float(P[PL<D>::KV_B + t]));
    }
  }
}

// ------------------------------------------------------------------------------------------
template <int D, int S>
static constexpr int smem_halves(int C) {
  int stage = S * (3 * D + 8);              // staged qkv
  int cross = (S + 2 * C) * (D + 8);        // staged cross q/k/v
  if (cross > stage) stage = cross;
  if (S * D > stage) stage = S * D;         // staged ffn-down h slice
  return NH * S * PLD * 2 + S * D + stage;  // score table + sA + staging
}

#define CF_FOR_EACH_INST(F) \
  F(640, 6, 4)              \
  F(640, 6, 8)              \
  F(1024, 6, 4)             \
  F(1024, 6, 8)

template <int D, int FM, int S, int TB>
static const void* kfun() {
  return (const void*)cf_fused_expert<D, FM, S, TB>;
}

template <int D, int FM, int S>
static const void* kfun_v(int tb) {
  return tb == 3 ? kfun<D, FM, S, 3>()
       : tb == 2 ? kfun<D, FM, S, 2>()
       : tb == 1 ? kfun<D, FM, S, 1>()
                 : kfun<D, FM, S, 0>();
}

template <int D, int FM, int S>
static int64_t max_grid_impl(int threads, int C, int tb) {
  int n = 0;
  const int sb = smem_halves<D, S>(C) * 2;
  if (sb > at::cuda::getCurrentDeviceProperties()->sharedMemPerBlockOptin) return 0;
  const void* f = kfun_v<D, FM, S>(tb);
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sb);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, threads, sb);
  return (int64_t)n * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
}

template <int D, int FM, int S>
static void launch_impl(const Args& a, int C, int G, int B, int threads, int tb) {
  const int sb = smem_halves<D, S>(C) * 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  const void* f = kfun_v<D, FM, S>(tb);
  cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, sb);
  const dim3 grid((unsigned)G, (unsigned)B);
  if (tb == 3)
    cf_fused_expert<D, FM, S, 3><<<grid, threads, sb, stream>>>(a);
  else if (tb == 2)
    cf_fused_expert<D, FM, S, 2><<<grid, threads, sb, stream>>>(a);
  else if (tb == 1)
    cf_fused_expert<D, FM, S, 1><<<grid, threads, sb, stream>>>(a);
  else
    cf_fused_expert<D, FM, S, 0><<<grid, threads, sb, stream>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Returns the resident-block cap for (D, FM, S, C), or 0 when the shape is not instantiated /
// does not fit in shared memory.  The Python side treats 0 as "fall back to PyTorch".
int64_t cf_max_grid(int64_t threads, int64_t D, int64_t FM, int64_t S, int64_t C, int64_t tb) {
#define F(d, fm, s)                                                     \
  if (D == d && FM == fm && S == s)                                     \
    return max_grid_impl<d, fm, s>((int)threads, (int)C, (int)tb);
  CF_FOR_EACH_INST(F)
#undef F
  return 0;
}

void cf_expert(torch::Tensor w, torch::Tensor kv, torch::Tensor xin, torch::Tensor xout,
               c10::optional<torch::Tensor> mask, torch::Tensor res, torch::Tensor qkv,
               torch::Tensor t1, torch::Tensor hf, torch::Tensor bar, int64_t D, int64_t FM,
               int64_t S, int64_t C, int64_t L, int64_t G, int64_t threads, int64_t stages,
               int64_t tb, int64_t B) {
  Args a;
  a.w = (const __half*)w.data_ptr();
  a.kv = (const __half*)kv.data_ptr();
  a.xin = (const __half*)xin.data_ptr();
  a.xout = (__half*)xout.data_ptr();
  a.mask = mask.has_value() ? (const __half*)mask->data_ptr() : nullptr;
  a.res = res.data_ptr<float>();
  a.qkv = (__half*)qkv.data_ptr();
  a.t1 = (__half*)t1.data_ptr();
  a.hf = (__half*)hf.data_ptr();
  a.bar = (unsigned long long*)bar.data_ptr();
  a.C = (int)C;
  a.L = (int)L;
  a.G = (int)G;
  a.stages = (int)stages;
  TORCH_CHECK(threads % 32 == 0 && threads <= (tb == 2 ? 1024 : tb == 3 ? 256 : 512),
              "cf_expert: threads exceeds the launch-bound variant");
  TORCH_CHECK(D % (threads / 32) == 0, "cf_expert: nwarps must divide D");
  // NCOL column tiling: every threadblock must consume a whole multiple of nwarps*NCOL columns
  // so no warp runs off the end and the ffn-down k-slice index stays block-uniform.
  TORCH_CHECK((D % (CF_NCOL * threads / 32) == 0) && (3 * D % (CF_NCOL * threads / 32) == 0)
                  && (FM * D % (CF_NCOL * threads / 32) == 0),
              "cf_expert: NCOL*nwarps (", CF_NCOL * threads / 32, ") must divide D, 3D and FM*D");
  TORCH_CHECK(C < PLD && S < PLD, "cf_expert: S and C must be < ", PLD);
  // The grid barrier spins, so EVERY block of EVERY slice has to be resident at once.  A too-big
  // grid is a silent hang, not an error, so the budget is checked here as well as in Python.
  TORCH_CHECK(B >= 1 && bar.numel() >= B, "cf_expert: bar needs one counter per batch slice (B=",
              B, ", got ", bar.numel(), ")");
#define F(d, fm, s)                                                            \
  if (D == d && FM == fm && S == s) {                                          \
    launch_impl<d, fm, s>(a, (int)C, (int)G, (int)B, (int)threads, (int)tb);    \
    return;                                                                    \
  }
  CF_FOR_EACH_INST(F)
#undef F
  TORCH_CHECK(false, "cf_expert: no instantiation for D=", D, " ffn_mult=", FM, " S=", S);
}

void cf_kv(torch::Tensor pw, torch::Tensor ctx, torch::Tensor out, int64_t D, int64_t C, int64_t L,
           int64_t G, int64_t threads, int64_t B) {
  const int sb = (int)((C + 7) / 8 * 8) * (int)D * 2;
  dim3 grid((unsigned)G, (unsigned)L, (unsigned)B);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (D == 640) {
    cudaFuncSetAttribute((const void*)cf_cross_kv<640>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, sb);
    cf_cross_kv<640><<<grid, (int)threads, sb, stream>>>(
        (const __half*)pw.data_ptr(), (const __half*)ctx.data_ptr(), (__half*)out.data_ptr(),
        (int)C, (int)L);
  } else {
    TORCH_CHECK(D == 1024, "cf_kv: unsupported D=", D);
    cudaFuncSetAttribute((const void*)cf_cross_kv<1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, sb);
    cf_cross_kv<1024><<<grid, (int)threads, sb, stream>>>(
        (const __half*)pw.data_ptr(), (const __half*)ctx.data_ptr(), (__half*)out.data_ptr(),
        (int)C, (int)L);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("expert", &cf_expert);
  m.def("cross_kv", &cf_kv);
  m.def("max_grid", &cf_max_grid);
}
