# Benchmarking protocol — read this before quoting any speedup

Every number in this project is a ratio against a baseline. Two things about that
baseline have silently corrupted results in the past. Both are cheap to get right
and expensive to discover late.

---

## 0. `vllm serve` compatibility. (Batch 1 is still the measurement of record.)

> **Read first, if you are deploying:**
> 1. **The TREE arm is stable under concurrency now** (the crash is fixed; re-verified here to
>    concurrency 64 at 4B, 982 requests, zero errors, acceptance flat at 2.40). But it is a
>    batch-1 technique: 1.35x at concurrency 1, 0.85x at 4, **0.14x at 64**, because its `K+1` is
>    42 query positions per request against a chain's 6. The batch cutoff below now covers it.
> 2. **Set `CF_TREE_GREEDY_GUARD=1` on any tree-mode server** — otherwise one `temperature>0`
>    request kills it permanently, for every client.
> 3. **`CF_SPEC_MAX_BATCH` is now ON by default, and you no longer have to pick N.** Without a
>    cutoff the 4B chain arm falls to 0.90x of the no-speculation baseline at concurrency 8,
>    0.74x at 16 and **0.44x at 64**; with it, 1.19x at batch 1 (bit-identical to the uncut arm)
>    and 0.94–0.96x under load. All six size × arm combinations are laddered — 4B **4**/**2**,
>    9B **4**/**3**, 27B **16**/**3** for chain/tree — and the default engages *only* those. A
>    target that was never laddered gets **no cutoff**, and says so; `=auto` opts into the
>    derived guess for one, `=0` turns it off.
> 4. **Do not serve 27B with speculation above concurrency ~16 at all.** There the binding cost
>    is not drafting and the cutoff cannot fix it: `--speculative-config` more than halves the
>    engine's KV cache, capping the 27B decode batch at ~28 against the base engine's 64.
> 5. **The draft cudagraph ladder now reaches `max_num_seqs`, not 32.** Nothing to set. It is
>    worth **+2.3% at concurrency 64** and nothing below it, and it turns the 29 mid-traffic
>    `torch.compile`s the unbucketed range used to trigger into one. `CF_DRAFT_BUCKETS` overrides
>    the ladder.
> 6. **`CF_SPEC_K_SCHEDULE` exists and is default OFF, because at 4B chain it loses.** A K=1 rung
>    beats the uncut K=5 arm under load (+14.6% at concurrency 64) and never beats the K=0 cutoff
>    at any batch. The measured reason is that the draft costs 19.4 ms at B=64 against a 14.1 ms
>    no-speculation step, so no verify-width lever can reach parity.

The published speedups and acceptance numbers are **batch-1 numbers**, taken through the
offline `LLM()` API (`vllm/test_plugin_native.py`, which also forces
`VLLM_ENABLE_V1_MULTIPROCESSING=0`), and that remains the measurement of record — sections 1–4
below. This section is about something different: whether the same code is **correct and
robust** under `vllm serve`, which is an engine core in its own process plus continuous
batching. It is not, yet, in the ways listed above; the rest of this section is the evidence.

The performance tables here are supporting context, not the headline. They are included
because a flag that disengages under batching is only acceptable while the result stays
*above* the no-speculation baseline, and at 4B it does not.

| | `vllm/bench_cf.sh` | `vllm/bench_serve.sh` |
|---|---|---|
| entry point | offline `LLM()` | `vllm serve` + HTTP |
| engine core | in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | spawned subprocess |
| batch | 1 | continuous, driven at concurrency 1/2/4/8/16 |
| prompts | `bench_data/*.jsonl`, 7 domains | RedHat AI `speculator_benchmarks` subset, 7 domains × 10 |
| acceptance from | the proposer object, in-process | the engine's `vllm:spec_decode_*` counters |
| use it for | **a fast regression gate** (~3 min at 4B, one process) | **the number you quote** |

Keep `bench_cf.sh`. It is the only harness that can read the proposer's own counters
without a metrics round-trip, it is much faster, and batch 1 is a real deployment point
(single-user, latency-bound). It is simply not *the* deployment point.

```bash
./vllm/bench_serve.sh 4b base 3 8601        # <size> <arm> <gpu> <port>
./vllm/bench_serve.sh 4b chain 3 8602
CF_SAMPLING_PROBE=1 ./vllm/bench_serve.sh 4b tree 3 8603
./vllm/bench_serve_report.py                # the table, all runs
./vllm/bench_serve_diff.py logs/bench_serve/4b_{base,chain}/serve_bench.json
```

For a ladder that goes past concurrency 16, use `vllm/serve_ladder.sh` instead — same arms, same
env table, but a **per-level request count** (`CF_LADDER="1:70,4:120,16:256,64:320"`) and two
pre-flight checks that `bench_serve.sh` does not have. Both exist because of the same failure:

> **Never start an arm on a GPU that is not yet empty.** vLLM sizes the KV cache as
> `gpu_memory_utilization × TOTAL − (whatever is already resident)`, so an engine that profiles
> while the previous arm is still tearing down does not run slowly — it runs with a **silently
> tiny KV cache**. Measured here: a 4B chain server started right after a 4B base server exited
> reported `Available KV cache memory: 1.43 GiB` / `Maximum concurrency … 2.53x` instead of the
> usual 41.38 GiB / 73.29x. It served `/health` fine, accepted every request, and read **505
> tok/s at "concurrency 8" while its decode batch never once exceeded 4** — every level above
> that measured the admission queue and reported it as the arm's throughput. Nothing in the
> throughput number says so; the only tell is one line in the startup log. `serve_ladder.sh`
> waits for the GPU to drain, then refuses to benchmark if the KV cache it got cannot hold the
> ladder it is about to be driven with.

Three things `bench_serve.sh` does that a naive serve script does not, each because
getting it wrong produced a wrong answer here first:

1. **It does not set `VLLM_ENABLE_V1_MULTIPROCESSING=0`.** Note that under `vllm serve`
   that variable does not even do what its name suggests — the API server and the engine
   core are separate processes either way (it gates the *offline* `LLMEngine`). Measuring
   with it set measures a flag that is doing nothing, in a configuration nobody deploys.
2. **It waits for a real request before reading the flag report.**
   `FlowDrafterProposer._build()` is called lazily from the first `propose()`, so a server
   that answers `/health` has *not yet resolved a single drafter flag*. Grepping a freshly
   started log finds only the API server's copy of the `[cf-defaults]` line — printed at
   import, in a process with no drafter in it — which states the *proposals* and cannot
   state the outcome. `bench_serve.sh` sends one completion, then greps the
   `(EngineCore pid=…)` copy, and refuses to benchmark if it is absent.
3. **It keeps every completion** so `bench_serve_diff.py` can gate on the arms having
   produced identical text before their throughputs are compared.

### The `[cf-defaults]` line is a build-time statement. It cannot report a batch gate.

Every gate in `chained_flow/defaults.py` is evaluated once, against the drafter and the
engine config, *before any request exists*. So the line says `cuda_block(D=640 …)` for the
whole life of a server on which the fused kernel runs on a minority of steps. Set
**`CF_BATCH_AUDIT=1`** (default off) for the per-step answer: it prints a histogram of the
decode batch and of the draft cudagraph bucket, and states outright what fraction of steps
could have run the batch-1-only kernel.

### What actually disengages above batch 1

Everything here is reported ON by the startup line in every case. Nothing below is a bug;
they are deliberate batch-1 specialisations that the flag report has no vocabulary for.

| flag | still on at batch > 1? | why not |
|---|---|---|
| `CF_CUDA_BLOCK` | **no, from B ≥ 2** — unless `CF_CUDA_BLOCK_BATCH=1`, and then only to B ≤ 8 (D=640) / 4 (D=1024) | `chunked_flow._cf_fused_runner` used to require `x.shape[0] == 1`; the drafter's `x.shape[0]` is the cudagraph *bucket*, so only bucket 1 qualified. `CF_CUDA_BLOCK_BATCH` (default OFF) instead runs the batch as `gridDim.y` slices — see below. Above the crossover it still falls back, and there the PyTorch stack is genuinely the faster path. |
| `CF_CUDA_PAIR` | **no, from B ≥ 2** — with batching, to the batch where each expert still gets ≥ 4 blocks per slice | rides on `CF_CUDA_BLOCK`. Splitting an already B-way-split machine two more ways starves both grids; measured, the pair loses to the serial fused kernel by 2× from B = 16. |
| `CF_GDN_DEFER` / `CF_GDN_BV` | **no, from B ≥ 2** (tree) | `tree_gdn.defer_rows(T)` caps at `CF_GDN_DEFER_MAXROWS` = 64 rows. An 8×5 tree is 41 rows per request, so B = 1 fits and B = 2 (82 rows) does not. The cap is a *memory* decision — a deferred stash pins that layer's k/v inside the cudagraph pool — not a correctness one. |
| draft cudagraph | degrades, but no longer **off** | buckets are powers of two **up to `max_num_seqs`** (`[1,2,4,8,16,32,64]` at the default 64); B = 5 replays the bucket-8 graph, i.e. three whole drafts on padding rows. It used to stop at 32, above which the draft ran eager *and* recompiled per batch size — see the ladder fix below. |
| `CF_DRAFT_EARLY` (side-stream prelaunch) | intermittent | the steady-state guard requires every input-batch slot to hold the *same request as last step*, which continuous batching breaks whenever a request joins or leaves. It is also tree-only: in chain mode `_prelaunch` returns immediately and the flag's remaining job is publishing the GPU counts for `CF_ASYNC_SPEC`. |
| `CF_TREE_FUSED_ATTN` | yes | its `N < 128` limit is the per-request tree width, not the batch. |
| `CF_TREE_FULLCG` | yes | a uniform multi-request decode still dispatches FULL. |
| `CF_SHORTLIST`, `CF_COMPILE`, `CF_TWOPASS_M`, `CF_PATH_TRIM`, `CF_FUSE_PATH`, `CF_RING_TRIM`, `CF_ASYNC_SPEC` | yes | batch-agnostic. |

#### `CF_CUDA_BLOCK_BATCH` — the fused kernel at batch, and where it stops paying (default OFF)

The kernel is written for one draft; a batch is B independent copies of it, so B rides on
`gridDim.y` and only the pointers move. Shared memory depends on `(D, S, C)` alone — `S` is a
draft's *row count*, never the batch — so there are no new template instantiations. Each slice
keeps its **own** grid-barrier counter (`bar + blockIdx.y`), which means the caller must hold
`G_per_slice × B` inside the resident-block cap or the kernel *hangs*; `grid_for` computes
`G = min(default, cap // B)` and refuses the batch when that reaches 0.

It works, and above a point it is the wrong thing to do. `integrate()`, compiled + cudagraphed,
ms, best fused config vs the PyTorch/cutlass fallback:

| B | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| D=640 torch | 2.035 | 2.195 | 2.234 | 2.516 | 3.438 | 3.771 | 4.421 |
| D=640 fused | **0.803** | **0.945** | **1.399** | 2.444 | 3.900 | 7.865 | 18.33 |
| ratio | 2.53× | 2.32× | 1.60× | 1.03× | 0.88× | 0.48× | 0.24× |
| D=1024 torch | 2.734 | 2.900 | 2.940 | 3.257 | 4.512 | 5.062 | — |
| D=1024 fused | **1.364** | **1.668** | **2.381** | 4.111 | 8.035 | 15.96 | — |
| ratio | 2.00× | 1.74× | 1.23× | 0.79× | 0.56× | 0.32× | — |

**ncu says the slicing did exactly what it was designed to do and then hit a different wall.**
DRAM throughput falls from 19.4 % to 1.5 % of peak (D=640, B = 1 → 64) — the 128 MB L2 does
dedupe the weight stream every slice shares — and the "no eligible instruction" stall falls from
0.29 to 0.04 per issue-active cycle, which is the latency-bound diagnosis being cured. What never
moves is **occupancy**: `sm__warps_active` is pinned at 33.33 % at *every* batch, because ~45 KB
(D=640) / 69 KB (D=1024) of shared memory allows one 16-warp block per SM. Issue-active tops out
at 55–61 %, IPC at 0.61, `sm__throughput` at 44–47 %. cutlass has neither constraint: it is not
co-residency-pinned, and at M = 8·B it reaches tensor cores where this kernel is committed to
scalar fp32-accumulate FMA at M = 8 per slice, forever. Widening the per-warp column tile
(`CF_CUDA_BLOCK_NCOL=4`) was tried against exactly this and lost: −8 % at B ≤ 4, +5 % at B ≥ 32,
with ptxas still spill-free — shared-memory bandwidth is not the binding constraint either.

So the flag hands every batch above the crossover back to cutlass (`cuda_block.batch_limit`).
**In-engine, 4B chain, `serve_ladder.sh`, same ladder and the same `4b_base_lad` baseline:**

| conc | base | `chain_cut4` | `chain_cut4batch` | `chain_cut8batch` |
|---|---|---|---|---|
| 1 | 138.3 | 165.6 a=1.86 1.20× | 165.4 a=1.86 1.20× | — |
| 4 | 498.0 | 500.0 a=1.86 1.00× | **529.9 a=1.87 1.06×** | 528.9 a=1.86 1.06× |
| 8 | 954.8 | 905.2 0.95× | 904.4 0.95× | **841.5 0.88×** |
| 16 | 1745.3 | 1660.8 0.95× | 1654.3 0.95× | 1648.1 0.95× |
| 32 | 2967.0 | 2836.4 0.96× | 2829.0 0.95× | — |
| 64 | 4519.8 | 4274.1 0.95× | 4264.9 0.94× | — |

One rung moves: concurrency 4, 1.00× → **1.06×**, at unchanged acceptance (1.86 → 1.87). Every
other rung is inside noise, because `CF_SPEC_MAX_BATCH=4` already stops the drafter above a
decode batch of 4, so buckets 2 and 4 are the only ones the kernel ever sees. **The cutoff
threshold does not move**: raising it to 8 *with* batching gives 0.88× at concurrency 8, worse
than the 0.95× the cutoff buys by not drafting at all — the batched kernel is only break-even
against cutlass at B = 8, which is not enough to make drafting at batch 5–8 profitable.

That narrowness is why the flag ships OFF: it is a +6 % at one rung of one arm, and the 9B/27B
ladders have not been re-run with it.

### The first request at a new batch size stalls the engine

`CF_COMPILE` compiles the flow net with `max-autotune-no-cudagraphs`, and the draft
cudagraph is captured lazily per bucket. Both happen **inside the serving loop**, the first
time a bucket is reached. Measured at 4B chain, from the engine's own step log: a **70–80 s
stall on the first step at each of buckets 2, 4, 8 and 16** (32–36 inductor autotune blocks
each). A benchmark must warm up *at the concurrency it is about to measure* or it charges
that stall to steady-state throughput — `bench_serve_drive.py` warms up with `3 × concurrency`
requests for exactly this reason, and got it wrong once (4B chain c=16 read 175.9 tok/s and a
20.0 s mean TTFT against 826 tok/s at c=8, purely from the un-warmed bucket-16 capture).

**Above bucket 32 it used to be much worse than "one stall per bucket," and that is FIXED.**
The buckets were `[1,2,4,8,16,32]`, so a decode batch of 33 or more had no bucket at all:
`_ctx_gpu` falls back to `bucket = B`, the flow net is compiled with `dynamic=False`, and
therefore **every distinct batch size in 33…max_num_seqs was its own compile**. That is the
shape of the 850-autotune-block storm seen on a first `guidellm --rate 64` pass — not six shapes
discovered once, but ~25, most of them discovered while the concurrency was decaying through the
thirties and forties. `FlowDrafterProposer._bucket_ladder` now runs the powers of two all the way
to `max_num_seqs` (plus `max_num_seqs` itself when it is not one), so the same range is **two**
shapes, and `CF_DRAFT_BUCKETS` overrides the ladder if you want it finer. See
"The drafter had no cudagraph above bucket 32" below for what that is worth in throughput.

**`CF_WARM_BUCKETS=1` (default off)** does the buckets up front: `FlowDrafterProposer._build()`
compiles and captures the draft graph for every bucket the batch cutoff can reach, against the
freshly zeroed ring, before the drafter serves anything. Because `_build()` is lazy it lands on
the *first request*, not on process start — which is still the difference between one slow
request at deploy time and an arbitrary request stalling later. Measured at 4B with
`CF_SPEC_MAX_BATCH=4`:

> `draft buckets warmed at startup: [1, 2, 4] in 51.3s` — the whole compile-and-capture bill for
> every shape the server can subsequently reach, paid once.

With the ladder now reaching `max_num_seqs` it covers an **uncut** server too — measured on the
same 4B chain engine with `CF_SPEC_MAX_BATCH=0`:

> `draft buckets warmed at startup: [1, 2, 4, 8, 16, 32, 64] in 125.4s`

i.e. ~18 s per rung, not the 70–80 s the per-shape figure above suggests (that number is one
inductor codegen *inside* a live engine, contending with it). Seven rungs for 125 s is what makes
the coarse top of the ladder the right trade: it replaces ~25 mid-traffic compiles.

### Measured, 2026-08-05 — the speedup is a batch-1 speedup

4B on one RTX PRO 6000, `--async-scheduling` on every arm, `--max-num-seqs 64`, 70 prompts ×
256 tokens with `ignore_eos`, server-side tok/s. `bench_serve.sh` at concurrency 1 reproduces
the offline batch-1 references (base 139.0 vs 140.0, chain 165.3 vs 157.8, tree 187.5 vs
188.9), so the two harnesses agree where they overlap.

**4B**

| conc | base | chain | vs base | accept | tree | vs base | accept |
|---|---|---|---|---|---|---|---|
| 1 | 139.0 | 165.3 | **1.19x** | 1.855 | 187.5 | **1.35x** | 2.395 |
| 2 | 256.8 | 273.4 | **1.06x** | 1.854 | *engine crash* | | |
| 4 | 485.6 | 494.5 | **1.02x** | 1.861 | *engine crash* | | |
| 8 | 929.6 | 836.0 | **0.90x** | 1.855 | *engine crash* | | |
| 16 | 1565.7 | 1213.8 | **0.78x** | 1.858 | *engine crash* | | |

**27B** — the same shape, but it never goes under water, because a 27B target forward at batch
16 is still bandwidth-bound per token and a 4B one is not:

| conc | base | chain | vs base | accept | tree | vs base | accept |
|---|---|---|---|---|---|---|---|
| 1 | 26.3 | 42.6 | **1.62x** | 1.953 | 48.3 | **1.84x** | 2.485 |
| 2 | 51.0 | 78.3 | **1.54x** | 1.949 | *engine crash* | | |
| 4 | 97.4 | 142.6 | **1.46x** | 1.955 | *engine crash* | | |
| 8 | 193.2 | 245.1 | **1.27x** | 1.955 | *engine crash* | | |
| 16 | 327.5 | 381.3 | **1.16x** | 1.947 | *engine crash* | | |

**The 4B chain arm goes BELOW the no-speculation baseline at concurrency ≥ 8** (0.90x, 0.78x).
That is the disengagement biting: the drafter still costs a full draft per step while its
batch-1 kernel is gone and the target forward has stopped being the bottleneck. A 4B server
expecting more than ~4 concurrent requests is faster with speculation turned **off**. 27B is
never in that regime on this hardware.

**Acceptance does not move** (1.854–1.861 across the whole ladder). The entire collapse is
draft *cost*, and roughly half of it is one flag. The A/B, same server config, `CF_CUDA_BLOCK`
the only variable:

| 4B chain | `CF_CUDA_BLOCK=1` | `CF_CUDA_BLOCK=0` | kernel worth |
|---|---|---|---|
| concurrency 1 | 165.3 | 149.3 | **+10.7%** |
| concurrency 4 | 494.5 | 495.7 | **+0.0%** |

That is the disengagement measured rather than read off the source: at concurrency 4 turning
the kernel *off* costs nothing, because the `x.shape[0] == 1` gate had already turned it off.
The rest of the collapse is structural and not ours to fix — a speculative step verifies
`(K+1) × B` tokens, so as `B` grows the target forward stops being bandwidth-bound per token
and the thing speculation exploits goes away.

### The drafter had no cudagraph above bucket 32 — fixed, and it is worth 2.3%

The profile in commit `81d0ecc` put the drafter at **24.2 ms of a 65.7 ms step at B=64**, the
larger half of the 0.56x deficit, and identified the missing cudagraph as the debt inside it:
**1901 of 2000 steps ran the draft with no cudagraph at all**, because the bucket ladder stopped
at 32. `FlowDrafterProposer._bucket_ladder` now runs powers of two to `max_num_seqs`
(`CF_DRAFT_BUCKETS` overrides). The A/B is one variable — the ladder — on the same GPU in the
same session, both arms `CF_SPEC_MAX_BATCH=0` so nothing else disengages:

| conc | base | chain, buckets **…32** | vs base | chain, buckets **…64** | vs base |
|---|---|---|---|---|---|
| 1 | 139.2 | 165.3 | **1.19x** | 165.4 | **1.19x** |
| 4 | 498.1 | 500.7 | 1.01x | 500.0 | 1.00x |
| 8 | 953.5 | 868.5 | 0.91x | 865.8 | 0.91x |
| 16 | 1763.2 | 1289.7 | 0.73x | 1292.7 | 0.73x |
| 32 | 2962.8 | 1749.6 | 0.59x | 1750.9 | 0.59x |
| 64 | 4523.9 | 1986.8 | **0.439x** | **2031.7** | **0.449x** |

**The mechanism is fully engaged and it is worth +2.3% at concurrency 64, and nothing anywhere
else.** `CF_BATCH_AUDIT` on the two arms: `no-cudagraph drafted steps` **769 → 0**, and the
unbucketed range that was **29 distinct `dynamic=False` compiles** (34, 35, 36, 37, 38, 39, 40,
41, 43, 45…64) is now the single bucket 64. Levels 1–32 are unchanged to within run-to-run noise
*by construction* — those buckets already existed, so not one step below 33 changes shape.

**+2.3% is the right answer and it was predictable before the ladder ran, which is what makes it
an attribution rather than a result.** `CF_DRAFTPROF=1 CF_WARM_BUCKETS=1` times the captured
graph against the eager (still `torch.compile`d) path at every bucket, in-engine:

| bucket | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| graph replay, whole draft (ms) | 2.675 | 4.390 | 4.900 | 5.816 | 7.867 | 11.810 | **19.446** |
| eager whole draft (ms) | 3.080 | 8.067 | 8.122 | 8.695 | 9.458 | 14.156 | **21.218** |

At bucket 64 the graph saves **1.77 ms of a 21.2 ms draft (8%)**, and the draft is about a third
of a 65.7 ms step — so ~2.7% predicted against +2.3% measured.

**So the profile's headline needs a correction, and it is the useful finding here.** The drafter
really is the larger half of the deficit at B=64, but the *missing cudagraph* was only 8% of the
drafter, not most of it. The remaining ~19 ms is real GPU work, and the kernel profile of the
captured bucket-64 graph says exactly where:

| | bucket 1 | bucket 64 |
|---|---|---|
| whole draft, captured | 2.675 ms | 19.446 ms |
| the 8-block stack | `cf_fused_expert`, **1.30 ms** (2 launches at S=8 + 2 at S=4) | 5 launches of a `cutlass wmma 32x` GEMM, **9.08 ms** |
| its share of the draft | 41% | **48%** |

That is the same batch-1 gate (`chunked_flow.py:296`, `x.shape[0] == 1`) showing up as **9.08 ms
per step** at B=64. Its roofline is not close: the block stack's weights are `16 × D²` per block
× 8 blocks × 2 Euler steps ≈ 210 MB at `D=640`, i.e. ~0.12 ms of streaming, and 512 rows through
it is ~107 GFLOP, i.e. ~0.36 ms of fp16 math. **Batching `cf_fused_expert` over `gridDim.y` is
therefore the next drafter fix, and it is worth ~25x more than the ladder was.** It is templated
`<D, FM, S, TB>` where `S` is one draft's row count, not the batch, and shared memory depends
only on `(D, S, C)`, so no new instantiations are needed — batch becomes a `gridDim.y` slice
(offset xin/xout/res/qkv/t1/hf/kv, and make the counting grid barrier `bar + blockIdx.y`;
`cf_cross_kv` already uses `blockIdx.y` and has no barrier). The constraint is co-residency for
the grid sync: `G_slice × B ≤ cf_max_grid` (~188 blocks here), so `G` comes from
`cf_max_grid / B` with an eager fallback above it. **NOT IMPLEMENTED.**

### `CF_SPEC_MAX_BATCH` — stop speculating above a decode batch (default ON where measured)

Taken past concurrency 16, the 4B chain arm does not level off, it keeps falling: **0.44x of the
no-speculation baseline at concurrency 64.** A server that is 2.3x slower under load because a
feature is enabled is not deployable, and "it helps at batch 1" is not a defence.

`CF_SPEC_MAX_BATCH=N` turns speculation off above a decode batch of N. It is **two patches, and
both are needed**:

| half | what it stops | why it cannot be the other one |
|---|---|---|
| `vllm_plugin/batch_cutoff.py` — wraps `AsyncScheduler._update_after_schedule` and zeroes `scheduler_output.num_spec_tokens_to_schedule` | the **target** verifying `(K+1) × B` positions | under async scheduling the drafter has **no say**: the scheduler writes `request.spec_token_ids = [-1] * K` before any draft exists, and vLLM never reads our returned ids back (`_copy_draft_token_ids_to_cpu` returns early when `use_async_scheduling`) |
| `FlowDrafterProposer.propose` — `_cut()` / `_no_draft()` | the **drafter** running at all | on its own the target would still be verifying a step's worth of zeros |

**Why not vLLM's own `num_speculative_tokens_per_batch_size`.** 0.25.1 *does* have a native
batch-size→K schedule (`SpeculativeConfig.num_speculative_tokens_per_batch_size` →
`Scheduler.dynamic_sd_lookup`), K=0 is a legal entry, and it lands in the same place. It was
rejected for one measured reason: setting it makes `VllmConfig.
_maybe_override_dynamic_sd_cudagraph_mode` downgrade `cudagraph_mode` from FULL_AND_PIECEWISE to
PIECEWISE **for the whole engine, including batch 1** — trading the high-concurrency fix for a
regression on the metric this project is judged on. Patching the scheduler decision alone leaves
full cudagraphs captured and dispatched for every step that still speculates.

**Measured, 4B**, same server config, `--max-num-seqs 64`, `serve_ladder.sh`:

| conc | base | chain | vs base | chain + `CF_SPEC_MAX_BATCH=4` | vs base |
|---|---|---|---|---|---|
| 1 | 139.0 | 165.2 | **1.19x** | 165.6 | **1.19x** |
| 4 | 498.8 | 498.6 | 1.00x | 500.0 | 1.00x |
| 8 | 952.9 | 855.7 | 0.90x | 905.2 | **0.95x** |
| 16 | 1764.4 | 1297.9 | 0.74x | 1660.8 | **0.94x** |
| 32 | 2968.3 | 1743.7 | 0.59x | 2836.4 | **0.96x** |
| 64 | 4518.6 | 1993.9 | **0.44x** | 4274.1 | **0.95x** |

**N=4 because that is where the curve crosses**, not because it is a round number: 1.19x at
concurrency 1, 1.00x at 4, 0.90x at 8. N is inclusive, so a batch of exactly 4 still speculates,
and the batch-1 number is **bit-identical** to the uncut arm (0/70 sequences differ).

**The threshold is picked from the target AND the arm**, because it is a measurement and both
ways of getting it wrong cost real throughput — N=4 on a 27B server measured 190.4 tok/s at
concurrency 8 against the uncut arm's 253.5, and N=16 on a 4B server leaves it under water from
concurrency 8 up. `_AUTO` is keyed on `(hidden_size, K+1)`, the hidden size read off the loaded
embedding rather than off a config. **All six combinations are now laddered**, each against its
own no-speculation baseline on the same GPU:

| | chain (`K+1` = 6) | 8×5 tree (`K+1` = 41) |
|---|---|---|
| 4B (2560) | **4** | **2** |
| 9B (4096) | **4** | **3** |
| 27B (5120) | **16** | **3** \* |

\* the 27B tree engine cannot reach a decode batch above 3 at all; see below.

Every entry is the last laddered decode batch still at or above parity, and the one next to it
is below — that is the rule, applied the same way six times. Resolution happens in `_build()`,
which is the first moment the target's size is known; until then `max_batch()` reads 0 and the
cutoff is inert — correct rather than merely tolerable, since the only steps that precede
`_build()` are the first few of the first request, at a decode batch of 1 that no threshold in
the table would cut.

**This is now the DEFAULT**, and the reason it can be is that it refuses to guess: a
`(hidden_size, K+1)` that is not in the table resolves to *no cutoff at all*, printing
`NOT MEASURED for hidden_size=… at verify width …`. `CF_SPEC_MAX_BATCH=auto` opts back into the
derived width rule for such a target, `=<n>` sets one directly, `=0` or `off` disables. The
asymmetry is the point: a derived N that is too low silently costs speedup and nothing in the
throughput number says so, so it is not a thing to acquire by accident.

#### `CF_SPEC_K_SCHEDULE` — a K-ladder instead of a K=0 cliff. Measured, and at 4B it LOSES.

The cutoff is a cliff. The roofline says there should be something between K=5 and K=0: `F.linear`
at this model's shapes runs at 138–162 TFLOP/s at M=64 against an M=2688 asymptote of 330–380, so
`B=64` is only half way to compute-bound and `M=128` — which is `B=64` at **K=1** — costs only
~1.33x the M=64 forward the no-speculation baseline pays. So a K=1 arm ought to have a chance of
staying above parity where K=5 provably cannot.

`CF_SPEC_K_SCHEDULE="4:full,16:1"` expresses that: rungs of `<max decode batch>:<K>`, implicit
K=0 above the last. It rides the **same** scheduler patch — `_update_after_schedule` already
writes `num_spec_tokens_to_schedule`, and writing `1` there instead of `0` needs no change in the
proposer, because `_prepare_input_ids` scatters `range(start, start + draft_len)` from
`start = prev_index * prev_num_spec_tokens`, i.e. the **first `draft_len` columns** of whatever
width tensor the proposer returned, and `prev_num_spec_tokens` is read off that tensor's own
width every step. For a chain those columns are exactly the first `draft_len` links. So this is
still not vLLM's `num_speculative_tokens_per_batch_size`, which would downgrade `cudagraph_mode`
to PIECEWISE engine-wide including batch 1.

**Measured, 4B chain, `CF_SPEC_K_SCHEDULE="64:1"` — K=1 at every decode batch, so the whole K=1
curve in one run — against the same-session base and the uncut K=5 arm:**

| conc | base | K=5, uncut | | K=1 | | K=0 cliff at N=4 |
|---|---|---|---|---|---|---|
| 1 | 139.2 | 165.4 | **1.19x** | 136.6 | 0.98x | **1.19x** |
| 4 | 498.1 | 500.0 | 1.00x | 424.0 | 0.85x | 1.00x |
| 8 | 953.5 | 865.8 | 0.91x | 749.5 | 0.79x | **0.95x** |
| 16 | 1763.2 | 1292.7 | 0.73x | 1229.1 | 0.70x | **0.94x** |
| 32 | 2962.8 | 1750.9 | 0.59x | 1776.1 | 0.60x | **0.96x** |
| 64 | 4523.9 | 2031.7 | 0.449x | 2329.4 | 0.515x | **0.95x** |

Acceptance is flat at **1.525–1.530** for K=1 against 1.849–1.865 for K=5, exactly as a chain
should behave.

**K=1 beats K=5 under load (+14.6% at concurrency 64) and it never once beats the K=0 cliff.**
There is therefore **no 4B chain threshold at which a K=1 rung is the right answer**, and
`_AUTO_K1` stays empty and the schedule stays default OFF.

**The reason is one number and it is not the verify width.** Per-step, at a decode batch of 64:

| | tokens/step | step |
|---|---|---|
| base | 1.000 | **14.1 ms** |
| K=1 | 1.525 | 40.0 ms |
| K=5 | 1.849 | 52.7 ms |

Dropping K from 5 to 1 removes 256 of 384 verify positions and buys 12.7 ms — the roofline was
right about that. It cannot be enough, because **the draft alone is 19.4 ms and the entire
no-speculation step is 14.1 ms**. No reduction in verify width can bring a step to parity when
the drafter's own fixed cost already exceeds the whole step it is trying to accelerate. The only
lever that reaches it is the DRAFT, which is what the cliff pulls — and what batching
`cf_fused_expert` would pull without giving up the acceptance.

**27B was the case with the best prior and it says the same thing.** A 27B target forward is far
more expensive relative to the same drafter, so if a K=1 rung wins anywhere it should win here.
`CF_SPEC_K_SCHEDULE="64:1"`, 27B chain, against a same-session base on the same GPU (recorded
K=5 absolutes in the third column, restated against *this* base):

| conc | base | K=1 | | K=5 (recorded 42.7 / 140.5 / 395.6 / 501.7 / 503.4) | K=0 cliff at N=16 |
|---|---|---|---|---|---|
| 1 | 26.3 | 34.7 | 1.32x | **1.62x** | **1.62x** |
| 4 | 100.1 | 122.8 | 1.23x | **1.40x** | **1.41x** |
| 16 | 355.9 | 376.2 | 1.06x | **1.11x** | **1.08x** |
| 32 | 622.4 | 515.9 | 0.83x | 0.81x | 0.82x |
| 64 | 887.9 | 520.6 | 0.59x | 0.57x | 0.52x |

K=1 is *worse* than K=5 at every level that is a real decode batch and ties it, inside noise, at
the two that are not — this engine's `max B` is **28**, and concurrency 64 has a **15.7 s** TTFT,
so those rows are the `--speculative-config` admission queue and no K threshold addresses them.
Acceptance is flat at 1.574–1.579.

**So `_AUTO_K1` is empty at both sizes on measurement, not for want of a ladder**, and
`CF_SPEC_K_SCHEDULE` ships as an instrument: it is how the next target gets laddered, and it is
the shape the schedule would take if a drafter ever became cheap enough for a middle rung to
exist. That is the same thing the 4B arithmetic says — the middle rung's existence is gated on
the DRAFT cost, not on the verify width.

**It also deletes the compile storm**, for free: with the cutoff at 4 the drafter is only ever
asked for buckets 1, 2 and 4, so those are the only three that ever compile — confirmed from
`draft_buckets.txt`, which lists exactly those three for the cutoff arm. The uncut arm at the
same ladder reached the unbucketed range and logged draft batches of 34, 37, 38, 42, 45, 46, 50,
54, 55, 59, 62, 63 and 64 — **every one of them its own `torch.compile`.**

**Correctness across the boundary.** The requirement is that a request in flight when drafting
switches off is not corrupted. Three pieces of evidence, strongest first:

1. **The two halves cannot disagree in the dangerous direction, by construction.** Both read the
   batch of the *same* step X — `_update_after_schedule` decides how many spec slots step X+1
   gets, and `propose()` produces the drafts that fill exactly those slots. The one thing that
   can differ is that the proposer counts rows that emitted a token while the scheduler counts
   every request it scheduled, and the former is a *subset* of the latter. So the proposer is
   only ever **more** willing to draft than the scheduler is to schedule; "slots allocated that
   the drafter did not fill" is unreachable. And if it were reachable, the skip returns a
   full-width tensor of zeros — drafts that lose, not a short row that would leave the previous
   step's tokens to be scattered.
2. **Below the cutoff it is bit-identical.** 4B, concurrency 1, cutoff arm vs uncut chain arm:
   **0 of 70 sequences differ.**
3. **Above it the divergence is the tie lottery, not corruption.** At concurrency 64 with the
   cutoff fully engaged, 48/320 sequences differ from base — inside the 10.0–17.1% null band
   measured by running the base arm against *itself* across concurrencies. Inspecting the
   divergent pairs: both continuations are coherent, grammatical and on-topic, branching
   mid-sentence at a plausible token. Corruption from a stale draft scatter does not look like
   that; it looks like token salad.

   Note the same-concurrency base-vs-base null (0.0% up to concurrency 8) is the **wrong** floor
   here and would flag this as a regression. It is a batch-shape-*preserving* null, and both
   speculating and de-speculating change the batch shape the target sees. `ladder_diff.py`
   computes the batch-shape-changing null instead, and says why.

**Read the ratios at concurrency 64 with a ±10% eye.** The two cutoff arms differ there (0.95x
at N=4, 0.85x at N=1) although both have speculation fully off at that batch, so that spread is
run-to-run variance on a shared box, not an N effect. The claim these numbers support is
"restored to roughly the no-speculation baseline", not a precise 5% deficit.

**The 4–6% that is left is not the drafter.** With the cutoff engaged there is no drafting at
all above N, yet the arm is still ~0.95x. About half of it is cudagraph dispatch: a step whose
requests have no draft tokens has query length 1, but a spec-configured engine sets
`uniform_decode_query_len = 1 + num_spec_tokens`, so that step cannot match a captured FULL
decode graph and falls to PIECEWISE, while the base arm (`num_spec_tokens = 0`) matches its own
FULL graph at query length 1. Measured with `CF_CGMODE=PIECEWISE` on the **base** arm — the same
engine, the same absence of speculation, only the graph mode changed:

| conc | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| base, FULL_AND_PIECEWISE | 139.0 | 498.8 | 952.9 | 1764.4 | 2968.3 | 4518.6 |
| base, PIECEWISE | 136.9 | 487.9 | 933.7 | 1727.0 | 2915.1 | 4487.6 |
| | 0.985x | 0.978x | 0.980x | 0.979x | 0.982x | 0.993x |

So ~2 points of the residual are a property of turning speculation off *inside a speculative
engine*, not of how it is turned off. The rest is the steps that still speculate: the decode
batch straddles N during ramp-up and drain, so at nominal concurrency 8 the audit records
batches of 3, 4, 7 and 8 in the same phase. That mixing is also visible in the acceptance —
the cutoff arm reports 2.00–2.17 against the uncut arm's 1.85, because the only steps that draft
are the low-batch ones.

### The TREE arm: stable under concurrency now, and it crosses much sooner than chain

The tree concurrency crash is fixed, so the tree arm can finally be laddered. Two results, and
they point opposite ways.

**It is stable.** 4B tree, concurrency 1 → 64, 982 requests: **zero errors at every level**, and
acceptance flat at 2.395–2.411 throughout. That is a materially wider verification than the
concurrency-8 the fix was signed off at, and nothing in it degrades.

**And it is a batch-1 technique, more sharply than chain is:**

| conc | base | tree | vs base | accept |
|---|---|---|---|---|
| 1 | 139.0 | 187.5 | **1.35x** | 2.395 |
| 2 | 256.8\* | 268.3 | 1.04x | 2.41 |
| 3 | ~378\*\* | 349.9 | 0.93x | 2.41 |
| 4 | 498.8 | 426.2 | 0.85x | 2.411 |
| 8 | 952.9 | 560.7 | 0.59x | 2.402 |
| 16 | 1764.4 | 623.4 | 0.35x | 2.400 |
| 32 | 2968.3 | 625.4 | 0.21x | 2.406 |
| 64 | 4518.6 | 624.3 | **0.14x** | 2.399 |

The chain arm is at parity at concurrency 4; the tree is already at 0.85x there and **1/7th of
base at 64**. That is not a defect in the tree — it is arithmetic. The scheduler hands the target
`K+1` query positions per request per step, and the 8×5 tree's `K+1` is **42 against a chain's
6**: at concurrency 4 the tree asks the target to verify 168 positions where the chain asks for
24. Acceptance being flat across the whole ladder is the proof that none of this is a drafting
problem.

Note also the plateau: from concurrency 16 the tree arm pins at ~625 tok/s and stops responding
to load at all, because a tree engine gets **117,537 KV tokens against the base engine's
1,112,818** — the lookahead reservation scales with `K`, so the tree cannot hold enough
concurrent requests to use the GPU.

With `CF_SPEC_MAX_BATCH=auto` (which resolves to **N=2** here, printed with its provenance):

| conc | base | tree | tree + cutoff | |
|---|---|---|---|---|
| 1 | 139.0 | 187.5 | **187.5** | identical — the whole 1.35x survives |
| 4 | 498.8 | 426.2 (0.85x) | 453.1 (**0.91x**) | |
| 8 | 952.9 | 560.7 (0.59x) | 642.9 (**0.67x**) | |
| 16 | 1764.4 | 623.4 (0.35x) | 1257.8 (**0.71x**) | 2.0x the uncut arm |
| 32 | 2968.3 | 625.4 (0.21x) | 1537.2 (**0.52x**) | 2.5x the uncut arm |
| 64 | 4518.6 | 624.3 (0.14x) | 1426.2 (**0.32x**) | 2.3x the uncut arm |

So for the tree the cutoff is **necessary but not sufficient**: it is worth 2.0–2.5x under load
and it costs nothing at batch 1, but it leaves the arm at 0.32–0.71x of base rather than the
0.94–0.96x the chain reaches. Do not serve a 4B tree above concurrency ~2 on the strength of the
cutoff alone.

**27B tree, now laddered — and it is the case that shows why deriving is not good enough:**

| conc | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| base | 26.3 | 51.0\* | 100.2 | 190.8 | 368.1 |
| 27B tree | 48.7 | 79.1 | 107.8 | 107.2 | 106.6 |
| | **1.85x** | **1.55x** | **1.08x** | 0.56x | **0.29x** |

**`(5120, 41) -> 3` is now measured and in `_AUTO`**, against a *derived* 2 that would have given
up the 1.08x. But read the third column carefully, because **the concurrency row and the decode
batch are not the same number here**, and the threshold is a decode-batch threshold:

| | conc 1 | conc 2 | conc 4 | conc 8 |
|---|---|---|---|---|
| decode batch actually run | 1 | 2 | **3** | **3** |

`CF_BATCH_AUDIT` on this ladder gives `decode batch B hist {1: 4251, 2: 2636, 3: 6613}` — **B
never reaches 4 at any offered load.** A 27B tree engine has **27,185 KV tokens** and cannot
admit a fourth request, so concurrency 4 ran three requests and queued the rest (TTFT 2.4 s), and
concurrency 8 ran *the same three* (TTFT 11.6 s). N is therefore 3 and not 4: batches 1, 2 and 3
are measured and all above parity, and batch 4 was never observed, so recording 4 would record a
measurement this ladder did not produce. On this engine the cutoff correctly never fires.

**The 0.56x at concurrency 8 is not a batch-size crossing and must not be read as one.** Same
decode batch as the 1.08x row, six times the queue. It is the `--speculative-config` KV
reservation, and no decode-batch threshold can address it — cutting at a batch of 3 would only
throw away the 2.50 acceptance that is carrying those three requests. `ladder_report.py` now
prints the largest decode batch each arm reached, next to the ladder it was driven with, so this
gap between offered load and decode batch is visible rather than something you have to know to
go and grep for. (\* concurrency-2 base from the earlier `27b_base` ladder.)

### 9B: both arms laddered, and the two thresholds are not the same number

`logs/bench_serve/9b_{base,chain,tree}_thr`, one RTX PRO 6000, `serve_ladder.sh`, each arm
against the no-speculation baseline on the same GPU. Drafter `Flow-Drafter-9B-v2`, confirmed
from the loaded path's hash `ed77e698e501423858effa9a596908a700876be7`.

| conc | 1 | 2 | 3 | 4 | 6 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|---|
| base | 83.7 | 157.7 | 236.8\* | 315.8 | 474.1\* | 632.4 | 1123.5 | 1983.8 | 3127.6 |
| chain | 107.3 | 188.5 | — | 336.1 | — | 578.4 | 908.8 | 1231.3 | 1463.2 |
| | **1.28x** | **1.20x** | | **1.06x** | | 0.91x | 0.81x | 0.62x | 0.47x |
| tree | 123.7 | 191.3 | 258.9 | 307.2 | 383.8 | 419.8 | 464.0 | 446.5 | 445.8 |
| | **1.48x** | **1.21x** | **1.09x** | 0.97x | 0.81x | 0.66x | 0.41x | 0.23x | 0.14x |

**Chain crosses between 4 and 8 → N=4. Tree crosses between 3 and 4 → N=3.** The tree's
concurrency 3 and 6 points were run separately (`9b_tree_mid`) for exactly this reason: 1.21x at
2 and 0.97x at 4 do not say where in between the crossing sits, and the answer is what separates
N=3 from the N=2 the 4B tree would suggest. Acceptance is flat across both ladders — 1.88–1.92
chain, 2.40–2.45 tree — so, again, none of the decline is drafting quality.

The **width rule would have derived 8 for the 9B chain**, which is to say it would have kept
speculating at a measured 0.91x. That is the case for measuring rather than deriving, stated as
a number rather than a principle.

\* base at concurrency 3 and 6 is interpolated, and here that is sound rather than a caveat: the
base arm's **per-request** throughput is flat at 78.8–79.1 tok/s from concurrency 2 through 8, so
base(3) and base(6) are 3× and 6× that to within a fraction of a percent. The tree points at 3
and 6 are measured. Note also that the 9B tree engine reaches a decode batch of only **17**
(131,606 KV tokens), so its concurrency 32 and 64 levels are the admission queue — the same
plateau the 4B tree shows, and no threshold applies to them.

**With the cutoff — run with `CF_SPEC_MAX_BATCH` UNSET, i.e. the shipping default, which resolved
`4` and `3` from the table and printed that it had:**

| conc | 1 | 2 | 3 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|---|
| chain uncut | 1.28x | 1.20x | — | 1.06x | 0.91x | 0.81x | 0.62x | **0.47x** |
| chain + cutoff (N=4) | **1.28x** | **1.20x** | — | **1.07x** | 0.94x | 0.92x | **0.96x** | **0.96x** |
| tree uncut | 1.48x | 1.21x | 1.09x | 0.97x | 0.66x | 0.41x | 0.23x | **0.14x** |
| tree + cutoff (N=3) | **1.48x** | **1.22x** | **1.09x** | 0.93x | **0.90x** | **0.92x** | — | — |

**At and below N the two arms are the same measurement**, which is the property the flag
promises: chain 107.5 vs 107.3, 188.8 vs 188.5, 336.4 vs 336.1 tok/s; tree 123.7 vs 123.7, 191.7
vs 191.3, 259.2 vs 258.9. Every pair is inside 0.3% — and concurrency 1 in particular is
untouched, so the published batch-1 numbers cannot move.

**The two halves are in exact lockstep, and the audit says so arithmetically** rather than by
argument. 9B tree cutoff arm, cumulative over the whole ladder: `11530 drafted, 12969 skipped by
CF_SPEC_MAX_BATCH`, against a decode-batch histogram of `{1: 5407, 2: 3676, 3: 2447, 4: 5724,
6: 3, 7: 94, 8: 4514, 11: 4, 13: 3, 15: 21, 16: 2607}`. The B ≤ 3 bins sum to **11,530** and the
B ≥ 4 bins to **12,969** — the drafter ran on exactly the steps at or below N and on no others.

**The 9B chain cutoff is the best result the flag has**: 0.47x → **0.96x** at concurrency 64,
against the 4B chain's 0.95x and the 27B chain's 0.52x. And unlike the 4B tree, the 9B *tree*
cutoff also reaches ~0.92x rather than stalling in the 0.67x range, because a 9B tree engine
holds 131,606 KV tokens against the 4B tree's 117,537 while serving a model whose base
throughput is 3.6× lower — so the KV reservation binds much later relative to the load.

**One honest cost at the boundary.** At concurrency 4 the tree cutoff reads 0.93x where the uncut
arm reads 0.97x: N=3 cuts a batch of 4, and at that batch turning speculation off is not yet a
win. It is the first batch above N, it is a ~4% effect on one rung, and the alternative (N=4)
would keep speculating at a measured 0.97x instead. Both are inside the band the ladder can
resolve; N=3 is the choice consistent with the rule used for the other five entries.

**The cutoff helps the tree but cannot bring it to parity, and that is the interesting part.**
The chain cutoff reaches 0.94–0.96x; the tree cutoff stalls at ~0.67x even though above N it is
doing no drafting at all. The reason is the table above: a 4B tree engine gets **117,537 KV
tokens against the base engine's 1,112,818**, so it cannot hold enough concurrent requests to
use the GPU whether it speculates or not. `CF_SPEC_MAX_BATCH` removes the drafting and verify
cost; it cannot give back memory that `--speculative-config` reserved before the first request
arrived. **That reservation is the next blocker, and it now gates both arms.**

\* concurrency 2 base is from the earlier `4b_base` ladder, not `base_lad6`, which did not run
that level. \*\* concurrency 3 base is interpolated between 2 and 4; the tree point is measured.
Both are flagged because the 2 and 3 rows are what set the tree threshold, and a threshold set
against an interpolated baseline should say so.

**Consequence for the threshold:** it cannot be keyed on model size alone. `CF_SPEC_MAX_BATCH=
auto` keys on `(hidden_size, K+1)` and gives the 4B tree **N=2** — the last batch still above
parity — against the 4B chain's N=4.

### 27B: the crossing is at 16, and above it the cutoff cannot help

| conc | base | chain | vs base | + `MAX_BATCH=4` (wrong N) | + `MAX_BATCH=auto` (N=16) | vs base |
|---|---|---|---|---|---|---|
| 1 | 26.3 | 42.7 | **1.62x** | 42.8 | 42.5 | **1.62x** |
| 4 | 100.2 | 140.5 | **1.40x** | 140.7 | 140.9 | **1.41x** |
| 8 | 190.8 | 253.5 | **1.33x** | 190.4 *(1.00x)* | 253.5 | **1.33x** |
| 16 | 368.1 | 395.6 | **1.07x** | 351.3 *(0.95x)* | 397.1 | **1.08x** |
| 32 | 630.7 | 501.7 | 0.80x | 513.1 | 518.9 | 0.82x |
| 64 | 1011.7 | 503.4 | **0.50x** | 521.2 | 521.4 | 0.52x |

Two things to read off this. **N=4 is the wrong threshold at 27B** — the `MAX_BATCH=4` column
throws away 1.33x at concurrency 8 and 1.07x at 16 for nothing, which is why `auto` keys the
threshold on the target rather than shipping one number. At N=16 the arm reproduces the uncut
column exactly at and below the cutoff (1.62x / 1.41x / 1.33x / 1.08x), which is the no-op
property the flag promises.

And **above 16 the cutoff barely moves the number** (0.80x → 0.82x, 0.50x → 0.52x), because at
27B the cost there is not drafting at all. `CF_BATCH_AUDIT` shows the 27B spec engine's decode
batch **never exceeds 28 at concurrency 64**, while the base engine runs all 64: enabling
`--speculative-config` more than halves the KV cache the engine gets for the same
`--gpu-memory-utilization` (27B: 332,946 → 150,845 tokens; 4B: 1,112,818 → 600,425), and on a
hybrid model that is a hard cap on concurrent requests. Raising `--gpu-memory-utilization` to
0.93 does lift it (193,783 tokens, 32 running requests) and then **OOMs**, because
`FlowDrafterProposer._build()` runs lazily on the first request — *after* vLLM has already sized
the KV cache against memory the drafter had not yet claimed.

**That is a separate production blocker from this one, and it is the binding one at 27B above
concurrency 16.** `CF_SPEC_MAX_BATCH` cannot address it: the allocation happens before any
request exists. At 4B chain it never binds (600,425 tokens is ~1,200 concurrent requests of this
dataset's length), which is why the 4B chain cutoff arm reaches 0.95x and the 27B one does not.

The tree makes it worse again, because the lookahead reservation scales with `K`:

| engine | KV tokens at the same `--gpu-memory-utilization` | vs base |
|---|---|---|
| 4B base | 1,112,818 | — |
| 4B chain (K=5) | 600,425 | 0.54x |
| 4B tree (K=41) | 117,537 | **0.11x** |
| 27B base | 332,946 | — |
| 27B chain (K=5) | 150,845 | 0.45x |
| 27B tree (K=41) | 27,185 | **0.08x** |

A 27B tree engine holds ~55 requests of this dataset's length against the base engine's ~680.
`serve_ladder.sh` refuses to ladder past what the KV can hold rather than report the admission
queue as the arm's throughput — which is how this was found.

**The tree arm does not survive concurrency ≥ 2 under `vllm serve`,** in either scheduling
mode. Bisected on 4B:

| tree config at concurrency 2 | result |
|---|---|
| default (`CF_ASYNC_SPEC=1`, `--async-scheduling`) | device assert `indexSelectSmallIndex: srcIndex < srcSelectDimSize` → `EngineDeadError`, server gone |
| `CF_CUDAGRAPH=0` (no draft cudagraph) | **same assert** — so it is not the captured graph, it is the index math |
| `CF_ASYNC_SPEC=0` + no `--async-scheduling` | **hangs**: `Running: 1 reqs`, 0 tok/s, engine idle at 12% CPU with the bucket-2 graph already captured |

**Reproduced identically at 27B** (same assert, same concurrency), so it is the tree serve path
and not a size-specific shape.

Batch > 1 *was* verified before — **offline**, with a static batch that starts and finishes
together. Continuous batching adds what that never exercised: requests joining and leaving
mid-flight, `InputBatch.condense()` re-mapping rows, slot migration. **Do not serve the tree
arm above concurrency 1.** The chain arm is unaffected and was driven to concurrency 16 at
both sizes without an error.

Note the two spec flags are coupled and cannot be varied independently on the fork: with
`CF_ASYNC_SPEC=0` the fork's `config/vllm.py` refuses `--async-scheduling` for `custom_class`
outright, so a spec arm must turn both off together (`CF_NO_ASYNC_SCHED=1` in `bench_serve.sh`).

### Correctness under concurrency — and why a token diff cannot be a pass/fail gate here

**vLLM is not batch-invariant.** The BASE arm, with no speculation anywhere, agrees with its own
concurrency-1 run on only 60/70 sequences at concurrency 16 (4B) and 62/70 (27B) — 256-token
greedy generations, same prompts, same server. This is the fp16-tie hazard of section 4, and
continuous batching multiplies it, because the batch a request is decoded in changes the
kernels it goes through.

So the honest test is a *rate against a null*, and the null is the base arm against itself:

| | chain vs base, per concurrency | null (base vs base across concurrency) |
|---|---|---|
| 4B  | 10.0 – 17.1% of sequences differ | 7.1 – 14.3% |
| 27B | 10.0 – 14.3% | 5.7 – 15.7% |

**The chain arm is inside the null band at every concurrency at both sizes** — there is no
evidence of a losslessness failure under continuous batching, and equally, a token diff at
n=70 could not have detected a small one. `bench_serve_diff.py --null <rate>` implements
exactly this comparison; without `--null` it reports and refuses to judge.

`InputBatch.condense()` — the slot-identity hazard — is handled: the proposer keys the ring on
`req_id` STRINGS, distinguishes migration (move the history) from recycle (reset it), snapshots
before moving because two requests can swap slots in one `condense()`, and prunes `_req_slot`
and `ctx_hist` against the live set every step. No leak found on the default path. (The
`CF_ORACLE` diagnostic dicts are keyed by `req_id` and never pruned — diagnostics only, off by
default, but do not leave it on in a long-running server.)

### `CF_TREE_CONV_NARROW` is ON by default, and the scheduler hole that kept it off is closed

The tree's GDN **conv** state used to be allocated at `conv_kernel-1 + num_speculative_tokens`
columns per slot, because that is what the CHAIN spec path needs. The tree conv kernel
(`tree_gdn._tree_conv_kernel`) keeps a per-NODE window and reads and writes only columns 0, 1, 2,
so 41 of 44 columns × 126 slots per request were never touched — and the conv width sets the mamba
page size, which sets the engine's **attention block size**. `CF_TREE_CONV_NARROW` reclaims them.

**That is a correctness lever, not a tuning one.** At the 8×5 tree the widened state moves the 4B
engine's attention block from base's 528 to 688, and *at 688 the tree deterministically decodes
different text from base*. So this flag decides whether the tree is lossless. It nevertheless
shipped OFF, because narrowing removes the sliding window the **stale-tree fallback** reads, and
that fallback could not be shown to be unreachable. It now can be.

#### What a stale row is, and the two places vLLM manufactures one

The tree's *shape* travels out of band: the proposer registers `req_id -> (tokens, parents,
depths)` and `_prepare_inputs` looks it up by req_id **and length**. A row whose lookup misses is a
STALE ROW — chain parents `[-1, 0, 1, …]` are synthesised for it — and a step in which *every* spec
row is stale has `TreeStep.branching == False`, which takes the GDN layer off the tree conv kernel
and onto the wide-window `causal_conv1d_update`. On a narrowed engine that call would read 41
columns that were never allocated, so it raises instead.

A row is stale exactly when the width the scheduler gave it is not the width the proposer drafted
it at. Stock vLLM 0.25.1 produces that in two places, both in `Scheduler.schedule`:

1. **`pad_spec_decode`.** A request scheduled out of the WAITING queue with `num_new_tokens == 1`
   is padded to the uniform spec width and given `[-1] * self.num_spec_tokens` slots — the
   engine's **static** K, not the cut `num_spec_tokens_to_schedule`, for a request that was not in
   the previous step's batch and so was never drafted. With `CF_SPEC_MAX_BATCH` engaged every
   RUNNING request has zero slots, so the newcomer is the *only* spec row and the step is all
   stale. vLLM itself disables this branch once K is dynamic (`… and self.dynamic_sd_lookup is
   None`); it stayed live for us only because our cutoff writes `num_spec_tokens_to_schedule` in
   `_update_after_schedule` rather than configuring `num_speculative_tokens_per_batch_size`, which
   we must not do because it downgrades `cudagraph_mode` engine-wide.
2. **Truncation.** `num_new_tokens` is clamped to `max_model_len - num_computed_tokens - 1` and
   `spec_token_ids` is then *shortened* to whatever survived, so any tree request that comes within
   `num_spec_tokens` of the context limit is served a SHORT tree the registry cannot match. At a
   decode batch of 1 that single row is the whole step. **This one owes nothing to the cutoff and
   fires at `CF_SPEC_MAX_BATCH=0`**; it was not in the previous analysis.

#### `num_new_tokens == 1` is common, and the old reproduction was looking for the wrong thing

The previous reproduction primed a prompt into the prefix cache and re-sent it under load, and
recorded zero stale rows. Neither reason was "the hole is closed":

- The prefix-cache hit is **block aligned** (`get_computed_blocks` caps the hit at `num_tokens - 1`
  and then rounds down to a block boundary), so a fully cached prompt only lands on
  `num_new_tokens == 1` when its length is exactly `k·block_size + 1`. At a 528-token block that is
  a 1-in-528 chance for an arbitrary prompt, and the script used one arbitrary prompt.
- vLLM turns prefix caching **off** for this hybrid + speculative engine anyway
  (`enable_prefix_caching=False` in the startup config line).

The trigger that needs neither is a **one-token prompt**: `num_new_tokens = num_tokens −
num_computed_tokens` is `1 − 0` for any fresh single-token request, on any engine, cache or no
cache. `vllm/cnd_stale_repro.py` now drives that, the block-aligned cache hit, and the
`max_model_len` truncation; `vllm/cnd_reach.sh` runs it against three server arms.

#### The fix: schedule the full drafted width, or schedule none

`chained_flow.vllm_plugin.batch_cutoff.install_slot_guard()`, installed unconditionally from the
plugin entry point — it is *not* gated on the cutoff, because hole 2 fires without it:

1. Give the scheduler instance an **identity `dynamic_sd_lookup`** (`self.num_spec_tokens` at every
   batch). vLLM's own padding guard stands down; the only other reader of that table seeds
   `num_spec_tokens_to_schedule` with exactly what it computed before, so the single place that
   decides K remains `_update_after_schedule`. It is set on the *instance*, so `SpeculativeConfig`
   is untouched and `_maybe_override_dynamic_sd_cudagraph_mode` — which reads the config, not this
   attribute — cannot downgrade the engine to PIECEWISE.
2. Before each `schedule()`, **drop `spec_token_ids`** on any RUNNING request whose full width
   would not survive the clamps. The `max_model_len` clamp is reproduced exactly; the token-budget
   one conservatively, because this walk cannot know which requests the real loop will skip —
   over-estimating costs a draft, under-estimating leaves a truncated tree.

A K-schedule rung of K=1 is the same failure by configuration (1 slot for a 40-node draft), so
`ladder()` now refuses a partial-K rung on a tree engine. `CF_SPEC_SLOT_GUARD=0` restores stock
behaviour, and exists so the old behaviour can be *measured*.

#### Why the fallback is unreachable — the argument, not the absence of a reproduction

`scheduled_spec_decode_tokens` is written in exactly two places and one of them is now dead. So on
any step with at least one spec row, every such row belongs to a request that

- had `spec_token_ids` set by `_update_after_schedule` in the previous step ⇒ was scheduled then
  and was **not** a prefill chunk;
- was therefore **not discarded** — `discard_request_mask` *is* the not-last-prefill-chunk mask —
  so it is in the proposer's `rows`;
- was scheduled a **non-zero** width ⇒ that step was under the cutoff ⇒ the proposer drafted, since
  `rows` is a subset of the scheduled requests and the proposer is therefore only ever *more*
  willing to draft than the scheduler is to schedule;
- was drafted, hence registered, at exactly `draft_width` — which is the width it is then
  scheduled, because the truncation branch can no longer shorten it.

Every spec row hits the registry. No all-stale step, no fallback. And the residual is **loud**: if
some path not covered above ever did produce one, the GDN call site raises with an actionable
message. That asymmetry is what makes the default flip right — narrow fails loudly and has never
been seen to fail, while wide fails **silently** and demonstrably does.

One correction to the record while we are here. `CF_TREE_FORCE_STALE=1` being 7/7 established that
the fallback is correct *when every step takes it*, i.e. against a conv state the chain path has
itself been maintaining. A **single** stale step in the middle of tree steps would read a window
the tree conv kernel never wrote — it writes columns 0, 1, 2 of the node's own slot and nothing
else — so the wide window is not obviously load-bearing there either. Making the step unreachable
is the better answer in both directions.

#### Measured — the hole is real, and the fix closes it

`vllm/cnd_reach.sh <gpu> 4b <tag> [env…]` serves the 4B tree arm with `CF_TREE_FALLBACK_LOG=1`
and drives `cnd_stale_repro.py` at it. The **positive control runs first on purpose**: an arm that
records zero stale rows proves nothing unless the same workload can be shown to produce them.

| arm | `CF_SPEC_SLOT_GUARD` | conv | `STALE STEP` lines | result |
|---|---|---|---|---|
| `ctl` | 0 (stock) | wide | **2** (`6/6 spec rows`) | every row stale — exactly the shape that takes `TreeStep.branching` to False |
| `ctl2` | 0 (stock) | wide | **5** (`6/6`, then four `1/1`) | reproduces; `1/1` is a single spec row that *is* the whole step |
| `haz` | 0 (stock) | narrow | 1 (`6/8`) | **the engine dies** — `Triton Error: device-side assert`, `EngineDeadError`, clients get 500s |
| `fix` / `fix2` | on (default) | narrow | **0** | 126 requests each, zero errors |
| `fixcut2` | on, `CF_SPEC_MAX_BATCH=2` | narrow | **0** | 126 requests, zero errors |
| `fixnocut` | on, `CF_SPEC_MAX_BATCH=0` | narrow | **0** | 126 requests, engine healthy |

Zero lines means zero stale steps, not zero logging: `note_spec_step` prints on the *first*
occurrence with `flush=True`, and every arm logged `[cf-tree-fallback] accounting ENABLED in this
process` from the engine core. The counter also covers exactly the steps that *can* be stale — the
canonical GPU-tree hand-off skips it, and that path requires every row to carry exactly `_Nc` spec
tokens *and* the published draft-time `req_id`s to equal this step's, which neither a padded
newcomer (41 slots, not in the published set) nor a truncated request (<40 slots) can satisfy.

The `haz` row is the one worth reading twice. Six **one-token prompts** sent into a loaded tree
server were each handed 41 spec slots nothing had drafted — and that is not merely the *all*-stale
step the conv width was argued about: a **partial**-stale step, where `branching` survives and the
tree GDN kernels do run, still killed the engine. So on a narrowed engine a stale row is fatal
*loudly*, which is the property that decides the default: the failure mode of narrowing is a dead
engine and an HTTP 500, and the failure mode of the wide default is silently different text.

#### Measured — losslessness, 7 domains × 256 greedy tokens, batch 1, 3 repeats

Reference is the **async-off base arm**, never another spec run, and the null comes first: base
against itself is **3/3 identical at 4B and 9B and 5/5 at 27B** in this session. (`base_aon`,
async scheduling ON, is *not* a valid reference — it flips 4B domain 4 at token 63 against
async-off base, reproducibly, on both repeats.)

| size | `CF_TREE_CONV_NARROW=1` (new default) | `=0` (old default) | base null |
|---|---|---|---|
| 4B  | **7/7, 7/7, 7/7** — and byte-identical run to run | 5/7, 5/7 — d3 @ tok 129 both times | 3/3 identical |
| 9B  | **7/7, 7/7, 7/7** — and byte-identical run to run | 6/7, 6/7 — d1 @ tok 44 both times | 3/3 identical |
| 27B | 7/7, 6/7, 6/7, 7/7, 7/7 — d4 @ tok 161 | 7/7, 6/7 — d4 @ tok 161 | 5/5 identical |

So narrowing makes the 4B and 9B tree **lossless and deterministic** — the widened block size *was*
the old default's divergences — while at 4B it is also *more* faithful to async-off base than
vLLM's own async base arm is (7/7 against `base_aon`'s 6/7).

**27B is unchanged by the flag and is not clean.** The 27B tree lands on either side of the
domain-4 / token-161 fp16 tie across its own repeats (3 of 5 identical to base, and r2/r3 differ
from r1 at exactly that position), and the WIDE arm does the same (1 of 2). Five async-off base
repeats never flip it, so **this is not inside a null established here** — it is a real
nondeterminism in the 27B tree arm, it is one token at one tie, and narrowing neither causes nor
fixes it. Do not read the 27B row as a losslessness result for either setting of the flag.

#### Measured — batch-1 throughput, pooled over 7 domains

| size | base (async off) | base_aon (deployment baseline) | tree, narrow | tree, wide |
|---|---|---|---|---|
| 4B  | 127.8 | 140.3 | **187.5 (1.336×)** | 177.9 (1.268×) |
| 9B  |  78.4 |  82.6 | **117.7 (1.426×)** | 116.8 (1.414×) |
| 27B |  25.8 |  26.2 | **46.6 (1.777×)**  |  46.6 (1.778×) |

Narrowing is worth +5.4% at 4B (the +30% KV blocks buy nothing at batch 1; the block-size change
does), +0.8% at 9B and nothing at 27B — and it is the lossless arm at all three. Every figure
reproduces the recorded headline (187.7 / 117.4 / 46.5 against 140.0 / 82.6 / 26.2).

**The chain arm is untouched**, which is the check that matters for a guard that installs on
*every* speculative engine and not only the tree: 4B chain reads **157.9** against the recorded
157.8. Both halves of the guard are unreachable at batch 1 by construction — padding requires a
non-empty running batch, truncation requires being within K of `max_model_len` — so this is a
measurement of something that was already provable.

#### Measured — the 4B concurrency ladder, `serve_ladder.sh`, narrowing on

| conc | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|
| base | 138.9 | 498.5 | 936.1 | 1763.8 | 2970.1 | 4525.8 |
| tree | 187.0 | 382.7 | 887.9 | 1653.0 | 1973.0 | 1983.2 |
| | **1.35×** | 0.77× | 0.95× | 0.94× | 0.66× | 0.44× |

Every rung reproduces the recorded narrow ladder (1.34 / 0.95 / 0.93 / 0.93 / 0.63 / 0.43) except
**c=4, which reads 0.77× against 0.95×**, with a 565 ms TTFT against 74 ms at c=8.

**That is not the slot guard, and it was worth checking rather than explaining away** — the guard
*does* take a step that admits a request into a running decode batch off the FULL cudagraph, and
c=4 is where joins are most frequent relative to the work. The low rungs, run twice with the guard
and once without:

| conc | base | tree, guard ON | tree, guard OFF (`CF_SPEC_SLOT_GUARD=0`) |
|---|---|---|---|
| 1 | 139.0 | 187.1 (1.35×) | 186.9 (1.34×) |
| 4 | 486.0 | 384.7 (0.79×), TTFT 550 ms | 389.6 (0.80×), TTFT 518 ms |
| 8 | 936.6 | 888.1 (0.95×) | 895.9 (0.96×) |

The guard costs 1.3% at c=4 and 0.9% at c=8, both inside run-to-run noise, and the c=4 deficit is
there with stock scheduler behaviour too. So **c=4 needs its own attribution** — the recorded run
read 466.6 tok/s there at a 62 ms TTFT and this engine reads 385–390 at ~530 ms, which is a change
somewhere else — but nothing on this page's ladder is attributable to the spec-slot guard.

### Sampled (`temperature > 0`) requests

- **chain: fine.** Verified against a live 4B server: a sampled request alone, a sampled
  request sharing the batch with a greedy one, and the server afterwards — all 200 OK, health
  green throughout. Chain mode never consults `SamplingMetadata`; vLLM's own rejection
  sampler handles the sampled case.
- **tree: ONE sampled request kills the server, permanently.** Measured against a live 4B
  tree server: greedy request 200 OK → one `temperature=0.8` request → **500**, `/health`
  **503**, and every subsequent *greedy* request 500 forever. The engine log carries our
  intended message —

  > `RuntimeError: chained-flow tree mode requires greedy sampling (temperature=0, no
  > logprobs, no penalties)… Run with VLLM_SPEC_TREE=0…`

  — but "fail fast with an actionable message" was designed for the offline `LLM()` path,
  where the exception reaches the caller. Under `vllm serve` it is raised inside the engine
  core's step loop, so it becomes `EngineDeadError` and takes the process down. **Any client
  that sends `temperature > 0` to a tree-mode server denies service to every other client.**
  A tree-mode server is therefore only deployable behind something that rejects non-greedy
  requests before they reach vLLM.

  `CF_NONGREEDY_CHAIN=1` opts into a per-step chain fallback that is **still shape-broken**
  (fixed-shape ring of `keep*depth+1` vs a chain of `draft_length`). Note `all_greedy` is a
  property of the whole batch, so under continuous batching *one* sampled request makes the
  step non-greedy for every request sharing it.

#### `CF_TREE_GREEDY_GUARD=1` — the mitigation (default OFF)

`chained_flow/vllm_plugin/greedy_guard.py`, installed from the existing `vllm.general_plugins`
entry point. In **tree mode only**, it wraps `AsyncLLM.add_request` — the single funnel every
serving front end goes through — and raises `ValueError` for a request whose `SamplingParams`
would fail the tree's preconditions. That happens **in the API-server process**, so the engine
core never sees it and vLLM maps it to a 4xx. Verified on a live 4B tree server:

| probe step | guard OFF (today's default) | guard ON |
|---|---|---|
| greedy request | 200 | 200 |
| **one `temperature=0.8` request** | **500 `EngineDeadError`** | **400, message says what to change** |
| `GET /health` after | **503** | **200** |
| greedy request after | **500** | **200** |
| greedy *batched with* a sampled one | **500** | **200** (the greedy one is served) |
| `GET /health` at the end | **503** | **200** |

Why reject rather than degrade to no-speculation for that request: "emit no draft" is
expressible on the synchronous path (return empty lists) but **not** under `CF_ASYNC_SPEC`,
which owes vLLM a `[num_reqs, draft_width]` GPU tensor — a short or absent draft there scatters
the previous step's tokens into `input_ids` silently, which is worse than an error. Making the
draft width dynamic is real surgery on the ring, the capture buckets and the tree hand-off.
Until that exists, refusing the request is the honest behaviour. It does **not** make tree mode
support sampling; it stops one client from taking the server away from everyone else.



**vLLM gives the base arm an engine feature that the speculative arm structurally
cannot have.** Unless you correct for it, every speedup you report is understated.

`scheduler_config.async_scheduling` defaults to `None`, and
`vllm/config/vllm.py:992-1040` then *auto-decides*:

- **base decoding** falls through the chain and is set to `True`;
- **`method="custom_class"` speculative decoding** hits the branch at `:1003` and is
  forced to `False`, logging:

  > `Async scheduling not supported with custom_class-based speculative decoding and will be disabled.`

Our proposer is a `custom_class` proposer. So the baseline overlaps its scheduler and
`_prepare_inputs` with the GPU, and our arm runs strictly serially. Measured cost of
that asymmetry (7-domain, batch 1, `maxtok` 256, 2 repeats, pooled):

| base arm | async ON (vLLM default) | async OFF | async is worth |
|---|---|---|---|
| 4B  | 140.0 | 126.7 | **+10.5%** |
| 9B  |  82.6 |  78.3 | **+5.5%**  |
| 27B |  26.2 |  25.8 | **+1.6%**  |

Monotone in model size, as expected: slower steps amortise host overhead better. It is
close to free at 27B and it is *most of the apparent deficit* at 4B.

### Consequence: quote two numbers, and label them

- **Deployment speedup** — base with async ON (vLLM's default). This is what a user
  actually gets by choosing our arm over stock vLLM. Use it in any external claim.
- **Like-for-like speedup** — base with async OFF, so both arms run the same engine.
  This is what the *speculation itself* is worth.

At the 2026-08-04 full stack: deployment 1.19 / 1.23 / 1.71 vs like-for-like
1.32 / 1.29 / 1.74 (4B / 9B / 27B). Reporting either one alone is misleading in a
different direction. **Never silently switch conventions between reports.**

The `chain` arm is *also* `custom_class`, so it is already async-off and is directly
comparable to `tree` with no correction. **Only the base arm ever needs the flag.**

### How to run it

`CF_ASYNC_SCHED=0|1` is plumbed through `vllm/test_plugin_native.py`. The `base` arm of
`vllm/bench_cf.sh` now defaults to **`0`** so the default comparison is like-for-like;
set `CF_ASYNC_SCHED=1` explicitly for the deployment number.

### Async scheduling for the spec path — `CF_ASYNC_SPEC=1` (default OFF)

**Status: shipped behind the flag, bit-exact, and it now recovers the FULL async prize —
+10.9% / +6.3% / +2.4% at 4B / 9B / 27B against the base arm's own +10.3% / +5.5% / +1.6%.**

`CF_ASYNC_SPEC=1` makes the proposer GPU-token-native, makes the tree GPU-RESIDENT, and
relaxes the guard at `config/vllm.py:971` *conditionally on that flag* (never blanket —
without it the proposer still reads the CPU token lists and would silently collapse to
accept 1.0). `vllm/bench_cf.sh` turns `CF_ASYNC_SCHED=1` on with it.

### Part 1 — the proposer becomes GPU-token-native

- `cnt` (tokens emitted) and `k0` (last committed token) come from the verify's own GPU
  output on **every** step, not just the `CF_DRAFT_EARLY` fast path — the CPU token lists
  are `[]` under async.
- `_draft_token_ids` is returned as a GPU **tensor** `[num_reqs, N]`. vLLM's async path
  never copies drafts to the host; `_prepare_input_ids` scatters that tensor into
  `input_ids.gpu`. A list-of-lists is silently ignored there.
- `valid_sampled_token_count_gpu` + `prev_sampled_token_ids` are published, so
  `update_num_computed_tokens_for_batch_change` can walk back the scheduler's optimistic
  "every draft accepted" `num_computed_tokens`. Skipping this is KV corruption, not a
  lost speedup.
- Two tree values were being read from `num_computed_tokens_cpu`, which async leaves
  optimistic: the sibling RoPE positions (now written as a shape-only **delta**
  `depth[i] - i`, provably identical in sync mode) and `ctx_lens` (now taken from the
  corrected GPU tensor). Both were silently wrong by the previous step's rejected-token
  count — **the symptom was requests terminating early, with no error**. Watch for that
  signature; it is easy to misattribute to the drafter.
- For a tree, `num_accepted_tokens` is not a count but the GDN carry-forward **column**
  (`gdn_next_accepted`); the generic async correction overwrites it with the chain value,
  so it is restored.
- `AsyncScheduler` is padded to `N`, not `N+1` — the `+1` is the spare mamba state
  column, never an emitted token.

### Part 2 — deleting the join: the tree goes GPU-resident

Part 1 alone was worth only +2.5% at 4B. The reason was measured, not guessed: the tree
PARENTS still had to reach the host before `_prepare_inputs` could build anything, so the
host blocked ~9.5 ms on the in-flight draft and then ran ~1.45 ms against a **drained**
GPU. Five host consumers of the parents had to go:

1. `tree_state.fill_static_step` — the ancestor mask (`not_anc`), a numpy walk;
2. `tree_gdn._layout_np` — the GDN `chain` table, a 41-row python loop;
3. `_calc_spec_decode_metadata` — `target_logits_indices = row_base + 1 + parents`;
4. `spec_decode_metadata.tree_parents` — an `async_tensor_h2d` from numpy;
5. `tree_state.lookup` — the registry itself.

All five are now device-side. The draft cudagraph's own output carries the parents
(`ch[:, N:]`), the proposer publishes that tensor via `tree_state.publish_gpu_tree`, and
ONE device walk table (`tree_state.device_walk`, one gather per level, ~16 launches total)
feeds both the ancestor mask and the GDN chain. Node DEPTHS never move at all: they are a
constant of the shape (level-major BFS, `depth[i] = i // keep`).

The registry (5) is bypassed by a **canonical predicate** evaluated from host-resident
scheduler metadata *before* anything touches the tree — every request carrying exactly
`N` drafts and `N+1` scheduled rows, the batch fitting the pad map, and the published
hand-off matching the current `req_ids` element-wise (the input batch can condense between
steps and silently re-map every row). It is deliberately a **superset** of the `_graphable`
predicate, so canonical ⟹ graphable and there is no path that consumes the device hand-off
and then discovers it needed the host copy; an assert makes any future drift loud.

Two guards make the bypass safe rather than merely fast:
- the hand-off is valid for exactly the ONE `_prepare_inputs` that follows it and is
  dropped unconditionally afterwards, so a stale tree can never be picked up by a later
  step whose batch happens to look the same;
- a non-canonical step still joins, and joins the staging of the step *immediately before
  it* — the host staging is deliberately **not** discarded on the canonical path. If
  nothing has been staged since the last join, `_async_register` clears the registry
  outright rather than let the length-only lookup accept an older step's tree. That is the
  same silent-malformed-draft shape as the `CF_TREE_DEPTH > draft_length-1` and
  `CF_TREE_KEEP > CF_TREE_TOPB` bugs.

### Measured

7 domains, batch 1, `maxtok` 256, pooled, full stack (`CF_TREE_KEEP=8 CF_TREE_DEPTH=5`):

| | spec sync | spec `CF_ASYNC_SPEC=1` | | base async OFF → ON |
|---|---|---|---|---|
| 4B  | 171.4 | **188.9** (+10.2%) | | 127.3 → 140.4 (+10.3%) |
| 9B  | 103.9 | **110.4** (+6.3%)  | | (+5.5%, docs) |
| 27B |  45.5 | **46.6** (+2.4%)   | | (+1.6%, docs) |

**The 9B row is a `Flow-Drafter-9B` (v1) number and the default is no longer v1.** 4B and 27B
moved to their `-v2` drafters when those were trained; the 9B line in `vllm/bench_cf.sh` was
never updated, so *every* 9B figure in this document predates 2026-08-05 and is a v1 figure.
Re-measured on 2026-08-05 (GPU 4, same protocol, 2 repeats, output-diffed against base):

| 9B, 8×5 tree | pooled accept | pooled tok/s | vs base async-off (78.6 fork / 78.5 pristine) |
|---|---|---|---|
| tree, v1 | 2.165 | 110.1 | 1.40x |
| **tree, v2** | **2.302** | **117.3** | **1.49x** |
| chain, v1 | 1.716 | 96.3 | 1.23x |
| **chain, v2** | **1.819** | **102.3** | **1.30x** |

v2 is +6.5% (tree) / +6.2% (chain) pooled over all 7 domains, and +7.8% / +6.0% over the
output-comparable 6 (tree drops `math_reasoning`, chain drops `writing` — see the token-diff
rule below). The accept gain is entirely on the free-form domains (`qa` +0.21, `summarization`
+0.30, `writing` +0.32 on the tree arm) with code/math flat-to-slightly-down, which is the
same shape as the offline `+0.09` that motivated the v2 retrain — this time the offline delta
*under*-stated the in-engine one. `bench_cf.sh` now defaults 9B to
`selimaktas/Flow-Drafter-9B-v2`; the numbers above the line are not restated because doing so
would silently mix drafters inside one table.

Run-to-run spread on the 4B sync arm across four measurements of the same code was
170.3–172.1 (~±0.5%), so treat anything under 1% here as noise. The sync arm's accept is
byte-identical to the pre-change reference on all 7 domains
(3.526 / 3.821 / 2.154 / 2.520 / 2.098 / 1.882 / 2.358): nothing on the default path
moved.

Host profile at 4B (`CF_VPROF=1`, poff 5, per engine step):

| | sync | async, join at top of `_prepare_inputs` | async, join moved late | async, **join deleted** |
|---|---|---|---|---|
| step wall | 13.66 ms | 13.76 ms | 13.20 ms | **12.30 ms** |
| where the host blocks | `parse_output` D2H, 9.0 ms | tree join, 9.8 ms | tree join, 9.5 ms | input-prep throttle, 9.0 ms |
| host work after the block | ~1.9 ms | ~1.9 ms | ~1.45 ms | ~0.6 ms |

12.30 ms against a GPU-bound estimate of ~11.7 ms, i.e. ~95% of the available headroom.
The block is now `synchronize_input_prep`, which is the *healthy* signature: the host is
running a step ahead and waiting on its own previous input-prep copies, not on the draft.

**Cudagraph**: `CF_DBG_CG=1` shows `FULL ntok=41 uniform=True` on 406 / 394 / 382 decode
dispatches at 4B / 9B / 27B against 4 `PIECEWISE ntok=41` each (the prefill-boundary
steps) — the device-side derivation does not push the step off FULL onto PIECEWISE. Check
this on any change here; that cliff was measured at +6–10 ms and would swamp the win.
`CF_ASYNC_PROBE=1` reports the same thing from the proposer's side: `device tree offered
200 | taken 198 | host joins 0`.

**Correctness**: sync vs `CF_ASYNC_SPEC=1`, **0 / 1599 token mismatches at 4B, 0 / 1366 at
9B**, and 6 of 7 domains bit-identical at 27B (the 7th diverges at index 202, the usual
fp16-tie level — base disagrees with *itself* across the same flag at index 63). Token
COUNTS are identical at every size and domain, which is the specific check for the
early-termination signature above. Accept is unchanged on every domain, and the 4B arm's
divergence from BASE is byte-for-byte the same as the sync arm's (163/1599, first at
domain 2 index 88), so the async path has not moved the relationship to base at all.

Note that under `CF_ASYNC_SPEC` the accept counter is accumulated on the GPU and drained
once per prompt set (the CPU token lists are empty every step), which can shift the
reported figure by one request-step; on a 45–67 token run that is visible in the second
decimal, on a 256-token run it is not.

### Not device-computable, and still on the host

Nothing on the draft path — all five consumers moved. What remains host-side by *choice*
rather than necessity: node depths and the sibling position deltas (constants of the tree
shape, so there is nothing to move), and the whole non-canonical fallback, which keeps the
original registry path intact for prefill, batch changes and any shape we did not draft.

### The FORK-FREE arm gets async too — measured, on a pristine vLLM

Everything above was measured on the forked vLLM. The **chain** arm is the shipping
default and runs on *unmodified* vLLM, and until 2026-08-05 its ~1.13x had never
been measured there: it was inferred from a forked-vLLM run. Two things had to
change before it could be.

1. **Something has to relax the guard on stock vLLM.** The fork patches only the
   explicit-request branch of `config/vllm.py`; the auto-decide branch at `:1007`
   still forces async off for `custom_class`. On stock vLLM that job belongs to
   `chained_flow/vllm_plugin/async_guard.py`, a **`vllm.general_plugins` entry
   point** that rebinds `NgramGPUTypes` inside `vllm.config.vllm` — the name is
   referenced at exactly those two guard sites and nowhere else — and then wraps
   `VllmConfig.__post_init__` to **raise** if the resolved `async_scheduling` is
   not `True`. Relaxing the *input* to the decision, never writing the answer
   afterwards: a post-hoc `async_scheduling = True` would skip the
   `disable_cascade_attn` branch at `:1060-1070` that reads it.
2. **`CF_ASYNC_SPEC` had to stop being a tree flag.** Requirements (1)-(3) above
   are vLLM's generic async contract; only (4), the tree-shape hand-off, needs the
   fork. `_install_async_hooks` used to import `vllm.v1.spec_decode.tree_state`
   unconditionally, which raises on stock vLLM and killed the engine at proposer
   construction — the reason the fork-free arm had never been run with async at all.

Measured 2026-08-05, pristine vLLM 0.25.1 (`RECORD` sha256: 0 modified, 0 added),
chained-flow installed as a wheel, GPU 6, batch 1, 7 domains, `maxtok` 256, pooled:

| 4B | tok/s | vs base async-on | vs base async-off |
|---|---|---|---|
| base, async **off** | 123.7 | | |
| base, async **on** (vLLM's default) | 139.6 | | |
| **chain, entry point ON** | **157.8** | **1.13x** | 1.28x |
| chain, `CF_ASYNC_SPEC=0` (guard untouched) | 148.7 | 1.07x | 1.20x |

**The 4B pooled figures above are contaminated by one domain, and the corrected
pair is below.** On domain 4 the two chain arms generated *different text* — a
1-ULP fp16 tie, resolved in opposite directions, see "The domain-4 gap" — so their
tok/s on that domain is not measuring the same work. Restricted to the six domains
whose output tokens are **byte-identical between the arms and to base**:

| 4B, six byte-identical domains | tok/s |
|---|---|
| **chain, entry point ON** | **171.8** |
| chain, `CF_ASYNC_SPEC=0` | 152.9 |

Both numbers are true and they are quoted for different purposes. **157.8 / 148.7 is
what the 7-prompt suite ran**; **171.8 / 152.9 is what the entry point is worth**,
because it is the only one of the two in which both arms decoded the same tokens.
Quote the pooled figure for suite-level reporting and the comparable-output figure
for any claim about the flag, and never mix them.

The same restriction moves the ratios, so both conventions are spelled out here
rather than left for a reader to recompute and disagree with:

| 4B, pooled over | base async-off | base async-on | chain | deployment | like-for-like |
|---|---|---|---|---|---|
| all 7 domains | 123.7 | 139.6 | 157.8 | **1.13x** | 1.28x |
| 6 comparable domains | 125.2 | 139.6 | 171.8 | **1.23x** | 1.37x |

**The external claim stays on the 7-domain pooled convention (1.13x) until someone
decides otherwise** — it is the conservative one and it is what the published suite
measures. Note that domain 4 is a tie lottery for *every* arm, not just ours: across
the same 7-domain sweep the two **base** arms differ from each other on exactly one
domain, and it is domain 4. So dropping it makes the base arms comparable too, which
is why the base async-off figure moves (123.7 → 125.2) while base async-on does not.

| 27B | tok/s | vs base async-on | vs base async-off |
|---|---|---|---|
| base, async **off** | 25.8 | | |
| base, async **on** (vLLM's default) | 26.2 | | |
| **chain, entry point ON** | **41.0** | **1.57x** | 1.59x |
| chain, `CF_ASYNC_SPEC=0` (guard untouched) | 39.7 | 1.52x | 1.54x |

Four things to read off those tables:

* **1.13x deployment at 4B is now a measurement, not an inference**, and it lands
  on the inferred value. 27B is 1.57x, above the ~1.5x expected.
* **The entry point is worth +12.4% on comparable output at 4B** (171.8 vs 152.9,
  six byte-identical domains), and +6.1% on the raw 7-domain pooled figure
  (157.8 vs 148.7). The two differ *only* because domain 4's arms decoded
  different text; the +6.1% understates the flag by half. Without the entry point
  the fork-free build is 1.07x — not nothing, but not a story either.
  At 27B it is **+3.3%** (41.0 vs 39.7) and that number needs no correction:
  **all seven** 27B domains are byte-identical between the arms and to base
  (checked, not assumed), so the pooled figure and the comparable-output figure
  are the same measurement there. Monotone in model size for the same reason the
  base arm's async prize is: slower steps amortise host overhead better.
* The fork-free synchronous chain (148.7 at 4B) reproduces the **forked** chain's
  148.0. The fork contributes nothing to the chain arm, which is the whole
  premise of shipping it fork-free.
* Both base arms reproduce the numbers in section 1 to the decimal (27B 25.8 /
  26.2), so the harness and the machine agree with the forked measurements and
  the chain figures above are not sitting on a moved baseline.

Making `_install_async_hooks` conditional is the only change on the TREE path, so
that path was re-measured on the fork as a regression check: **4B tree 189.3
tok/s** against the 188.9 recorded above, with the startup line reading
`async_spec(engine async ON, GPU-resident tree)` and every tree flag on. The join
still installs.

### The domain-4 gap — RESOLVED, and it was never a bug in our code

At 4B, **domain 4** read accept **1.261** with `CF_ASYNC_SPEC=1` and **1.652** with
it off (111.8 vs 130.6 tok/s), while the other six domains agreed to ~0.01. It
reproduced to the third decimal in a fresh process, so it was not run-to-run noise.
It was carried here for a day as an unexplained behavioural difference. It is
neither unexplained nor a difference in the *code path*:

**A single fp16 tie at output token 59 sends the two arms into different text, and
the two continuations differ enormously in how draftable they are.** Base-mode
top-2 target logits at that position:

```
tok   top1        top2        gap       gap/ULP   id1     id2
 58   19.31250    18.87500    0.43750     28.00    513     369
 59   17.03125    17.01562    0.01562      1.00   1330    1048   <-- exactly 1 ULP
 63   18.45312    18.43750    0.01562      1.00     13      11   <-- exactly 1 ULP
```

`1330` and `1048` are precisely the two tokens the two arms emit at index 59. Only
4 of 256 positions in this generation sit within 1 ULP, and two of them land at the
sentence boundary where the model decides whether to open a `<think>` block:

* the branch taken by sync **and by both base arms** opens
  `\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:** ...` — repetitive
  markdown that quotes the prompt verbatim. Accept **1.845** over its 110 remaining
  steps.
* the branch taken by the async arm continues in free prose. Accept **1.291** over
  its 158 remaining steps.

Up to the branch the arms are **bit-identical**: their emitted-count sequences match
exactly for the first 45 steps / 53 tokens, at accept **1.178 in both**.

**The flag is not what decides the branch.** Same prompt, same drafter, only the
listed knob changed:

| variant | `CF_ASYNC_SPEC` | tok/s | accept | token 59 | branch |
|---|---|---|---|---|---|
| chain K=5 (x2 processes) | off | 130.7 / 131.4 | 1.652 | 1330 | `<think>` |
| chain K=5, `CF_CUDA_BLOCK=0` | off | 115.2 | 1.652 | 1330 | `<think>` |
| chain K=4 / K=6 | off | 129.8 / 122.1 | 1.641 | 1330 | `<think>` |
| chain K=5 (x2 processes) | **on** | 111.9 / 111.9 | **1.261 / 1.266** | 1048 | prose |
| chain K=5, `CF_TWOPASS_M=2048` | **on** | 117.0 | **1.286** | 1048 | prose |
| **chain K=6** | **on** | **136.1** | **1.627** | **1330** | `<think>` |

Accept sorts perfectly by which side of the tie the run landed on and not at all by
the flag. One extra draft column takes the *async* arm from 1.261 to 1.627 and from
111.9 to 136.1 tok/s — the fastest of every variant tried.

**What ruled out a plugin bug** (this is the part to reuse; it is the same family as
the four silent async bugs above, and it is how you tell a real one from this):

* Per-step draft dump scored against the actual output: **`ACCEPTED == ORACLE` on
  202/202 async steps and 154/154 sync steps.** The number of drafts vLLM accepted
  equals the number of *our* drafted tokens that matched the emitted text, every
  step, in both arms. A misaligned, short or stale draft tensor cannot produce that
  equality — so this single check retires the whole `draft_width` / `self.N` /
  wrong-width-scatter class of hypothesis. In chain mode
  `draft_width == K == num_speculative_tokens`, so that change is a no-op there
  anyway and the `_async_draft_tensor` assert would have fired.
* The drafter's `k0` equalled the real committed token on **every** step in both
  arms — no phase error, no one-step lag.
* vLLM's own hybrid-model bookkeeping is correct under async: `num_accepted_tokens`
  equalled the previous step's emitted count on every step, **0 violations** of
  202 / 154, and `num_computed_tokens` is correctly walked back from the
  `valid_sampled_token_count` we publish. (Worth knowing *why*:
  `_prepare_inputs` fills `num_accepted_tokens` with 1 under async when
  `mamba_cache_mode != "align"`, and `update_num_computed_tokens_for_batch_change`
  then restores it from our published counts. The GDN/causal-conv state shift reads
  that value, so if we ever stop publishing `valid_sampled_token_count_gpu`
  correctly it becomes a real hybrid-state corruption. It currently does not.)
* Whole-sweep output diff, from the benchmark JSONs rather than a probe: at 4B the
  async and sync chain arms are **byte-identical on 6 of 7 domains, and both are
  byte-identical to base**. Token 59 of domain 4 is the *only* divergence in 1551
  tokens. Run the same diff on the two **base** arms and you get the same answer:
  they differ on exactly one domain, and it is domain 4 (at index 63, the other
  1-ULP tie). The prompt is a tie lottery for every arm; our arm is not special.

**The tree path and 27B are unaffected — but they are not immune.** 4B tree at
poff=4 on the fork reads 1.977 (async) vs 1.969 (sync), and all seven 27B chain
domains are byte-identical between the arms and to base. That is because those arms
happened to land on the *same* side of the tie, not because a tree or a bigger model
cannot flip one. A tree changes the verify batch shape, so it rounds differently and
lands wherever it lands; the next 1-ULP tie in front of a high-contrast branch will
bite whichever arm it likes.

The async arm remains the default. Do not quote domain-4 4B accept or tok/s as a
comparison between the flags — the two arms benchmarked different text. Use the
protocol rule in section 4 to catch the next one automatically.

### The flag report has to survive the engine-core spawn

vLLM starts its engine core in a **spawned subprocess**, and `defaults.apply()` runs in both.
By the time the child runs, every default the parent wrote is an ordinary environment variable,
indistinguishable from a caller's request — so the child reported the whole table as `explicit`
and printed

> `[cf-defaults] CF_CUDA_BLOCK was requested but CANNOT ENGAGE ... Any A/B against it is meaningless.`

about a default it had proposed itself. Three flags shouted that on a plain Quickstart run.
A banner that fires when nothing is wrong is worse than no banner: it is how the real one stops
being read. `apply()` now writes the same `CF_DEFAULTS_FROM_SHELL` provenance marker the `--sh`
emitter uses, so the child can tell our defaults from the caller's requests, and a value the
caller changes *after* the parent applied it still reads as a request.

The same bug had a second, silent half: an inherited `CF_SHORTLIST` read as explicit, which
skips the candidate search — in the one process that actually loads the drafter. The
checkpoint's own `shortlist.pt` could never have won there.

Confirm on any run that it actually engaged. Two lines, both in the engine log:

```
[cf-plugin] vLLM async-scheduling guard relaxed for method='custom_class' (vllm 0.25.1); ...
INFO ... [vllm.py:1042] Asynchronous scheduling is enabled.
[cf-defaults] ON: ... async_spec(engine async ON, chain) ... | build=stock-vllm (chain only), async-guard relaxed
```

Absence of the first line means the entry point never fired — almost always
because the package is on `PYTHONPATH` rather than installed, which creates no
entry points. That is a silent 6% and it is exactly why the plugin raises rather
than logs when the resolved value disagrees with it.

Driver: `vllm/bench_forkfree.sh <size> [maxtok]`, and `CF_PY=<venv>/bin/python`
selects the vLLM in `bench_cf.sh`.

### The published number was not the number a `pip install` produced

Two flags were doing work that the benchmark had and the shipped package did not.
Neither showed up as a failure; both showed up as a smaller ratio, which is the
hardest kind of difference to notice.

1. **The shortlist was a path into this repo.** `CF_SHORTLIST` defaulted to
   `<repo>/out/flow/shortlist_q3527b.pt`, which no wheel contains, and
   `bench_forkfree.sh` *set it explicitly*. So every measurement here ran the
   62,642-row head and every pip user ran the full 248,320-row one — worth
   **145.7 vs 157.8 tok/s (1.04x vs 1.13x)** at 4B in the clean room. The list is
   keyed by token id, so it is a property of the Qwen3.5 *vocabulary*, not of a
   model or a checkpoint: 250 KB as int32, one file for 4B/9B/27B. It now ships
   inside the wheel (`chained_flow/data/shortlist_qwen3_5.pt`) and is the default.
2. **`CF_COMPILE` was hard-coded to 1 in `bench_cf.sh`'s spec arms** and defaulted
   to 0 in the proposer, and it was not in the defaults table, so it never appeared
   on the `[cf-defaults]` line. A pip user was on an uncompiled flow net (9.79 vs
   5.26 ms per draft at 27B) with nothing anywhere saying so. It is now a
   capability-gated default like everything else, and **`bench_cf.sh` no longer
   sets it** — if the script sets it, the table cannot report on it.

Measured 2026-08-05 on **GPU 6** (GPU 5, the usual one, was occupied by a foreign
job), pristine vLLM 0.25.1, chained-flow installed as a wheel, batch 1, 7 domains,
`maxtok` 256, pooled. The chain arm ran with **no CF_* environment variables set
at all** beyond `CF_DRAFTER_DIR` — i.e. exactly what `pip install` gives you:

| 4B | tok/s | vs base async-on |
|---|---|---|
| base, async **on** (vLLM's default) | 140.3 | |
| **chain, fresh install, no env vars** | **157.5** | **1.12x** |
| chain, `CF_SHORTLIST=` (full head) | 148.5 | 1.06x |

157.5 against the 157.8 reference, and it now needs no environment. The shortlist
is worth **+6.1%** pooled on this box.

27B, same conditions, also with no environment variables — this is the run that
exercises the packaged list's **vocab guard** at a second model size (the guard
compares the recorded `vocab_size=248320` against the loaded `lm_head`, and 4B /
9B / 27B share it):

| 27B | tok/s | vs base async-on |
|---|---|---|
| base, async **on** | 26.2 | |
| **chain, fresh install, no env vars** | **41.0** | **1.57x** |

Both reproduce the recorded figures exactly (26.2 / 41.0).

Same run restricted to the six domains whose output tokens are byte-identical to
base (domain 4 is the 1-ULP tie lottery described below, and it flipped exactly as
before — first divergence at index 59):

| 4B, six byte-identical domains | tok/s |
|---|---|
| base | 140.4 |
| **chain, fresh install** | **171.3** (1.22x) |

**Accept did not move.** The packaged list and the repo list are the same ids, and
the A/B says so at the level that matters: outputs **bit-identical on all 7
domains**, per-domain accept identical on 6 of 7 (2.9565 / 3.0233 / 1.5299 /
1.9922 / 1.4971 / 1.8633) and the seventh differing by 0.0049 — one request-step,
which is the known `CF_ASYNC_SPEC` counter-drain artifact and not a behavioural
difference, since the two runs emitted the same tokens.

Regression arms on the fork, same day, same GPU, `CF_TREE_KEEP=8 CF_TREE_DEPTH=5`:
**4B tree 187.9** (reference 188.9, inside the documented ±0.5% spread) and
**27B tree 46.6** (reference 46.6, exact). Tree per-domain accept
3.450 / 3.824 / 2.144 / 2.520 / 2.106 / 1.882 / 2.364 against the recorded
3.526 / 3.821 / 2.154 / 2.520 / 2.098 / 1.882 / 2.358 — every domain within 0.01
except the 67-token domain 0, where one request-step is worth 0.05.

**Batch-1 regression check for the bucket-ladder and K-schedule work**, `bench_cf.sh`, 4B,
`CF_MAXTOK=256`, 7 domains, GPU 3, same session as the ladders above:

| arm | tok/s | reference |
|---|---|---|
| base (async off) | 127.6 | 126.7 / 127.3 |
| **chain** | **157.8** | **157.8** (exact) |
| 8×5 tree (`CF_ASYNC_SPEC=1`) | 178.6 | 188.9 — **see the note below, it is not this work** |

`chain` reproduces the recorded figure exactly and the serve path agrees — concurrency 1 reads
**165.3** with the old ladder and **165.4** with the new one, against the recorded 165.2. That
is what the check is for, and it is also true *by construction*: `_bucket_ladder` still starts
at 1, `_ctx_gpu` picks bucket 1 for a batch of 1 under either ladder, and the K-schedule is off
unless `CF_SPEC_K_SCHEDULE` is set. **No step at a decode batch of 1 changes shape, kernel or
output.**

**The 8×5 tree arm reads 178.6 against the recorded 188.9, and the bucket ladder cannot be the
cause.** Three pieces of evidence, and they should be read together rather than as a defence:

1. **Batch 1 takes bucket 1 under either ladder**, so the tree draft ran the identical captured
   graph; and the K-schedule is off unless `CF_SPEC_K_SCHEDULE` is set.
2. **The chain arm in the same session, same GPU, same harness invocation, is exact** (157.8
   against 157.8). A drifted box or a drifted harness would have moved it too.
3. The installed vLLM's `model_executor/layers/mamba/mamba_utils.py` was **modified at 20:11**,
   between the recorded reference and every measurement in this section, by concurrent work on
   the tree's GDN conv-state width — which sets the engine's block size, and whose own commit
   message records that fp16-tie outcomes moved with it. Per-domain accept lands at
   3.450 / 3.824 / 2.030 / 2.457 / 1.680 / 1.882 / 2.364: **domains 0, 1, 5 and 6 are byte-equal
   to the last recorded regression arm** and the three that differ are the long-running ones.

That is an attribution, not a clearance: **the 4B tree batch-1 number needs re-establishing
against the settled engine**, and until it is, quote 188.9 only with this note attached.

One trap found while doing it: **`bench_cf.sh`'s `tree` arm does not default to the 8×5 tree
the published number is from.** Run as-is it resolves `K=16 width=4`, a 4×4 tree, whose
per-domain accept (3.091 / 3.421 / 1.830 / 2.402 / 1.977 / 1.766 / 2.185) is well below the 8×5
reference list above while its pooled tok/s (190.4) is close to it. Comparing those accepts to
the recorded ones would read as a large regression that is only a different tree. Set
`CF_TREE_KEEP=8 CF_TREE_DEPTH=5` explicitly for any tree number meant to be compared.

### The shortlist is guarded, not trusted

A shortlist is a list of integers. Point it at a model with a different
vocabulary and every id names a different token: the list is not suboptimal, it
is nonsense, and the only symptom is a quietly lower accept. The old loader
clamped ids into range (`sl[sl < V]`) and carried on, which is that failure mode
exactly. The payload now records the `vocab_size` it was built against,
`chained_flow.shortlist.check` **refuses** a mismatch, and the fallback is the
full head with the reason printed — slower, never wrong. Resolution order is
`CF_SHORTLIST` (explicit, and `CF_SHORTLIST=` means "full head, deliberately"),
then `<drafter checkpoint>/shortlist.pt` (now actually reachable: it is in the
HF `allow_patterns` and is resolved *after* the snapshot download, so the
documented "drop it next to the checkpoint" works for a repo id), then a source
checkout's `out/flow/`, then the packaged list. The `[cf-defaults]` line names
which one won and its row count, read off the built head rather than off the env
var.

### The K x prompt-length sweep (the K+1 cudagraph collision)

A prompt of exactly `num_speculative_tokens + 1` tokens satisfies vLLM's
*shape-only* uniform-decode test, so a **prefill** is dispatched to the FULL
decode cudagraph; on a hybrid model that graph holds the recurrent GDN step
instead of the chunked prefill scan and the request is corrupt from token 0.
`flow_proposer._install_uniform_decode_guard` makes the test about phase instead.
The collision sits on the `plen == K+1` diagonal, so the regression test has to
sweep both axes rather than benchmark one shape:

```bash
./vllm/bench_prefill_guard.sh 4b            # K in {3..8} x plen 1..32, vs base
```

Result, 2026-08-05, GPU 6, pristine vLLM 0.25.1 + the wheel, 4B, greedy, 32 output
tokens per cell: **160 of 160 cells byte-identical to base, 0 degenerate**, and the
`plen == K+1` diagonal is indistinguishable from every other cell.

Two traps this harness has to avoid, both of which produced a wrong verdict first:

* **Prompts must be built from raw token ids.** Tokenizing text and hoping for
  `K+1` tokens misses the only cell that matters.
* **Base degenerates on its own at `plen == 1`** — a single token with nothing to
  condition on, and it repeats id 0, which is *exactly* the corruption signature.
  Judged absolutely, the sweep reported FAIL on every K. Degeneracy is evidence
  only where base did not do it too. (Same family as the fp16-tie rule in section
  3: compare against base, never against an absolute.)

**The sweep also found a second defect**, which is what a sweep is for: at `K=8`
against a `draft_length=8` drafter the engine started, loaded, and then died on the
first decode step with `draft is (32, 7) but 8 columns were declared to vLLM` — a
shape assertion about an internal buffer, minutes after the mistake, naming
neither the knob nor the value. A chain emits `draft_length - 1` tokens (depth 0
reconstructs the already-committed token), exactly like the tree, and the tree had
a clear `ValueError` for it while the chain had none. It does now:
`num_speculative_tokens=8 but this drafter can only emit 7 chain tokens`. K=8 is
therefore reported as REFUSED rather than run, and 3..7 are the sweep's K axis on
this drafter.

---

## 2. Aggregation: pooled vs mean-of-domain

`vllm/test_plugin_native.py` prints per-domain tok/s **and** a pooled figure
(total tokens ÷ total seconds). They differ a lot, and mixing them has produced fake
regressions and fake wins:

| 27B 8×5, same run | pooled | mean-of-domain |
|---|---|---|
| tok/s  | 38.4  | 44.0  |
| accept | 2.384 | 2.737 |

That is **1.47× vs 1.68× from one measurement**. This went unnoticed for days because
*base* tok/s barely varies by domain (26.2 either way), so only the numerator silently
changed convention. **Always state which one you are quoting.** Pooled is the more
conservative and is the default in current reports.

The same error crept into an external comparison: our unweighted-mean accept was
compared against a competitor's pooled figure. Like-for-like the gap was materially
worse than recorded. Compare like with like.

---

## 3. `RedHatAI/speculator_benchmarks`, per domain — the published numbers

These are the figures on the three v2 model cards. Measured 2026-08-06 on
RTX PRO 6000 Blackwell, `guidellm 0.6.0` driving `vllm serve` (**vLLM 0.25.1**) over the
OpenAI HTTP API.

### Read this first: the dataset is not what the README says

The published snapshot ships **9 `.jsonl` files**, not the seven this project used
previously and not the eight its README documents:

- **`tool_call.jsonl` (200 rows) is undocumented.** It is absent from the dataset README
  and had never been benchmarked here. It is a normal, well-behaved domain.
- **`writing.jsonl` and `question.jsonl` are byte-identical** — the same blob
  (`md5 f776f462`, both symlinking `e493606516e7…`). The README calls #4 "MT_bench" and #8
  "Writing"; in this snapshot they are the same MT-bench prompts.
  **Anyone reproducing our older "7-domain" numbers needs to know the old `writing`
  column was MT-bench content, not prose.** It is dropped as a duplicate below, and
  `question` is kept under the README's own name for that content.

So: **8 distinct domains**. `load_dataset("RedHatAI/speculator_benchmarks")` still fails
(HumanEval's schema differs, so `datasets` tries to concatenate the files into one split);
load per-file with `data_files=`, which is also what the RedHat harness does — it runs one
benchmark per file and never pools.

### THE TRAP: every RedHat config ships `DATASET=".../math_reasoning.jsonl"`

`math_reasoning` is the single most favourable domain at every size. Running the harness
**as shipped** and quoting the result reports:

| size | math_reasoning (as shipped) | honest pooled (8 domains) | overstatement |
|---|---|---|---|
| 4B  | 1.83x | 1.29x | +42% |
| 9B  | 1.92x | 1.40x | +37% |
| 27B | **2.51x** | **1.75x** | **+43%** |

Quote the pooled row. If you quote a single domain, name it.

### Settings

Batch/concurrency **1** (guidellm `synchronous` profile, no rate; measured
`request_concurrency` mean 0.9998–0.9999, max 1.000). Greedy (`temperature=0`). A **fixed
256 output tokens** per request (`output_tokens_count` → `max_completion_tokens` +
`ignore_eos`) so every arm does identical work — this also removes the HumanEval-EOS
variability that once made identical repeats read 233 vs 277 tok/s. **Async scheduling ON
for every arm including base** (`CF_ASYNC_SCHED=1`), so the baseline is not handicapped.
25 prompts per domain, **3 repeats**, one guidellm benchmark **per domain** (the only way
to attribute a Prometheus counter delta to a domain). Arm: **chain, K=5**, shipping
defaults, shortlist head resolved from the package. Drafters confirmed by printed hash:
4B `9c3962cc…`, 9B `ed77e698e501423858effa9a596908a700876be7`, 27B `59cbe06e…`.

Every arm ran against a **frozen copy of `src/`** (`CF_SRC=…/src_snapshot`, which defaults
to the repo working tree and is a no-op otherwise). Without it, another agent editing
`drafters/` between the base and chain arms silently makes them non-comparable.

### Metric definitions

- **acceptance** = mean accepted length = `1 + num_accepted_tokens / num_drafts`, from the
  delta of vLLM's own Prometheus counters across the domain's window. **The bonus token is
  included**, so a non-speculative baseline is 1.00 by definition and K=5 caps at 6.00.
  Same formula as `speculators/tests/e2e/run_vllm.py` and vLLM's "Mean acceptance length".
- **speedup** = chain tok/s ÷ base tok/s, where tok/s = `sum(output tokens) / wall
  duration`, **prefill in the denominator**.
- For a **tree** arm `num_draft_tokens` is the tree's *node count*, so vLLM's "Avg Draft
  acceptance rate" and its per-position rates are meaningless there. Mean accepted length
  is the metric that stays comparable across draft geometries.

### Results

Acceptance (tokens/step, bonus included) and speedup vs the no-speculation baseline:

| domain | 4B acc | 4B speedup | 9B acc | 9B speedup | 27B acc | 27B speedup |
|---|---|---|---|---|---|---|
| HumanEval | 2.29 | 1.47x | 2.39 | 1.61x | 2.42 | 2.00x |
| math_reasoning | 2.88 | 1.83x | 2.85 | 1.92x | 3.05 | 2.51x |
| qa | 1.93 | 1.24x | 1.96 | 1.34x | 2.01 | 1.67x |
| question (MT-bench) | 1.98 | 1.28x | 2.05 | 1.40x | 2.09 | 1.74x |
| rag | 2.03 | 1.29x | 2.10 | 1.42x | 2.13 | 1.74x |
| summarization | 1.95 | 1.24x | 2.00 | 1.35x | 2.07 | 1.70x |
| tool_call | 2.13 | 1.35x | 2.16 | 1.45x | 2.13 | 1.75x |
| translation | 1.45 | **0.94x** | 1.49 | 1.02x | 1.57 | 1.31x |
| **POOLED (8 domains)** | **2.02** | **1.29x** | **2.06** | **1.40x** | **2.12** | **1.75x** |

Underlying tok/s (pooled, prefill included):

| size | base | chain | pooled speedup | decode-only speedup (prefill excluded) |
|---|---|---|---|---|
| 4B  | 138.27 | 178.43 | 1.290x | 1.330x |
| 9B  |  81.99 | 114.68 | 1.399x | 1.438x |
| 27B |  26.26 |  45.95 | 1.750x | 1.802x |

**`translation` is a regression at 4B (0.94x) and bare parity at 9B (1.02x).** Speculation
costs a draft pass every step; where acceptance is low it does not pay for itself. The
pooled figures above already carry those losses.

### Confidence

- **Baselines reproduce.** 138.27 / 81.99 / 26.26 against the previously recorded
  138.9 / 82.4 / 26.3. The 4B gate against the legacy pooled subset read 138.66 vs 138.92
  (−0.18%) before any new number was read.
- **Spread is negligible**: 0.0–0.4% per domain, **≤0.1% pooled** over 3 complete repeats
  (27B chain read 45.95 on all three).
- **Base is deterministic**: repeat 1 vs repeat 2 was 200/200 byte-identical at every size.
- **Equal work verified, not assumed**: all 3600 requests emitted exactly 256 tokens,
  0 errors.
- **Divergence from base** (fp16 1-ULP tie-flips, spread across all domains rather than
  concentrated): 4B 27/200, 9B 17/200, 27B 13/200. Throughput stays comparable because the
  token count is fixed regardless of which continuation is taken.

### These are concurrency-1 numbers

Section 0 above is the load story and it is the one that matters for deployment: the 4B
chain arm falls to 0.90x at concurrency 8 and 0.44x at 64 without `CF_SPEC_MAX_BATCH`, and
27B should not be served with speculation above concurrency ~16 at all. Do not quote the
table above as though it holds on a busy server.

### Reproducing

```bash
vllm/specbench_domains.py -o logs/specbench_dom/data --per-domain 25 --output-tokens 256
CF_SRC=<frozen copy of src> SPECBENCH_REPEATS=3 \
  vllm/specbench_dom_matrix.sh "27b:base 27b:chain 9b:base 9b:chain 4b:base 4b:chain"
vllm/specbench_dom_report.py --size 27b --json-out logs/specbench_dom/report_27b.json
```

`specbench_dom_report.py` pools only domains **and complete repeats** present in every arm,
and prints what it excluded — an in-flight repeat covers a subset of domains, and pooling it
produces a fake run-to-run "spread" that is really a domain-mix difference (it moved 27B
pooled from a contaminated 1.793x to the correct 1.750x).

---

## 4. Other hazards worth knowing before you trust an A/B

- **Confirm the flag actually engaged.** A shape gate (`hidden_size == 640`) made
  `CF_CUDA_BLOCK` a silent no-op at 9B/27B for hours, producing confident, meaningless
  A/Bs. Print the state actually used, not the env var — e.g. `ring_slots=6 of 41`.
- **Baseline against the compiled + cudagraphed path, never eager.** An eager baseline
  once inflated a drafter kernel to "4.3×" when its real end-to-end effect was +2.6%.
  A second bug in the same script popped the wrong cache key, so the flag was never
  even enabled inside the timing loop.
- **Greedy decoding of an fp16-logit model is ill-posed.** vLLM keeps logits in the
  model dtype; at |logit| ≈ 24 the fp16 ULP is 0.015625, and 0.3–0.9% of tokens sit
  within 1 ULP of a tie. Base mode itself is not reproducible at those tokens. So a
  **single-run token diff is never a signal at any size**; 4B diverges from base at 2/7
  domains with no flags set at all. Prefer a within-process device-side element tally,
  which cancels per-process kernel selection exactly; compare token equality against
  **base**, never against another spec run.
- **Diff the OUTPUT TOKENS before comparing accept or tok/s across arms. Exclude or
  flag any domain whose outputs diverge.** This is the companion rule to the one
  above, and it is the one that costs you numbers rather than confidence. Two arms
  that decoded different text did not measure the same work, so *nothing* downstream
  of the divergence is comparable — not accept, not tok/s, not the pooled figure they
  feed. One 1-ULP flip anywhere in a generation is enough: at 4B domain 4 a flip at
  token 59 (gap = 0.015625, exactly 1 ULP) put one arm in a repetitive `<think>` block
  and the other in free prose, and dragged the *pooled* 7-domain entry-point figure
  from +12.4% down to +6.1% — an artifact that looked for a day like a fifth silent
  async bug. Practically: the harness already saves every set's token ids to
  `/tmp/cf_native_<mode><tag>.json`, so the check is a few lines against those files
  and costs no GPU time. Report the pooled figure over the domains that match, say
  how many were dropped and why, and treat a *new* divergence as a bug to investigate
  rather than a domain to drop. We have now been bitten by fp16 ties three separate
  times; this rule is the one that would have caught all three cheaply.
- **`CF_CUDA_BLOCK=1` is nondeterministic outside cudagraph capture** (2.70% of tree
  nodes flip run-to-run). Offline bit-exactness checks must run with it OFF or against
  a control.
- **Offline window sweeps understate accept damage.** A candidate-head width that
  looked like −0.010 offline (5 domains × 1000 windows) collapsed summarization
  2.116 → 1.627 in-engine, because a greedy run compounds one lost candidate over a
  whole trajectory. **Validate anything that changes candidates in-engine.**
- **Check for foreign jobs.** `nvidia-smi` before trusting a sweep. Tells for
  contention: non-monotonic cost curves, *same accept but lower tok/s*, or a base value
  disagreeing with a known-good one. Accept is compute-independent — if accept
  reproduces but throughput does not, suspect contention, not the code.
- **Never `pkill -f <pattern>`** on a shared box; it is not GPU-scoped and has killed
  another run mid-flight. Scope cleanup to your own PIDs.
