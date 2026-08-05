# Benchmarking protocol — read this before quoting any speedup

Every number in this project is a ratio against a baseline. Two things about that
baseline have silently corrupted results in the past. Both are cheap to get right
and expensive to discover late.

---

## 0. `vllm serve` compatibility. (Batch 1 is still the measurement of record.)

> **Read first, if you are deploying:**
> 1. **Do not serve the TREE arm above concurrency 1** — the engine core dies at concurrency 2
>    (device assert) or hangs (with async off), at 4B and 27B alike.
> 2. **Set `CF_TREE_GREEDY_GUARD=1` on any tree-mode server** — otherwise one `temperature>0`
>    request kills it permanently, for every client.
> 3. **At 4B, turn speculation off above ~4 concurrent requests** — the chain arm drops to
>    0.90x/0.78x of the no-speculation baseline at concurrency 8/16. 27B never does.

The published speedups and acceptance numbers are **batch-1 numbers**, taken through the
offline `LLM()` API (`vllm/test_plugin_native.py`, which also forces
`VLLM_ENABLE_V1_MULTIPROCESSING=0`), and that remains the measurement of record — sections 1–3
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
| `CF_CUDA_BLOCK` | **no, from B ≥ 2** | `chunked_flow._cf_fused_runner` requires `x.shape[0] == 1`; the drafter's `x.shape[0]` is the cudagraph *bucket*, so only bucket 1 qualifies. Falls back to the bit-identical PyTorch block stack. |
| `CF_CUDA_PAIR` | **no, from B ≥ 2** | rides on `CF_CUDA_BLOCK`. |
| `CF_GDN_DEFER` / `CF_GDN_BV` | **no, from B ≥ 2** (tree) | `tree_gdn.defer_rows(T)` caps at `CF_GDN_DEFER_MAXROWS` = 64 rows. An 8×5 tree is 41 rows per request, so B = 1 fits and B = 2 (82 rows) does not. The cap is a *memory* decision — a deferred stash pins that layer's k/v inside the cudagraph pool — not a correctness one. |
| draft cudagraph | degrades, then off above 32 | buckets are `[1,2,4,8,16,32]`; B = 5 replays the bucket-8 graph, i.e. three whole drafts on padding rows. Above 32 there is no bucket and the draft runs eager. |
| `CF_DRAFT_EARLY` (side-stream prelaunch) | intermittent | the steady-state guard requires every input-batch slot to hold the *same request as last step*, which continuous batching breaks whenever a request joins or leaves. It is also tree-only: in chain mode `_prelaunch` returns immediately and the flag's remaining job is publishing the GPU counts for `CF_ASYNC_SPEC`. |
| `CF_TREE_FUSED_ATTN` | yes | its `N < 128` limit is the per-request tree width, not the batch. |
| `CF_TREE_FULLCG` | yes | a uniform multi-request decode still dispatches FULL. |
| `CF_SHORTLIST`, `CF_COMPILE`, `CF_TWOPASS_M`, `CF_PATH_TRIM`, `CF_FUSE_PATH`, `CF_RING_TRIM`, `CF_ASYNC_SPEC` | yes | batch-agnostic. |

### The first request at a new batch size stalls the engine

`CF_COMPILE` compiles the flow net with `max-autotune-no-cudagraphs`, and the draft
cudagraph is captured lazily per bucket. Both happen **inside the serving loop**, the first
time a bucket is reached. Measured at 4B chain, from the engine's own step log: a **70–80 s
stall on the first step at each of buckets 2, 4, 8 and 16** (32–36 inductor autotune blocks
each). A benchmark must warm up *at the concurrency it is about to measure* or it charges
that stall to steady-state throughput — `bench_serve_drive.py` warms up with `3 × concurrency`
requests for exactly this reason, and got it wrong once (4B chain c=16 read 175.9 tok/s and a
20.0 s mean TTFT against 826 tok/s at c=8, purely from the un-warmed bucket-16 capture).

For a deployment the fix is to capture every bucket at startup rather than on demand; that
is not implemented.

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
greedy generations, same prompts, same server. This is the fp16-tie hazard of section 3, and
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
protocol rule in section 3 to catch the next one automatically.

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

## 3. Other hazards worth knowing before you trust an A/B

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
