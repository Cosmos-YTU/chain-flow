# Benchmarking protocol — read this before quoting any speedup

Every number in this project is a ratio against a baseline. Two things about that
baseline have silently corrupted results in the past. Both are cheap to get right
and expensive to discover late.

---

## 1. The async-scheduling trap (this one invalidated real, published-internally numbers)

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

| 27B | tok/s | vs base async-on | vs base async-off |
|---|---|---|---|
| base, async **off** | 25.8 | | |
| base, async **on** (vLLM's default) | 26.2 | | |
| **chain, entry point ON** | **41.0** | **1.57x** | 1.59x |
| chain, `CF_ASYNC_SPEC=0` (guard untouched) | 39.7 | 1.52x | 1.54x |

Four things to read off those tables:

* **1.13x deployment at 4B is now a measurement, not an inference**, and it lands
  on the inferred value. 27B is 1.57x, above the ~1.5x expected.
* **The entry point is worth +6.1%** on the 4B chain arm (157.8 vs 148.7).
  Without it the fork-free build is 1.07x — not nothing, but not a story either.
  At 27B it is +3.3% (41.0 vs 39.7), monotone in model size for the same reason
  the base arm's async prize is: slower steps amortise host overhead better.
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

### One unresolved thing, deliberately not smoothed over

At 4B, **domain 4** reads accept **1.261** with `CF_ASYNC_SPEC=1` and **1.652**
with it off, and its throughput moves the same way (111.8 vs 130.6 tok/s). The
other six domains agree to ~0.01.

A repeat run in a fresh process reproduced the async arm to the third decimal on
**every** domain (2.957 / 3.023 / 1.530 / 1.992 / **1.261** / 1.515 / 1.863,
pooled 157.6 vs 157.8), so this is **not** fp16-tie noise — it is a deterministic
difference in what the two paths draft on that one domain. What it is not:

* an early-termination / KV-corruption signature — every domain emitted an
  **identical token count** under both flags, at both model sizes;
* a 27B problem — there the two arms' accepts agree to ~0.01 on all seven
  domains (2.812/3.935/1.576/1.810/1.554/1.631 sync vs
  2.765/3.952/1.570/1.823/1.550/1.631 async).

The async arm is still much faster overall and remains the default, but the
domain-4 gap is a real behavioural difference on the 4B chain and has not been
explained. Do not quote per-domain 4B accept under `CF_ASYNC_SPEC` as though it
were the sync figure.

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
