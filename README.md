# chained-flow

A flow-matching **hidden-state drafter** for lossless speculative decoding in vLLM.
Instead of a second transformer, the drafter integrates a small conditional flow
over the target model's own final hidden states, then decodes the trajectory
through the frozen `lm_head`. Verification is vLLM's, unchanged, so the output
distribution is the target model's.

## Training

To train or continue a drafter (Turkish or otherwise), see **[TRAINING.md](TRAINING.md)** — clone,
`uv venv`, `uv pip install -r requirements.txt`, fetch the prompts, run one script. That path is
CUDA-13 and deliberately does **not** install vLLM: training imports it nowhere, and keeping the
two environments separate avoids a second torch pin.

## Install

```bash
pip install chained-flow          # pulls vllm==0.25.*
```

That is the **default build: fork-free**. It runs the *chain* path on unmodified
vLLM and needs no patching. The install registers a `vllm.general_plugins` entry
point ([`chained_flow/vllm_plugin/async_guard.py`](https://github.com/Zeuss5/chained-flow/blob/main/src/chained_flow/vllm_plugin/async_guard.py))
that relaxes one guard so a `custom_class` proposer is allowed to keep vLLM's
async scheduling, which vLLM otherwise hands to the baseline and denies to us.
Measured worth on the chain arm: **+6.1% at 4B**. That is why the install has to
be a real install — a bare `PYTHONPATH` creates no entry points and the plugin
never fires.

Measured on pristine vLLM 0.25.1, batch 1, 7 domains, 256 tokens, pooled,
**with no environment variables set beyond `CF_DRAFTER_DIR`**:

| | base (vLLM default) | chain | speedup |
|---|---|---|---|
| Qwen3.5-4B  | 139.6 tok/s | **157.8** | **1.13x** |
| Qwen3.5-27B |  26.2 tok/s |  **41.0** | **1.57x** |

How that number was taken, and the two baseline mistakes that have corrupted it
before, are in the benchmarking protocol. It ships in the wheel:

```bash
chained-flow docs        # docs/BENCHMARKING.md — read it before quoting any speedup
```

Check what you got:

```bash
chained-flow info
```

It prints the vLLM build in use, whether the async guard was relaxed, whether the
fused CUDA kernel compiles on this box, which shortlist would be used, whether
flashinfer's kernels are prebuilt, and the resolved state of every flag.

## What you have to supply

Exactly two things. Everything else has a default that is gated and printed.

1. **The target model** — any Qwen3.5 checkpoint a drafter was trained for.
2. **`CF_DRAFTER_DIR`** — the drafter checkpoint: a local directory, or a Hugging
   Face repo id downloaded on first use.

| target model | drafter |
|---|---|
| `Qwen/Qwen3.5-4B`  | `ytu-ce-cosmos/Flow-Drafter-Qwen3.5-4B` |
| `Qwen/Qwen3.5-9B`  | `ytu-ce-cosmos/Flow-Drafter-Qwen3.5-9B` |
| `Qwen/Qwen3.5-27B` | `ytu-ce-cosmos/Flow-Drafter-Qwen3.5-27B` |

## Quickstart

```python
# quickstart.py  —  CF_DRAFTER_DIR=ytu-ce-cosmos/Flow-Drafter-Qwen3.5-4B python quickstart.py
from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model="Qwen/Qwen3.5-4B",
        dtype="float16",
        speculative_config={
            "method": "custom_class",
            "model": "chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
            "num_speculative_tokens": 5,
        },
    )
    out = llm.generate(
        ["Explain how gradient descent works, and why the learning rate matters."],
        SamplingParams(temperature=0, max_tokens=256),
    )
    print(out[0].outputs[0].text)


# REQUIRED, not style: vLLM 0.25 starts its engine core in a `spawn`ed subprocess, which
# re-imports this file. Without the guard the module-level `LLM(...)` runs again in the
# child and the process dies in multiprocessing before generating anything.
if __name__ == "__main__":
    main()
```

```bash
CF_DRAFTER_DIR=ytu-ce-cosmos/Flow-Drafter-Qwen3.5-4B python quickstart.py
```

Everything else is defaulted and **capability-gated**: eleven optimisation flags
are proposed, each gate is evaluated against the real process (does the CUDA
extension build? does the drafter's shape fit the kernel? was the shortlist built
for this vocabulary? did the engine actually enable async scheduling?), and the
resolved state of all of them is printed on one `[cf-defaults]` line at startup.
See [`defaults.py`](https://github.com/Zeuss5/chained-flow/blob/main/src/chained_flow/defaults.py).
A flag that could not engage says so; nothing here fails silently.

### First run is slow, and only some of that is ours

* **flashinfer** JIT-compiles its sampling kernels on first use unless the
  prebuilt cache is installed. It is not on PyPI (it is built per CUDA version),
  so `pip` cannot pull it for you and vLLM will not warn you. Minutes, and it
  needs `nvcc`. Skip it with:

  ```bash
  pip install flashinfer-jit-cache --extra-index-url https://flashinfer.ai/whl/cu130/   # match your CUDA
  ```

* **our fused CUDA kernel** JIT-compiles once (~60 s) — `chained-flow build-kernel`
  precompiles it, see [CUDA kernel](#cuda-kernel).
* **`CF_COMPILE`** (default on) `torch.compile`s the drafter's flow net on the
  first draft; worth 9.79 → 5.26 ms per draft at 27B.

`chained-flow info` tells you which of these are still pending.

### The shortlist

The drafter decodes through the target's frozen 248k-row `lm_head`, but drafting
only ever consumes the top few candidates per position — so ~45% of the draft is
weight traffic for rows the beam never looks at. A **shortlist** restricts the
head to the 62,642 ids that actually occur. It is quality-free: the drafter only
*proposes*, and vLLM verifies every token, so a missing id can cost acceptance
and can never cost correctness.

**It ships in the wheel** (250 KB, int32) and is used automatically — it is keyed
by token id, so one file serves 4B/9B/27B, which share the Qwen3.5 vocabulary.
Without it, 4B measures 145.7 tok/s (1.04x) instead of 157.8 (1.13x).

You only touch this for a vocabulary we do not ship one for:

* the packaged list is **refused, loudly, on a vocab-size mismatch** — it falls
  back to the full head rather than silently indexing the wrong tokens;
* `chained-flow build-shortlist --tokenizer <model> --vocab-size <lm_head rows> --ids '<glob>.pt' --jsonl '<glob>.jsonl'`
  builds one. Pass `--vocab-size` explicitly: the guard compares it against the
  model's `lm_head`, and a tokenizer under-reports whenever that head is padded
  (Qwen3.5: 248,044 vs 248,320);
* put it next to the drafter checkpoint as `shortlist.pt` (this works for an HF
  drafter repo too — it is downloaded with the weights) or point `CF_SHORTLIST`
  at it. `CF_SHORTLIST=` (empty) forces the full head.

The `[cf-defaults]` line reports which one won and its row count.

## The two arms

|  | **chain** (default) | **tree** (opt-in) |
|---|---|---|
| vLLM | unmodified, `pip install` only | requires the fork |
| verification | vLLM's linear chain | branching tree verify |
| sampling | any | greedy only |
| enable with | nothing | `VLLM_SPEC_TREE=1` + the patch |

The chain path builds the same draft tree from one flow pass and beam-searches
it, but emits the single best **path** as a linear chain, because vLLM V1 has no
tree-verify hook. That keeps the tree's token-*selection* benefit and gives up
only its multi-branch acceptance.

### Enabling the tree path

The tree needs a patched vLLM (a tree-aware verify, tree-shaped GDN recurrence
and attention, and a KV/state hand-off none of which upstream exposes). One
command, on the vLLM of the interpreter you run it with:

```bash
chained-flow tree-patch                # what it would touch, and the current state
chained-flow tree-patch --apply        # patch this environment's vLLM
chained-flow tree-patch --status       # is it applied? is every other file still stock?
chained-flow tree-patch --revert       # restore it byte-identically
```

`--apply` and `--revert` both take `--dry-run`, which stages and verifies the
whole change in memory and writes nothing.

It refuses rather than half-doing anything, because a **half-applied fork does
not raise — it drafts wrongly, at full speed**:

* the target is the vLLM this interpreter would `import`, resolved through the
  import system, not a guessed `site-packages` path;
* the version gate is hard and doubled — `vllm.__version__` *and* the dist-info
  must both read `0.25.1` and must agree with each other;
* before and after, **every one of the wheel's 4502 files is hashed against
  `vllm-0.25.1.dist-info/RECORD`**. After `--apply` the expected diff is exactly
  6 modified + 8 added; after `--revert`, zero. Any other number is reported
  file by file;
* the diff is applied at exact line numbers with zero fuzz, every file is built
  in memory first, and one bad hunk aborts before a single byte is written —
  there is no `.rej` path;
* `--revert` *reverse-applies* the same hunks and re-hashes against RECORD, so
  it needs no saved backup and still works after you have upgraded pip packages
  around it. A file it cannot prove it would restore exactly is refused, not
  guessed at;
* re-running `--apply` on an already-patched install says so and exits 0.

`--status` exits 0 applied / 1 not applied / 3 inconsistent, so it is usable in
a script. `chained-flow tree-patch --show` prints the patch file's path and the
manual `patch -p1` route, for patching a *different* interpreter's vLLM.

Then run with `VLLM_SPEC_TREE=1`, `CF_TREE_KEEP` × `CF_TREE_DEPTH` nodes, and
`num_speculative_tokens = CF_TREE_KEEP*CF_TREE_DEPTH + 1` (the extra column is a
spare mamba-state slot, never an emitted token). `chained-flow info` reports
`vLLM build : FORKED` once the patch is in, and the `[cf-defaults]` line lists
`gdn_defer gdn_bv tree_fused_attn tree_fullcg` under **ON**; without the patch
those flags report `fork_missing` and are **not offered** rather than silently
ignored.

The patch touches 6 upstream files and adds 8 (five `.py` and the three `.cu`
sources their JIT loaders compile), all but two under `vllm/v1/spec_decode/`. It
is generated against 0.25.1 exactly; the version pin is deliberate.

## CUDA kernel

The drafter's block stack has a hand-written fused CUDA kernel (`CF_CUDA_BLOCK`,
default on). It is **JIT-compiled on first use** via torch's extension loader,
takes ~60 s once per machine, and is then cached in
`~/.cache/torch_extensions` (`CF_KERNEL_DIR` overrides). Precompile it if you do
not want that stall inside your first engine start:

```bash
chained-flow build-kernel
```

If the toolchain is missing or the drafter's shape is not one the kernel is
instantiated for, it falls back to the **bit-identical** PyTorch block stack
(~2x slower draft) and says why. It never hard-fails.

## Benchmarking

**Before quoting any speedup, read the benchmarking protocol** — `chained-flow docs`,
or [docs/BENCHMARKING.md](https://github.com/Zeuss5/chained-flow/blob/main/docs/BENCHMARKING.md).
The baseline is where this project has been wrong before.

**`vllm serve` is the measurement path**; the offline harness is the regression gate.
Wins in this repo are gated on a decode batch of 1 and disengage under continuous
batching, so a batch-1 number is not a deployment number — see section 0 of the doc.

```bash
./vllm/bench_serve.sh 4b base 3 8601   # <size> <arm> <gpu> <port>; concurrency 1..16
./vllm/bench_serve.sh 4b chain 3 8602  # over a real HTTP server, engine core in its own process
./vllm/bench_serve_report.py           # the table
./vllm/bench_serve_diff.py logs/bench_serve/4b_{base,chain}/serve_bench.json
```

```bash
./vllm/bench_cf.sh 4b base            # batch-1 regression gate: baseline
./vllm/bench_cf.sh 4b chain           # fork-free arm
./vllm/bench_cf.sh 4b tree            # needs the patch
./vllm/bench_forkfree.sh 4b 256       # all arms, both baselines, one run
```

`CF_PY=<venv>/bin/python` selects which vLLM to run against.
`CF_BATCH_AUDIT=1` makes the proposer report the decode batch it is *actually* running
at, which is the only way to tell whether the batch-1-gated flags engaged.
`vllm/serve_ladder.sh` is `bench_serve.sh` for ladders past concurrency 16: per-level
request counts, and it refuses to run on a GPU that has not finished draining (an engine
that profiles against a busy GPU gets a silently tiny KV cache and then benchmarks its own
admission queue).

**Serving under concurrency: `CF_SPEC_MAX_BATCH` (default ON where the threshold is measured).**
Speculation is a batch-1 win. Above a decode batch of N it is a loss — the target verifies
`(K+1) x B` positions for an acceptance that cannot pay for them, while the drafter's batch-1
kernels have already disengaged. Measured at 4B, chain vs no speculation: 1.19x at concurrency 1,
1.00x at 4, 0.90x at 8, **0.44x at 64**. With the cutoff the same ladder reads 1.19x / 1.00x /
0.95x / 0.95x — and the batch-1 number is bit-identical to the uncut arm.

The flag works in two halves that must both be on: the scheduler stops allocating speculative
slots (so the target stops verifying) and the proposer stops drafting. **N is per model size AND
per arm** — the scheduler hands the target `K+1` query positions per request, so an 8×5 tree
saturates it seven times sooner than a chain. `_AUTO` is keyed on `(hidden_size, K+1)` and every
size × arm this project publishes a number for is now laddered against its own no-speculation
baseline on the same GPU.

**The default resolves that table and nothing else.** A combination that was never laddered
resolves to *no cutoff at all*, with the reason printed — a derived N that is too low silently
costs speedup, and that is not something to acquire by accident. `CF_SPEC_MAX_BATCH=auto` opts
back into the derived guess for an unmeasured target, `=<n>` sets one directly, and `=0` (or
`off`) disables it. `CF_WARM_BUCKETS=1` pairs with it to move the drafter's per-batch-shape
`torch.compile` out of live traffic.

Two baseline hazards have corrupted results here before:

1. **async scheduling.** Without the entry point above, vLLM gives the base arm a
   feature it force-disables for `custom_class` speculative decoding, understating
   every speedup by **+10.5% / +5.5% / +1.6%** at 4B / 9B / 27B. `bench_cf.sh`'s
   base arm defaults to `CF_ASYNC_SCHED=0` (like-for-like); pass
   `CF_ASYNC_SCHED=1` for the deployment number. **Quote both, labelled.**
2. **pooled vs mean-of-domain.** The harness prints both and they differ a lot —
   one 27B run reads 1.47x pooled and 1.68x mean-of-domain. Always say which.

Greedy decoding of an fp16-logit model is ill-posed at ~0.3-0.9% of tokens, so a
single-run token diff is never a signal; see the doc for the verification protocol.

## Current components

- `FrozenLMWrapper`: one frozen tokenizer/model wrapper that exposes final hidden
  states, logits, LM-head projection, prefill, cached forward, and greedy next
  token.
- `ChainedFlowContext`: owns the single shared backbone instance. Drafters and
  verifiers receive this wrapper by dependency injection.
- `SpeculativeVerifier`: verifies draft tokens with the frozen AR path and crops
  cache state so only accepted tokens remain committed.
- `generate_with_drafter`: Orthrus-style greedy speculative loop with timing and
  per-step acceptance stats.
- `ARDrafter`: correctness/debug baseline.
- `HiddenMLPDrafter`: first trainable drafter baseline. When configured with a
  VAE checkpoint it predicts future latent states, then decodes them back to LM
  hidden states with the frozen VAE decoder.
- `vae`: compact per-token hidden-state VAE architectures (`mlp`,
  `residual_mlp`, `low_rank`) behind a shared interface.
- `training.losses`: combined hidden, logit/token, and verifier-surrogate
  losses for hidden-state drafter training.
- `data.windows`: token window helpers with explicit teacher hidden-state
  alignment.

## Training losses

`training.losses.compute_drafter_loss` groups every term into four
categories.

### Latent losses

These are used when a drafter owns a frozen VAE and predicts future latent
states instead of raw hidden states.

- `latent.mse`: mean squared error between predicted and VAE-encoded teacher
  latents.
- `latent.cos`: cosine distance between predicted and VAE-encoded teacher
  latents.

### Hidden losses

These keep predicted states close to the frozen LM's final hidden-state manifold.

- `hidden.mse`: mean squared error between predicted and teacher hidden states.
- `hidden.cos`: cosine distance between predicted and teacher hidden directions.
- `hidden.norm`: activation-norm matching between predicted and teacher states.

### Logit / token losses

These use the frozen LM head to check what predicted hidden states decode to.

- `logit.ce`: position-weighted cross entropy against future token ids.

### Verifier-based losses

These are differentiable surrogates for accepted-prefix length.

- `verifier.expected_accept`: maximizes an approximation of expected accepted
  tokens using cumulative target-token probabilities under
  `lm_head(pred_hidden)`.

Default combined loss:

```text
L =
  1.0  * hidden.mse
+ 0.2  * hidden.cos
+ 0.05 * hidden.norm
+ 0.2  * logit.ce
+ 0.1  * verifier.expected_accept
```

VAE-backed drafters add latent losses only when `lambda_latent_mse` or
`lambda_latent_cos` are nonzero. The VAE is frozen during drafter training.

## Teacher-state datasets

Teacher collection stores K-independent full sequences. Each row contains:

- `input_ids`: tokenized formatted prompt+response text.
- `final_hidden`: frozen LM final hidden states for every token position.
- `example_id`, `source`, `split`: tracing and dataset-mixing metadata.
- `text`: the exact decoded prompt plus model-generated response used for
  collection.
- `prompt_text`: the exact formatted prompt seen by the model.
- `generated_text`: the exact decoded model output after the prompt.
- `format_name`, `model_id`, `hidden_dtype`, `num_tokens`: reproducibility and
  filtering metadata.
- `prompt_length`: number of prompt tokens before greedy model generation
  starts.

Training samples windows dynamically:

```text
require t >= prompt_length - 1
context_hidden = final_hidden[t-m+1 : t+1]
target_hidden  = final_hidden[t : t+K]
future_tokens  = input_ids[t+1 : t+K+1]
```

Hidden MLP training example:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_hidden_mlp.py \
  --dataset_path teacher_states/gsm8k-qwen35-08b-smoke \
  --output_dir checkpoints/hidden-mlp-smoke \
  --per_device_train_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 1e-4 \
  --logging_steps 10 \
  --save_steps 100 \
  --windows_per_epoch 32 \
  --window_seed 0 \
  --local_files_only true
```

To train the drafter in VAE latent space, pass a trained VAE checkpoint:

```yaml
vae_dir: /path/to/hidden-vae-checkpoint
lambda_latent_mse: 1.0
lambda_latent_cos: 0.2
```

The training script uses Hugging Face `Trainer` and `TrainingArguments`. It does
not read training configuration from `.env`; pass CLI/dataclass arguments or a
single YAML config file.

Smoke YAML config example:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_hidden_mlp.py train_configs/smoke_mlp.yaml
```

Hidden VAE smoke training:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/train_vae.py train_configs/vae/smoke_vae.yaml
```

GSM8K collection is prompt-only: the script asks the frozen model to generate
the response greedily, then stores hidden states for that generated sequence.

GSM8K collection example:

```bash
cp .env.example .env
UV_CACHE_DIR=.uv-cache uv run python scripts/collect_teacher_states.py collect_configs/smoke_gsm8k.yaml
```

The script loads `.env` before project imports so `HF_TOKEN`, `HF_HOME`, and
similar Hugging Face environment variables are available. Collection settings
come from CLI args or a YAML config, not `.env`.

Collection first writes/pushes a temporary answer-only dataset using a `_tmp`
prefix, then runs hidden-state extraction and writes the final dataset.
Use `dataset_start` and `dataset_end` for half-open dataset shards such as
`[0:1024]` and `[1024:2048]`; `limit` remains a backward-compatible fallback.
Use `generation_batch_size` and `hidden_batch_size` in YAML configs to tune the
two phases separately; `batch_size` remains the fallback for both.

To skip generation and extract hidden states from a previously saved `_tmp`
answer dataset:

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/collect_teacher_states.py \
  --answer-dataset-path teacher_states/_tmp_gsm8k-qwen35-08b-smoke \
  --output-dir teacher_states/gsm8k-qwen35-08b-smoke \
  --model-id Qwen/Qwen3.5-0.8B \
  --storage-dtype float16 \
  --dtype float16
```

`--answer-dataset-path` accepts either a local `save_to_disk` path or a Hugging
Face dataset repo id. Use `--answer-dataset-split` for HF repos when the split
is not `train`.

Set `device: cuda` or `device: cuda:0` in collection/training YAML configs to
load the frozen model on CUDA. Use `device: auto` to let Transformers choose a
device map.

For CUDA collection, set `dtype: float16` in
`collect_configs/smoke_gsm8k.yaml`.

## Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

