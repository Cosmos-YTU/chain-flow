# chained-flow

A flow-matching **hidden-state drafter** for lossless speculative decoding in vLLM.
Instead of a second transformer, the drafter integrates a small conditional flow
over the target model's own final hidden states, then decodes the trajectory
through the frozen `lm_head`. Verification is vLLM's, unchanged, so the output
distribution is the target model's.

## Install

```bash
pip install chained-flow          # pulls vllm==0.25.*
```

That is the **default build: fork-free**. It runs the *chain* path on unmodified
vLLM and needs no patching. The install registers a `vllm.general_plugins` entry
point ([`chained_flow/vllm_plugin/async_guard.py`](src/chained_flow/vllm_plugin/async_guard.py))
that relaxes one guard so a `custom_class` proposer is allowed to keep vLLM's
async scheduling, which vLLM otherwise hands to the baseline and denies to us.
Measured worth on the chain arm: **+6.1% at 4B**. That is why the install has to
be a real install — a bare `PYTHONPATH` creates no entry points and the plugin
never fires.

Measured on pristine vLLM 0.25.1, batch 1, 7 domains, 256 tokens, pooled:

| | base (vLLM default) | chain | speedup |
|---|---|---|---|
| Qwen3.5-4B  | 139.6 tok/s | **157.8** | **1.13x** |
| Qwen3.5-27B |  26.2 tok/s |  **41.0** | **1.57x** |

Check what you got:

```bash
chained-flow info
```

It prints the vLLM build in use, whether the async guard was relaxed, whether the
fused CUDA kernel compiles on this box, and the resolved state of every flag.

## Quickstart

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3.5-4B",
    dtype="float16",
    speculative_config={
        "method": "custom_class",
        "model": "chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
        "num_speculative_tokens": 5,
    },
)
print(llm.generate(["Explain gradient descent simply."],
                   SamplingParams(temperature=0, max_tokens=256))[0].outputs[0].text)
```

The drafter checkpoint comes from `CF_DRAFTER_DIR` — a local directory or a
Hugging Face repo id, downloaded on first use:

| target model | drafter |
|---|---|
| `Qwen/Qwen3.5-4B`  | `selimaktas/Flow-Drafter-4B-v2` |
| `Qwen/Qwen3.5-9B`  | `selimaktas/Flow-Drafter-9B` |
| `Qwen/Qwen3.5-27B` | `selimaktas/Flow-Drafter-Qwen3.5-27B-v2` |

```bash
CF_DRAFTER_DIR=selimaktas/Flow-Drafter-4B-v2 python your_script.py
```

One thing worth setting explicitly: `CF_SHORTLIST=<a .pt of token ids>`. Without
it the drafter scores the full 248k-row `lm_head` at every depth, which is ~40%
of the draft spent on rows the beam never looks at. `scripts/build_shortlist.py`
builds one; drop it next to the checkpoint as `shortlist.pt` and it is picked up
automatically. A run without one says so at startup.

Everything else is defaulted and **capability-gated**: ten optimisation flags are
proposed, each gate is evaluated against the real process (does the CUDA
extension build? does the drafter's shape fit the kernel? did the engine actually
enable async scheduling?), and the resolved state of all of them is printed on one
`[cf-defaults]` line at startup. See
[`defaults.py`](src/chained_flow/defaults.py). A flag that could not engage says
so; nothing here fails silently.

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
and attention, and a KV/state hand-off none of which upstream exposes):

```bash
chained-flow tree-patch      # prints the patch path and the exact commands
```

which amounts to:

```bash
cd "$(python -c 'import vllm,os;print(os.path.dirname(os.path.dirname(vllm.__file__)))')"
patch -p1 --dry-run < .../chained_flow/patches/vllm-0.25.1-chained-flow-tree.patch
patch -p1         < .../chained_flow/patches/vllm-0.25.1-chained-flow-tree.patch
```

Then run with `VLLM_SPEC_TREE=1`, `CF_TREE_KEEP` × `CF_TREE_DEPTH` nodes, and
`num_speculative_tokens = CF_TREE_KEEP*CF_TREE_DEPTH + 1` (the extra column is a
spare mamba-state slot, never an emitted token). `chained-flow info` reports
`vLLM build : FORKED` once the patch is in; without it the tree flags report
`fork_missing` and are **not offered** rather than silently ignored.

The patch touches 6 upstream files and adds 5, all under
`vllm/v1/spec_decode/`. It is generated against 0.25.1 exactly; the version pin
is deliberate.

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

**Before quoting any speedup, read [docs/BENCHMARKING.md](docs/BENCHMARKING.md).**
The baseline is where this project has been wrong before.

```bash
./vllm/bench_cf.sh 4b base            # baseline
./vllm/bench_cf.sh 4b chain           # fork-free arm
./vllm/bench_cf.sh 4b tree            # needs the patch
./vllm/bench_forkfree.sh 4b 256       # all arms, both baselines, one run
```

`CF_PY=<venv>/bin/python` selects which vLLM to run against.

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

