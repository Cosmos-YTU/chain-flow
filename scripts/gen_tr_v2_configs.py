"""Generate the Turkish-v2 teacher-state collection configs for 4B / 9B / 27B.

WHY THE THREE SIZES DIFFER
--------------------------
4B and 27B v1 both collected the SAME 29,100 rows out of `bench_data_tr/*.train.jsonl`.
`bench_data_tr_v2` was rebuilt with the same seed and larger per-source `train` counts, and the
builder fills HOLDOUT first from a seeded shuffle -- so the v2 files are a strict PREFIX EXTENSION
of the v1 files (verified byte-for-byte, holdouts identical). Their existing teacher states stay
valid and only the TAIL needs collecting.

9B v1 is the exception: it collected from `data/turkish_prompts/*.jsonl` (14,746 rows,
`format_name: raw_prompt`), a different and smaller prompt set. There is no prefix relationship,
so unifying 9B onto the same corpus means collecting all 56,699 rows. That is the price of making
the three sizes comparable; it was NOT comparable before.
"""
import json, os

# (source stem, source tag, v1 rows already collected for 4B/27B, rows available in v2)
SRC = [
    ("tr-instructurca",         "tr_instruct",   24000, 39264),
    ("tr-function-calling-20k", "tr_funccall",    3000,  8000),
    ("tr-tool-calling-10k",     "tr_toolcall",     900,  7290),
    ("tr-multiturn",            "tr_multiturn",   1200,  2145),
]

# WHICH ROWS TO TAKE. Both presets are prefix ranges of the same v2 files, so switching costs
# nothing already collected.
#
#   instruct  -- all instructurca and all multiturn; tool/function held at their v1 counts.
#   varied    -- everything available.
#
# `instruct` is the DEFAULT and it is the one to use unless you have a reason not to. The served
# per-domain results say why: function-calling 2.48x and tool-calling 2.43x are already the
# STRONGEST domains (accept ~3.1), while instruct 1.84x and multiturn 1.62x (accept 2.24 / 1.97)
# are what drags the pooled 1.94x down. `varied` pours 11,390 rows into the two strong domains and
# drops instructurca from 58% of prompt tokens to 38% -- it optimises where the drafter is already
# good, and dilutes where it is not. It is also 42% more collection (19.1M vs 11.0M prompt tokens).
#
# The "diverse data lifts weak domains" result from the English v2 mix does NOT argue for `varied`
# here: that gain came from adding NEW KINDS of data (creative prose, summarization, translation),
# not from more rows of domains already represented.
PRESETS = {
    "instruct": {"tr_instruct": 39264, "tr_funccall": 3000, "tr_toolcall": 900, "tr_multiturn": 2145},
    "varied":   {"tr_instruct": 39264, "tr_funccall": 8000, "tr_toolcall": 7290, "tr_multiturn": 2145},
}
# per-size batch sizes copied from the v1 configs that are known to fit
# Collection batch sizes. The v1 values (27B: gen 24 / hid 12) were sized for 97 GB cards and are
# the single biggest lever on collection time that costs no new code.
#
# Collection is HF `model.generate()` -- NOT vLLM. The generation phase is 256 sequential decode
# steps per batch against ONE forward for the hidden-state pass, so it dominates, and HF generate
# pads every sequence to the longest in its batch and runs without CUDA graphs or continuous
# batching. vLLM would be materially faster for that phase (it cannot do the hidden-state pass --
# that needs `output_hidden_states`), but it is a separate venv because of the torch pin and a code
# path that does not exist yet. Raising the batch is the version of that win available today:
# at 27B bf16 the weights are ~54 GB and the KV for batch 256 at ~860 tokens is ~26 GB, so ~80 GB
# against a B300's ~288.
#
# !! gen and hid DO NOT SCALE TOGETHER. Raising them in step OOMs phase 2. !!
#
# Phase 1 (generate) holds weights + KV: ~0.5 GB/sequence at 27B, so gen=256 runs comfortably and
# was measured at 2.48 it/s.
#
# Phase 2 (`output_hidden_states=True`) materialises EVERY layer's hidden states at once, plus the
# activations to produce them -- `torch_chunk_gated_delta_rule` does a full
# `transpose(1,2).contiguous()` copy. MEASURED at 27B: hid=128 put 266 GB in use on a 267 GB card
# and OOMed, i.e. ~1.66 GB per sequence, about 3x what the hidden-state tensor alone accounts for.
# Do not derive this from layers x tokens x hidden; that underestimates it by 3x.
#
#   hid=32  -> ~54 GB weights + ~53 GB  = ~107 GB   (default; still 2.7x the v1 value of 12)
#   hid=64  -> ~54 GB weights + ~106 GB = ~160 GB   (workable, less margin for long prompts)
#   hid=128 -> OOM, observed
#
# Prompt length varies a lot by source (tool-calling averages 825 tokens against function-calling's
# 563), so the margin at hid=64 is thinner on some shards than others.
#
# Override per run without regenerating: CF_GEN_BS / CF_HID_BS.
_GEN = int(os.environ.get("CF_GEN_BS", "0"))
_HID = int(os.environ.get("CF_HID_BS", "0"))
SIZES = {
    "4btr":  dict(model="Qwen/Qwen3.5-4B",  gen=_GEN or 512, hid=_HID or 128, reuse=True),
    "9btr":  dict(model="Qwen/Qwen3.5-9B",  gen=_GEN or 256, hid=_HID or 64,  reuse=False),
    "tr27b": dict(model="Qwen/Qwen3.5-27B", gen=_GEN or 256, hid=_HID or 32,  reuse=True),
}
# Shard size bounds HOST memory as well as wall-clock: phase 2 accumulates every row and only
# writes at the end of a shard, so peak RAM is proportional to this. With final_hidden stored as
# a numpy array a 27B row is ~8.8 MB (860 tokens x 5120 x fp16), so 2000 rows is ~18 GB plus the
# Arrow conversion. It was 4000 with rows held as nested PYTHON floats -- ~140 MB each, ~560 GB
# per shard -- which is what took a machine down. Override with CF_SHARD_ROWS.
SHARD_ROWS = int(os.environ.get("CF_SHARD_ROWS", "2000"))

# Warm start. `continue` resumes the SHIPPED Turkish checkpoint -- "carry on from where it left
# off" -- which is what you want when the goal is a better Turkish drafter. `parent` restarts from
# the English v2 checkpoint, which is what v1 did and is the only option that keeps a clean
# v1-vs-v2 comparison of the DATA (continuing confounds more data with more epochs on the old rows).
INIT = {
    "continue": {"4btr": "selimaktas/Flow-Drafter-4B-tr",
                 "9btr": "selimaktas/Flow-Drafter-9B-tr",
                 "tr27b": "selimaktas/Flow-Drafter-Qwen3.5-27B-tr"},
    "parent":   {"4btr": "selimaktas/Flow-Drafter-4B-v2",
                 "9btr": "selimaktas/Flow-Drafter-9B-v2",
                 "tr27b": "selimaktas/Flow-Drafter-Qwen3.5-27B-v2"},
}
# Per-size geometry and the 2xB300 batch shape.
#
# `eff` is that size's OWN v1 effective batch (per_device x accum x 2 GPUs) and is preserved
# exactly, so only the microbatch/accumulation SPLIT changes and the optimisation trajectory is
# untouched. They differ per size -- 4B 12,288, 9B 24,576, 27B 6,144 -- so there is no single rule;
# `assert mb * accum * 2 == eff` below is what keeps a hand edit from silently changing the run.
# Doubling effective batch at a constant LR is a different experiment, not a speed knob.
#
# The microbatch is far larger than v1 because the fused head removes the [B, K, 248320] activation
# that forced the 27B microbatch down to 64, and a B300 has ~288 GB.
# UNVERIFIED ON B300 (no card was free to test): if a run OOMs, halve `mb` and double `accum` --
# the assert will tell you immediately if the pair stops matching.
GEO = {
    "4btr":  dict(model="Qwen/Qwen3.5-4B",  ed=640,  ls=640,  vi=1920, mb=2048, accum=3,  lr=1.5e-4,
                  eff=12288, vae="out/vae/ckpts/transformer-hidden-4bx-2560-latent640-fp16"),
    "9btr":  dict(model="Qwen/Qwen3.5-9B",  ed=1024, ls=1024, vi=3072, mb=2048, accum=6,  lr=1.5e-4,
                  eff=24576, vae="out/vae/ckpts/transformer-hidden-9bx-4096-latent1024-fp16"),
    "tr27b": dict(model="Qwen/Qwen3.5-27B", ed=1024, ls=1024, vi=4096, mb=512,  accum=6,  lr=1.0e-4,
                  eff=6144, vae="out/vae/ckpts/transformer-hidden-q3527bx-5120-latent1024-fp16"),
}
for _s, _g in GEO.items():
    assert _g["mb"] * _g["accum"] * 2 == _g["eff"], (
        f"{_s}: {_g['mb']} x {_g['accum']} x 2 = {_g['mb']*_g['accum']*2}, "
        f"but v1's effective batch was {_g['eff']}")


def write_train_config(size, preset, init_mode, epochs):
    g = GEO[size]
    path = f"train_configs/recovered/joint_{size}_v2_{preset}.yaml"
    with open(path, "w") as f:
        f.write(f"""# Turkish v2 ({preset} preset, warm start: {init_mode}). Generated by
# scripts/gen_tr_v2_configs.py -- edit that, not this.
#
# Batch shape targets 2xB300 (~288 GB each). Effective batch is v1's 6144 windows either way, so
# only throughput changes, not the optimisation path. The fused head (CF_FUSED_HEAD=1, default on)
# is what makes a microbatch this size fit: it removes the [B, K, 248320] logits tensor that held
# the 27B microbatch at 64. Both are UNVERIFIED on B300 -- halve mb and double accum on an OOM.
#
# Dense checkpointing on purpose. The v1 sweep found the two Turkish axes saturate at very
# different points -- in-distribution accept plateaued near 0.7 epochs while out-of-distribution
# accept kept climbing to ~2.5 -- so only a per-checkpoint ladder locates the knee.
# save_total_limit sits ABOVE the checkpoint count: set below it, the trainer silently deletes the
# band the ladder needs and leaves a plausible curve that cannot find the optimum.
model_id: {g['model']}
context_size: 8
draft_length: 8
chunk_size: 4
expert_dim: {g['ed']}
num_heads: 8
ffn_multiplier: 6
num_drafter_layers: 8
num_flow_steps: 2
init_mode: delta
detach_previous_chunks: true
markov_rank: 256
path_order: 8
path_ffn_multiplier: 2
lambda_cov: 0.3
cov_b: 8
cov_margin: 1.0
tree_top_b: 8
tree_max_nodes: 64
architecture: tree_flow
device: cuda
vae_dir: {g['vae']}
latent_size: {g['ls']}
vae_type: transformer_hidden
vae_intermediate_size: {g['vi']}
vae_num_layers: 2
vae_num_heads: 8
vae_max_sequence_length: 16
train_vae: true
lambda_vae_recon: 0.5
init_from: {INIT[init_mode][size]}
dataset_path: data/flow_cache/stage1_{size}_v2{preset}_mix_k4
dataset_split: train
materialize_rows: true
lambda_flow: 0.2
lambda_hidden: 0.25
lambda_cos: 0.1
lambda_ce: 0.3
lambda_accept: 0.15
gamma: 0.8
output_dir: out/flow/ckpts/tree-vae-joint-{size}v2{preset}-{g['ls']}-k8-l8
learning_rate: {g['lr']}
num_train_epochs: {epochs}
per_device_train_batch_size: {g['mb']}
gradient_accumulation_steps: {g['accum']}
lr_scheduler_type: constant
save_steps: 400
logging_steps: 25
save_total_limit: 16
dataloader_num_workers: 4
report_to: none
bf16: true
ddp_find_unused_parameters: true
""")
    return path


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default="instruct")
    ap.add_argument("--init", choices=sorted(INIT), default="continue")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--no-reuse", action="store_true",
                    help="collect EVERY row instead of only the tail. Required on a machine that "
                         "does not have the v1 flow cache -- the tail-only plan assumes v1's rows "
                         "are already represented by that cache, and it is 124 GB of DERIVED data "
                         "that is not published anywhere.")
    opts = ap.parse_args()
    plan = PRESETS[opts.preset]
    if opts.no_reuse:
        for _c in SIZES.values():
            _c["reuse"] = False
    print(f"preset={opts.preset}  target rows/source={plan}\n")
    for size, cfg in SIZES.items():
        out = f"collect_configs/stage1_{size}_v2"
        # CLEAR FIRST. Shard filenames encode their row range, so changing preset or reuse mode
        # produces a DIFFERENT set of names and the old ones survive alongside the new. That is not
        # cosmetic: `multiturn_0_2145` next to a leftover `multiturn_1200_2145` collects rows
        # [1200,2145) twice and silently duplicates them into the training cache. Nothing downstream
        # would notice -- the shards are valid, the concat succeeds, the row count merely looks high.
        if os.path.isdir(out):
            for stale in os.listdir(out):
                os.remove(os.path.join(out, stale))
        os.makedirs(out, exist_ok=True)
        n = 0
        for stem, src, v1_end, v2_max in SRC:
            v2_end = min(plan[src], v2_max)
            start = v1_end if cfg["reuse"] else 0
            if start >= v2_end:
                continue
            a = start
            while a < v2_end:
                b = min(a + SHARD_ROWS, v2_end)
                name = f"{src.replace('tr_','')}_{a}_{b}"
                with open(f"{out}/{name}.yaml", "w") as fh:
                    fh.write(
                        f"# Turkish v2, {cfg['model']}. Rows [{a},{b}) of "
                        f"bench_data_tr_v2/{stem}.train.jsonl.\n"
                        f"# Prompts are chat-templated PREFIXES; continuations are generated by the\n"
                        f"# REAL target (the corpus's own completions came from Qwen3.6-35B-A3B-FP8\n"
                        f"# and are discarded). enable_thinking=False is baked into the prompt text.\n"
                        f"model_id: {cfg['model']}\n"
                        f"dataset_name: json\n"
                        f"data_files: bench_data_tr_v2/{stem}.train.jsonl\n"
                        f"dataset_config: null\nsplit: train\n"
                        f"dataset_start: {a}\ndataset_end: {b}\n"
                        f"source: {src}\nformat_name: pretemplated\n"
                        f"generation_max_new_tokens: 256\n"
                        f"generation_batch_size: {cfg['gen']}\n"
                        f"hidden_batch_size: {cfg['hid']}\n"
                        f"storage_dtype: float16\ndtype: float16\ndevice: cuda:0\n"
                        f"output_dir: teacher_states/stage1-{size}-v2-{name}\n")
                n += b - a
                a = b
        # Overlap is the failure that duplicating rows would cause, so assert on the RANGES rather
        # than trusting that clearing the directory was enough.
        import collections as _c
        _by = _c.defaultdict(list)
        for _f in os.listdir(out):
            _d = open(os.path.join(out, _f)).read()
            _src = [l.split(": ")[1] for l in _d.splitlines() if l.startswith("source: ")][0]
            _s0 = int([l.split(": ")[1] for l in _d.splitlines() if l.startswith("dataset_start:")][0])
            _e0 = int([l.split(": ")[1] for l in _d.splitlines() if l.startswith("dataset_end:")][0])
            _by[_src].append((_s0, _e0))
        for _src, _rs in _by.items():
            _rs.sort()
            for (_a1, _b1), (_a2, _b2) in zip(_rs, _rs[1:]):
                if _a2 < _b1:
                    raise SystemExit(f"{out}: overlapping ranges for {_src}: "
                                     f"[{_a1},{_b1}) and [{_a2},{_b2}) would collect rows twice")
        tc = write_train_config(size, opts.preset, opts.init, opts.epochs)
        print(f"{out:38s} shards={len(os.listdir(out)):3d}  rows={n:6d}"
              f"  ({'tail only' if cfg['reuse'] else 'FULL recollect'})  -> {tc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
