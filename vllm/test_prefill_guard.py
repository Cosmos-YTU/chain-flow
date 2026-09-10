"""Regression test for the K+1 prefill / uniform-decode cudagraph collision.

THE BUG (upstream vLLM 0.25.1, reproduced with stock `method="ngram"`): a request whose PROMPT
is exactly ``num_speculative_tokens + 1`` tokens long schedules exactly ``K+1`` tokens in one
row, which satisfies `GPUModelRunner._is_uniform_decode`'s shape-only test
(``max_num_scheduled_tokens == 1 + num_speculative_tokens and num_tokens == max * num_reqs``).
The batch is then dispatched to the FULL *decode* cudagraph. On a hybrid model (Qwen3.5 = GDN
linear attention + full attention) the captured decode graph holds the recurrent gated-delta-rule
STEP, not the chunked prefill scan, so the replay computes the wrong linear-attention output and
leaves a wrong recurrent state: the request is corrupt from its very first emitted token.

`chain_flow.vllm_plugin.flow_proposer._install_uniform_decode_guard` fixes it by making
`uniform_decode` a statement about PHASE (does any row still have prompt tokens left to compute?)
rather than about shape. This file is the test that it stays fixed -- and, because the collision
is at ``plen == K+1`` for EVERY K, that means sweeping both axes:

    K in {3..8}  x  prompt length in {1..32}  x  {base, spec}

The signature to look for is stark and K-dependent: at ``plen == K+1`` and nowhere else, the
generation degenerates (measured: repeated token id 0). So the check is not "the arms agree on
average", it is "the diagonal is not special".

    CF_MODE=base ./test_prefill_guard.py           # reference tokens, one process
    CF_MODE=spec CF_K=5 ./test_prefill_guard.py    # one K
    ./bench_prefill_guard.sh                       # the whole sweep + the comparison

Prompts are built as raw token ids so the length is EXACT -- tokenizing text and hoping for
``K+1`` tokens is how a sweep silently misses the one cell that matters.
"""
import importlib.util
import json
import os
import sys

if importlib.util.find_spec("chain_flow") is None:
    sys.path.insert(0, "/home/shadeform/chain-flow/src")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams  # noqa: E402

MODE = os.environ.get("CF_MODE", "base")
MODEL = os.environ.get("CF_MODEL", "Qwen/Qwen3.5-4B")
K = int(os.environ.get("CF_K", "5"))
GMU = float(os.environ.get("CF_GMU", "0.55"))
MAXTOK = int(os.environ.get("CF_MAXTOK", "32"))
PLENS = [int(x) for x in os.environ.get("CF_PLENS", ",".join(str(i) for i in range(1, 33))).split(",")]
OUT = os.environ.get("CF_OUT_JSON",
                     f"/tmp/cf_prefill_{MODE}{'' if MODE == 'base' else f'_K{K}'}.json")

kw = dict(model=MODEL, gpu_memory_utilization=GMU, max_model_len=2048, dtype="float16",
          max_num_seqs=64)
if MODE == "spec":
    kw["speculative_config"] = {"method": "custom_class",
                                "model": "chain_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
                                "num_speculative_tokens": K}
_as = os.environ.get("CF_ASYNC_SCHED")
if _as is not None:
    kw["async_scheduling"] = _as == "1"
llm = LLM(**kw)

tk = llm.get_tokenizer()
# One long, ordinary passage; every prompt is its first `n` ids. Using a PREFIX family (rather
# than n unrelated prompts) means the only thing varying down the sweep is the length, which is
# the variable under test.
SEED_TEXT = ("The study of numerical optimisation begins with a simple question: given a "
             "function and a starting point, which direction should you move, and how far, to "
             "make the value smaller? Gradient descent answers it by following the negative "
             "gradient, and the learning rate decides the step size. Too large and the "
             "iterates oscillate or diverge; too small and progress stalls long before the "
             "minimum is reached. Everything else in the field is a refinement of that trade.")
seed_ids = tk(SEED_TEXT, add_special_tokens=False)["input_ids"]
assert len(seed_ids) >= max(PLENS), f"seed text is only {len(seed_ids)} tokens"

sp = SamplingParams(temperature=0, max_tokens=MAXTOK)
prompts = [{"prompt_token_ids": seed_ids[:n]} for n in PLENS]
llm.generate(prompts, sp)                      # warmup: our draft cudagraph is captured lazily

res = {}
for n, o in zip(PLENS, llm.generate(prompts, sp)):
    ids = list(o.outputs[0].token_ids)
    res[str(n)] = ids

with open(OUT, "w") as f:
    json.dump({"mode": MODE, "K": K, "model": MODEL, "maxtok": MAXTOK, "out": res}, f)
print(f"[prefill-guard] {MODE} K={K}: wrote {len(res)} prompt lengths -> {OUT}", flush=True)

# A degenerate generation is the bug's signature, so flag it here as well as in the comparison
# -- but it is only EVIDENCE when base did not do the same thing. A 1-token prompt degenerates in
# base mode too (nothing to condition on), which is why the verdict belongs to the comparison
# step and this line says "check base" rather than "corrupted".
for n, ids in res.items():
    if len(set(ids)) == 1:
        print(f"[prefill-guard] plen={n} emitted ONE repeated token id {ids[0]} x{len(ids)} "
              f"-- the K+1 corruption signature IF base does not do it too (it does at plen=1)",
              flush=True)
