"""Tie-probe: dump the target model's top-k logits for the row that produces each
emitted token, in BOTH base (M=1) and tree-verify (M=num_rows) paths.

Purpose: test whether the 27B poff=4 token-161 divergence is a floating-point
argmax TIE (verify GEMM at M=41 vs base GEMM at M=1) or a real indexing bug.

Env: same as bench_cf.sh plus
  CF_PROBE_OUT   where to write the json (default /tmp/cf_probe_<mode><tag>.json)
  CF_PROBE_TOPK  how many logits to record per row (default 8)
Nothing here changes a default: it is a separate driver, all hooks are local.
"""
import os, sys, time, json, glob, itertools

sys.path.insert(0, "/home/shadeform/chained-flow/src")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

TOPK = int(os.environ.get("CF_PROBE_TOPK", "8"))
MODE = os.environ.get("CF_MODE", "base")
MODEL = os.environ.get("CF_MODEL", "Qwen/Qwen3.5-27B")
K = int(os.environ.get("CF_K", "41"))
GMU = float(os.environ.get("CF_GMU", "0.85"))

REC = {"on": False, "steps": []}

# CF_PROBE_FP32=1: compute the lm_head projection in float32 so the emitted logits
# carry ~2**13 finer resolution than the fp16 default.  Diagnostic only -- it is slow
# and it is NOT a default; it exists to ask whether the near-ties survive fp32.
if os.environ.get("CF_PROBE_FP32", "0") == "1":
    import torch.nn.functional as _F
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    def _get_logits_fp32(self, hidden_states, lm_head, embedding_bias):
        w = lm_head.weight.float()
        lg = _F.linear(hidden_states.float(), w,
                       None if embedding_bias is None else embedding_bias.float())
        return lg[..., : self.org_vocab_size]

    LogitsProcessor._get_logits = _get_logits_fp32
    print("[probe] lm_head forced to float32", flush=True)


def _topk_cpu(logits):
    """[rows, vocab] -> (vals[rows,TOPK] float64 list, ids[rows,TOPK] list)."""
    v, i = torch.topk(logits.float(), TOPK, dim=-1)
    return v.double().cpu().tolist(), i.cpu().tolist()


# ---------------------------------------------------------------- base hook
from vllm.v1.sample.sampler import Sampler

_orig_greedy = Sampler.greedy_sample


def _greedy(logits):
    if REC["on"]:
        vals, ids = _topk_cpu(logits)
        REC["steps"].append({"kind": "sampler", "dtype": str(logits.dtype),
                             "rows": int(logits.shape[0]), "vals": vals, "ids": ids})
    return _orig_greedy(logits)


Sampler.greedy_sample = staticmethod(_greedy)

# ---------------------------------------------------------------- tree hook
import vllm.v1.sample.rejection_sampler as _rs

_orig_tree = _rs.tree_rejection_sample


def _tree(metadata, logits):
    out = _orig_tree(metadata, logits)
    if REC["on"]:
        from vllm.v1.spec_decode import tree_state
        cur = tree_state.get_current()
        vals, ids = _topk_cpu(logits)
        rec = {"kind": "tree", "dtype": str(logits.dtype), "rows": int(logits.shape[0]),
               "vals": vals, "ids": ids,
               "draft": metadata.draft_token_ids.cpu().tolist(),
               "parents": metadata.tree_parents.cpu().tolist(),
               "cu_draft": metadata.cu_num_draft_tokens.cpu().tolist(),
               "cu_sampled": metadata.cu_num_sampled_tokens.cpu().tolist(),
               "sampled": out.cpu().tolist()}
        if cur is not None and cur.accepted_path is not None:
            rec["path"] = cur.accepted_path.cpu().tolist()
            rec["alen"] = cur.accepted_len.cpu().tolist()
        REC["steps"].append(rec)
    return out


_rs.tree_rejection_sample = _tree
# forward() resolves the name from the module global, so the rebind above is enough,
# but rebind the imported alias too in case a future edit imports it by value.

# ---------------------------------------------------------------- prompts
_pd = os.environ.get("CF_PROMPTS", "/home/shadeform/chained-flow/bench_data")
_OFFS = [int(x) for x in os.environ.get("CF_POFF", "4").split(",")]
_N = int(os.environ.get("CF_BATCH", "1"))
_per = []
for f in sorted(glob.glob(os.path.join(_pd, "*.jsonl"))):
    _per.append([json.loads(l)["prompt"] for l in open(f) if l.strip()])
_rr = [p for grp in itertools.zip_longest(*_per) for p in grp if p]
SETS = [[_rr[(i + o) % len(_rr)] for i in range(_N)] for o in _OFFS]
PROMPTS = SETS[0]

sp = SamplingParams(temperature=0.0, max_tokens=int(os.environ.get("CF_MAXTOK", "256")))
kw = dict(model=MODEL, gpu_memory_utilization=GMU, max_model_len=2048, dtype="float16",
          enforce_eager=os.environ.get("CF_EAGER", "0") == "1", max_num_seqs=64)
if MODE == "spec":
    kw["speculative_config"] = {"method": "custom_class",
                                "model": "chained_flow.vllm_plugin.flow_proposer.FlowDrafterProposer",
                                "num_speculative_tokens": K}
llm = LLM(**kw)

llm.generate(PROMPTS, sp)
llm.generate(PROMPTS, sp)

sets = []
for _off, _ps in zip(_OFFS, SETS):
    REC["on"] = True
    REC["steps"] = []
    t0 = time.time()
    outs = llm.generate(_ps, sp)
    dt = time.time() - t0
    REC["on"] = False
    toks = [list(o.outputs[0].token_ids) for o in outs]
    sets.append({"poff": _off, "tokens": toks, "n": sum(len(t) for t in toks),
                 "secs": dt, "steps": REC["steps"]})
    print(f"\n[probe:{MODE}] poff={_off} {sets[-1]['n']} tok / {dt:.2f}s, "
          f"{len(REC['steps'])} steps", flush=True)

out_path = os.environ.get("CF_PROBE_OUT",
                          f"/tmp/cf_probe_{MODE}{os.environ.get('CF_TAG', '')}.json")
json.dump({"mode": MODE, "topk": TOPK, "sets": sets}, open(out_path, "w"))
print(f"\n[probe:{MODE}] DONE -> {out_path}", flush=True)
