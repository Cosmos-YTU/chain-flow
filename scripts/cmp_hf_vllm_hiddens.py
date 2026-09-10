"""Are HF-collected hiddens (what the drafter trains on) numerically the same as vLLM's (what it
sees at inference)? Asserted earlier, never measured. If they differ materially, the v1 cache must be
regenerated from vLLM; if not, regeneration buys nothing and only the lag matters.

Same prompt, greedy, both engines; compare the final post-norm hidden per position.
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, "/home/shadeform/chain-flow/src")

PROMPT = "Q: What is 15*23? A: Let's think step by step."
N = int(os.environ.get("CMP_MAXTOK", "32"))
MODEL = os.environ.get("CMP_MODEL", "Qwen/Qwen3.5-4B")


def vllm_hiddens():
    from vllm import LLM, SamplingParams
    from chain_flow.vllm_plugin.flow_proposer import _install_hidden_state_hook, _STASH
    llm = LLM(model=MODEL, gpu_memory_utilization=0.55, max_model_len=2048, dtype="float16",
              max_num_seqs=8)
    _install_hidden_state_hook()
    grabbed = []
    runner = None
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    orig = GPUModelRunner.propose_draft_token_ids

    out = llm.generate([PROMPT], SamplingParams(temperature=0, max_tokens=N))
    toks = list(out[0].outputs[0].token_ids)
    # no spec config -> propose_draft_token_ids may not fire; grab via a direct forward hook instead
    return toks, grabbed, llm


def main():
    mode = os.environ.get("CMP_MODE", "hf")
    outp = os.environ.get("CMP_OUT", "/tmp/cmp")
    if mode == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda").eval()
        ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
        gen = m.generate(ids, max_new_tokens=N, do_sample=False)
        with torch.inference_mode():
            o = m(gen, output_hidden_states=True)
        h = o.hidden_states[-1][0].float().cpu()          # post-final-norm, [T, D]
        torch.save({"ids": gen[0].cpu(), "h": h}, f"{outp}_hf.pt")
        print("hf saved", tuple(h.shape))
    elif mode == "vllm":
        # Capture hiddens by wrapping the model's forward inside the vLLM worker.
        from vllm import LLM, SamplingParams
        cap = {}
        llm = LLM(model=MODEL, gpu_memory_utilization=0.55, max_model_len=2048, dtype="float16",
                  max_num_seqs=8, enforce_eager=True)

        def hook(runner):
            model = runner.model
            orig_fwd = model.forward

            def wrapped(*a, **kw):
                out = orig_fwd(*a, **kw)
                t = out[0] if isinstance(out, tuple) else out
                cap.setdefault("rows", []).append(t.detach().float().cpu())
                return out
            model.forward = wrapped

        llm.llm_engine.engine_core.engine_core.model_executor.collective_rpc(
            "apply_model", args=(lambda m: None,))
        try:
            llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner  # noqa
        except Exception:
            pass
        import chain_flow.vllm_plugin.flow_proposer as fp
        fp._install_hidden_state_hook()
        d = torch.load(f"{outp}_hf.pt")
        ids = d["ids"].tolist()
        from vllm import TokensPrompt
        out = llm.generate([TokensPrompt(prompt_token_ids=ids[:-1])],
                           SamplingParams(temperature=0, max_tokens=1))
        rows = cap.get("rows", [])
        print("vllm captured tensors:", [tuple(r.shape) for r in rows][:4])
        if rows:
            torch.save({"h": rows[0]}, f"{outp}_vllm.pt")
    else:
        a = torch.load(f"{outp}_hf.pt")["h"]
        b = torch.load(f"{outp}_vllm.pt")["h"]
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
        rel = ((a - b).norm(dim=-1) / a.norm(dim=-1))
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        print(f"positions={n}")
        print(f"  ||hf||  mean={a.norm(dim=-1).mean():.3f}   ||vllm|| mean={b.norm(dim=-1).mean():.3f}")
        print(f"  relative L2 error: mean={rel.mean():.4e} max={rel.max():.4e}")
        print(f"  cosine similarity: mean={cos.mean():.6f} min={cos.min():.6f}")


if __name__ == "__main__":
    main()
