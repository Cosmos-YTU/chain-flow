"""Parse the 4B + 9B + 27B accept-bench and vLLM-speedup logs into docs/deployment_results.md."""
import re, os, time
ROOT = "/home/shadeform/chain-flow"

MODELS = ["4B", "9B", "27B"]
ACCEPT_LOGS = {"4B": f"{ROOT}/logs/deploy_eval.log", "9B": f"{ROOT}/logs/deploy_eval_9b.log",
               "27B": f"{ROOT}/logs/deploy_eval_27b.log"}
SPEED_LOGS = {"4B": f"{ROOT}/vllm/test_compiled_4b_cgdraft.log", "9B": f"{ROOT}/logs/speedup_9b.log",
              "27B": f"{ROOT}/logs/speedup_27b.log"}
DOM_LABEL = {"gsm8k_heldout": "gsm8k held-out (in-distribution)", "math_reasoning": "math_reasoning (structured)",
             "HumanEval": "HumanEval (code)", "qa": "qa (short free-form)", "writing": "writing (prose)",
             "summarization": "summarization (prose)"}
DOM_ORDER = ["gsm8k_heldout", "math_reasoning", "HumanEval", "writing", "qa", "summarization"]
PROMPT_LABEL = {0: "math (step-by-step)", 1: "code (fibonacci)", 2: "short factual (\"capital of France\")", 3: "prose (ocean)"}


def parse_accept(path):
    out = {}
    if not os.path.exists(path): return out
    for line in open(path, errors="ignore"):
        m = re.search(r"\[(\w+)\] LIVE tree_accept=([0-9.]+)", line)
        if m: out[m.group(1)] = float(m.group(2))
    return out


def parse_speed(path):
    rows = []
    if not os.path.exists(path): return rows, None
    txt = open(path, errors="ignore").read()
    for m in re.finditer(r"\[(\d)\] AR ([0-9.]+) t/s \| spec ([0-9.]+) t/s \(([0-9.]+)x\) \| accept ([0-9.]+)", txt):
        rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))))
    mean = re.search(r"MEAN AR ([0-9.]+) t/s \| spec ([0-9.]+) t/s \| SPEEDUP ([0-9.]+)x \| accept ([0-9.]+)", txt)
    nat = re.search(r"vLLM-native reference:?\s*([0-9.]+) tok/s", txt)
    meand = dict(ar=float(mean.group(1)), spec=float(mean.group(2)), sx=float(mean.group(3)), acc=float(mean.group(4))) if mean else None
    if meand and nat: meand["native"] = float(nat.group(1))
    return rows, meand


lines = ["# Flow-Drafter — deployment results (RTX PRO 6000 Blackwell)\n",
         f"_generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_\n",
         "All numbers are **lossless** (spec output identical to the base model). Live tree-accept = real greedy",
         "tree acceptance verified against the backbone; speedup = spec vs autoregressive in the same compiled-forward",
         "framework (both cudagraphed). `vLLM-native` = fully-optimized base decode (reference).\n"]

# 1. acceptance table
lines.append("## 1. Acceptance — RedHatAI/speculator_benchmarks + held-out gsm8k\n")
lines.append("Live tree-accept (tokens accepted per verify pass):\n")
accepts = {m: parse_accept(ACCEPT_LOGS[m]) for m in MODELS}
cols = [m for m in MODELS if accepts[m]]  # only models that actually have measured data
lines.append("| domain | " + " | ".join(cols) + " |")
lines.append("|" + "---|" * (len(cols) + 1))
for d in DOM_ORDER:
    row = [DOM_LABEL.get(d, d)] + [f"{accepts[m][d]:.2f}" if d in accepts[m] else "—" for m in cols]
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# 2. speedup table
lines.append("## 2. vLLM speedup — compiled-forward + full-cudagraph verify + cudagraph draft\n")
for model in MODELS:
    rows, mean = parse_speed(SPEED_LOGS[model])
    if not rows and not mean:
        continue  # skip models with no speedup data yet
    lines.append(f"### {model}\n")
    lines.append("| prompt | accept | AR t/s | spec t/s | speedup |")
    lines.append("|---|---|---|---|---|")
    for pi, ar, spec, sx, acc in sorted(rows):
        lines.append(f"| {PROMPT_LABEL.get(pi, pi)} | {acc:.2f} | {ar:.0f} | {spec:.0f} | {sx:.2f}× |")
    if mean:
        lines.append(f"| **mean** | {mean['acc']:.2f} | {mean['ar']:.0f} | {mean['spec']:.0f} | **{mean['sx']:.2f}×** |")
        if "native" in mean:
            lines.append(f"\n_vLLM-native reference: {mean['native']:.0f} tok/s._\n")
    lines.append("")

lines.append("## Notes\n")
lines.append("- Accept tracks **output predictability**, not strict in/out-of-distribution: structured math/code stay high, free-form prose dips.")
lines.append("- Speedup wins clearly on high-accept traffic (code/math); low-accept prose is where the mean is dragged.")
lines.append("- Bigger base → larger win (base decode is memory-bound & scales with weight size; the latent-flow draft is base-decoupled).")

os.makedirs(f"{ROOT}/docs", exist_ok=True)
open(f"{ROOT}/docs/deployment_results.md", "w").write("\n".join(lines) + "\n")
print(f"wrote {ROOT}/docs/deployment_results.md")
