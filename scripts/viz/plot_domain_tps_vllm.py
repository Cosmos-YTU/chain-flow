"""Per-domain THROUGHPUT (tok/s) vs stock vLLM, RedHatAI speculator-benchmark domains, batch 1.

Data: the final FULL-cudagraph sweep (/tmp/cf_native_*_fcg.json), CF_POFF=0..6 with CF_BATCH=1, so
each set is one domain in sorted(bench_data/*.jsonl) order.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMS = ["HumanEval", "math_reasoning", "qa", "rag", "summarization", "translation", "writing"]
LAB = {"HumanEval": "code", "math_reasoning": "math", "qa": "qa", "rag": "rag",
       "summarization": "summ", "translation": "transl", "writing": "writing"}
ARMS = [("base", "stock vLLM", "#cbd5e1"),
        ("chain", "chain (K=4)", "#93c5fd"),
        ("tree", "tree (16-node)", "#2563eb")]
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"


def arm(size, tag):
    mode = "base" if tag == "base" else "spec"
    d = json.load(open(f"/tmp/cf_native_{mode}_{size}_{tag}_fcg.json"))
    return {s["poff"]: s for s in d["sets"]}


fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.6), dpi=150, sharex=True)
fig.patch.set_facecolor("white")
x = np.arange(len(DOMS))
w = 0.27

for ax, (size, title) in zip(axes, [("27b", "Qwen3.5-27B"), ("4b", "Qwen3.5-4B")]):
    ax.set_facecolor("white")
    base = arm(size, "base")
    top = 0
    for j, (tag, lbl, c) in enumerate(ARMS):
        a = arm(size, tag)
        vals = [a[i]["tps"] for i in range(7)]
        top = max(top, max(vals))
        off = (j - 1) * w
        bars = ax.bar(x + off, vals, w, label=lbl, color=c, edgecolor="white", lw=1, zorder=3)
        for b, v, i in zip(bars, vals, range(7)):
            sp = "" if tag == "base" else f"\n{v / base[i]['tps']:.2f}x"
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.012, f"{v:.0f}{sp}",
                    ha="center", va="bottom", fontsize=7.2, color=ink, linespacing=1.2)
    bmean = np.mean([base[i]["tps"] for i in range(7)])
    tmean = np.mean([arm(size, "tree")[i]["tps"] for i in range(7)])
    ax.axhline(bmean, color="#dc2626", lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.set_title(f"{title}    stock {bmean:.0f} tok/s  →  tree {tmean:.0f} tok/s"
                 f"    ·    mean {tmean / bmean:.2f}x",
                 fontsize=11.5, color=ink, fontweight="bold", pad=8, loc="left")
    ax.set_ylabel("throughput (tok/s)", fontsize=10, color=ink)
    ax.set_ylim(0, top * 1.22)
    ax.tick_params(colors=muted, length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.yaxis.grid(True, color=grid, lw=1, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=9.5, loc="upper right", frameon=False, ncol=3)
    for t in leg.get_texts():
        t.set_color(ink)

axes[1].set_xticks(x)
axes[1].set_xticklabels([LAB[d] for d in DOMS], fontsize=10.5, color=ink)

fig.suptitle("Chained-Flow speculative decoding — per-domain throughput vs stock vLLM",
             fontsize=14.5, color=ink, fontweight="bold", x=0.5, y=0.985, ha="center")
fig.text(0.5, 0.008,
         "RedHatAI speculator-benchmark domains · batch 1, greedy, max_tokens 256 · one prompt per "
         "domain · published HF drafters, bit-exact vs stock.\n"
         "Red dashed = that model's mean stock throughput. Note the two panels have very different "
         "y-scales (27B ~26 tok/s, 4B ~140 tok/s).",
         ha="center", fontsize=8.5, color=muted, linespacing=1.5)
fig.tight_layout(rect=[0, 0.035, 1, 0.965])
out = "/home/shadeform/chain-flow/docs/vllm_domain_tps.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)

for size in ("27b", "4b"):
    for tag, lbl, _ in ARMS:
        a = arm(size, tag)
        v = [a[i]["tps"] for i in range(7)]
        print(f"{size:4s} {lbl:16s} " + " ".join(f"{LAB[d]}={s:.1f}" for d, s in zip(DOMS, v))
              + f"  | mean {np.mean(v):.1f} tok/s")
