"""Per-domain vLLM speedup vs stock, RedHatAI speculator-benchmark domains, batch 1.

Data: the final FULL-cudagraph sweep (/tmp/cf_native_*_fcg.json), CF_POFF=0..6 with CF_BATCH=1, so
each set is one domain in sorted(bench_data/*.jsonl) order. Speedup = spec t/s / base t/s per domain.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOMS = ["HumanEval", "math_reasoning", "qa", "rag", "summarization", "translation", "writing"]
LAB = {"HumanEval": "code", "math_reasoning": "math", "qa": "qa", "rag": "rag",
       "summarization": "summ", "translation": "transl", "writing": "writing"}
ARMS = [("chain", "chain (K=4)", "#93c5fd"), ("tree", "tree (16-node)", "#2563eb")]
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"


def sets(path):
    d = json.load(open(path))
    return {s["poff"]: s for s in d["sets"]}


def arm(size, tag):
    mode = "base" if tag == "base" else "spec"
    return sets(f"/tmp/cf_native_{mode}_{size}_{tag}_fcg.json")


fig, axes = plt.subplots(2, 1, figsize=(11.5, 8.4), dpi=150, sharex=True)
fig.patch.set_facecolor("white")
x = np.arange(len(DOMS))
w = 0.38

for ax, (size, title) in zip(axes, [("27b", "Qwen3.5-27B"), ("4b", "Qwen3.5-4B")]):
    ax.set_facecolor("white")
    base = arm(size, "base")
    for j, (tag, lbl, c) in enumerate(ARMS):
        a = arm(size, tag)
        vals = [a[i]["tps"] / base[i]["tps"] for i in range(7)]
        acc = [a[i].get("accept") for i in range(7)]
        off = (j - 0.5) * w
        bars = ax.bar(x + off, vals, w, label=lbl, color=c, edgecolor="white", lw=1, zorder=3)
        for b, v, ac in zip(bars, vals, acc):
            txt = f"{v:.2f}x" + (f"\nacc {ac:.1f}" if ac else "")
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, txt, ha="center", va="bottom",
                    fontsize=7.5, color=ink, linespacing=1.25)
    mean = np.mean([arm(size, "tree")[i]["tps"] / base[i]["tps"] for i in range(7)])
    ax.axhline(1.0, color="#dc2626", lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.set_title(f"{title}    stock vLLM {np.mean([base[i]['tps'] for i in range(7)]):.0f} tok/s"
                 f"    ·    tree mean {mean:.2f}x",
                 fontsize=11.5, color=ink, fontweight="bold", pad=8, loc="left")
    ax.set_ylabel("speedup vs stock vLLM", fontsize=10, color=ink)
    ax.tick_params(colors=muted, length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.yaxis.grid(True, color=grid, lw=1, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(fontsize=9.5, loc="upper right", frameon=False, ncol=2)
    for t in leg.get_texts():
        t.set_color(ink)

axes[0].set_ylim(0, 3.5)
axes[1].set_ylim(0, 1.6)
axes[1].set_xticks(x)
axes[1].set_xticklabels([LAB[d] for d in DOMS], fontsize=10.5, color=ink)

fig.suptitle("Chained-Flow speculative decoding — per-domain speedup vs stock vLLM",
             fontsize=14.5, color=ink, fontweight="bold", x=0.5, y=0.985, ha="center")
fig.text(0.5, 0.008,
         "RedHatAI speculator-benchmark domains · batch 1, greedy, max_tokens 256 · one prompt per "
         "domain · published HF drafters, bit-exact vs stock.\n"
         "Red dashed = 1.0x break-even. 27B clears it on every domain; 4B does not — its draft costs "
         "about as much as the base decode it replaces.",
         ha="center", fontsize=8.5, color=muted, linespacing=1.5)
fig.tight_layout(rect=[0, 0.035, 1, 0.965])
out = "/home/shadeform/chained-flow/docs/vllm_domain_speedup.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)

for size in ("27b", "4b"):
    base = arm(size, "base")
    for tag, lbl, _ in ARMS:
        a = arm(size, tag)
        v = [a[i]["tps"] / base[i]["tps"] for i in range(7)]
        print(f"{size:4s} {lbl:16s} " + " ".join(f"{d[:5]}={s:.2f}" for d, s in zip(DOMS, v))
              + f"  | mean {np.mean(v):.2f}x")
