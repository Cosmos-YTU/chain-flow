"""v1 vs v2 vLLM speedup across ALL 7 RedHatAI domains, for 4B and Qwen3.5-27B."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

D = json.load(open("/tmp/evalA_speedup_data.json"))
DOMS = ["math_reasoning", "HumanEval", "rag", "translation", "qa", "writing", "summarization"]
DLAB = {"math_reasoning": "math", "HumanEval": "code", "rag": "rag", "translation": "transl",
        "qa": "qa", "writing": "writing", "summarization": "summ"}
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"

fig, axes = plt.subplots(2, 1, figsize=(12.5, 9), dpi=150, sharex=True)
fig.patch.set_facecolor("white")
PANELS = [("4B", "4B v1", "4B v2", "#93c5fd", "#2563eb", 495),
          ("Qwen3.5-27B", "27B v1", "27B v2", "#fdba74", "#ea580c", 101)]
x = np.arange(len(DOMS)); w = 0.4

for ax, (title, k1, k2, c1, c2, nat) in zip(axes, PANELS):
    ax.set_facecolor("white")
    for k, c, off in [(k1, c1, -w/2), (k2, c2, +w/2)]:
        vals = [D[k][d][0] for d in DOMS]
        ll = [D[k][d][2] == D[k][d][3] for d in DOMS]
        bars = ax.bar(x + off, vals, w, label=k.split()[-1], color=c, edgecolor="white", linewidth=1, zorder=3)
        for b, v, is_ll in zip(bars, vals, ll):
            if not is_ll: b.set_hatch("///")
            ax.text(b.get_x()+b.get_width()/2, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=ink)
    ax.axhline(1.0, color=muted, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.set_title(f"{title}   (native {nat} t/s)", fontsize=12, color=ink, fontweight="bold", pad=6, loc="left")
    ax.set_ylim(0, 1.5); ax.set_yticks(np.arange(0, 1.6, 0.5)); ax.tick_params(colors=muted, length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid); ax.yaxis.grid(True, color=grid, lw=1, zorder=0); ax.set_axisbelow(True)
    ax.set_ylabel("speedup (x)", fontsize=10, color=ink)
    leg = ax.legend(fontsize=9.5, loc="upper right", frameon=False, ncol=2)
    for tt in leg.get_texts(): tt.set_color(ink)

axes[1].set_xticks(x); axes[1].set_xticklabels([DLAB[d] for d in DOMS], fontsize=10, color=ink)
fig.suptitle("Flow-Drafter v1 vs v2 — vLLM speedup across all 7 RedHatAI domains",
             fontsize=14, color=ink, fontweight="bold", x=0.5, y=0.99, ha="center")
fig.text(0.5, 0.005, "4 prompts/domain (long summarization/rag truncated to fit). Dashed = 1.0x breakeven. "
         "Hatched bars = fp16 argmax-tie divergence on some tokens (not fully bit-exact). "
         "v2's diverse data lifts the free-form domains (writing/qa/summ/rag); math/code ~flat.",
         ha="center", fontsize=8.5, color=muted)
fig.tight_layout(rect=[0, 0.02, 1, 0.98])
out = "/home/shadeform/chain-flow/docs/alldomain_speedup_v1_vs_v2.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
