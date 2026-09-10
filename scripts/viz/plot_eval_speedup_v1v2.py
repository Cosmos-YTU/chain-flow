"""v1 vs v2 vLLM speedup on REAL eval prompts (RedHatAI domains) for 4B and Qwen3.5-27B.
Shows the v2 diverse-data win that the 4 generic prompts hid: v2 >= v1 on every domain, biggest on writing."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json

D = json.load(open("/tmp/eval_speedup_data.json"))
DOMS = ["writing", "qa", "translation", "math_reasoning"]
DLAB = {"writing": "writing", "qa": "qa", "translation": "translation", "math_reasoning": "math\n(control)"}
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150, sharey=True)
fig.patch.set_facecolor("white")
PANELS = [("4B", "4B v1", "4B v2", "#93c5fd", "#2563eb", 495),
          ("Qwen3.5-27B", "27B v1", "27B v2", "#fdba74", "#ea580c", 101)]
x = np.arange(len(DOMS)); w = 0.38

for ax, (title, k1, k2, c1, c2, nat) in zip(axes, PANELS):
    ax.set_facecolor("white")
    for k, c, off in [(k1, c1, -w/2), (k2, c2, +w/2)]:
        vals = [D[k][d][0] for d in DOMS]
        bars = ax.bar(x + off, vals, w, label=k.split()[-1], color=c, edgecolor="white", linewidth=1, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=ink)
    ax.axhline(1.0, color=muted, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.set_xticks(x); ax.set_xticklabels([DLAB[d] for d in DOMS], fontsize=9.5, color=ink)
    ax.set_title(f"{title}   (native {nat} t/s)", fontsize=11.5, color=ink, fontweight="bold", pad=8)
    ax.set_ylim(0, 1.5); ax.tick_params(colors=muted, length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid); ax.yaxis.grid(True, color=grid, lw=1, zorder=0); ax.set_axisbelow(True)
    leg = ax.legend(fontsize=9.5, loc="upper left", frameon=False)
    for tt in leg.get_texts(): tt.set_color(ink)

axes[0].set_ylabel("vLLM decode speedup vs autoregressive  (x)", fontsize=10.5, color=ink)
axes[0].text(-0.46, 1.015, "1.0x breakeven", ha="left", va="bottom", fontsize=8, color=muted, style="italic")
fig.suptitle("Flow-Drafter v1 vs v2 — vLLM speedup on REAL eval prompts (RedHatAI domains)",
             fontsize=13.5, color=ink, fontweight="bold", x=0.5, y=1.02, ha="center")
fig.text(0.5, -0.03, "5 prompts/domain, all lossless (writing 27B: fp16 tie on ~29/240 tokens, equal for v1 & v2). "
         "v2 (expanded diverse data) ≥ v1 on every domain, biggest on writing — the win the 4 generic prompts couldn't show.",
         ha="center", fontsize=8.5, color=muted)
fig.tight_layout()
out = "/home/shadeform/chain-flow/docs/eval_speedup_v1_vs_v2.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
