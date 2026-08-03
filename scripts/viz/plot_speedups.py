"""Grouped-bar chart of the final lossless vLLM decode speedups (4B/9B/27B) across prompt types.
Model encoded as a sequential blue ramp (light 4B -> dark 27B) since size is the ordered story;
horizontal breakeven line at 1.0x separates net speedup from drafting overhead."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

prompts = ["math\n(step-by-step)", "code\n(fibonacci)", "short\nfactual", "prose\n(ocean)", "MEAN"]
data = {  # model -> per-prompt speedup, last is mean
    "4B  (native 495 t/s)":  [0.87, 1.38, 0.67, 0.91, 0.97],
    "9B  (native 309 t/s)":  [1.77, 1.57, 0.95, 1.02, 1.27],
    "27B (native 101 t/s)":  [1.64, 1.98, 1.21, 1.02, 1.46],
}
# sequential blue ramp: light -> dark encodes 4B -> 27B (bigger = darker = faster)
colors = ["#93c5fd", "#3b82f6", "#1e3a8a"]
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"

x = np.arange(len(prompts))
w = 0.26
fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for i, (label, vals) in enumerate(data.items()):
    off = (i - 1) * w
    bars = ax.bar(x + off, vals, w, label=label, color=colors[i],
                  edgecolor="white", linewidth=1.0, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5,
                color=ink, fontweight="bold" if v == vals[-1] else "normal")

# breakeven line at 1.0x (lossless: below = draft overhead, above = net speedup)
ax.axhline(1.0, color=muted, lw=1.6, ls=(0, (5, 4)), zorder=2)
ax.text(-0.45, 1.02, "1.0x  breakeven (lossless)", ha="left", va="bottom",
        fontsize=8.5, color=muted, style="italic")

# shade the MEAN group to set it apart
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f9fafb", zorder=0)

ax.set_xticks(x); ax.set_xticklabels(prompts, fontsize=9.5, color=ink)
ax.set_ylabel("vLLM decode speedup vs autoregressive  (x)", fontsize=10.5, color=ink)
ax.set_ylim(0, 2.2)
ax.set_yticks(np.arange(0, 2.3, 0.5))
ax.tick_params(colors=muted, length=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(grid)
ax.yaxis.grid(True, color=grid, lw=1, zorder=0)
ax.set_axisbelow(True)

ax.set_title("Flow-Drafter — final lossless vLLM speedups on RTX PRO 6000 Blackwell",
             fontsize=13.5, color=ink, fontweight="bold", pad=32, loc="left")
ax.text(0, 1.045, "Bigger base -> larger win: memory-bound base decode grows with model size, "
        "while the latent-flow draft stays base-decoupled (~11ms fixed).",
        transform=ax.transAxes, fontsize=9.5, color=muted)

leg = ax.legend(title="base model", fontsize=9.5, title_fontsize=9.5,
                loc="upper left", frameon=False, ncol=1, bbox_to_anchor=(0.005, 0.99))
leg.get_title().set_color(ink)
for t in leg.get_texts():
    t.set_color(ink)

fig.tight_layout()
out = "docs/vllm_speedups_4b_9b_27b.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
