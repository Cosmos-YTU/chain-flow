"""Grouped-bar chart of final lossless vLLM decode speedups across 4B / 9B / 27B(Qwen3.6) /
27B(Qwen3.5), parsed directly from the speedup logs. Series with no log yet are skipped."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re, os

ROOT = "/home/shadeform/chained-flow"
# label -> (speedup log, bar color)   colors: size ramp light->dark, two 27Bs as distinct hues
SERIES = [
    ("4B",            f"{ROOT}/vllm/test_compiled_4b_cgdraft.log", "#93c5fd"),
    ("9B",            f"{ROOT}/logs/speedup_9b.log",               "#3b82f6"),
    ("27B (Qwen3.6)", f"{ROOT}/logs/speedup_27b.log",             "#1e3a8a"),
    ("27B (Qwen3.5)", f"{ROOT}/logs/speedup_q3527b.log",          "#f59e0b"),
]
PROMPTS = ["math\n(step-by-step)", "code\n(fibonacci)", "short\nfactual", "prose\n(ocean)", "MEAN"]
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"


def parse(path):
    """return [p0,p1,p2,p3, mean] speedups, or None if log missing/empty."""
    if not os.path.exists(path):
        return None, None
    txt = open(path, errors="ignore").read()
    rows = {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"\[(\d)\] AR [0-9.]+ t/s \| spec [0-9.]+ t/s \(([0-9.]+)x\)", txt)}
    mean = re.search(r"SPEEDUP ([0-9.]+)x", txt)
    nat = re.search(r"vLLM-native reference:?\s*([0-9.]+) tok/s", txt)
    if len(rows) < 4 or not mean:
        return None, None
    vals = [rows[i] for i in range(4)] + [float(mean.group(1))]
    return vals, (float(nat.group(1)) if nat else None)


data = []
for label, path, color in SERIES:
    vals, nat = parse(path)
    if vals is None:
        print(f"skip {label}: no speedup data at {path}")
        continue
    leg = f"{label}  (native {nat:.0f} t/s)" if nat else label
    data.append((leg, vals, color))

n = len(data)
x = np.arange(len(PROMPTS))
w = 0.8 / n
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=150)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

for i, (leg, vals, color) in enumerate(data):
    off = (i - (n - 1) / 2) * w
    bars = ax.bar(x + off, vals, w, label=leg, color=color, edgecolor="white", linewidth=1.0, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7.8, color=ink, fontweight="bold" if v == vals[-1] else "normal")

ax.axhline(1.0, color=muted, lw=1.6, ls=(0, (5, 4)), zorder=2)
ax.text(-0.46, 1.02, "1.0x  breakeven (lossless)", ha="left", va="bottom", fontsize=8.5,
        color=muted, style="italic")
ax.axvspan(x[-1] - 0.5, x[-1] + 0.5, color="#f9fafb", zorder=0)

ax.set_xticks(x); ax.set_xticklabels(PROMPTS, fontsize=9.5, color=ink)
ax.set_ylabel("vLLM decode speedup vs autoregressive  (x)", fontsize=10.5, color=ink)
ax.set_ylim(0, 2.3); ax.set_yticks(np.arange(0, 2.4, 0.5))
ax.tick_params(colors=muted, length=0)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(grid)
ax.yaxis.grid(True, color=grid, lw=1, zorder=0); ax.set_axisbelow(True)

ax.set_title("Flow-Drafter — final lossless vLLM speedups (4B / 9B / 27B x2) on RTX PRO 6000",
             fontsize=13, color=ink, fontweight="bold", pad=32, loc="left")
ax.text(0, 1.045, "Both 27B drafters (Qwen3.6 post-trained & Qwen3.5 base) land together — same recipe, "
        "one trained 2-GPU DDP; accept tracks base output-predictability, not training.",
        transform=ax.transAxes, fontsize=9, color=muted)

leg = ax.legend(title="base model", fontsize=9, title_fontsize=9, loc="upper left",
                frameon=False, ncol=1, bbox_to_anchor=(0.005, 0.99))
leg.get_title().set_color(ink)
for t in leg.get_texts():
    t.set_color(ink)

fig.tight_layout()
out = f"{ROOT}/docs/vllm_speedups_4models.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out, "with", n, "series")
