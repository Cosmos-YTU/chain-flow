"""Two-panel v1-vs-v2 vLLM speedup comparison for 4B and Qwen3.5-27B, per prompt + mean.
Parsed from the four speedup logs. Non-lossless cells (fp16 tie divergence) are hatched + footnoted."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import re, os

ROOT = "/home/shadeform/chained-flow"
LOGS = {"4B v1": f"{ROOT}/vllm/test_compiled_4b_cgdraft.log", "4B v2": f"{ROOT}/logs/speedup_4bx.log",
        "27B v1": f"{ROOT}/logs/speedup_q3527b.log", "27B v2": f"{ROOT}/logs/speedup_q3527bx.log"}
PROMPTS = ["math", "code", "short\nfactual", "prose", "MEAN"]
ink, muted, grid = "#1f2937", "#6b7280", "#e5e7eb"


def parse(path):
    t = open(path, errors="ignore").read()
    sx, ll = {}, {}
    for m in re.finditer(r"\[(\d)\] AR [0-9.]+ t/s \| spec [0-9.]+ t/s \(([0-9.]+)x\).*?lossless\(vs native\) (\d+)/48", t):
        i = int(m.group(1)); sx[i] = float(m.group(2)); ll[i] = int(m.group(3))
    mean = re.search(r"SPEEDUP ([0-9.]+)x", t)
    vals = [sx.get(i, 0) for i in range(4)] + [float(mean.group(1)) if mean else 0]
    lossless = [ll.get(i, 48) for i in range(4)] + [48]
    return vals, lossless


D = {k: parse(p) for k, p in LOGS.items()}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=150, sharey=True)
fig.patch.set_facecolor("white")

PANELS = [("4B", "4B v1", "4B v2", "#93c5fd", "#2563eb", 495),
          ("Qwen3.5-27B", "27B v1", "27B v2", "#fdba74", "#ea580c", 101)]
x = np.arange(len(PROMPTS)); w = 0.38

for ax, (title, k1, k2, c1, c2, nat) in zip(axes, PANELS):
    ax.set_facecolor("white")
    for j, (k, c, off) in enumerate([(k1, c1, -w/2), (k2, c2, +w/2)]):
        vals, ll = D[k]
        bars = ax.bar(x + off, vals, w, label=k.split()[-1], color=c, edgecolor="white", linewidth=1, zorder=3)
        for b, v, l in zip(bars, vals, ll):
            if l < 48:  # non-lossless: hatch + mark
                b.set_hatch("////"); b.set_alpha(0.55)
                ax.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}*", ha="center", va="bottom", fontsize=8, color=muted)
            else:
                ax.text(b.get_x()+b.get_width()/2, v+0.03, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=ink, fontweight="bold" if PROMPTS[list(bars).index(b)]=="MEAN" else "normal")
    ax.axhline(1.0, color=muted, lw=1.5, ls=(0, (5, 4)), zorder=2)
    ax.axvspan(x[-1]-0.5, x[-1]+0.5, color="#f9fafb", zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(PROMPTS, fontsize=9.5, color=ink)
    ax.set_title(f"{title}   (native {nat} t/s)", fontsize=11.5, color=ink, fontweight="bold", pad=8)
    ax.set_ylim(0, 2.2); ax.tick_params(colors=muted, length=0)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid); ax.yaxis.grid(True, color=grid, lw=1, zorder=0); ax.set_axisbelow(True)
    leg = ax.legend(title="", fontsize=9.5, loc="upper right", frameon=False)
    for tt in leg.get_texts(): tt.set_color(ink)

axes[0].set_ylabel("vLLM decode speedup vs autoregressive  (x)", fontsize=10.5, color=ink)
axes[0].text(-0.46, 1.02, "1.0x breakeven", ha="left", va="bottom", fontsize=8, color=muted, style="italic")
fig.suptitle("Flow-Drafter v1 vs v2 (expanded diverse data) — vLLM speedup on 4 generic prompts",
             fontsize=13.5, color=ink, fontweight="bold", x=0.5, y=1.02, ha="center")
fig.text(0.5, -0.04, "Speedup is ~unchanged v1→v2 on these 4 prompts — they sit in domains v1 already handled; "
         "v2's accept gains are on writing/qa/summarization (see accept tables), which these prompts don't exercise.\n"
         "*4B-v2 math: lossless 13/48 — an fp16 argmax-tie divergence from the v2 tree on this prompt (not shown as a valid speedup).",
         ha="center", fontsize=8.5, color=muted)
fig.tight_layout()
out = f"{ROOT}/docs/speedup_v1_vs_v2.png"
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
