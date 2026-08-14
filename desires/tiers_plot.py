"""Chart the tier-uniform (directional push) vs tier-antisymmetric (disruption) decomposition
of object-vector steering, from results/{mode}/objects_steer.json. No model needed.

Run: python tiers_plot.py  ->  results/tier_components.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).parent / "results"
LAYERS, COLORS = [7, 11, 14, 18, 21], ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
SRC_COLOR = {"cent": "#2a78d6", "same": "#eb6834", "rand": "#1baf7a"}
SRC_LABEL = {"cent": "centered vector", "same": "raw vector", "rand": "random control"}
RAINBOW = {"red": "#d03b3b", "orange": "#e07b39", "yellow": "#d9a800", "green": "#3a8f3a",
           "blue": "#2a78d6", "indigo": "#4b3f9e", "violet": "#8a5bb8"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def comp(rows, src, cf, L):
    d = {t: sum(r["delta"] for r in rows
                if (r["source"], r["coef"], r["layer"], r["tier"]) == (src, cf, L, t)) / 7
         for t in ["worst", "neutral", "best"]}
    return sum(d.values()) / 3, (d["worst"] - d["best"]) / 2  # uniform, antisym


def style(ax, title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color=MUTED, lw=1)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK)


fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), sharex="col")
fig.set_facecolor(SURFACE)
for r, mode in enumerate(["inherent", "modifier"]):
    rows = json.load(open(RES / mode / "objects_steer.json"))
    for c, (key, name) in enumerate([(0, "uniform component (directional push)"),
                                     (1, "antisymmetric component (disruption)")]):
        ax = axes[r][c]
        style(ax, name if r == 0 else "")
        for src in ["cent", "same", "rand"]:
            for cf, ls in [(1.0, "-"), (2.0, "--")]:
                ys = [comp(rows, src, cf, L)[key] for L in LAYERS]
                ax.plot(LAYERS, ys, ls, color=SRC_COLOR[src], lw=2, marker="o", ms=5,
                        label=f"{SRC_LABEL[src]}, coef {cf:g}" if r == 0 else None)
        ax.set_xticks(LAYERS)
        if c == 0:
            ax.set_ylabel(f"{mode} items\nΔ mean-logprob toward color", fontsize=10, color=INK)
        if r == 1:
            ax.set_xlabel("steered layer", fontsize=9, color=MUTED)
    # per-color uniform push at the cleanest config: centered, L21, coef 1
    ax = axes[r][2]
    style(ax, "per-color push — centered, L21, coef 1" if r == 0 else "")
    vals = [sum(x["delta"] for x in rows if (x["source"], x["coef"], x["layer"], x["color"])
                == ("cent", 1.0, 21, col)) / 3 for col in COLORS]
    ax.bar(range(7), vals, color=[RAINBOW[c] for c in COLORS], width=0.62,
           edgecolor=SURFACE, linewidth=2)
    ax.set_xticks(range(7), [c[:3] for c in COLORS], color=MUTED)
    if r == 1:
        ax.set_xlabel("steered color", fontsize=9, color=MUTED)

handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, frameon=False,
           labelcolor=INK, bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Object-vector steering decomposed: push vs disruption (mean over 7 colors × 3 tiers × 20 pairs)",
             fontsize=12, color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(RES / "tier_components.png", dpi=150, facecolor=SURFACE)
print("wrote", RES / "tier_components.png")
