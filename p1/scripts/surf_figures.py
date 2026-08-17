"""SURF probeloop figures (f36, f37), following CHARTS.md via root chartstyle.

f36: probe-hardening behavioral convergence — held-out menu choice rate of each
     cycle's top-20 probe-selected items, per model, with the pre-loop (E2 v0)
     baselines and each model's revealed-arm ceiling for reference.
f37: the prequential scissors — on each cycle's fresh discoveries, the current
     (just-retrained) probe's correlation vs the frozen v0 control. The widening
     gap is the probe hardening; v0's decay shows the search escaping its
     training support.

Usage: uv run python scripts/surf_figures.py   (CPU)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1.parent))
from chartstyle import (INK2, MODEL_COLORS, MODEL_LABELS, MUTED,  # noqa: E402
                        bounded_axis, save, setup, style)

MODELS = ["qwen25-7b", "llama31-8b", "qwen3-4b"]
E2_BASE = {"qwen25-7b": 0.512, "llama31-8b": 0.658}   # E2 arm-P referee (pre-loop v0)
R_CEIL = {"qwen25-7b": 0.704, "llama31-8b": 0.958}    # E2 arm-R (revealed-arm) reference
FIGS = P1 / "results" / "figures"


def cycles(model):
    return json.loads((P1 / "results" / "surf" / "probeloop" / model /
                       "cycles.json").read_text())


def f36():
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for m in MODELS:
        cs = cycles(m)
        xs = [c["cycle"] for c in cs]
        ys = [c["t3_top20_mean"] for c in cs]
        if m in E2_BASE:
            xs, ys = [0] + xs, [E2_BASE[m]] + ys
        ax.plot(xs, ys, color=MODEL_COLORS[m], lw=1.8, marker="o", ms=4,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        ax.annotate(MODEL_LABELS[m], (xs[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=MODEL_COLORS[m])
        if m in R_CEIL:
            ax.plot([2.75, 3.55], [R_CEIL[m]] * 2, color=MODEL_COLORS[m],
                    lw=0.9, ls=":", zorder=2)
    ax.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    ax.annotate("chance vs mid-utility anchor", (0.02, 0.505), fontsize=7.5,
                color=MUTED, va="bottom")
    ax.annotate("revealed-arm ceilings", (2.78, 0.985), fontsize=7,
                color=MUTED, va="top")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["v0\n(E2)", "cycle 1", "cycle 2", "cycle 3"])
    ax.set_xlim(-0.2, 4.2)
    bounded_axis(ax, "y", 0.45, 1.0)
    ax.set_ylabel("held-out choice rate, top-20 probe-selected items")
    ax.set_title("Probe hardening: behavioral validity of probe-guided search",
                 fontsize=10, color=INK2, loc="left")
    style(ax, grid_axis="y")
    save(fig, FIGS / "f36_probeloop_referee.png")


def f37():
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
    for ax, m in zip(axes, MODELS):
        cs = cycles(m)
        xs = [c["cycle"] for c in cs]
        cur = [c["per_probe"][f"v{c['cycle']}"]["raw"]["pearson"] for c in cs]
        v0 = [c["per_probe"]["v0"]["raw"]["pearson"] for c in cs]
        ax.plot(xs, cur, color=MODEL_COLORS[m], lw=1.8, marker="o", ms=4,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3,
                label="current probe")
        ax.plot(xs, v0, color=MUTED, lw=1.4, ls="--", marker="o", ms=3.5,
                markerfacecolor="white", markeredgewidth=1.1, zorder=2,
                label="frozen v0")
        ax.set_title(MODEL_LABELS[m], fontsize=9, color=MODEL_COLORS[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"c{x}" for x in xs])
        style(ax, grid_axis="y")
        bounded_axis(ax, "y", 0.2, 1.0)
    axes[0].set_ylabel("r vs measured $\\mu$,\nfresh discoveries")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Prequential scissors: each cycle's probe vs the frozen v0 control",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGS / "f37_probeloop_scissors.png")


if __name__ == "__main__":
    setup()
    f36()
    f37()
