"""Chart styling (neutrals from the repo-root chart standards — see /CHARTS.md) and
the two lens figures: the layer x mode token grid and the agreement curves.

MODE_COLOR encodes lens mode (logit/j/r), not model identity: lens figures are
single-model, with the model named in the title."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from chartstyle import GRID, INK, MUTED, SURFACE, bounded_axis, save as _std_save  # noqa: E402

MODE_COLOR = {"logit": "#898781", "j": "#2a78d6", "r": "#1baf7a"}
MODE_LABEL = {"logit": "logit lens", "j": "j-lens", "r": "r-lens"}


def style(ax, title=""):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK)


def save(fig, path):
    _std_save(fig, path)


def token_grid(readout, path, title, top=3):
    """Layer rows x mode columns; each cell shows the top `top` tokens of that lens readout.
    readout = evals.readout_modes output. The bottom row is the model's own top tokens."""
    modes = [m for m in ("logit", "j", "r") if m in readout]
    layers = sorted(readout[modes[0]])
    nrows = len(layers) + 1
    fig, ax = plt.subplots(figsize=(3.6 * len(modes) + 1.2, 0.34 * nrows + 1.2))
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, len(modes))
    ax.set_ylim(0, nrows)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK)

    def cell(x, y, toks, color):
        text = "  ".join(repr(t)[1:-1] for t, _ in toks[:top])
        ax.text(x + 0.04, y, text[:52], fontsize=7.5, color=color, va="center", family="monospace")

    for ci, m in enumerate(modes):
        ax.text(ci + 0.04, nrows - 0.35, MODE_LABEL[m], fontsize=9.5, color=MODE_COLOR[m],
                weight="bold", family="monospace")
        for ri, l in enumerate(layers):
            y = (len(layers) - ri) * (nrows - 1.9) / len(layers) + 0.9
            cell(ci, y, readout[m][l], INK)
            if ci == 0:
                lab = "emb" if l == -1 else f"L{l}"
                ax.text(-0.16, y, lab, fontsize=7.5, color=MUTED, va="center", ha="right",
                        family="monospace")
    y = 0.35
    ax.text(-0.16, y, "out", fontsize=7.5, color=MUTED, va="center", ha="right", family="monospace")
    cell(0, y, readout["model"], MODE_COLOR["j"])
    save(fig, path)


def agreement_curves(agree, path, title):
    """agree = evals.agreement output: {mode: {layer: fraction}}."""
    fig, ax = plt.subplots(figsize=(7, 4))
    style(ax, title)
    for m in ("logit", "j", "r"):
        if m not in agree:
            continue
        layers = sorted(agree[m])
        ax.plot([l for l in layers], [agree[m][l] for l in layers], color=MODE_COLOR[m],
                lw=2, marker="o", ms=3, label=MODE_LABEL[m])
    ax.set_xlabel("layer (-1 = embedding)", color=MUTED, fontsize=9)
    ax.set_ylabel("top-1 agreement with final output", color=MUTED, fontsize=9)
    bounded_axis(ax, "y")
    ax.legend(frameon=False, fontsize=9)
    save(fig, path)
