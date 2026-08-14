"""Shared chart styling: the repo's palette and axis conventions."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for drivers)

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
RAINBOW = {"red": "#d03b3b", "orange": "#e07b39", "yellow": "#d9a800", "green": "#3a8f3a",
           "blue": "#2a78d6", "indigo": "#4b3f9e", "violet": "#8a5bb8"}
SERIES = {7: "#2a78d6", 11: "#eb6834", 14: "#1baf7a", 18: "#eda100", 21: "#e87ba4"}
SRC_COLOR = {"cent": "#2a78d6", "same": "#eb6834", "rand": "#1baf7a"}
SRC_LABEL = {"cent": "centered vector", "same": "raw vector", "rand": "random control"}


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


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print("wrote", path)
