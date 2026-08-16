"""Chart standards for this repo: fixed per-model colors, shared neutrals, axis/save helpers.

Single source of truth — the rules and the palette validation record live in CHARTS.md.
p1/ and lenses/ import this module via a two-line sys.path shim to the repo root
(mirroring p1/_day1.py); desires/ is frozen and keeps its historical copy.

Model colors follow "hue = model family, lightness = size within family":
Qwen2.5 is an ordinal violet ramp (H≈293, darker = larger, 0.5B→32B), Qwen3 is
gold (H≈75), Llama is Meta blue (H≈258). CVD-validated — re-check any change
with scripts/validate_palette.py before shipping it.
"""
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

# ---- per-model identity (fixed assignment, never cycled) ----------------------------------------
MODEL_COLORS = {
    # core roster
    "qwen3-4b":   "#D9951E",   # Qwen3 family: gold
    "llama31-8b": "#468EFA",   # Meta blue, relit to sit inside the violet ramp's gap
    "qwen25-7b":  "#864BF9",   # Qwen2.5 violet ramp, size step 7B
    "qwen25-32b": "#5D12BD",   # Qwen2.5 violet ramp, darkest (largest)
    # Qwen2.5 tiny sizes (same ordinal violet ramp, lighter = smaller)
    "qwen25-05b": "#B7A5FB",
    "qwen25-15b": "#A58AFA",
    "qwen25-3b":  "#956CFB",
}
MODEL_LABELS = {
    "qwen3-4b": "Qwen3-4B",
    "llama31-8b": "Llama-3.1-8B",
    "qwen25-7b": "Qwen2.5-7B",
    "qwen25-32b": "Qwen2.5-32B",
    "qwen25-05b": "Qwen2.5-0.5B",
    "qwen25-15b": "Qwen2.5-1.5B",
    "qwen25-3b": "Qwen2.5-3B",
}
MODEL_ORDER = list(MODEL_COLORS)
# the Qwen2.5 size ladder, smallest -> largest (ordinal: read identity from lightness + labels)
QWEN25_SIZES = ["qwen25-05b", "qwen25-15b", "qwen25-3b", "qwen25-7b", "qwen25-32b"]

# ---- neutrals & semantic colors (dataviz reference palette, light mode) -------------------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
ACCENT = "#2a78d6"                                  # generic single-series mark (not a model)
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"     # diverging
DARK, LIGHT = "#1c5cab", "#86b6ef"                  # paired dumbbell/compare marks
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", SEQ)


def setup():
    """Apply the shared rcParams. Call once, after the backend is set."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    })


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def bounded_axis(ax, axis="y", lo=0.0, hi=1.0, frac=0.04):
    """Padded limits for a bounded quantity (share, rate, probability): marks at the
    bound render whole instead of being sliced by the spine. Ticks stay natural."""
    pad = (hi - lo) * frac
    (ax.set_ylim if axis == "y" else ax.set_xlim)(lo - pad, hi + pad)


def save(fig, path):
    import matplotlib.pyplot as plt
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    try:
        shown = path.relative_to(Path.cwd())
    except ValueError:
        shown = path
    print("wrote", shown)
