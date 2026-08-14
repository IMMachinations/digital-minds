"""Presentation charts from existing result files (no model needed).

Run: python plots.py [--which {inherent,tiers,all}]
  inherent -> results/inherent/inh_{preferences,push_vs_disruption,vector_geometry,
              color_response}.png   (four inherent-mode figures)
  tiers    -> results/tier_components.png   (push/disruption decomposition, both modes)
"""
import argparse

import torch

from lib.data import COLORS
from lib.harness import STEER_LAYERS
from lib.io import load_json
from lib.paths import RESULTS
from lib.plotting import (GRID, INK, MUTED, RAINBOW, SRC_COLOR, SRC_LABEL, SURFACE, plt,
                          save, style)
from lib.stats import tier_components, tier_deltas
from lib.tiers import signed

SRC = {"cent": ("#2a78d6", "centered vector"), "same": ("#eb6834", "raw vector"),
       "rand": ("#1baf7a", "random control")}


def inherent_figures():
    from matplotlib.colors import LinearSegmentedColormap

    res = RESULTS / "inherent"
    objs = load_json(res / "objects.json")
    old = load_json(res / "preferences.json")
    steer = load_json(res / "objects_steer.json")
    vp = torch.load(res / "vectors_obj.pt")

    # ---- A. what the preferences are ------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.set_facecolor(SURFACE)
    M = [[sum(signed(c["diff"], c, r) for c in objs if {c["color_a"], c["color_b"]} == {r, col}) / 20
          if r != col else 0 for col in COLORS] for r in COLORS]
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    vm = max(abs(v) for row in M for v in row)
    a1.imshow(M, cmap=cmap, vmin=-vm, vmax=vm)
    a1.set_xticks(range(7), COLORS, rotation=45, ha="right", color=MUTED, fontsize=9)
    a1.set_yticks(range(7), COLORS, color=MUTED, fontsize=9)
    for i in range(7):
        for j in range(7):
            if i != j:
                a1.text(j, i, f"{M[i][j]:+.2f}", ha="center", va="center", fontsize=8, color=INK)
    a1.set_title("Head-to-head: mean log-prob edge of row color's\nobjects over column color's (red = row wins)",
                 fontsize=10.5, color=INK)
    style(a2)
    x = [c["diff"] for c in old]
    y = [c["diff"] for c in objs]
    r = torch.corrcoef(torch.stack([torch.tensor(x), torch.tensor(y)]))[0, 1]
    a2.scatter(x, y, s=12, color="#2a78d6", alpha=0.45, linewidths=0)
    a2.set_xlabel("A/B-letter logit diff (old measurement)", fontsize=9, color=MUTED)
    a2.set_ylabel("object-token mean log-prob diff", fontsize=9, color=MUTED)
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_title(f"The two measurements agree on the same 420 pairs\n(each dot one comparison; r = {r:+.2f})",
                 fontsize=10.5, color=INK)
    fig.suptitle("Inherent-object preferences, read directly off the object tokens", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, res / "inh_preferences.png")

    # ---- B. push vs disruption ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    fig.set_facecolor(SURFACE)
    for ax, key, title in [
            (axes[0][0], 0, "Uniform component = mean Δ across tiers\na real preference shift moves every tier"),
            (axes[0][1], 1, "Antisymmetric component = (Δworst − Δbest)/2\nall three sources overlap: disruption is generic")]:
        style(ax)
        for src, (col, lab) in SRC.items():
            for cf, ls in [(1.0, "-"), (2.0, "--")]:
                ys = [tier_components(tier_deltas(steer, src, cf, L))[key] for L in STEER_LAYERS]
                ax.plot(STEER_LAYERS, ys, ls, color=col, lw=2, marker="o", ms=5,
                        label=f"{lab}, coef {cf:g}")
        ax.set_xticks(STEER_LAYERS)
        ax.set_xlabel("steered layer", fontsize=9, color=MUTED)
        ax.set_title(title, fontsize=10.5, color=INK)
    axes[0][0].set_ylabel("Δ mean log-prob toward steered color", fontsize=9, color=MUTED)
    axes[0][0].set_ylim(top=0.68)  # headroom so the legend clears the curves
    axes[0][0].legend(fontsize=8.5, frameon=False, labelcolor=INK, loc="upper left", ncol=2)
    for ax, (L, cf), verdict in [(axes[1][0], (11, 2.0), "mirror shape: pure disruption"),
                                 (axes[1][1], (21, 1.0), "flat, small: push without disruption (centered only)")]:
        style(ax)
        for k, (src, (col, lab)) in enumerate(SRC.items()):
            d = tier_deltas(steer, src, cf, L)
            ax.bar([i + (k - 1) * 0.25 for i in range(3)], list(d.values()), width=0.23,
                   color=col, label=lab)
        ax.set_xticks(range(3), ["worst 20", "neutral 20", "best 20"], color=MUTED, fontsize=9)
        ax.set_xlabel("example tier (by baseline preference)", fontsize=9, color=MUTED)
        ax.set_title(f"Layer {L}, coef {cf:g} — {verdict}", fontsize=10.5, color=INK)
    axes[1][0].set_ylabel("Δ mean log-prob toward steered color", fontsize=9, color=MUTED)
    axes[1][0].legend(fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle("Steering inherent comparisons with object-derived vectors: directional push vs generic disruption",
                 fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, res / "inh_push_vs_disruption.png")

    # ---- C. vector geometry ---------------------------------------------------------------------
    V = torch.stack([vp["vecs"][c] for c in COLORS])       # [7, 28, d], unit norm
    C = torch.stack([vp["centered"][c] for c in COLORS])
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    fig.set_facecolor(SURFACE)
    off = ~torch.eye(7, dtype=bool)
    for ax in (c1, c2):
        style(ax)
        ax.set_xlabel("layer", fontsize=9, color=MUTED)
    nL = V.shape[1]
    c1.plot(range(nL), [(V[:, L] @ V[:, L].T)[off].mean() for L in range(nL)],
            color="#eb6834", lw=2, label="raw vectors")
    c1.plot(range(nL), [(C[:, L] @ C[:, L].T)[off].mean() for L in range(nL)],
            color="#2a78d6", lw=2, label="centered vectors")
    c1.set_ylim(-0.3, 1.05)
    c1.set_title("Mean pairwise cosine between the 7 color vectors\nraw ≈ 1: they are one shared direction",
                 fontsize=10.5, color=INK)
    c1.legend(fontsize=9, frameon=False, labelcolor=INK, loc="center right")
    c2.plot(range(nL), [(V[:, L] - V[:, L].mean(0)).norm(dim=-1).mean() for L in range(nL)],
            color="#2a78d6", lw=2)
    c2.set_title("Color-specific residual as a fraction of vector norm\nthe part centering keeps: 3–8%",
                 fontsize=10.5, color=INK)
    fig.suptitle("Why raw vectors steer like random noise: color identity is a tiny residual on a huge shared component",
                 fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, res / "inh_vector_geometry.png")

    # ---- D. per-color stability -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    fig.set_facecolor(SURFACE)
    style(ax)
    grand = []
    for i, col in enumerate(COLORS):
        pts = []
        for L in (11, 14, 18, 21):
            ds = [r["delta"] for r in steer
                  if (r["source"], r["coef"], r["color"], r["layer"]) == ("cent", 2.0, col, L)]
            pts.append(sum(ds) / len(ds))  # uniform push at this layer (mean of 3 tiers)
        ax.scatter([i] * 4, pts, s=48, color=RAINBOW[col], alpha=0.85, linewidths=0, zorder=3)
        ax.plot([i - 0.22, i + 0.22], [sum(pts) / 4] * 2, color=INK, lw=2, zorder=4)
        grand += pts
    ax.axhline(sum(grand) / len(grand), color=MUTED, lw=1.2, ls="--")
    ax.text(6.45, sum(grand) / len(grand) + 0.02, "7-color mean", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks(range(7), COLORS, color=MUTED, fontsize=9)
    ax.set_ylabel("uniform push, Δ mean log-prob", fontsize=9, color=MUTED)
    ax.set_title("Per-color uniform push from its own centered vector, coef 2\n"
                 "(dots: steered layers 11/14/18/21; black tick = that color's mean)",
                 fontsize=10.5, color=INK)
    fig.suptitle("The average push is positive, but per-color effects are unstable across layers",
                 fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, res / "inh_color_response.png")


def tiers_figure():
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7), sharex="col")
    fig.set_facecolor(SURFACE)
    for r, mode in enumerate(["inherent", "modifier"]):
        rows = load_json(RESULTS / mode / "objects_steer.json")
        for c, (key, name) in enumerate([(0, "uniform component (directional push)"),
                                         (1, "antisymmetric component (disruption)")]):
            ax = axes[r][c]
            style(ax, name if r == 0 else "")
            for src in ["cent", "same", "rand"]:
                for cf, ls in [(1.0, "-"), (2.0, "--")]:
                    ys = [tier_components(tier_deltas(rows, src, cf, L))[key]
                          for L in STEER_LAYERS]
                    ax.plot(STEER_LAYERS, ys, ls, color=SRC_COLOR[src], lw=2, marker="o", ms=5,
                            label=f"{SRC_LABEL[src]}, coef {cf:g}" if r == 0 else None)
            ax.set_xticks(STEER_LAYERS)
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
    fig.savefig(RESULTS / "tier_components.png", dpi=150, facecolor=SURFACE)
    print("wrote", RESULTS / "tier_components.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["inherent", "tiers", "all"], default="all")
    args = ap.parse_args()
    if args.which in ("inherent", "all"):
        inherent_figures()
    if args.which in ("tiers", "all"):
        tiers_figure()
