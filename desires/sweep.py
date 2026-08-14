"""Dense coefficient sweep of prefer-vector steering, decomposed by letter, with a chart.

Run: python sweep.py [modifier|inherent]  (after experiment.py; reuses flip.py's machinery)
Saves results/{mode}/sweep.json and results/{mode}/sweep.png.
"""
import json

import torch

import experiment as E  # loads model; picks up mode from sys.argv[1]
import flip as F
from data import COLORS

COEFS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # x residual norm; denser than E.COEFS


def run():
    rows = []
    for name, suffix in F.FRAMINGS.items():
        n_suf = len(E.tok(F.reframe(F.COMPS[0], suffix)).input_ids) - \
            len(E.tok(F.reframe(F.COMPS[0], "")).input_ids)
        for color in COLORS:
            worst = F.worst_by_prefer(color)
            wp = [F.reframe(c, suffix) for c in worst]
            base = F.sides(E.last_logits(wp), worst, color)
            for L in E.STEER_LAYERS:
                for cf in COEFS:
                    vec = (F.VECS[color][L] * cf * F.RESID_NORMS[L]).to("cuda", torch.bfloat16)
                    st = F.sides(E.last_logits(wp, steer=(L, vec), n_suffix=n_suf), worst, color)
                    rows.append(dict(
                        framing=name, color=color, layer=L, coef=cf,
                        delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                        d_own=st["own"] - base["own"], d_opp=st["opp"] - base["opp"],
                        d_own_lp=st["own_lp"] - base["own_lp"],
                        d_opp_lp=st["opp_lp"] - base["opp_lp"]))
        print(f"swept {name}")
    (E.OUT / "sweep.json").write_text(json.dumps(rows, indent=1))
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SERIES = {7: "#2a78d6", 11: "#eb6834", 14: "#1baf7a", 18: "#eda100", 21: "#e87ba4"}
    INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
    metrics = [("delta", "Δ logit diff (own − opp)"),
               ("d_own", "Δ own-letter logit"), ("d_opp", "Δ opposite-letter logit")]

    fig, axes = plt.subplots(len(F.FRAMINGS), 3, figsize=(11.5, 9), sharex=True, sharey="row")
    fig.set_facecolor(SURFACE)
    for r, framing in enumerate(F.FRAMINGS):
        for c, (key, label) in enumerate(metrics):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.grid(True, color=GRID, lw=0.8)
            ax.axhline(0, color=MUTED, lw=1)
            ax.tick_params(colors=MUTED, labelsize=9)
            for L in E.STEER_LAYERS:
                ys = [sum(x[key] for x in rows
                          if (x["framing"], x["layer"], x["coef"]) == (framing, L, cf)) / len(COLORS)
                      for cf in COEFS]
                ax.plot([0] + COEFS, [0] + ys, color=SERIES[L], lw=2, marker="o", ms=5,
                        label=f"layer {L}")
            if r == 0:
                ax.set_title(label, fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f'"{framing}" framing', fontsize=11, color=INK)
            if r == len(F.FRAMINGS) - 1:
                ax.set_xlabel("steering coefficient (× residual norm)", fontsize=9,
                              color=MUTED)
    axes[0][0].legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(f"Prefer-vector steering vs. magnitude — {E.MODE} items, "
                 "mean over 7 colors × 20 worst pairs", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(E.OUT / "sweep.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {E.OUT / 'sweep.png'}")


if __name__ == "__main__":
    cache = E.OUT / "sweep.json"  # delete it to force a fresh sweep
    plot(json.loads(cache.read_text()) if cache.exists() else run())
