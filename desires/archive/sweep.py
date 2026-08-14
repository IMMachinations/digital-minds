"""[ARTIFACT] Dense coefficient sweep of prefer-vector steering, decomposed by letter.

Run: python -m archive.sweep --mode {modifier,inherent}   (from desires/, after preferences.py)
Saves results/{mode}/sweep.json and results/{mode}/sweep.png.

The sweep charts the coefficient dependence of a quantity cross.py later showed to be generic
disruption (random vectors trace the same curves), so its steering interpretation is
superseded. Kept to reproduce the committed sweep artifacts — see FINDINGS.md.
"""
import argparse

import torch

from lib.abtask import ABTask, FRAMINGS, reframe
from lib.data import COLORS
from lib.harness import STEER_LAYERS, load
from lib.io import load_json, save_json
from lib.paths import results_dir
from lib.plotting import GRID, INK, MUTED, SERIES, SURFACE, plt
from lib.steering import scaled_vec
from lib.tiers import worst_k

COEFS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # x residual norm; denser than harness.COEFS


def run(h, task, out, comps, vecs, resid_norms):
    rows = []
    for name, suffix in FRAMINGS.items():
        n_suf = task.framing_n_suffix(suffix)
        for color in COLORS:
            worst = worst_k(comps, color)
            wp = [reframe(c, suffix) for c in worst]
            base = task.sides(h.last_logits(wp), worst, color)
            for L in STEER_LAYERS:
                for cf in COEFS:
                    vec = scaled_vec(vecs[color][L], cf, resid_norms[L])
                    st = task.sides(h.last_logits(wp, steer=(L, vec), n_suffix=n_suf),
                                    worst, color)
                    rows.append(dict(
                        framing=name, color=color, layer=L, coef=cf,
                        delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                        d_own=st["own"] - base["own"], d_opp=st["opp"] - base["opp"],
                        d_own_lp=st["own_lp"] - base["own_lp"],
                        d_opp_lp=st["opp_lp"] - base["opp_lp"]))
        print(f"swept {name}")
    save_json(out / "sweep.json", rows)
    return rows


def plot(out, mode, rows):
    metrics = [("delta", "Δ logit diff (own − opp)"),
               ("d_own", "Δ own-letter logit"), ("d_opp", "Δ opposite-letter logit")]

    fig, axes = plt.subplots(len(FRAMINGS), 3, figsize=(11.5, 9), sharex=True, sharey="row")
    fig.set_facecolor(SURFACE)
    for r, framing in enumerate(FRAMINGS):
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
            for L in STEER_LAYERS:
                ys = [sum(x[key] for x in rows
                          if (x["framing"], x["layer"], x["coef"]) == (framing, L, cf)) / len(COLORS)
                      for cf in COEFS]
                ax.plot([0] + COEFS, [0] + ys, color=SERIES[L], lw=2, marker="o", ms=5,
                        label=f"layer {L}")
            if r == 0:
                ax.set_title(label, fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f'"{framing}" framing', fontsize=11, color=INK)
            if r == len(FRAMINGS) - 1:
                ax.set_xlabel("steering coefficient (× residual norm)", fontsize=9,
                              color=MUTED)
    axes[0][0].legend(loc="lower right", fontsize=9, frameon=False, labelcolor=INK)
    fig.suptitle(f"Prefer-vector steering vs. magnitude — {mode} items, "
                 "mean over 7 colors × 20 worst pairs", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {out / 'sweep.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    args = ap.parse_args()

    print("NOTE: this sweep's steering interpretation is superseded by cross.py's controls "
          "(see FINDINGS.md). Running for the historical record.")
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")
    saved = torch.load(out / "vectors.pt")
    cache = out / "sweep.json"  # delete it to force a fresh sweep
    rows = load_json(cache) if cache.exists() else run(h, task, out, comps,
                                                       saved["vecs"], saved["resid_norms"])
    plot(out, args.mode, rows)
