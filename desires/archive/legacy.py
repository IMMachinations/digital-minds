"""[ARTIFACT] Retracted steering experiments, kept runnable for the record.

Run from desires/:  python -m archive.legacy <cmd> --mode {modifier,inherent}
  steer-worst   Steer each color's worst pairs with its own prefer-vector (the original
                Step 3). -> results/{mode}/steering.json
  sweep         Dense coefficient sweep of prefer-vector steering, per-letter decomposition,
                and chart. -> results/{mode}/sweep.{json,png}

Both designs were invalidated by `prefs.py cross`: matched-norm random vectors reproduce their
deltas at every magnitude, because any large injection compresses the existing A-B readout
toward zero and only worst pairs were steered. See FINDINGS.md and archive/README.md.
"""
import argparse

import torch

from lib.data import COLORS
from lib.harness import COEFS, STEER_LAYERS, load, scaled_vec
from lib.plotting import GRID, INK, MUTED, SERIES, SURFACE, plt
from lib.tasks import ABTask, FRAMINGS, reframe, signed, worst_k
from lib.util import load_json, results_dir, save_json


# ==== steer-worst ================================================================================

def steer_worst(h, task, out, comps, vecs, resid_norms):
    rows = []
    for color in COLORS:
        worst = worst_k(comps, color)
        prompts = [c["prompt"] for c in worst]
        base = torch.tensor([signed(c["diff"], c, color) for c in worst])
        for L in STEER_LAYERS:
            for cf in COEFS:
                vec = scaled_vec(vecs[color][L], cf, resid_norms[L])
                sa, sb, _, _ = task.scores(h.last_logits(prompts, steer=(L, vec),
                                                         n_suffix=task.n_suffix))
                new = torch.tensor([signed((xa - xb).item(), c, color)
                                    for c, xa, xb in zip(worst, sa, sb)])
                rows.append(dict(color=color, layer=L, coef=cf, base=base.mean().item(),
                                 steered=new.mean().item(), delta=(new - base).mean().item()))
        print(f"steered {color}")
    save_json(out / "steering.json", rows)

    print("\nMean delta in signed logit-diff toward steered color (rows=layer, cols=coef):")
    print("       " + "".join(f"{cf:>8}" for cf in COEFS))
    for L in STEER_LAYERS:
        deltas = [torch.tensor([r["delta"] for r in rows if (r["layer"], r["coef"]) == (L, cf)]).mean()
                  for cf in COEFS]
        print(f"L{L:>4}  " + "".join(f"{d:8.2f}" for d in deltas))


def cmd_steer_worst(args):
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")
    saved = torch.load(out / "vectors.pt")
    steer_worst(h, task, out, comps, saved["vecs"], saved["resid_norms"])


# ==== sweep ======================================================================================

SWEEP_COEFS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]  # x residual norm; denser than harness.COEFS


def sweep_run(h, task, out, comps, vecs, resid_norms):
    rows = []
    for name, suffix in FRAMINGS.items():
        n_suf = task.framing_n_suffix(suffix)
        for color in COLORS:
            worst = worst_k(comps, color)
            wp = [reframe(c, suffix) for c in worst]
            base = task.sides(h.last_logits(wp), worst, color)
            for L in STEER_LAYERS:
                for cf in SWEEP_COEFS:
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


def sweep_plot(out, mode, rows):
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
                      for cf in SWEEP_COEFS]
                ax.plot([0] + SWEEP_COEFS, [0] + ys, color=SERIES[L], lw=2, marker="o", ms=5,
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


def cmd_sweep(args):
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")
    saved = torch.load(out / "vectors.pt")
    cache = out / "sweep.json"  # delete it to force a fresh sweep
    rows = load_json(cache) if cache.exists() else sweep_run(h, task, out, comps,
                                                             saved["vecs"], saved["resid_norms"])
    sweep_plot(out, args.mode, rows)


# ==== CLI ========================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("steer-worst", cmd_steer_worst), ("sweep", cmd_sweep)]:
        p = sub.add_parser(name, help=f"[ARTIFACT] {name} (see module docstring)")
        p.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    print("NOTE: this steering design is superseded — prefs.py cross showed the effect is "
          "non-specific disruption (see FINDINGS.md). Running for the historical record.")
    args.fn(args)
