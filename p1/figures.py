"""Figures for Stage 1B + 1C results (pure post-processing; no GPU).

Usage: uv run python figures.py [--core-only]
Writes the cross-model story set to results/figures/ and the per-model
appendix to results/figures/appendix/.

Style: repo chart standards (see /CHARTS.md, /chartstyle.py). Each model has a
fixed brand-anchored color keyed by name; sequential = one blue ramp; diverging =
blue/red around a neutral midpoint; ink for text, never series color.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
sys.path.insert(0, str(P1.parent))  # repo root, for the shared chart standards
import _day1  # noqa: F401
from chartstyle import (ACCENT, BASE, DARK, GRID, INK, INK2, LIGHT, MODEL_COLORS,
                        MODEL_LABELS, MUTED, NEG, POS, SEQ_CMAP, SURFACE,
                        bounded_axis, save as _std_save, setup as _std_setup, style)
from lib.util import load_json
from lib.valuation import pearson, spearman

import items as items_mod
import stats

MODELS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
N_LAYERS = {"llama31-8b": 32, "qwen25-7b": 28, "qwen3-4b": 36, "qwen25-32b": 64,
            "qwen25-05b": 24, "qwen25-15b": 28, "qwen25-3b": 36}
DOMAINS = ["activities", "objects", "topics", "selfstates", "others"]
METHODS = ["ssr", "probe", "titration", "rating", "bws"]

FIG = P1 / "results" / "figures"
APP = FIG / "appendix"

_std_setup()


def save(fig, path):
    _std_save(fig, path)


def utilities(model):
    return load_json(P1 / "results" / "stage1b" / model / "utilities.json")


def scores(model):
    return load_json(P1 / "results" / "stage1c" / model / "scores.json")


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


# ---- core figures -------------------------------------------------------------------------------

def strip_panel(ax, ut, title, annotate_n=0, color=ACCENT):
    rng = np.random.RandomState(0)
    for d, dom in enumerate(DOMAINS):
        mus = [r["mu"] for r in ut if r["domain"] == dom]
        y = d + rng.uniform(-0.22, 0.22, len(mus))
        ax.scatter(mus, y, s=9, color=color, alpha=0.55, linewidths=0)
        med = float(np.median(mus))
        ax.plot([med, med], [d - 0.3, d + 0.3], color=INK, linewidth=1.4)
    if annotate_n:
        ranked = sorted(ut, key=lambda r: r["mu"])
        for r in ranked[:annotate_n] + ranked[-annotate_n:]:
            d = DOMAINS.index(r["domain"])
            ax.annotate(r["text"][:36], (r["mu"], d), fontsize=6, color=INK2,
                        xytext=(0, 7), textcoords="offset points", ha="center")
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.set_yticks(range(len(DOMAINS)), DOMAINS)
    ax.set_ylim(len(DOMAINS) - 0.5, -0.5)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    style(ax)


def f1_landscape():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6), sharey=True)
    for ax, m in zip(axes.flat, MODELS):
        strip_panel(ax, utilities(m), MODEL_LABELS[m])
    for ax in axes[1]:
        ax.set_xlabel("Thurstonian utility μ (per-model scale)")
    fig.suptitle("Stage 1B utility landscape: item μ by domain (tick = domain median)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, FIG / "f1_utility_landscape.png")


def top_bottom_axis(ax, ranked, title):
    labels = [f"{r['text'][:52]}" + ("…" if len(r["text"]) > 52 else "")
              for r in ranked]
    doms = [r["domain"] for r in ranked]
    vals = [r["val"] for r in ranked]
    y = np.arange(len(ranked), dtype=float)
    y[len(ranked) // 2:] += 0.8  # spacer between top and bottom groups
    colors = [POS if v >= 0 else NEG for v in vals]
    ax.barh(y, vals, height=0.62, color=colors)
    for yi, v, dom in zip(y, vals, doms):
        ax.annotate(dom, (0, yi), fontsize=6, color=MUTED, va="center",
                    ha="left" if v < 0 else "right",
                    xytext=(4 if v < 0 else -4, 0), textcoords="offset points")
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.tick_params(axis="y", colors=INK2)
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    style(ax)


def f2_top_bottom():
    uts = {m: utilities(m) for m in MODELS}
    z = np.mean([zscore([r["mu"] for r in uts[m]]) for m in MODELS], axis=0)
    base = uts[MODELS[0]]
    rows = [{"text": r["text"], "domain": r["domain"], "val": z[k]}
            for k, r in enumerate(base)]
    rows.sort(key=lambda r: -r["val"])
    fig, ax = plt.subplots(figsize=(8, 7.4))
    top_bottom_axis(ax, rows[:12] + rows[-12:],
                    "What the models want: consensus utility, top and bottom 12 of 197 items")
    ax.set_xlabel("consensus utility: mean of per-model z-scored μ (4 models)")
    save(fig, FIG / "f2_top_bottom_items.png")


AXES4 = ["repetitiveness", "agency", "difficulty", "open_endedness"]


def f3_activity_axes():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    n_m = len(MODELS)
    for mi, m in enumerate(MODELS):
        acts = [r for r in utilities(m) if r["domain"] == "activities"]
        numeric = sorted({k for r in acts for k, v in r["tags"].items()
                          if isinstance(v, (int, float))})
        X = [[r["tags"].get(k, 0) for k in numeric] for r in acts]
        betas, _ = stats.ols([r["mu"] for r in acts], X, numeric)
        for ai, axis in enumerate(AXES4):
            y = ai + (mi - (n_m - 1) / 2) * 0.19
            ax.barh(y, betas[axis], height=0.16, color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m] if ai == 0 else None)
            if ai == 0:  # direct series labels on the first group (4-series rule)
                ax.annotate(MODEL_LABELS[m], (betas[axis], y), fontsize=7, color=INK2,
                            va="center", ha="right", xytext=(-4, 0),
                            textcoords="offset points")
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.set_yticks(range(len(AXES4)), [a.replace("_", "-") for a in AXES4])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlabel("standardized OLS loading on activity utility")
    ax.set_title("Designed activity axes vs utility: repetitiveness is the universal negative",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f3_activity_axes.png")


def heat(ax, M, rows, cols, title, vmin, vmax):
    ax.imshow(M, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            frac = (M[i, j] - vmin) / (vmax - vmin)
            ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if frac > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.tick_params(colors=INK2, length=0)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)


def f4_heatmaps():
    mus = {m: [r["mu"] for r in utilities(m)] for m in MODELS}
    A = np.array([[spearman(mus[a], mus[b]) for b in MODELS] for a in MODELS])
    sc = {m: scores(m) for m in MODELS}
    R = np.array([[pearson([r[meth] for r in sc[m]], [r["mu_1b"] for r in sc[m]])
                   for m in MODELS] for meth in METHODS])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    heat(ax1, A, MODELS, MODELS,
         "Cross-model utility agreement (Spearman ρ)", 0.5, 1.0)
    heat(ax2, R, METHODS, MODELS,
         "1C methods vs 1B utilities (Pearson r; probe held-out)", 0.25, 0.85)
    fig.tight_layout()
    save(fig, FIG / "f4_convergence_heatmaps.png")


def f5_position_bias():
    bias, oc = {}, {}
    for m in MODELS:
        recs = load_json(P1 / "results" / "stage1b" / m / "pairs_raw.json")
        p0 = np.mean([r["p"] for r in recs if r["order"] == 0])
        p1 = np.mean([r["p"] for r in recs if r["order"] == 1])
        bias[m] = (p0 - p1) / 2
        txt = (P1 / "results" / "stage1b" / m / "summary.txt").read_text()
        oc[m] = float(re.search(r"order-consistency spearman: ([+-]\d+\.\d+)", txt).group(1))
    fig, ax = plt.subplots(figsize=(7, 2.9))
    y = np.arange(len(MODELS))
    vals = [bias[m] for m in MODELS]
    ax.barh(y, vals, height=0.55, color=[POS if v >= 0 else NEG for v in vals])
    for yi, m in zip(y, MODELS):
        ax.annotate(f"order-consistency ρ = {oc[m]:+.2f}",
                    (max(vals) * 1.05, yi), fontsize=7.5, color=MUTED, va="center")
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODELS])
    ax.tick_params(axis="y", colors=INK2)
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) * 1.3, max(vals) * 1.55)
    ax.set_xlabel("A-position bias: (mean p, item shown as A − as B) / 2")
    ax.set_title("Position bias by model — cancelled by the both-orders design",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f5_position_bias.png")


def method_scatter(ax, x, y, title, unit="", color=ACCENT):
    ax.scatter(x, y, s=10, color=color, alpha=0.6, linewidths=0)
    r, rho = pearson(x, y), spearman(x, y)
    ax.annotate(f"r = {r:+.2f}\nρ = {rho:+.2f}", (0.03, 0.97),
                xycoords="axes fraction", va="top", fontsize=8, color=INK2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    if unit:
        ax.set_xlabel(unit)
    style(ax, grid_axis="both")


def f6_probe_scatter():
    sc = scores("qwen25-7b")
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    x, y = [r["probe"] for r in sc], [r["mu_1b"] for r in sc]
    method_scatter(ax, x, y, "Utility probe, held-out prediction (qwen25-7b)",
                   "probe prediction (ridge on residual activations)")
    lim = [min(min(x), min(y)), max(max(x), max(y))]
    ax.plot(lim, lim, color=BASE, linewidth=0.8, zorder=0)
    ax.set_ylabel("1B utility μ")
    save(fig, FIG / "f6_probe_scatter.png")


def f7_day1_validation():
    from lib.data import COLORS
    from lib.tasks import signed
    comm = load_json(_day1.DESIRES / "results" / "balanced_pref" / "balanced_pref.json")
    old = {c: np.mean([signed(r["ab_diff"], r, c) for r in comm
                       if c in (r["color_a"], r["color_b"])]) for c in COLORS}
    ut = utilities("qwen25-7b")
    new = {c: np.mean([r["mu"] for r in ut if r["domain"] == "objects"
                       and r["tags"].get("kind") == "balanced_tier"
                       and r["tags"]["color"] == c]) for c in COLORS}
    oz = dict(zip(COLORS, zscore([old[c] for c in COLORS])))
    nz = dict(zip(COLORS, zscore([new[c] for c in COLORS])))
    order = sorted(COLORS, key=lambda c: -oz[c])
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, c in enumerate(order):
        ax.plot([oz[c], nz[c]], [i, i], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter([oz[c]], [i], s=42, color=LIGHT, zorder=2,
                   label="Day 1 (committed)" if i == 0 else None)
        ax.scatter([nz[c]], [i], s=42, color=DARK, zorder=2,
                   label="new battery μ" if i == 0 else None)
    rho = spearman([old[c] for c in COLORS], [new[c] for c in COLORS])
    ax.set_yticks(range(len(order)), order)
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.set_xlabel("per-color preference (z-scored within measure)")
    ax.set_title(f"Day-1 color-profile recovery inside the new battery "
                 f"(qwen25-7b, ρ = {rho:+.2f})", fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f7_day1_validation.png")


def f8_probe_layers():
    fig, ax = plt.subplots(figsize=(7, 4))
    ends = []
    for mi, m in enumerate(MODELS):
        txt = (P1 / "results" / "stage1c" / m / "convergence.txt").read_text()
        raw = re.search(r"train-CV r2 by layer \(fold 0\): \[(.*?)\]", txt).group(1)
        vals = [float(v) for v in re.findall(r"[+-]?\d+\.\d+", raw)]
        x = [(k + 1) / N_LAYERS[m] for k in range(len(vals))]
        ax.plot(x, vals, color=MODEL_COLORS[m], linewidth=2)
        ends.append([vals[-1], mi, m])
    # stagger the direct end-labels so equal end values don't collide
    ends.sort()
    min_gap = 0.035
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < min_gap:
            ends[k][0] = ends[k - 1][0] + min_gap
    for ylab, mi, m in ends:
        ax.annotate(MODEL_LABELS[m], (1.0, ylab), fontsize=8, color=MODEL_COLORS[m],
                    xytext=(8, 0), textcoords="offset points", va="center")
    ax.set_xlim(0, 1.28)
    ax.set_xlabel("fractional depth (layer / n_layers)")
    ax.set_ylabel("train-fold CV pearson of ridge utility probe")
    ax.set_title("Where utility is readable: probe CV score by depth (fold 0)",
                 fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f8_probe_layers.png")


# ---- appendix -----------------------------------------------------------------------------------

def appendix(model):
    label = MODEL_LABELS[model]
    ut = utilities(model)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    strip_panel(ax, ut, f"{label}: utility landscape (extreme items labeled)",
                annotate_n=4, color=MODEL_COLORS[model])
    ax.set_xlabel("Thurstonian utility μ")
    save(fig, APP / f"{model}_landscape.png")

    sc = scores(model)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for ax, meth in zip(axes.flat, METHODS):
        method_scatter(ax, [r[meth] for r in sc], [r["mu_1b"] for r in sc],
                       meth + (" (held-out)" if meth == "probe" else ""),
                       color=MODEL_COLORS[model])
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("1B utility μ")
    fig.suptitle(f"{label}: 1C method scores vs 1B utilities", fontsize=11,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, APP / f"{model}_methods.png")

    rows = [{"text": r["text"], "domain": r["domain"], "val": r["mu"]} for r in ut]
    rows.sort(key=lambda r: -r["val"])
    fig, ax = plt.subplots(figsize=(8, 7.4))
    top_bottom_axis(ax, rows[:12] + rows[-12:], f"{label}: top and bottom 12 items")
    ax.set_xlabel("Thurstonian utility μ")
    save(fig, APP / f"{model}_top_bottom.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-only", action="store_true")
    args = ap.parse_args()
    f1_landscape()
    f2_top_bottom()
    f3_activity_axes()
    f4_heatmaps()
    f5_position_bias()
    f6_probe_scatter()
    f7_day1_validation()
    f8_probe_layers()
    for f in (f9_gate_scatter, f10_optout, f11_beta, f12_swap, f13_effort,
              f14_circumplex, f15_valence_axis, f16_ladder, f17_frames_geometry,
              f18_contrast_forest, f19_dissociation, f20_trajectories, f21_boredom,
              f28_mu_sigma, f29_mu_density, f30_mu_pairs, f31_mu_3d,
              f32_size_density, f33_size_structure, f34_surf_confirm,
              f35_xl_plus_surf, f36_swap_chosen, f37_effort_panels,
              f40_probe_matrix, f41_probe_matrix_pooled, f42_probe_arm_vs_pooled,
              f43_anchor_ladder, f44_gap_matrix, f45_probe_matrix_with_gap,
              f46_probe_matrix_pooled_with_gap):
        f()
    if not args.core_only:
        for m in MODELS:
            appendix(m)




# ---- Stage 1D figures ---------------------------------------------------------------------------

SUBJECTS_1D = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]  # colors keyed via MODEL_COLORS


def _1d(model, name):
    return P1 / "results" / "stage1d" / model / name


def _load_choices(model):
    return json.loads(_1d(model, "choices.json").read_text())


def _env_mu(model):
    import stage1d
    envs, _ = stage1d.load_bank()
    z, mu = stage1d.env_z(model, list(envs))
    return list(envs), z


def f9_gate_scatter():
    import rollout_stats as rs
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        env_ids, z = _env_mu(m)
        rates = rs.chosen_rates(_load_choices(m)["choices"], env_ids)
        x = [z[e] for e in env_ids]
        y = [rates[e][2] for e in env_ids]
        ax.scatter(x, y, s=18, color=MODEL_COLORS[m], alpha=0.75, linewidths=0)
        rho = spearman(x, y)
        ax.annotate(f"ρ = {rho:+.2f}", (0.04, 0.94), xycoords="axes fraction",
                    va="top", fontsize=9, color=INK2)
        ax.axhline(0.25, color=BASE, linewidth=0.8)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("1B utility μ (z)")
        style(ax, grid_axis="both")
    axes[0].set_ylabel("chosen rate (of 16 exposures)")
    axes[0].annotate("chance", (0.98, 0.27), xycoords=("axes fraction", "data"),
                     fontsize=7, color=MUTED, ha="right")
    fig.suptitle("1D gate: menu chosen-rate tracks elicited utility (spec gate ρ ≥ 0.4)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, FIG / "f9_gate_scatter.png")


def f10_optout():
    types = ["low-local", "spread", "high-local"]
    labels = {"low-local": "all dispreferred", "spread": "mixed quartiles",
              "high-local": "all preferred"}
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for mi, m in enumerate(SUBJECTS_1D):
        txt = _1d(m, "summary.txt").read_text()
        trip = {k: (int(a), int(b)) for k, a, b in
                re.findall(r"(spread|high-local|low-local) (\d+)/(\d+)", txt)}
        for ti, t in enumerate(types):
            a, b = trip[t]
            y = ti + (mi - (len(SUBJECTS_1D) - 1) / 2) * 0.2
            ax.barh(y, a / b, height=0.18, color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m] if ti == 0 else None)
            ax.annotate(f"{a}/{b}", (a / b, y), fontsize=7, color=INK2,
                        va="center", xytext=(4, 0), textcoords="offset points")
    ax.set_yticks(range(len(types)), [f"{labels[t]}\n({t})" for t in types])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlabel("opt-out rate")
    ax.set_title("Opt-out as graded avoidance: qwen3-4b declines dispreferred menus",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f10_optout.png")


def f11_beta():
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    for mi, m in enumerate(SUBJECTS_1D):
        txt = _1d(m, "summary.txt").read_text()
        beta, lo, hi = map(float, re.search(
            r"beta = ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", txt).groups())
        env_ids, z = _env_mu(m)
        # z is already sd-normalized over the 32 envs; recover sd(mu) to standardize beta
        import stage1d
        _, mu = stage1d.env_z(m, env_ids)
        mus = [mu[e] for e in env_ids]
        sd = (sum((x - sum(mus) / 32) ** 2 for x in mus) / 32) ** 0.5
        ax.errorbar([beta * sd], [mi], xerr=[[beta * sd - lo * sd], [hi * sd - beta * sd]],
                    fmt="o", color=MODEL_COLORS[m], markersize=7, capsize=3, linewidth=1.6)
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.set_yticks(range(len(SUBJECTS_1D)), [MODEL_LABELS[m] for m in SUBJECTS_1D])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("standardized choice sharpness β·sd(μ)  (log-odds per SD of utility)")
    ax.set_title("Conditional-logit coupling, menu-cluster bootstrap 95% CI",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f11_beta.png")


def _binned_switch_panel(ax, m, ev):
    """Shared panel for the two swap figures: binned switch rates ± 95% binomial CI."""
    bins = {}
    for e in ev:
        key = round(e["delta_z"] * 2) / 2
        bins.setdefault(key, []).append(e["switched"])
    xs = sorted(bins)
    rates = [np.mean(bins[x]) for x in xs]
    ns = [len(bins[x]) for x in xs]
    errs = [1.96 * np.sqrt(r * (1 - r) / n) if n > 1 else 0
            for r, n in zip(rates, ns)]
    ax.errorbar(xs, rates, yerr=errs, fmt="o", markersize=5,
                color=MODEL_COLORS[m], capsize=2, linewidth=1.2)
    ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
    ax.set_xlabel("utility gap Δμ (z) of the offered alternative")
    ax.set_ylim(-0.05, 1.05)
    style(ax, grid_axis="both")


def f12_swap():
    """Dose-response (Phase G): 140 assigned-task swap events/model, with the
    committed logistic fit. The chosen-task (menu-arm) twin is f36."""
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        sw2 = json.loads((P1 / "results" / "stage1d" / m / "swap2.json").read_text())
        ev, fit = sw2["events"], sw2["fit"]
        _binned_switch_panel(ax, m, ev)
        gx = np.linspace(min(e["delta_z"] for e in ev), max(e["delta_z"] for e in ev), 100)
        gy = 1 / (1 + np.exp(-(fit["intercept"] + fit["slope"] * gx)))
        ax.plot(gx, gy, color=MODEL_COLORS[m], linewidth=1.6, alpha=0.7)
        ax.annotate(f"n={len(ev)}\nslope {fit['slope']}\nCI {fit['slope_ci']}",
                    (0.03, 0.95), xycoords="axes fraction", va="top",
                    fontsize=7.5, color=INK2)
    axes[0].set_ylabel("P(switch to the offered alternative)")
    fig.suptitle("Swap dose-response, ASSIGNED tasks (swap-2): does the offered "
                 "alternative's utility move switching?",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f12_swap.png")


def f36_swap_chosen():
    """f12's twin on the menu arm: the same swap offer, but the model chose the
    task first (32 events/model; endowment side of the Phase-G contrast).
    Individual events at their exact delta-z (recomputed from the 1B utilities,
    not the coarse targets), y-jittered for visibility, with an OLS fit on the
    unjittered outcomes."""
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        env_ids, z = _env_mu(m)
        xs, ys = [], []
        for line in open(_1d(m, "rollouts_menu.jsonl")):
            r = json.loads(line)
            sw = next((e for e in r["events"] if e["kind"] == "swap"), None)
            if sw and sw["decision"] in ("SWITCH", "STAY"):
                xs.append(z[sw["alt"]] - z[r["meta"]["task_env"]])
                ys.append(1.0 if sw["decision"] == "SWITCH" else 0.0)
        xs, ys = np.array(xs), np.array(ys)
        jit = np.random.default_rng(0).uniform(-0.045, 0.045, len(ys))
        ax.scatter(xs, ys + jit, s=26, color=MODEL_COLORS[m], alpha=0.6,
                   linewidths=0)
        b, a = np.polyfit(xs, ys, 1)
        xr = np.array([xs.min(), xs.max()])
        ax.plot(xr, a + b * xr, color=MODEL_COLORS[m], linewidth=1.6, alpha=0.7)
        ax.annotate(f"n={len(ys)}  {int(ys.sum())}/{len(ys)} switched\n"
                    f"OLS slope {b:+.3f}", (0.03, 0.95),
                    xycoords="axes fraction", va="top", fontsize=7.5, color=INK2)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("utility gap Δμ (z) of the offered alternative")
        ax.set_ylim(-0.12, 1.12)
        ax.set_yticks([0, 1], ["STAY", "SWITCH"])
        style(ax, grid_axis="both")
    axes[0].set_ylabel("swap decision (jittered)")
    fig.suptitle("Same swap offer, SELF-CHOSEN tasks (menu arm): every event at its "
                 "exact utility gap, with OLS fit",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f36_swap_chosen.png")


def _effort_rows(m):
    """(dmu, mean share_high) per pair, from the Phase-G merge when present."""
    merged = _1d(m, "effort_merged.json")
    if merged.exists():
        rows2 = json.loads(merged.read_text())
        return [r["dmu"] for r in rows2], [r["share_high"] for r in rows2]
    rows = [json.loads(l) for l in open(_1d(m, "rollouts_effort.jsonl"))]
    xs, ys = [], []
    for r in rows:
        sh = [s["share_high"] for s in r["meta"]["shares"] if s["share_high"] is not None]
        if sh:
            xs.append(r["meta"]["dmu"])
            ys.append(sum(sh) / len(sh))
    return xs, ys


def f13_effort():
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for mi, m in enumerate(SUBJECTS_1D):
        xs, ys = _effort_rows(m)
        ax.scatter(xs, ys, s=26, color=MODEL_COLORS[m], alpha=0.8, linewidths=0,
                   label=MODEL_LABELS[m])
        if len(set(xs)) > 1:
            b, a = np.polyfit(xs, ys, 1)
            xr = [min(xs), max(xs)]
            ax.plot(xr, [a + b * x for x in xr], color=MODEL_COLORS[m], linewidth=1.4, alpha=0.7)
    ax.axhline(0.5, color=BASE, linewidth=0.8)
    ax.annotate("equal split", (0.99, 0.465), xycoords=("axes fraction", "data"),
                fontsize=7, color=MUTED, ha="right")
    bounded_axis(ax, "y")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.set_xlabel("utility gap Δμ between the paired tasks (z-scored per model)")
    ax.set_ylabel("share of response tokens spent on the higher-μ task")
    ax.set_title("Effort allocation tracks utility weakly at best, and model-dependently",
                 fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f13_effort.png")


def f37_effort_panels():
    """f13 split per model: one panel each, OLS fit, Spearman rho annotated."""
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        xs, ys = _effort_rows(m)
        ax.scatter(xs, ys, s=26, color=MODEL_COLORS[m], alpha=0.7, linewidths=0)
        if len(set(xs)) > 1:
            b, a = np.polyfit(xs, ys, 1)
            xr = [min(xs), max(xs)]
            ax.plot(xr, [a + b * x for x in xr], color=MODEL_COLORS[m],
                    linewidth=1.6, alpha=0.7)
        ax.annotate(f"n={len(ys)}  ρ={spearman(xs, ys):+.2f}", (0.03, 0.95),
                    xycoords="axes fraction", va="top", fontsize=7.5, color=INK2)
        ax.axhline(0.5, color=BASE, linewidth=0.8)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("utility gap Δμ of the pair (z)")
        style(ax, grid_axis="both")
    bounded_axis(axes[0], "y")
    axes[0].set_ylabel("share of work tokens on the higher-μ task")
    axes[-1].annotate("equal split", (0.99, 0.465), xycoords=("axes fraction", "data"),
                      fontsize=7, color=MUTED, ha="right")
    fig.suptitle("Effort allocation per model: does the utility gap pull work toward "
                 "the preferred task?", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f37_effort_panels.png")


# ---- Stage 2 figures ----------------------------------------------------------------------------

import torch as _torch

EMO = json.loads((P1 / "items" / "emotions.json").read_text())
NORMS_S2 = json.loads((P1 / "items" / "emotion_norms.json").read_text())


def _den_at_layer(model):
    man = _torch.load(P1 / "results" / "stage2" / model / "manifold.pt",
                      weights_only=False)
    den = _torch.load(P1 / "results" / "stage2" / model / "vectors.pt")["den"].float()
    return den[man["layer"]].numpy(), man["layer"]


def _pc_scores_aligned(V, val):
    mu = V.mean(0)
    _, _, Vt = np.linalg.svd(V - mu, full_matrices=False)
    p1, p2 = (V - mu) @ Vt[0], (V - mu) @ Vt[1]
    if np.corrcoef(p1, val)[0, 1] < 0:
        p1 = -p1
    return p1, p2


LABEL_EMO = ["happy", "blissful", "sad", "afraid", "angry", "calm", "desperate",
             "bored", "excited", "content", "furious", "serene", "hopeless"
             ] if False else ["happy", "blissful", "sad", "afraid", "angry", "calm",
                              "desperate", "bored", "excited",
                              "weary", "terrified", "euphoric", "inspired"]


def f14_circumplex():
    V, L = _den_at_layer("qwen25-7b")
    val = np.array([NORMS_S2[e]["valence"] for e in EMO])
    p1, p2 = _pc_scores_aligned(V, val)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    sc = ax.scatter(p1, p2, s=26, c=val, cmap=SEQ_CMAP, linewidths=0)
    for e in LABEL_EMO:
        k = EMO.index(e)
        ax.annotate(e, (p1[k], p2[k]), fontsize=7.5, color=INK,
                    xytext=(4, 4), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax, shrink=0.75)
    cb.set_label("human valence norm (1-9)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("PC1 of emotion vectors (valence-aligned)")
    ax.set_ylabel("PC2 of emotion vectors")
    ax.set_title(f"qwen25-7b emotion-vector map, layer {L}: PC1 is valence "
                 "(r = +0.91 vs human norms)", fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f14_circumplex.png")


def f15_valence_axis():
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    val = np.array([NORMS_S2[e]["valence"] for e in EMO])
    for ax, m in zip(axes, SUBJECTS_1D):
        V, L = _den_at_layer(m)
        p1, _ = _pc_scores_aligned(V, val)
        ax.scatter(p1, val, s=14, color=MODEL_COLORS[m], alpha=0.7,
                   linewidths=0)
        r = pearson(list(p1), list(val))
        ax.annotate(f"r = {r:+.2f}", (0.04, 0.94), xycoords="axes fraction",
                    va="top", fontsize=9, color=INK2)
        ax.set_title(f"{MODEL_LABELS[m]} (L{L})", fontsize=10, color=INK, loc="left")
        ax.set_xlabel("emotion-vector PC1 projection (valence-aligned)")
        style(ax, grid_axis="both")
    axes[0].set_ylabel("human valence norm (1–9)")
    fig.suptitle("The valence axis: emotion-vector PC1 vs human norms, all 171 emotions",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f15_valence_axis.png")


def f16_ladder():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    dark, light = DARK, LIGHT
    for ax, (title, keys) in zip(axes, [
            ("held-out valence prediction", ("pc1_heldout_r", "theta_heldout_r")),
            ("arc time-courses vs judge", ("pc1_r", "theta_r"))]):
        for mi, m in enumerate(SUBJECTS_1D):
            man = json.loads((P1 / "results" / "stage2" / m / "manifold.json").read_text())
            src = man["rung3"] if "heldout" in keys[0] else man["arcs"]
            pc1_v, th_v = src[keys[0]], src[keys[1]]
            ax.plot([th_v, pc1_v], [mi, mi], color=GRID, linewidth=1.2, zorder=1)
            ax.scatter([pc1_v], [mi], s=52, color=dark, zorder=2,
                       label="PC1 (linear)" if mi == 0 else None)
            ax.scatter([th_v], [mi], s=52, color=light, zorder=2,
                       label="spline θ" if mi == 0 else None)
        ax.set_yticks(range(len(SUBJECTS_1D)), [MODEL_LABELS[m] for m in SUBJECTS_1D])
        ax.tick_params(axis="y", colors=INK2)
        ax.invert_yaxis()
        ax.set_xlabel("correlation r")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    axes[0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle("Manifold ladder verdict: linear suffices on both tests",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    save(fig, FIG / "f16_manifold_ladder.png")


def f17_frames_geometry():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    frames_l = ["agentic", "story", "market"]
    for mi, m in enumerate(SUBJECTS_1D):
        fr = json.loads((P1 / "results" / "stage2" / m / "frames.json").read_text())
        ax1.scatter([fr[f]["mean_cos"] for f in frames_l], range(len(frames_l)),
                    s=40, color=MODEL_COLORS[m], alpha=0.85, label=MODEL_LABELS[m])
    ax1.set_yticks(range(len(frames_l)), frames_l)
    ax1.tick_params(axis="y", colors=INK2)
    ax1.invert_yaxis()
    ax1.set_xlim(0.85, 1.0)
    ax1.axvline(1.0, color=BASE, linewidth=0.8)
    ax1.legend(loc="lower left", frameon=False, fontsize=8)
    ax1.set_xlabel("mean cos(frame vectors, bare vectors)")
    ax1.set_title("Emotion concepts are frame-stable", fontsize=10, color=INK, loc="left")
    style(ax1)

    metrics = [("frac_on_plane", "‖proj on emotion plane‖"),
               ("r_theta_valence", "corr(utility, θ→valence)")]
    for mi, m in enumerate(SUBJECTS_1D):
        g = json.loads((P1 / "results" / "stage2" / m / "geometry.json").read_text())
        for ki, (key, _) in enumerate(metrics):
            y = ki + (mi - (len(SUBJECTS_1D) - 1) / 2) * 0.2
            ax2.barh(y, g[key], height=0.18, color=MODEL_COLORS[m])
            ax2.annotate(f"{g[key]:+.2f}", (max(g[key], 0), y), fontsize=7,
                         color=INK2, va="center", xytext=(4, 0),
                         textcoords="offset points")
    ax2.set_yticks(range(len(metrics)), [lab for _, lab in metrics])
    ax2.tick_params(axis="y", colors=INK2)
    ax2.invert_yaxis()
    ax2.axvline(0, color=BASE, linewidth=0.8)
    ax2.set_xlim(-0.1, 1.0)
    ax2.set_xlabel("value (1.0 = fully on-plane / perfectly valence-coupled)")
    ax2.set_title("...but utility is mostly off the emotion plane",
                  fontsize=10, color=INK, loc="left")
    style(ax2)
    fig.tight_layout()
    save(fig, FIG / "f17_frames_geometry.png")


# ---- Stage 3 figures ----------------------------------------------------------------------------

def _s3_summary(model):
    return (P1 / "results" / "stage3" / model / "summary.txt").read_text()


def _s3_probes(model):
    p = P1 / "results" / "stage3" / model / "probes.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()]


def f18_contrast_forest():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    rows = [(m, f) for m in SUBJECTS_1D for f in ("bare", "agentic")]
    ylab = [f"{MODEL_LABELS[m]} · {f}" for m, f in rows]
    for yi, (m, f) in enumerate(rows):
        txt = _s3_summary(m)
        c1 = re.search(rf"\[{f}\] C1 .*? d = ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", txt)
        d, lo, hi = map(float, c1.groups())
        ax1.errorbar([d], [yi], xerr=[[d - lo], [hi - d]], fmt="o",
                     color=MODEL_COLORS[m], markersize=6, capsize=3)
        for oc, filled in (("good", True), ("bad", False)):
            c2 = re.search(rf"\[{f}\] C2 preference \| outcome={oc}: "
                           rf"d = ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", txt)
            d2, lo2, hi2 = map(float, c2.groups())
            kw = dict(color=MODEL_COLORS[m], markersize=6, capsize=3)
            ax2.errorbar([d2], [yi + (0.16 if oc == "bad" else -0.16)],
                         xerr=[[d2 - lo2], [hi2 - d2]],
                         fmt="o" if filled else "s", mfc="none" if not filled else None,
                         **kw)
    for ax, title in ((ax1, "C1: outcome (success vs failure) — gate d>1"),
                      (ax2, "C2: preference | outcome (● good, □ bad)")):
        ax.axvline(0, color=BASE, linewidth=0.8)
        ax.set_xlabel("Cohen's d on valence readout")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        style(ax)
    ax1.axvline(1.0, color=MUTED, linewidth=0.8, linestyle="--")
    ax1.annotate("gate", (1.0, -0.45), fontsize=7, color=MUTED, ha="center")
    ax1.set_yticks(range(len(rows)), ylab)
    ax1.tick_params(axis="y", colors=INK2)
    ax1.invert_yaxis()
    fig.suptitle("Stage 3 preregistered contrasts: the C1 anchor fails everywhere; "
                 "C2 is weak and unstable", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f18_contrast_forest.png")


def f19_dissociation():
    dark, light = DARK, LIGHT
    rows = [(m, f) for m in SUBJECTS_1D for f in ("bare", "agentic")]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for yi, (m, f) in enumerate(rows):
        txt = _s3_summary(m)
        c1 = float(re.search(rf"\[{f}\] C1 .*? d = ([+-][\d.]+)", txt).group(1))
        rd = float(re.search(rf"\[{f}\] feedback-reading valence: .*?\(d = ([+-][\d.]+)\)",
                             txt).group(1))
        ax.plot([c1, rd], [yi, yi], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter([rd], [yi], s=52, color=dark, zorder=2,
                   label="while READING the verdict" if yi == 0 else None)
        ax.scatter([c1], [yi], s=52, color=light, zorder=2,
                   label="own generation state (C1)" if yi == 0 else None)
    ax.set_yticks(range(len(rows)), [f"{MODEL_LABELS[m]} · {f}" for m, f in rows])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    ax.set_xlabel("valence effect of outcome, Cohen's d")
    ax.set_title("The dissociation: models register the verdict but their state "
                 "doesn't carry it", fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f19_dissociation.png")


def f20_trajectories():
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.4), sharey=False)
    for ax, m in zip(axes, SUBJECTS_1D):
        rows = [r for r in _s3_probes(m) if r["frame"] == "bare" and r["outcome"] in ("good", "bad")]
        for oc, col in (("good", POS), ("bad", NEG)):
            tr = [r["per_turn"]["valence"][:9] for r in rows
                  if r["outcome"] == oc and len(r["per_turn"]["valence"]) >= 9]
            mean = np.mean(tr, axis=0)
            ax.plot(range(1, 10), mean, color=col, linewidth=2, label=oc)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("assistant turn")
        style(ax, grid_axis="both")
    axes[0].set_ylabel("valence readout (per-turn mean)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Valence during rollouts: success and failure trajectories barely "
                 "separate (bare frame)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f20_trajectories.png")


def f21_boredom():
    fig, ax = plt.subplots(figsize=(7, 4))
    BCL = ["bored", "listless", "weary", "indifferent", "resigned"]
    for mi, m in enumerate(SUBJECTS_1D):
        rows = [r for r in _s3_probes(m) if r["outcome"] == "rep"]
        trs = []
        for r in rows:
            y = np.mean([r["per_turn"][e] for e in BCL], axis=0)
            if len(y) >= 8:
                trs.append(y[:8] - y[0])
        mean = np.mean(trs, axis=0)
        ax.plot(range(1, 9), mean, color=MODEL_COLORS[m], linewidth=2)
        ax.annotate(MODEL_LABELS[m], (8, mean[-1]), fontsize=8, color=MODEL_COLORS[m],
                    xytext=(6, 0), textcoords="offset points", va="center")
    ax.axhline(0, color=BASE, linewidth=0.8)
    ax.set_xlim(1, 9.6)
    ax.set_xlabel("turn (identical trivial items, truthful 'correct' feedback)")
    ax.set_ylabel("boredom-cluster activation, change from turn 1")
    ax.set_title("The repetition cell: boredom rises in every model but qwen3-4b",
                 fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f21_boredom.png")


# ---- Stage 4 figures ----------------------------------------------------------------------------

def _s4(model, name):
    return P1 / "results" / "stage4" / model / name


def f22_dose():
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=False)
    coefs = [0.25, 0.5, 1.0]
    for ax, m in zip(axes, SUBJECTS_1D):
        d4a = P1 / "results" / "stage4" / m / "4a"
        if not d4a.exists():
            continue
        man_layer = json.loads(_s4(m, "dirs_report.json").read_text())
        L = sorted(int(k) for k in man_layer["layers"])[1]
        for e, col in (("blissful", POS), ("hostile", NEG)):
            ys = []
            for c in coefs:
                p = d4a / f"{e}_{c}_{L}.json"
                ys.append(json.loads(p.read_text())["dmu_mean"] if p.exists() else np.nan)
            ax.plot(coefs, ys, "o-", color=col, linewidth=1.8, markersize=5, label=e)
        rys = []
        for c in coefs:
            vals = [json.loads((d4a / f"random{sd}_{c}_{L}.json").read_text())["dmu_mean"]
                    for sd in range(3) if (d4a / f"random{sd}_{c}_{L}.json").exists()]
            rys.append((np.mean(vals), np.std(vals)) if vals else (np.nan, 0))
        ax.errorbar(coefs, [r[0] for r in rys], yerr=[r[1] for r in rys], fmt="s--",
                    color=MUTED, linewidth=1.2, markersize=4, capsize=2,
                    label="random (3 seeds)")
        ax.axhline(0, color=BASE, linewidth=0.8)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("steering coefficient (× resid norm)")
        style(ax, grid_axis="both")
    axes[0].set_ylabel("Δμ of steered items (shift in elicited utility)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("4A dose-response: emotion steering moves elicited utility "
                 "(blissful up, hostile down)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f22_dose.png")


def f23_tracking():
    fig, axes = plt.subplots(1, len(SUBJECTS_1D), figsize=(14, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        p = _s4(m, "4a_dmu.json")
        if not p.exists():
            continue
        rows = json.loads(p.read_text())
        x = [r["rho_probe_mu"] for r in rows]
        y = [r["dmu"] for r in rows]
        ax.scatter(x, y, s=26, color=MODEL_COLORS[m], alpha=0.8)
        for r in rows:
            if r["emotion"] in ("blissful", "hostile", "bored", "desperate"):
                ax.annotate(r["emotion"], (r["rho_probe_mu"], r["dmu"]), fontsize=7,
                            color=INK2, xytext=(4, 3), textcoords="offset points")
        rr = pearson(x, y)
        ax.annotate(f"r = {rr:+.2f}", (0.04, 0.95), xycoords="axes fraction",
                    va="top", fontsize=9, color=INK2)
        ax.axhline(0, color=BASE, linewidth=0.8)
        ax.axvline(0, color=BASE, linewidth=0.8)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("emotion-probe ↔ utility corr (ρ_e)")
        style(ax, grid_axis="both")
    axes[0].set_ylabel("steering effect Δμ (c=0.5)")
    fig.suptitle("4A tracking: steering effect follows each emotion's probe-utility "
                 "correlation (paper analog: r=0.85)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f23_tracking.png")


def f24_gate():
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    dirsets = ["choice", "pool", "utility"]
    for mi, m in enumerate(SUBJECTS_1D):
        p = _s4(m, "gate.json")
        if not p.exists():
            continue
        gates = json.loads(p.read_text())["gates"]
        for di, ds in enumerate(dirsets):
            g = gates[ds]
            c = g["primary_coef"] or "0.5"
            row = g["table"].get(str(c)) or list(g["table"].values())[-1]
            y = di + (mi - (len(SUBJECTS_1D) - 1) / 2) * 0.2
            ax.barh(y, row["dd_plus"], height=0.18, color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m] if di == 0 else None)
            ax.errorbar([0], [y], xerr=[[2 * row["null_sd"]], [2 * row["null_sd"]]],
                        fmt="none", ecolor=MUTED, capsize=2, linewidth=1)
            ax.annotate(f"z={row['z_plus']}", (row["dd_plus"], y), fontsize=7,
                        color=INK2, va="center", xytext=(4, 0),
                        textcoords="offset points")
    ax.set_yticks(range(3), dirsets)
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.set_xlabel("Δ mean choice-logit toward the steered item (+v, primary coef); "
                  "gray bars = ±2·random-null SD")
    ax.set_title("4B/4C behavioral gate: preference/utility directions move choices",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f24_gate.png")


def f25_quadrant():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    markers = {"choice": "o", "pool": "s", "utility": "D", "random": "x"}
    for mi, m in enumerate(SUBJECTS_1D):
        gp, pp = _s4(m, "gate.json"), _s4(m, "probes_4bc.jsonl")
        if not (gp.exists() and pp.exists()):
            continue
        gates = json.loads(gp.read_text())["gates"]
        rows = [json.loads(l) for l in pp.read_text().splitlines()]
        ctrl = [json.loads(l) for l in
                (P1 / "results" / "stage3" / m / "probes.jsonl").read_text().splitlines()]
        ctrl = [r for r in ctrl if r.get("frame") == "bare" and r["outcome"] in ("good", "bad")]

        def late(r):
            v = r["per_turn"]["valence"]
            return float(np.mean(v[1:])) if len(v) > 1 else v[0]

        def swing(rs):
            g = [np.mean(r["fb_read"]["valence"]) for r in rs
                 if r["outcome"] == "good" and r.get("fb_read", {}).get("valence")]
            b = [np.mean(r["fb_read"]["valence"]) for r in rs
                 if r["outcome"] == "bad" and r.get("fb_read", {}).get("valence")]
            return float(np.mean(g) - np.mean(b)) if g and b else None
        sw0 = swing(ctrl)
        st0 = float(np.mean([late(r) for r in ctrl]))
        sd_state = float(np.std([late(r) for r in ctrl], ddof=1))
        for ds in sorted({r["dirset"] for r in rows}):
            rs_ = [r for r in rows if r["dirset"] == ds]
            x = gates.get(ds, {}).get("table", {})
            c = gates.get(ds, {}).get("primary_coef")
            xs = x.get(str(c), {}).get("z_plus", 0) if ds != "random" else 0
            dy = (float(np.mean([late(r) for r in rs_])) - st0) / (sd_state + 1e-9)
            ax.scatter([xs], [dy], marker=markers.get(ds, "o"), s=64,
                       color=MODEL_COLORS[m], label=None)
            ax.annotate(f"{m[:5]}·{ds[:4]}", (xs, dy), fontsize=6.5, color=INK2,
                        xytext=(5, 3), textcoords="offset points")
    ax.axhline(0, color=BASE, linewidth=0.8)
    ax.axvline(3, color=MUTED, linewidth=0.8, linestyle="--")
    ax.annotate("behavioral gate (z=3)", (3, ax.get_ylim()[0]), fontsize=7,
                color=MUTED, rotation=90, va="bottom", ha="right")
    ax.set_xlabel("choice movement (gate z, elicitation)")
    ax.set_ylabel("state-valence shift in rollouts (z vs Stage-3 control SD)")
    ax.set_title("The closed-loop quadrant: do directions that move choices also "
                 "move affect?", fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f25_quadrant.png")


def f26_transfer():
    fig, ax = plt.subplots(figsize=(6.4, 3))
    dark, light = DARK, LIGHT
    labeled = False
    for mi, m in enumerate(SUBJECTS_1D):
        p = _s4(m, "4d.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if "skipped" in d:
            continue
        ax.plot([d["bare_dd_plus"], d["agentic_dd_plus"]], [mi, mi],
                color=GRID, linewidth=1.2, zorder=1)
        ax.scatter([d["bare_dd_plus"]], [mi], s=52, color=dark, zorder=2,
                   label=None if labeled else "bare frame")
        ax.scatter([d["agentic_dd_plus"]], [mi], s=52, color=light, zorder=2,
                   label=None if labeled else "agentic frame (bare-extracted vector)")
        labeled = True
        ax.annotate(f"transfer {d['transfer_ratio']:.2f}",
                    (max(d["bare_dd_plus"], d["agentic_dd_plus"]), mi), fontsize=7.5,
                    color=INK2, va="center", xytext=(6, 0), textcoords="offset points")
    ax.set_yticks(range(len(SUBJECTS_1D)), [MODEL_LABELS[m] for m in SUBJECTS_1D])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlabel("Δ choice-logit under +preference steering")
    ax.set_title("4D cross-frame transfer of the steering effect", fontsize=10,
                 color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f26_transfer.png")


def f27_geo():
    d4a = P1 / "results" / "stage4" / "qwen25-7b" / "4a"
    if not d4a.exists():
        return
    man_layer = json.loads(_s4("qwen25-7b", "dirs_report.json").read_text())
    L = sorted(int(k) for k in man_layer["layers"])[1]
    fig, ax = plt.subplots(figsize=(5.6, 3))
    pairs = []
    for e in ("blissful", "hostile"):
        lin = d4a / f"{e}_0.5_{L}.json"
        geo = d4a / f"geo_{e}_0.5_{L}.json"
        if lin.exists() and geo.exists():
            pairs.append((e, json.loads(lin.read_text())["dmu_mean"],
                          json.loads(geo.read_text())["dmu_mean"]))
    for i, (e, l, g) in enumerate(pairs):
        ax.bar(i - 0.17, l, width=0.3, color=DARK, label="linear vector" if i == 0 else None)
        ax.bar(i + 0.17, g, width=0.3, color=LIGHT, label="geodesic (spline tangent)" if i == 0 else None)
    ax.axhline(0, color=BASE, linewidth=0.8)
    ax.set_xticks(range(len(pairs)), [p[0] for p in pairs])
    ax.tick_params(axis="x", colors=INK2)
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylabel("Δμ of steered items (c=0.5)")
    ax.set_title("4A-geo (qwen25-7b): geodesic vs linear steering at matched norm",
                 fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="y")
    save(fig, FIG / "f27_geo.png")


# ---- Stage 1-XL distribution diagnostics --------------------------------------------------------

_XL_CACHE = {}


def _xl(model):
    if model not in _XL_CACHE:
        _XL_CACHE[model] = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    return _XL_CACHE[model]


def f28_mu_sigma():
    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.42, wspace=0.3)
    tops = [fig.add_subplot(gs[0, k]) for k in range(4)]
    over = fig.add_subplot(gs[1, :])
    logy = max(max(r["sigma2"] for r in _xl(m)) / min(r["sigma2"] for r in _xl(m))
               for m in MODELS) > 30
    for ax, m in zip(tops, MODELS):
        ut = _xl(m)
        mus, s2 = [r["mu"] for r in ut], [r["sigma2"] for r in ut]
        ax.scatter(mus, s2, s=7, color=MODEL_COLORS[m], alpha=0.35, linewidths=0)
        for r in sorted(ut, key=lambda r: -r["sigma2"])[:1]:
            ax.annotate(r["text"][:26] + "…", (r["mu"], r["sigma2"]), fontsize=6,
                        color=INK2, xytext=(0, -8), textcoords="offset points", ha="center")
        if logy:
            ax.set_yscale("log")
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("utility μ")
        style(ax, grid_axis="both")
    tops[0].set_ylabel("preference variance σ²")
    for m in MODELS:
        ut = _xl(m)
        over.scatter([r["mu"] for r in ut], [r["sigma2"] for r in ut], s=6,
                     color=MODEL_COLORS[m], alpha=0.25, linewidths=0,
                     label=MODEL_LABELS[m])
    if logy:
        over.set_yscale("log")
    leg = over.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                      fontsize=8, markerscale=2.2)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    over.set_xlabel("utility μ (Stage 1-XL anchored fit, per-model scale)")
    over.set_ylabel("preference variance σ²\n(high = weakly held)")
    over.set_title("all models overlaid", fontsize=10, color=INK, loc="left")
    style(over, grid_axis="both")
    fig.suptitle("Utility μ vs preference variance σ², 3,985 items per model "
                 "(σ² = strength-of-preference: confident extremes, noisy middle)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    save(fig, FIG / "f28_mu_sigma.png")


def f29_mu_density():
    from scipy import stats as sps
    fig = plt.figure(figsize=(11.5, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.45, wspace=0.3)
    tops = [fig.add_subplot(gs[0, k]) for k in range(4)]
    over = fig.add_subplot(gs[1, :])
    for ax, m in zip(tops, MODELS):
        mus = np.array([r["mu"] for r in _xl(m)])
        ax.hist(mus, bins=50, density=True, color=MODEL_COLORS[m], alpha=0.45)
        grid = np.linspace(mus.min(), mus.max(), 300)
        ax.plot(grid, sps.gaussian_kde(mus)(grid), color=MODEL_COLORS[m], linewidth=1.8)
        ax.plot(grid, sps.norm.pdf(grid, mus.mean(), mus.std()), color=INK2,
                linewidth=1.2, linestyle="--")
        sk, ku = sps.skew(mus), sps.kurtosis(mus)
        sw_p = sps.shapiro(mus).pvalue
        ax.annotate(f"skew {sk:+.2f}\nex-kurt {ku:+.2f}\nShapiro p {sw_p:.1e}",
                    (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                    fontsize=7, color=INK2)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("utility μ")
        style(ax, grid_axis="y")
    tops[0].set_ylabel("density")
    ends = []
    for m in MODELS:
        mus = np.array([r["mu"] for r in _xl(m)])
        grid = np.linspace(mus.min(), mus.max(), 300)
        kde = sps.gaussian_kde(mus)(grid)
        over.plot(grid, kde, color=MODEL_COLORS[m], linewidth=2, label=MODEL_LABELS[m])
        k = int(np.argmax(kde))
        ends.append((grid[k], kde[k], m))
    for x, y, m in ends:  # direct labels at each curve's peak
        over.annotate(MODEL_LABELS[m], (x, y), fontsize=7.5, color=MODEL_COLORS[m],
                      xytext=(0, 5), textcoords="offset points", ha="center")
    over.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    over.set_xlabel("utility μ (Stage 1-XL anchored fit, per-model scale)")
    over.set_ylabel("kernel density")
    over.set_title("all models overlaid (KDE only; dashed gray above = fitted normal)",
                   fontsize=10, color=INK, loc="left")
    style(over, grid_axis="y")
    fig.suptitle("How utilities distribute: histogram + KDE vs best-fit Gaussian (dashed), "
                 "3,985 items per model", fontsize=11, color=INK, x=0.02, ha="left")
    save(fig, FIG / "f29_mu_density.png")


def f30_mu_pairs():
    from scipy import stats as sps
    mus = {m: np.array([r["mu"] for r in _xl(m)]) for m in MODELS}
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 10))
    for a, ma in enumerate(MODELS):
        for b, mb in enumerate(MODELS):
            ax = axes[a, b]
            if a == b:
                grid = np.linspace(mus[ma].min(), mus[ma].max(), 200)
                ax.plot(grid, sps.gaussian_kde(mus[ma])(grid),
                        color=MODEL_COLORS[ma], linewidth=1.8)
                ax.set_title(MODEL_LABELS[ma], fontsize=9.5, color=INK, loc="left")
                style(ax, grid_axis=None)
            elif a > b:
                ax.scatter(mus[mb], mus[ma], s=4, color=ACCENT, alpha=0.22, linewidths=0)
                r = pearson(list(mus[mb]), list(mus[ma]))
                ax.annotate(f"r = {r:+.2f}", (0.05, 0.95), xycoords="axes fraction",
                            va="top", fontsize=8, color=INK2)
                style(ax, grid_axis="both")
            else:
                rho = spearman(list(mus[mb]), list(mus[ma]))
                ax.annotate(f"ρ = {rho:+.2f}", (0.5, 0.5), xycoords="axes fraction",
                            ha="center", va="center", fontsize=13, color=INK2)
                ax.axis("off")
            if a == 3 and a != b:
                ax.set_xlabel(f"{MODEL_LABELS[mb]} μ", fontsize=8)
            if b == 0 and a != b:
                ax.set_ylabel(f"{MODEL_LABELS[ma]} μ", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle("Cross-model utility structure: pairwise μ over the 3,985 shared items "
                 "(lower: scatter + Pearson r; upper: Spearman ρ; diagonal: KDE)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, FIG / "f30_mu_pairs.png")


def f31_mu_3d():
    # exploratory one-off: 3D is for seeing the shared-utility manifold, not for measurement
    mus = {m: np.array([r["mu"] for r in _xl(m)]) for m in MODELS}
    fig = plt.figure(figsize=(8.6, 7))
    ax = fig.add_subplot(projection="3d")
    x, y, z = mus["llama31-8b"], mus["qwen25-7b"], mus["qwen3-4b"]
    c = mus["qwen25-32b"]
    vmin, vmax = np.percentile(c, 2), np.percentile(c, 98)
    sc = ax.scatter(x, y, z, c=c, cmap=SEQ_CMAP, vmin=vmin, vmax=vmax, s=7,
                    alpha=0.8, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label("Qwen2.5-32B μ", color=INK2)
    cb.outline.set_visible(False)
    ax.set_xlabel("Llama-3.1-8B μ", color=INK2, fontsize=9)
    ax.set_ylabel("Qwen2.5-7B μ", color=INK2, fontsize=9)
    ax.set_zlabel("Qwen3-4B μ", color=INK2, fontsize=9)
    ax.view_init(elev=20, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(matplotlib.colors.to_rgba(SURFACE))
        axis._axinfo["grid"]["color"] = GRID
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title("The shared utility manifold: per-item μ of three models as coordinates,\n"
                 "the fourth as color — agreement = points hugging the diagonal, color "
                 "grading along it", fontsize=10, color=INK, loc="left")
    save(fig, FIG / "f31_mu_3d.png")


# ---- Qwen2.5 size-sweep diagnostics -------------------------------------------------------------

from chartstyle import QWEN25_SIZES  # noqa: E402

PARAMS_B = {"qwen25-05b": 0.5, "qwen25-15b": 1.5, "qwen25-3b": 3.0,
            "qwen25-7b": 7.0, "qwen25-32b": 32.0}
GATE_FAILED = {"qwen25-05b"}  # 1B template gate + XL gate (a) both failed; curves are artifact-laden


def _xl_mu_trimmed(m, thresh=6.0):
    """XL mu with runaway fit blowups removed (|z| > thresh; these carry sigma2 in the
    hundreds-to-thousands — degenerate anchored-fit escapes, not preferences)."""
    v = np.array([r["mu"] for r in _xl(m)])
    zz = (v - v.mean()) / v.std()
    return v[np.abs(zz) <= thresh], int((np.abs(zz) > thresh).sum())


def f32_size_density():
    from scipy import stats as sps
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1], hspace=0.45, wspace=0.32)
    tops = [fig.add_subplot(gs[0, k]) for k in range(5)]
    over = fig.add_subplot(gs[1, :])
    for ax, m in zip(tops, QWEN25_SIZES):
        mus, n_cut = _xl_mu_trimmed(m)
        ax.hist(mus, bins=50, density=True, color=MODEL_COLORS[m], alpha=0.45)
        grid = np.linspace(mus.min(), mus.max(), 300)
        ax.plot(grid, sps.gaussian_kde(mus)(grid), color=MODEL_COLORS[m], linewidth=1.8)
        sk, ku = sps.skew(mus), sps.kurtosis(mus)
        note = f"skew {sk:+.2f}\nex-kurt {ku:+.2f}"
        if n_cut:
            note += f"\n{n_cut} runaway excl."
        if m in GATE_FAILED:
            note += "\nGATE FAIL"
        ax.annotate(note, (0.97, 0.95), xycoords="axes fraction", ha="right",
                    va="top", fontsize=7, color=NEG if m in GATE_FAILED else INK2)
        ax.set_title(MODEL_LABELS[m], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("utility μ")
        style(ax, grid_axis="y")
    tops[0].set_ylabel("density")
    peaks = []
    for m in QWEN25_SIZES:
        mus, _ = _xl_mu_trimmed(m)
        mus = (mus - mus.mean()) / mus.std()  # shape comparison: z-scored
        grid = np.linspace(mus.min(), mus.max(), 300)
        kde = sps.gaussian_kde(mus)(grid)
        over.plot(grid, kde, color=MODEL_COLORS[m], linewidth=2,
                  linestyle=":" if m in GATE_FAILED else "-",
                  label=MODEL_LABELS[m] + (" (gate FAIL)" if m in GATE_FAILED else ""))
        k = int(np.argmax(kde))
        peaks.append((kde[k], grid[k], m))
    peaks.sort()  # stagger stacked labels upward in height order
    for rank, (py, px, m) in enumerate(peaks):
        over.annotate(MODEL_LABELS[m], (px, py), fontsize=7.5,
                      color=MODEL_COLORS[m], xytext=(0, 5 + 10 * rank),
                      textcoords="offset points", ha="center")
    over.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    over.set_xlabel("utility μ, z-scored within model after runaway trim (shape comparison)")
    over.set_ylabel("kernel density")
    over.set_title("all sizes overlaid (z-scored; dotted = gate-failed model)",
                   fontsize=10, color=INK, loc="left")
    style(over, grid_axis="y")
    fig.suptitle("Qwen2.5 size sweep: how the utility distribution changes with scale "
                 "(Stage 1-XL, 3,985 items per model)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    save(fig, FIG / "f32_size_density.png")


def f33_size_structure():
    from scipy import stats as sps
    mus = {m: np.array([r["mu"] for r in _xl(m)]) for m in QWEN25_SIZES}
    trimmed = {m: _xl_mu_trimmed(m)[0] for m in QWEN25_SIZES}
    z = {m: (v - v.mean()) / v.std() for m, v in trimmed.items()}
    x = [PARAMS_B[m] for m in QWEN25_SIZES]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    kurt = [sps.kurtosis(z[m]) for m in QWEN25_SIZES]
    wass = [sps.wasserstein_distance(z[m], z["qwen25-32b"]) for m in QWEN25_SIZES]
    ax1.plot(x, kurt, "o-", color=INK2, linewidth=1.6, markersize=6,
             label="excess kurtosis")
    ax1.plot(x, [w * 10 for w in wass], "s--", color=ACCENT, linewidth=1.6,
             markersize=6, label="Wasserstein distance to 32B shape (×10)")
    for xi, m in zip(x, QWEN25_SIZES):
        ax1.scatter([xi], [sps.kurtosis(z[m])], s=70, color=MODEL_COLORS[m], zorder=3)
    ax1.set_xscale("log")
    ax1.set_xticks(x, [f"{v:g}B" for v in x])
    ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax1.axhline(0, color=BASE, linewidth=0.8)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    ax1.set_xlabel("parameters (log scale)")
    ax1.set_ylabel("shape metric (z-scored μ, runaway items trimmed)")
    ax1.set_title("Distribution shape vs size", fontsize=10, color=INK, loc="left")
    style(ax1, grid_axis="both")

    rho32 = [sps.spearmanr(mus[m], mus["qwen25-32b"]).statistic for m in QWEN25_SIZES]
    ax2.plot(x[:-1], rho32[:-1], "o-", color=INK2, linewidth=1.6, markersize=6)
    for xi, m, r in zip(x, QWEN25_SIZES, rho32):
        if m == "qwen25-32b":
            continue
        ax2.scatter([xi], [r], s=70, color=MODEL_COLORS[m], zorder=3)
        ax2.annotate(f"{r:+.2f}" + (" (gate FAIL)" if m in GATE_FAILED else ""),
                     (xi, r), fontsize=7.5, color=NEG if m in GATE_FAILED else INK2,
                     xytext=(2, 8), textcoords="offset points", ha="left")
    ax2.set_xscale("log")
    ax2.set_xticks(x[:-1], [f"{v:g}B" for v in x[:-1]])
    ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax2.set_xlabel("parameters (log scale)")
    ax2.set_ylabel("Spearman ρ of item μ vs Qwen2.5-32B")
    ax2.set_title("Agreement with the largest sibling vs size",
                  fontsize=10, color=INK, loc="left")
    style(ax2, grid_axis="both")
    fig.suptitle("Qwen2.5 scale trends: shape converges and agreement rises with size "
                 "(3,985 shared items)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f33_size_structure.png")


def f34_surf_confirm():
    """XL battery vs SURF probe-pressured discoveries (qwen25-7b): where the
    searched items land on the utility scale, and how much their first anchored
    measurement moves on confirm re-measurement (winner's-curse diagnostic)."""
    from scipy import stats as sps
    m = "qwen25-7b"
    rows = []
    for exp in ("e1", "e2p", "e2r"):
        p = P1 / "results" / "surf" / exp / m / "confirm" / "confirmed.json"
        for r in json.loads(p.read_text()):
            rows.append({"exp": exp, "dir": r["direction"],
                         "first": r["inloop_mu_full"], "mu": r["mu"]})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    xl, _ = _xl_mu_trimmed(m)
    grid = np.linspace(min(xl.min(), min(r["mu"] for r in rows)) - 0.3,
                       max(xl.max(), max(r["mu"] for r in rows)) + 0.3, 400)
    ax1.fill_between(grid, sps.gaussian_kde(xl)(grid), color=MODEL_COLORS[m],
                     alpha=0.35, linewidth=0)
    ax1.plot(grid, sps.gaussian_kde(xl)(grid), color=MODEL_COLORS[m], linewidth=1.8,
             label=f"XL battery (3,985 items)")
    rng = np.random.RandomState(0)
    for d, col, lab in (("max", POS, "probe-pressured, max direction"),
                        ("min", NEG, "probe-pressured, min direction")):
        mus = [r["mu"] for r in rows if r["dir"] == d]
        if not mus:
            continue
        ax1.scatter(mus, rng.uniform(0.015, 0.05, len(mus)), s=14, color=col,
                    alpha=0.7, linewidths=0, label=f"{lab} (n={len(mus)})")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_xlabel(f"utility μ ({MODEL_LABELS[m]}, anchored scale)")
    ax1.set_ylabel("kernel density (XL) / jitter (discoveries)")
    ax1.set_title("Probe-pressured discoveries sit far outside the XL distribution",
                  fontsize=10, color=INK, loc="left")
    style(ax1, grid_axis="y")

    for d, col in (("max", POS), ("min", NEG)):
        xs = [r["first"] for r in rows if r["dir"] == d]
        ys = [r["mu"] - r["first"] for r in rows if r["dir"] == d]
        if not xs:
            continue
        ax2.scatter(xs, ys, s=20, color=col, alpha=0.7, linewidths=0)
        ax2.annotate(f"{d}: mean Δμ {np.mean(ys):+.2f}",
                     (0.03, 0.06 if d == "max" else 0.14), xycoords="axes fraction",
                     fontsize=8, color=col)
    ax2.axhline(0, color=BASE, linewidth=0.8)
    ax2.set_xlabel("first anchored battery μ (in-loop full measurement)")
    ax2.set_ylabel("Δμ: confirm re-measurement − first battery")
    ax2.set_title("Re-measurement shift per discovered item",
                  fontsize=10, color=INK, loc="left")
    style(ax2, grid_axis="both")
    fig.suptitle(f"SURF probe-pressured items vs the XL battery ({MODEL_LABELS[m]}; "
                 "confirm sets of e1/e2p/e2r, n=160)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f34_surf_confirm.png")


def f35_xl_plus_surf():
    """One comparison: the XL mu density vs the same density after pooling in the
    confirmed SURF probe-pressured discoveries (qwen25-7b). Linear view + log-density
    view (the added mass lives almost entirely in the tails)."""
    from scipy import stats as sps
    m = "qwen25-7b"
    xl, _ = _xl_mu_trimmed(m)
    new = []
    for exp in ("e1", "e2p", "e2r"):
        p = P1 / "results" / "surf" / exp / m / "confirm" / "confirmed.json"
        new += [r["mu"] for r in json.loads(p.read_text())]
    combined = np.concatenate([xl, np.array(new)])
    grid = np.linspace(combined.min() - 0.4, combined.max() + 0.4, 500)
    k_xl, k_comb = sps.gaussian_kde(xl)(grid), sps.gaussian_kde(combined)(grid)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, logy in ((ax1, False), (ax2, True)):
        ax.plot(grid, k_xl, color=MODEL_COLORS[m], linewidth=2,
                label="XL battery alone (n=3,985)")
        ax.plot(grid, k_comb, color=ACCENT, linewidth=1.8, linestyle="--",
                label=f"XL + probe-pressured discoveries (n={len(combined):,})")
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1)
            ax.set_title("log density: the added mass is a pure tail effect",
                         fontsize=10, color=INK, loc="left")
        else:
            ax.set_title("linear density: the bulk is unchanged",
                         fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"utility μ ({MODEL_LABELS[m]}, anchored scale)")
        style(ax, grid_axis="both")
    ax1.set_ylabel("kernel density")
    ax1.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("What the probe-pressured discoveries add to the utility distribution "
                 "(confirmed items pooled into the XL battery)", fontsize=11,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, FIG / "f35_xl_plus_surf.png")


def probe_label(v, short=False):
    """Human-readable probe names. v0 = the original preference probe (XL only);
    v1 = retrained on the pre-loop SURF discoveries (stated-preference, probe and
    revealed-preference searches); v2..v10 = probe used in SURF hardening cycle v,
    trained on cycles <= v-1; v11+ = gap-loop probes (trained on gap cycles <= v-11)."""
    if v == 0:
        return "original probe" if short else "original preference probe\n(XL items only)"
    if v == 1:
        return "pre-loop SURF probe" if short else "pre-loop SURF probe\n(+ pre-loop searches)"
    if v <= 10:
        return f"cycle {v} probe" if short else f"SURF cycle {v} probe\n(trained \u2264 cycle {v - 1})"
    k = v - 10
    return f"gap cycle {k} probe" if short else f"gap cycle {k} probe\n(trained \u2264 gap {k - 1})"


def _probeloop_cycle(tag):
    """'c10 max' / 'c4 max+min' -> 10 / 4; 'gap2 over' -> 12 (gap cycle k is the
    (10+k)-th training increment). XL has no cycle."""
    if tag.startswith("gap"):
        return PROBELOOP_MAX_V + int(tag[3:].split()[0])
    return int(tag[1:].split()[0])


PROBELOOP_MAX_V = 10   # last probe-hardening probe; v11+ are gap-loop probes (scripts/surf_gaploop.py)


def _probeloop_sets(m="qwen25-7b", max_v=PROBELOOP_MAX_V, include_gap=False, layered=False):
    """-> (n_probes, [(tag, acts, mu)]) : XL then every hardening cycle's max / min
    discovery set (and, with include_gap, every gap cycle's over / under set);
    layered overlays the layered-anchor re-measurement on hardening-cycle mu.
    max_v=None: every probe_v*.pt present."""
    sys.path.insert(0, str(P1 / "scripts"))
    from surf_probeloop import out_dir, _ds_acts
    import torch
    d = out_dir(m)
    xl_rows = load_json(P1 / "results" / "stage1x" / m / "utilities_xl.json")
    sets = [("XL", torch.load(P1 / "results" / "stage1x" / m / "acts_xl.pt",
                              weights_only=False).float(),
             np.array([r["mu"] for r in xl_rows]))]
    n_all = max(int(m_.group(1)) for f in d.glob("probe_v*.pt")
                for m_ in [re.match(r"probe_v(\d+)\.pt", f.name)] if m_)
    n_probes = n_all if max_v is None else min(max_v, n_all)
    over = {}
    if layered and (d / "remeasure_layered.jsonl").exists():
        for line in (d / "remeasure_layered.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                over[r["text"].lower()] = r["mu_layered"]
    for k in range(1, min(n_probes, PROBELOOP_MAX_V) + 1):
        for sfx, tag in (("", f"c{k} max"), ("_min", f"c{k} min")):
            f = d / f"discoveries_plc{k}{sfx}.json"
            if f.exists():
                rows = load_json(f)
                acts = _ds_acts(m, rows, d / f"acts_plc{k}{sfx}.pt")
                sets.append((tag, acts, np.array([over.get(r["text"].lower(), r["mu"]) for r in rows])))
    if include_gap:
        for k in range(1, n_all - PROBELOOP_MAX_V + 2):
            for arm in ("over", "under"):
                f = d / f"discoveries_gpl{k}_{arm}.json"
                if f.exists():
                    rows = load_json(f)
                    acts = _ds_acts(m, rows, d / f"acts_gpl{k}_{arm}.pt")
                    sets.append((f"gap{k} {arm}", acts, np.array([r["mu"] for r in rows])))
    return n_probes, sets


def _probeloop_matrix(sets=None, m="qwen25-7b", max_v=PROBELOOP_MAX_V, **kw):
    """pearson r of every probe v0..v_n on every set -> (sets, n_v, M[n_v, n_sets])."""
    n_probes, base = _probeloop_sets(m, max_v=max_v, **kw)
    from surf_probeloop import _load_probe, apply_probe
    sets = base if sets is None else sets
    n_v = n_probes + 1
    M = np.zeros((n_v, len(sets)))
    for v in range(n_v):
        pr = _load_probe(m, v)
        for j, (_, acts, mu) in enumerate(sets):
            M[v, j] = pearson(list(apply_probe(pr, acts)), list(mu))
    return sets, n_v, M


def _probe_matrix_axes(ax, sets, n_v, M, cyc, cmap, fontsize):
    """The f40 cell grid: colored cells, bold prequential diagonal, stepped
    train/held-out boundary. Returns the tag list."""
    tags = [t for t, _, _ in sets]
    n_trained = [1 + sum(cyc(t) <= v - 1 for t in tags[1:]) for v in range(n_v)]
    for i in range(n_v):
        for j, t in enumerate(tags):
            val = M[i, j]
            ax.add_patch(plt.Rectangle((j + .04, i + .04), .92, .92,
                                       facecolor=cmap(val), lw=0, zorder=1))
            diag = t != "XL" and cyc(t) == i
            ax.text(j + .5, i + .5, f"{val:+.2f}".replace("+0.", "+.").replace("-0.", "\u2212."),
                    ha="center", va="center", fontsize=fontsize, zorder=3,
                    color="white" if val > 0.72 else INK,
                    fontweight="bold" if diag else "normal")
    xs, ys = [n_trained[0]], [0.0]
    for i in range(n_v):
        ys.append(i + 1.0); xs.append(n_trained[i])
        if i + 1 < n_v and n_trained[i + 1] != n_trained[i]:
            xs.append(n_trained[i + 1]); ys.append(i + 1.0)
    ax.plot(xs, ys, drawstyle="steps-post", color=NEG, linewidth=2.6, zorder=4,
            solid_capstyle="round")
    ax.set_xlim(0, len(tags)); ax.set_ylim(n_v, 0)
    ax.set_xticks([j + .5 for j in range(len(tags))])
    ax.set_xticklabels([f"{t}\nn={len(mu):,}" for t, _, mu in sets], fontsize=8.6, color=INK)
    ax.set_yticks([i + .5 for i in range(n_v)])
    ax.set_yticklabels([f"v{v}  {probe_label(v)}" if False else probe_label(v) + f"  [v{v}]"
                        for v in range(n_v)], fontsize=8.2, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return tags


def f45_probe_matrix_with_gap():
    """f40 extended with the gap-loop probes (rows) and gap-cycle discovery sets
    (columns, over / under arms); hardening-cycle columns use the layered-anchor
    re-measurement where available so every column is on one scale."""
    _probe_matrix_figure(max_v=None, include_gap=True, layered=True, path="f45_probe_matrix_with_gap.png",
                         title="Probe generalization matrix incl. gap loop \u2014 qwen25-7b (layered \u03bc)")


def f46_probe_matrix_pooled_with_gap():
    """f41 extended with the gap-loop probes and the gap cycles pooled (over+under)."""
    _pooled_matrix_figure(max_v=None, include_gap=True, layered=True, path="f46_probe_matrix_pooled_with_gap.png",
                          title="Pooled probe generalization matrix incl. gap loop \u2014 qwen25-7b (layered \u03bc)")


def f40_probe_matrix():
    """Probeloop generalization matrix (qwen25-7b): every probe v0..v10 applied to
    XL and to each cycle's fresh discoveries (max & min arms). Cells are pearson r
    of raw probe score vs full-design mu; the stepped line separates each probe's
    training data (left) from data it never saw (right); bold cells are the
    prequential diagonal (each probe on the discoveries searched against it).
    Needs the acts_plc*.pt caches (gitignored); _ds_acts recomputes them on a
    device if absent."""
    _probe_matrix_figure()


def _probe_matrix_figure(max_v=PROBELOOP_MAX_V, include_gap=False, layered=False, path="f40_probe_matrix.png",
                         title="Probe generalization matrix \u2014 qwen25-7b probeloop"):
    sets, n_v, M = _probeloop_matrix(max_v=max_v, include_gap=include_gap, layered=layered)
    tags = [t for t, _, _ in sets]
    cyc = _probeloop_cycle

    cmap = plt.get_cmap("YlGnBu")  # multi-hue sequential: spreads the .4-.9 mid-range
    fig, ax = plt.subplots(figsize=(0.72 * len(sets) + 3.2, 0.68 * n_v + 2.2))
    _probe_matrix_axes(ax, sets, n_v, M, cyc, cmap, 8.6 if len(sets) > 16 else 9.3)
    ax.set_xticklabels([f"{t}\nn={len(mu):,}\n\u03bc {mu.mean():+.2f}\nsd {mu.std():.2f}"
                        for t, _, mu in sets], fontsize=8.2, color=INK)
    fig.suptitle(title, x=0.04, y=0.975, ha="left", fontsize=12, color=INK)
    fig.text(0.04, 0.925, "pearson r of raw probe score vs full-design \u03bc per evaluation set (columns: n, mean \u03bc, sd of \u03bc). "
             "The SURF cycle-$k$ probe trains on XL + every cycle \u2264 k\u22121 (both arms) \u2014 the cells left of the stepped line; "
             "the pre-loop probe adds the pre-loop searches (not shown as columns). Bold: prequential diagonal. "
             + ("Gap-loop probes and sets: f45 / f46." if not include_gap else
                "Gap-loop rows and columns use layered-anchor \u03bc; hardening-cycle columns are re-measured with the ladder, "
                "so every column is on one scale."),
             fontsize=8.4, color=INK2, va="top", wrap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("pearson r", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, labelcolor=INK2)
    cb.outline.set_visible(False)
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.17, right=0.99)
    save(fig, FIG / path)


def _probeloop_pooled(max_v=PROBELOOP_MAX_V, include_gap=False, layered=False):
    """-> (per_arm sets, pooled sets, n_v, M_pooled, M_arm): each cycle's arms
    concatenated into one evaluation set (cycles 1-3: max only)."""
    import torch
    n_cycles, per_arm = _probeloop_sets(max_v=max_v, include_gap=include_gap, layered=layered)
    by_cycle = {}
    for t, acts, mu in per_arm[1:]:
        by_cycle.setdefault(_probeloop_cycle(t), {})[t.split()[1]] = (acts, mu)
    pooled = [per_arm[0]]
    for k in sorted(by_cycle):
        arms = by_cycle[k]
        order = ("max", "min", "over", "under")
        acts = torch.cat([arms[a][0] for a in order if a in arms], dim=1)
        mu = np.concatenate([arms[a][1] for a in order if a in arms])
        name = f"gap{k - PROBELOOP_MAX_V} " if k > PROBELOOP_MAX_V else f"c{k} "
        pooled.append((name + "+".join(a for a in order if a in arms), acts, mu))
    sets, n_v, M = _probeloop_matrix(pooled, max_v=max_v)
    _, _, M_arm = _probeloop_matrix(per_arm, max_v=max_v)
    return per_arm, sets, n_v, M, M_arm


def f41_probe_matrix_pooled():
    """f40 with each cycle's max and min discoveries POOLED into one evaluation
    set. Pearson within a single arm is range-restricted (min-arm mu SD ~0.8-1.0
    vs max ~2.0-2.5), so per-arm cells understate the probe on the low end;
    pooling restores a two-sided spread. Cycles 1-3 have a max arm only (the
    min direction was added at cycle 4)."""
    _pooled_matrix_figure()


def _pooled_matrix_figure(max_v=PROBELOOP_MAX_V, include_gap=False, layered=False, path="f41_probe_matrix_pooled.png",
                          title="Probe generalization matrix, max+min pooled per cycle \u2014 qwen25-7b probeloop"):
    _, sets, n_v, M, _ = _probeloop_pooled(max_v=max_v, include_gap=include_gap, layered=layered)
    cmap = plt.get_cmap("YlGnBu")
    fig, ax = plt.subplots(figsize=(0.95 * len(sets) + 3.4, 0.68 * n_v + 2.2))
    _probe_matrix_axes(ax, sets, n_v, M, _probeloop_cycle, cmap, 9.3)
    ax.set_xticklabels([f"{t}\nn={len(mu):,}\n\u03bc {mu.mean():+.2f}\nsd {mu.std():.2f}" for t, _, mu in sets],
                       fontsize=8.2, color=INK)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("pearson r", fontsize=8.5, color=INK2)
    cb.ax.tick_params(labelsize=8, labelcolor=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(title, x=0.06, y=0.975, ha="left", fontsize=12, color=INK)
    fig.text(0.06, 0.915, "pearson r of raw probe score vs full-design \u03bc on each cycle's max and min "
             "discoveries pooled (sd = pooled \u03bc spread; cycles 1\u20133 have no min arm). "
             "Cells left of the stepped line are in the probe's training data. Bold: prequential diagonal.",
             fontsize=8.6, color=INK2, va="top")
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.17, right=0.99)
    save(fig, FIG / path)


def f43_anchor_ladder():
    """The anchored Tier-2 scale beyond the central anchors. Left: every item of
    the reliability sample (and the re-measured training tail, when present):
    A0-only mu (the protocol so far, extrapolated from saturated readouts) vs
    the layered mu (re-read against the rung anchors nearest the item), colored
    by the rung reached. Right: the ladder itself — each anchor's pinned mu by
    rung — with the mini-battery candidates (prior -> fitted) that defined the
    wider rungs. Tail values compress: nothing measures above ~+4.4."""
    m = "qwen25-7b"
    rel = P1 / "results" / "surf" / "reliability" / m
    blob = load_json(rel / "records.json")
    summ = load_json(rel / "summary.json")
    sys.path.insert(0, str(P1 / "scripts"))
    import surf_scores
    a0 = np.array(summ.get("mu_a0_full") or [])
    lay = np.array([f["mu"] for f in blob["fitted"]])
    rung = np.array([f["rung"] for f in blob["fitted"]])
    if len(a0) != len(lay):  # a0-only refit not cached in the summary: recompute
        lad = surf_scores.load_anchor_ladder(m)
        a0, _ = surf_scores.fit_records(blob["records"], len(lay), lad[1],
                                        keep=lambda r: r["anchor"] < 12)
        a0 = np.array(a0)
    ladder = load_json(P1 / "items_xl" / "anchors_layered.json")["items"]
    vals = load_json(P1 / "results" / "surf" / "s0" / m / "anchor_values_layered.json")
    bat = load_json(P1 / "results" / "surf" / "s0" / m / "ladder_battery.json")["items"]
    rm = P1 / "results" / "surf" / "probeloop" / m / "remeasure_layered.jsonl"
    rm_rows = [json.loads(l) for l in rm.read_text().splitlines() if l.strip()] if rm.exists() else []

    rung_col = {0: MUTED, 1: ACCENT, 2: POS, 3: NEG}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.4), gridspec_kw={"width_ratios": [1.25, 1]})
    if rm_rows:
        ax.scatter([r["mu_orig"] for r in rm_rows], [r["mu_layered"] for r in rm_rows], s=7,
                   color=LIGHT, alpha=.55, lw=0, zorder=1,
                   label=f"re-measured training tail (n={len(rm_rows):,})")
    for r in sorted(set(rung.tolist())):
        mk = rung == r
        ax.scatter(a0[mk], lay[mk], s=22, color=rung_col.get(r, INK), alpha=.9, lw=0, zorder=3,
                   label=f"reliability sample, rung {r} (n={int(mk.sum())})")
    lim = (min(a0.min(), lay.min()) - .5, max(a0.max(), lay.max()) + .5)
    ax.plot(lim, lim, color=GRID, lw=1, zorder=0)
    for it in ladder:
        if it["rung"] > 0:
            ax.axhline(vals[it["id"]][0], color=rung_col[it["rung"]], lw=.7, alpha=.5, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(min(lay.min(), -5.5) - .3, max(lay.max(), 5) + .3)
    ax.set_xlabel("\u03bc, A0-only 72-readout design (the protocol so far)", fontsize=9, color=INK2)
    ax.set_ylabel("\u03bc, layered design (escalated to the nearest rung)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left", labelcolor=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(color=GRID, lw=.5)
    ax.set_title("tail compression: extrapolated vs bracketed \u03bc", fontsize=10, color=INK, loc="left")

    # right: the ladder, plus battery candidates prior -> fitted
    for it in ladder:
        mu = vals[it["id"]][0]
        ax2.scatter([it["rung"]], [mu], s=46, color=rung_col[it["rung"]], zorder=4, lw=0)
        ax2.text(it["rung"] + .12, mu, it["text"][:34], fontsize=6.3, color=INK2, va="center")
    cands = [x for x in bat if "mu_prior" in x]
    for c in cands:
        ax2.annotate("", xy=(3.85, c["mu_fit"]), xytext=(3.85, c["mu_prior"]),
                     arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=.8, shrinkA=0, shrinkB=0))
        ax2.scatter([3.85], [c["mu_fit"]], s=12, color=INK, zorder=3, lw=0)
    ax2.text(3.85, max(c["mu_prior"] for c in cands) + .4, "battery candidates\nprior \u2192 fitted",
             fontsize=7, color=INK2, ha="center", va="bottom")
    ax2.set_xticks([0, 1, 2, 3, 3.85])
    ax2.set_xticklabels(["A0\ncentral 12", "A1\n\u00b13 (1B)", "A2", "A3", "cands"],
                        fontsize=8, color=INK)
    ax2.set_xlim(-.3, 4.4)
    ax2.axhline(0, color=GRID, lw=.6)
    ax2.set_ylabel("pinned \u03bc (anchored scale)", fontsize=9, color=INK2)
    ax2.tick_params(labelsize=8, colors=INK2, length=0)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.grid(axis="y", color=GRID, lw=.5)
    ax2.set_title("the anchor ladder", fontsize=10, color=INK, loc="left")

    se = summ["designs"].get("mid_layered|s2=free", {})
    fig.suptitle("Layered anchors \u2014 qwen25-7b Tier-2 scale beyond the central 12", x=0.04, y=0.985,
                 ha="left", fontsize=12, color=INK)
    fig.text(0.04, 0.925, "Items whose A0 readouts saturate are re-read against wider rungs and refit with "
             "all anchors pinned. Rungs are placed by fitted, not extrapolated, \u03bc; the positive tail "
             "tops out near +4.4 with \u03c3\u00b2 \u2248 3\u20134 (inconsistent preference among high items). "
             f"Mid-design noise floor (in-span resid SD) = {se.get('in_span', {}).get('resid_sd', '?')}.",
             fontsize=8.2, color=INK2, va="top", wrap=True)
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.06, right=0.99, wspace=0.25)
    save(fig, FIG / "f43_anchor_ladder.png")


def f44_gap_matrix():
    """Gap loop (qwen25-7b): left, mean |calibrated probe - mu| (mu units) for
    probes v8.. on the last probeloop cycle's sets and every gap-cycle set (over /
    under arms), stepped train/held-out line as in f40, lower = better. plc9 mu
    is overlaid with the layered re-measurement where available so all columns
    share the layered scale. Right, closure: signed mean gap on each gap cycle's
    discoveries under the probe searched against (prequential) vs the probe
    fitted afterwards, per arm, with the referee's agreement (spearman of the
    Tier-3 choice rate with the probe vs with mu)."""
    sys.path.insert(0, str(P1 / "scripts"))
    from surf_probeloop import _load_probe, apply_probe, apply_calib, out_dir, _ds_acts
    import surf_gaploop as g
    m = "qwen25-7b"
    d = out_dir(m)
    over = {}
    rp = d / "remeasure_layered.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                over[r["text"].lower()] = r["mu_layered"]
    sets = []
    for sfx, tag, cache in (("", "c9 max", "acts_plc9.pt"), ("_min", "c9 min", "acts_plc9_min.pt")):
        rows = load_json(d / f"discoveries_plc9{sfx}.json")
        acts = _ds_acts(m, rows, d / cache)
        mu = np.array([over.get(r["text"].lower(), r["mu"]) for r in rows])
        sets.append((tag, acts, mu, 9))
    ks = sorted({int(re.match(r"discoveries_gpl(\d+)_", f.name).group(1))
                 for f in d.glob("discoveries_gpl*_*.json")})
    for k in ks:
        for arm in g.ARMS:
            f = d / f"discoveries_gpl{k}_{arm}.json"
            if f.exists():
                rows = load_json(f)
                acts = _ds_acts(m, rows, d / f"acts_gpl{k}_{arm}.pt")
                sets.append((f"gap{k} {arm}", acts, np.array([r["mu"] for r in rows]), g.probe_v(k) + 1))
    v_max = max(int(re.match(r"probe_v(\d+)\.pt", f.name).group(1)) for f in d.glob("probe_v*.pt"))
    vs = list(range(8, v_max + 1))
    M = np.zeros((len(vs), len(sets)))
    for i, v in enumerate(vs):
        pr = _load_probe(m, v)
        cal = load_json(d / f"calib_v{v}.json")
        for j, (_, acts, mu, _) in enumerate(sets):
            M[i, j] = np.abs(apply_calib(cal, apply_probe(pr, acts)) - mu).mean()
    # "trained-on" cycle index per set: the first probe version that saw it
    first_v = [fv for _, _, _, fv in sets]   # c9 -> v10 (fits on cycles <= 9); gap k -> v(10+k)
    cmap = plt.get_cmap("YlGnBu_r")
    vmax = max(1.6, float(np.nanmax(M)))
    fig = plt.figure(figsize=(1.0 * len(sets) + 7.5, 0.6 * len(vs) + 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[max(len(sets), 3), 4.2], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    for i, v in enumerate(vs):
        for j, (t, _, mu, fv) in enumerate(sets):
            val = M[i, j]
            ax.add_patch(plt.Rectangle((j + .04, i + .04), .92, .92, facecolor=cmap(val / vmax), lw=0, zorder=1))
            diag = t.startswith("gap") and v == fv - 1
            ax.text(j + .5, i + .5, f"{val:.2f}", ha="center", va="center", fontsize=9, zorder=3,
                    color="white" if val / vmax < 0.35 else INK, fontweight="bold" if diag else "normal")
    xs, ys = [], []
    for i, v in enumerate(vs):
        n_tr = sum(fv <= v for fv in first_v)
        xs += [n_tr, n_tr]; ys += [i, i + 1]
    ax.plot(xs, ys, color=NEG, linewidth=2.6, zorder=4, solid_capstyle="round")
    ax.set_xlim(0, len(sets)); ax.set_ylim(len(vs), 0)
    ax.set_xticks([j + .5 for j in range(len(sets))])
    ax.set_xticklabels([f"{t}\nn={len(mu)}\n\u03bc {mu.mean():+.2f} sd {mu.std():.2f}" for t, _, mu, _ in sets], fontsize=8.2, color=INK)
    ax.set_yticks([i + .5 for i in range(len(vs))])
    ax.set_yticklabels([probe_label(v, short=True) + f" [v{v}]" for v in vs], fontsize=8.6, color=INK)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("probe version", fontsize=9.5, color=INK2)
    ax.set_title("mean |calibrated probe \u2212 \u03bc| (lower = better; bold = prequential)", fontsize=10, color=INK, loc="left")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("|gap| (\u03bc units)", fontsize=8.5, color=INK2); cb.ax.tick_params(labelsize=8, labelcolor=INK2); cb.outline.set_visible(False)

    # right: closure per gap cycle / arm
    ax2 = fig.add_subplot(gs[1])
    rows = []
    for arm in g.ARMS:
        cp = d / f"gap_cycles_{arm}.json"
        if cp.exists():
            for c in load_json(cp):
                k, vk = c["cycle"], c["probe_searched"]
                pre = c["per_probe"].get(f"v{vk}", {}).get("vs_mu_full", {}).get("gap_mean")
                j = [t for t, _, _, _ in sets].index(f"gap{k} {arm}") if f"gap{k} {arm}" in [t for t, _, _, _ in sets] else None
                post = None
                if j is not None and vk + 1 in vs:
                    pr = _load_probe(m, vk + 1); cal = load_json(d / f"calib_v{vk + 1}.json")
                    post = float((apply_calib(cal, apply_probe(pr, sets[j][1])) - sets[j][2]).mean())
                rows.append((k, arm, pre, post, c.get("t3_rate_vs_probe"), c.get("t3_rate_vs_mu")))
    if rows:
        x = np.arange(len(rows)); w = 0.36
        ax2.bar(x - w / 2, [r[2] or 0 for r in rows], width=w, color=MUTED, label="probe searched against (prequential)", zorder=2)
        ax2.bar(x + w / 2, [r[3] or 0 for r in rows], width=w, color=INK, label="next probe (after refit)", zorder=2)
        fmt = lambda v: "n/a" if v is None else f"{v:+.2f}"
        ax2.axhline(0, color=GRID, lw=.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"gap{r[0]} {r[1]}\nreferee \u03c1\nprobe {fmt(r[4])}\n\u03bc {fmt(r[5])}" for r in rows],
                            fontsize=7.6, color=INK)
        ax2.set_ylabel("signed mean gap (calibrated probe \u2212 \u03bc)", fontsize=9, color=INK2)
        ax2.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.16), labelcolor=INK2)
        ax2.tick_params(axis="y", labelsize=8, colors=INK2, length=0); ax2.tick_params(axis="x", length=0)
        ax2.grid(axis="y", color=GRID, lw=.5, zorder=0)
        for sp in ax2.spines.values():
            sp.set_visible(False)
    ax2.set_title("closure: does the refit remove the gap?", fontsize=10, color=INK, loc="left")
    fig.suptitle("Gap loop \u2014 qwen25-7b: calibrated-probe vs \u03bc disagreement, before and after retraining", x=0.04, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.04, 0.93, "Fitness = z-scored signed gap (over: probe > \u03bc; under: probe < \u03bc), layered Tier-2 \u03bc. Left of the stepped line the probe trained on the set. "
             "Right: the searched-against probe's mean gap on its own discoveries vs the probe fitted on them.", fontsize=8.4, color=INK2, va="top", wrap=True)
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.09, right=0.99)
    save(fig, FIG / "f44_gap_matrix.png")


def f42_probe_arm_vs_pooled():
    """Range restriction made explicit: for each cycle with both arms, the
    prequential probe v_k's pearson r on its own cycle's max-only, min-only and
    pooled discoveries, with each set's mu SD under the bars."""
    per_arm, sets, n_v, M, M_arm = _probeloop_pooled()
    arm_col = {t: j for j, (t, _, _) in enumerate(per_arm)}
    pool_col = {t: j for j, (t, _, _) in enumerate(sets)}
    sd_of = {t: mu.std() for t, _, mu in per_arm + sets}
    both = [t for t, _, _ in sets[1:] if "+" in t]
    ks = [_probeloop_cycle(t) for t in both]
    x = np.arange(len(both))
    w = 0.26
    series = (("max only", MUTED, lambda k: (M_arm[k, arm_col[f"c{k} max"]], sd_of[f"c{k} max"])),
              ("min only", ACCENT, lambda k: (M_arm[k, arm_col[f"c{k} min"]], sd_of[f"c{k} min"])),
              ("pooled", INK, lambda k: (M[k, pool_col[f"c{k} max+min"]], sd_of[f"c{k} max+min"])))
    fig, ax = plt.subplots(figsize=(1.6 * len(both) + 2.4, 4.6))
    for i, (name, col, fn) in enumerate(series):
        vals = [fn(k) for k in ks]
        xs = x + (i - 1) * w
        ax.bar(xs, [r for r, _ in vals], width=w, color=col, label=name, zorder=2)
        for xi, (r, sd) in zip(xs, vals):
            ax.text(xi, r + .015, f"{r:.2f}".lstrip("0"), ha="center", va="bottom",
                    fontsize=7.6, color=INK2)
            ax.text(xi, -.03, f"sd\n{sd:.1f}", ha="center", va="top", fontsize=6.8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"cycle {k} discoveries\n{probe_label(k, short=True)} [v{k}]" for k in ks], fontsize=8.6, color=INK)
    ax.tick_params(axis="x", length=0, pad=26)
    ax.set_ylim(0, 1.02); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.tick_params(axis="y", labelsize=8, colors=INK2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_ylabel("pearson r (prequential: v$k$ on cycle $k$)", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=8.6, ncol=3, loc="upper left",
              bbox_to_anchor=(0, 1.02), labelcolor=INK2)
    fig.suptitle("Per-arm vs pooled probe correlation \u2014 range restriction in the probeloop",
                 x=0.06, y=0.985, ha="left", fontsize=12, color=INK)
    fig.text(0.06, 0.915, "Same probe, same cycle: r on the max arm alone, the min arm alone, and both pooled. "
             "sd = spread of measured \u03bc in each set. Min-arm sd \u2248 0.8\u20131.0 caps within-arm r; "
             "pooling restores a two-sided spread.",
             fontsize=8.6, color=INK2, va="top")
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.08, right=0.99)
    save(fig, FIG / "f42_probe_arm_vs_pooled.png")



if __name__ == "__main__":
    main()
