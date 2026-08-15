"""Figures for Stage 1B + 1C results (pure post-processing; no GPU).

Usage: uv run python figures.py [--core-only]
Writes the cross-model story set to results/figures/ and the per-model
appendix to results/figures/appendix/.

Style: light mode, dataviz reference palette. Categorical slots 1-4 map to the
models in fixed order (never cycled); sequential = one blue ramp; diverging =
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
import _day1  # noqa: F401
from lib.util import load_json
from lib.valuation import pearson, spearman

import items as items_mod
import stats

MODELS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
N_LAYERS = {"llama31-8b": 32, "qwen25-7b": 28, "qwen3-4b": 36, "qwen25-32b": 64}
DOMAINS = ["activities", "objects", "topics", "selfstates", "others"]
METHODS = ["ssr", "probe", "titration", "rating", "bws"]

# dataviz reference palette (light mode)
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # models, fixed order
POS, NEG, MID = "#2a78d6", "#e34948", "#f0efec"           # diverging
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)

FIG = P1 / "results" / "figures"
APP = FIG / "appendix"

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


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path.relative_to(P1))


def utilities(model):
    return load_json(P1 / "results" / "stage1b" / model / "utilities.json")


def scores(model):
    return load_json(P1 / "results" / "stage1c" / model / "scores.json")


def zscore(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std()


# ---- core figures -------------------------------------------------------------------------------

def strip_panel(ax, ut, title, annotate_n=0):
    rng = np.random.RandomState(0)
    for d, dom in enumerate(DOMAINS):
        mus = [r["mu"] for r in ut if r["domain"] == dom]
        y = d + rng.uniform(-0.22, 0.22, len(mus))
        ax.scatter(mus, y, s=9, color=SLOTS[0], alpha=0.55, linewidths=0)
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
        strip_panel(ax, utilities(m), m)
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
    ax.set_xlabel("mean z-scored μ across 4 models")
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
            ax.barh(y, betas[axis], height=0.16, color=SLOTS[mi],
                    label=m if ai == 0 else None)
            if ai == 0:  # direct series labels on the first group (4-series rule)
                ax.annotate(m, (betas[axis], y), fontsize=7, color=INK2,
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
    ax.set_yticks(y, MODELS)
    ax.tick_params(axis="y", colors=INK2)
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlim(min(vals) * 1.3, max(vals) * 1.55)
    ax.set_xlabel("A-position bias: (mean p, item shown as A − as B) / 2")
    ax.set_title("Position bias by model — cancelled by the both-orders design",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f5_position_bias.png")


def method_scatter(ax, x, y, title, unit=""):
    ax.scatter(x, y, s=10, color=SLOTS[0], alpha=0.6, linewidths=0)
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
        ax.scatter([oz[c]], [i], s=42, color="#86b6ef", zorder=2,
                   label="Day 1 (committed)" if i == 0 else None)
        ax.scatter([nz[c]], [i], s=42, color="#1c5cab", zorder=2,
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
        ax.plot(x, vals, color=SLOTS[mi], linewidth=2)
        ends.append([vals[-1], mi, m])
    # stagger the direct end-labels so equal end values don't collide
    ends.sort()
    min_gap = 0.035
    for k in range(1, len(ends)):
        if ends[k][0] - ends[k - 1][0] < min_gap:
            ends[k][0] = ends[k - 1][0] + min_gap
    for ylab, mi, m in ends:
        ax.annotate(m, (1.0, ylab), fontsize=8, color=SLOTS[mi],
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
    ut = utilities(model)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    strip_panel(ax, ut, f"{model}: utility landscape (extreme items labeled)",
                annotate_n=4)
    ax.set_xlabel("Thurstonian utility μ")
    save(fig, APP / f"{model}_landscape.png")

    sc = scores(model)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4))
    for ax, meth in zip(axes.flat, METHODS):
        method_scatter(ax, [r[meth] for r in sc], [r["mu_1b"] for r in sc],
                       meth + (" (held-out)" if meth == "probe" else ""))
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("1B utility μ")
    fig.suptitle(f"{model}: 1C method scores vs 1B utilities", fontsize=11,
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, APP / f"{model}_methods.png")

    rows = [{"text": r["text"], "domain": r["domain"], "val": r["mu"]} for r in ut]
    rows.sort(key=lambda r: -r["val"])
    fig, ax = plt.subplots(figsize=(8, 7.4))
    top_bottom_axis(ax, rows[:12] + rows[-12:], f"{model}: top and bottom 12 items")
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
    for f in (f9_gate_scatter, f10_optout, f11_beta, f12_swap, f13_effort):
        f()
    if not args.core_only:
        for m in MODELS:
            appendix(m)


if __name__ == "__main__":
    main()


# ---- Stage 1D figures ---------------------------------------------------------------------------

SUBJECTS_1D = ["llama31-8b", "qwen25-7b", "qwen3-4b"]  # slots 1-3, same hues as elsewhere


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
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
    for ax, m in zip(axes, SUBJECTS_1D):
        env_ids, z = _env_mu(m)
        rates = rs.chosen_rates(_load_choices(m)["choices"], env_ids)
        x = [z[e] for e in env_ids]
        y = [rates[e][2] for e in env_ids]
        ax.scatter(x, y, s=18, color=SLOTS[SUBJECTS_1D.index(m)], alpha=0.75, linewidths=0)
        rho = spearman(x, y)
        ax.annotate(f"ρ = {rho:+.2f}", (0.04, 0.94), xycoords="axes fraction",
                    va="top", fontsize=9, color=INK2)
        ax.axhline(0.25, color=BASE, linewidth=0.8)
        ax.set_title(m, fontsize=10, color=INK, loc="left")
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
            y = ti + (mi - 1) * 0.24
            ax.barh(y, a / b, height=0.2, color=SLOTS[mi],
                    label=m if ti == 0 else None)
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
                    fmt="o", color=SLOTS[mi], markersize=7, capsize=3, linewidth=1.6)
    ax.axvline(0, color=BASE, linewidth=0.8)
    ax.set_yticks(range(len(SUBJECTS_1D)), SUBJECTS_1D)
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.set_xlabel("standardized choice sharpness β·sd(μ)  (log-odds per SD of utility)")
    ax.set_title("Conditional-logit coupling, menu-cluster bootstrap 95% CI",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f11_beta.png")


def f12_swap():
    conds = [("upgrade offered", lambda s: s["delta_z"] > 0.15),
             ("lateral", lambda s: abs(s["delta_z"]) <= 0.15),
             ("downgrade offered", lambda s: s["delta_z"] < -0.15)]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for mi, m in enumerate(SUBJECTS_1D):
        swaps = _load_choices(m)["swaps"]
        for ci, (label, f) in enumerate(conds):
            rs_ = [s for s in swaps if f(s)]
            rate = sum(s["switched"] for s in rs_) / len(rs_) if rs_ else 0.0
            y = ci + (mi - 1) * 0.24
            ax.barh(y, rate, height=0.2, color=SLOTS[mi], label=m if ci == 0 else None)
            ax.annotate(f"{sum(s['switched'] for s in rs_)}/{len(rs_)}", (rate, y),
                        fontsize=7, color=INK2, va="center", xytext=(4, 0),
                        textcoords="offset points")
    ax.set_yticks(range(3), [c[0] for c in conds])
    ax.tick_params(axis="y", colors=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8)
    ax.set_xlabel("P(switch) at the turn-5 swap offer")
    ax.set_title("Swap offers: switching is prior-driven, not Δμ-driven",
                 fontsize=10, color=INK, loc="left")
    style(ax)
    save(fig, FIG / "f12_swap.png")


def f13_effort():
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for mi, m in enumerate(SUBJECTS_1D):
        rows = [json.loads(l) for l in open(_1d(m, "rollouts_effort.jsonl"))]
        xs, ys = [], []
        for r in rows:
            sh = [s["share_high"] for s in r["meta"]["shares"] if s["share_high"] is not None]
            if sh:
                xs.append(r["meta"]["dmu"])
                ys.append(sum(sh) / len(sh))
        ax.scatter(xs, ys, s=26, color=SLOTS[mi], alpha=0.8, linewidths=0, label=m)
        if len(set(xs)) > 1:
            b, a = np.polyfit(xs, ys, 1)
            xr = [min(xs), max(xs)]
            ax.plot(xr, [a + b * x for x in xr], color=SLOTS[mi], linewidth=1.4, alpha=0.7)
    ax.axhline(0.5, color=BASE, linewidth=0.8)
    ax.annotate("equal split", (0.99, 0.465), xycoords=("axes fraction", "data"),
                fontsize=7, color=MUTED, ha="right")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_xlabel("utility gap Δμ (z) between the two tasks")
    ax.set_ylabel("token share on the higher-μ task")
    ax.set_title("Effort allocation tracks utility weakly at best, and model-dependently",
                 fontsize=10, color=INK, loc="left")
    style(ax, grid_axis="both")
    save(fig, FIG / "f13_effort.png")
