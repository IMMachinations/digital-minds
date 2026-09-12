"""SURF probeloop figures (f36, f37), following CHARTS.md via root chartstyle.

f36: probe-hardening behavioral convergence — held-out menu choice rate of each
     cycle's top-20 probe-selected items, per model, with the pre-loop (E2 v0)
     baselines and each model's revealed-arm ceiling for reference.
f37: retrained vs frozen probe — on each cycle's fresh discoveries, the
     correlation with measured mu of the probe just retrained on all earlier
     cycles vs the original (pre-loop) probe held fixed. The widening gap is the
     probe hardening; the frozen probe's decay shows the search escaping its
     training support ("prequential scissors" in the SURF report).

Usage: uv run python scripts/surf_figures.py   (CPU)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def f37_retrained_vs_frozen():
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
    for ax, m in zip(axes, MODELS):
        cs = cycles(m)
        xs = [c["cycle"] for c in cs]
        cur = [c["per_probe"][f"v{c['cycle']}"]["raw"]["pearson"] for c in cs]
        v0 = [c["per_probe"]["v0"]["raw"]["pearson"] for c in cs]
        ax.plot(xs, cur, color=MODEL_COLORS[m], lw=1.8, marker="o", ms=4,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3,
                label="retrained probe (this cycle)")
        ax.plot(xs, v0, color=MUTED, lw=1.4, ls="--", marker="o", ms=3.5,
                markerfacecolor="white", markeredgewidth=1.1, zorder=2,
                label="original probe (frozen)")
        ax.set_title(MODEL_LABELS[m], fontsize=9, color=MODEL_COLORS[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"c{x}" for x in xs])
        if max(xs) > 3:   # cycles 4+ ran on the laptop with a Sonnet generator
            ax.axvline(3.5, color=MUTED, lw=0.8, ls=":", zorder=1)
            ax.annotate("c4+: Sonnet generator", (3.6, 0.99), fontsize=6.5,
                        color=MUTED, ha="left", va="top")
        style(ax, grid_axis="y")
        bounded_axis(ax, "y", 0.0, 1.0)
    axes[0].set_ylabel("Pearson r vs measured $\\mu$\n(cycle's fresh discoveries)")
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle("Probe hardening: retrained probe vs the frozen original, "
                 "scored on each cycle's new discoveries",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGS / "f37_retrained_vs_frozen.png")


def f38():
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(P1 / "scripts"))
    from surf_probeloop import apply_probe, _load_probe, probe_path
    from surf_revealed_probe import _labels, _panel_acts

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.2),
                                   gridspec_kw={"width_ratios": [1.1, 1.6]})
    # (a) behavioral AUC on the 388-item labeled panel (qwen25-7b)
    m = "qwen25-7b"
    items, rate, _ = _labels(m)
    acts = _panel_acts(m, items)
    y = (np.asarray(rate) > 0.5).astype(int)
    scores = {}
    for v in range(4):
        if probe_path(m, v).exists():
            scores[f"v{v}"] = apply_probe(_load_probe(m, v), acts)
    rp = torch.load(P1 / "results" / "surf" / "revealed" / m / "revealed_probe.pt",
                    weights_only=False)
    scores["revealed\nprobe\n(in-sample)"] = apply_probe(rp, acts)
    scores["stated $\\mu$"] = np.array([r["mu"] for r in items])
    names = list(scores)
    aucs = [roc_auc_score(y, scores[k]) for k in names]
    cols = [MODEL_COLORS[m]] * 4 + [MUTED, INK2]
    ax1.bar(range(len(names)), aucs, color=cols, width=0.62)
    for i, a in enumerate(aucs):
        ax1.annotate(f"{a:.2f}", (i, a), ha="center", va="bottom", fontsize=7.5,
                     color=INK2)
    ax1.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=7.5)
    bounded_axis(ax1, "y", 0.45, 1.0)
    ax1.set_ylabel("ROC AUC")
    ax1.set_title(f"(a) predicting behavior ({MODEL_LABELS[m]}):\n"
                  "chosen >50% in menu test, n=388 panel", fontsize=8.5,
                  color=INK2, loc="left")
    style(ax1, grid_axis="y")

    # (b) mu-median-split AUC on fresh discoveries: current probe vs frozen v0
    offs = {"qwen25-7b": -0.12, "llama31-8b": 0.0, "qwen3-4b": 0.12}
    for m in MODELS:
        cur_a, v0_a, xs = [], [], []
        for c in (1, 2, 3):
            ds = json.loads((P1 / "results" / "surf" / "probeloop" / m /
                             f"discoveries_plc{c}.json").read_text())
            acts = torch.load(P1 / "results" / "surf" / "probeloop" / m /
                              f"acts_plc{c}.pt", weights_only=False).float()
            mu = np.array([r["mu"] for r in ds])
            yb = (mu > np.median(mu)).astype(int)
            cur = apply_probe(_load_probe(m, c), acts)
            v0s = apply_probe(_load_probe(m, 0), acts)
            cur_a.append(roc_auc_score(yb, cur))
            v0_a.append(roc_auc_score(yb, v0s))
            xs.append(c + offs[m])
        ax2.plot(xs, cur_a, color=MODEL_COLORS[m], lw=1.8, marker="o", ms=4,
                 markerfacecolor="white", markeredgewidth=1.4, zorder=3)
        ax2.plot(xs, v0_a, color=MODEL_COLORS[m], lw=1.1, ls="--", marker="o",
                 ms=3, markerfacecolor="white", markeredgewidth=1.0, alpha=0.55,
                 zorder=2)
        ax2.annotate(MODEL_LABELS[m], (xs[-1], cur_a[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7.5,
                     color=MODEL_COLORS[m])
    ax2.axhline(0.5, color=MUTED, lw=0.7, ls="--")
    ax2.plot([], [], color=INK2, lw=1.8, label="current probe")
    ax2.plot([], [], color=INK2, lw=1.1, ls="--", alpha=0.55, label="frozen v0")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["cycle 1", "cycle 2", "cycle 3"])
    ax2.set_xlim(0.7, 3.9)
    bounded_axis(ax2, "y", 0.45, 1.0)
    ax2.set_title("(b) separating above/below-median $\\mu$\non fresh discoveries",
                  fontsize=8.5, color=INK2, loc="left")
    style(ax2, grid_axis="y")
    fig.tight_layout()
    save(fig, FIGS / "f38_probe_aucs.png")


def f39():
    """The question-form artifact: stated-channel-only, qwen25-7b-only."""
    import numpy as np
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.1),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    # (a) standardized question-minus-declarative gap per measurement channel
    rows = json.loads((P1 / "results" / "surf" / "e2" / "qwen25-7b" /
                       "referee.json").read_text())["rows"]
    q = np.array([r["question_form"] for r in rows], bool)
    chans = [("stated $\\mu$", "mu_t2"), ("internal probe", "probe_mu"),
             ("revealed choice", "t3_rate")]
    rng = np.random.RandomState(1)
    ds, los, his = [], [], []
    for _, key in chans:
        v = np.array([r[key] for r in rows], float)
        def smd(mask, vals):
            a, b = vals[mask], vals[~mask]
            sd = np.concatenate([a - a.mean(), b - b.mean()]).std(ddof=2) + 1e-9
            return (a.mean() - b.mean()) / sd
        ds.append(smd(q, v))
        reps = []
        for _ in range(2000):
            idx = rng.randint(0, len(v), len(v))
            if 3 <= q[idx].sum() <= len(v) - 3:
                reps.append(smd(q[idx], v[idx]))
        los.append(np.percentile(reps, 2.5))
        his.append(np.percentile(reps, 97.5))
    ypos = np.arange(len(chans))[::-1]
    ax1.barh(ypos, ds, xerr=[np.array(ds) - los, np.array(his) - ds],
             color=[MODEL_COLORS["qwen25-7b"], MUTED, MUTED], height=0.55,
             error_kw={"ecolor": INK2, "lw": 1.0, "capsize": 2.5})
    ax1.axvline(0, color=INK2, lw=0.8)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([c for c, _ in chans], fontsize=8.5)
    ax1.set_xlabel("question $-$ declarative gap (SMD, 95% CI)")
    ax1.set_title("(a) the inflation lives only in the\nstated letter-logit channel",
                  fontsize=8.5, color=INK2, loc="left")
    style(ax1, grid_axis="x")
    # (b) lineage check: E1 max items re-measured natively per model (z-units)
    per = {}
    e1 = json.loads((P1 / "results" / "surf" / "e1" / "qwen25-7b" / "confirm" /
                     "confirmed.json").read_text())
    per["qwen25-7b"] = [(r["text"].rstrip().endswith("?"), r["mu"])
                       for r in e1 if r["direction"] == "max"]
    for m in ("llama31-8b", "qwen3-4b"):
        tr = json.loads((P1 / "results" / "surf" / "transfer" / m /
                         "e1_items.json").read_text())
        per[m] = [(r["question_form"], r["mu_target"])
                  for r in tr if r["direction"] == "max"]
    for i, m in enumerate(per):
        sd = np.std([r["mu"] for r in json.loads(
            (P1 / "results" / "stage1x" / m / "utilities_xl.json").read_text())])
        qv = np.mean([v for isq, v in per[m] if isq]) / sd
        dv = np.mean([v for isq, v in per[m] if not isq]) / sd
        y = len(per) - 1 - i
        ax2.plot([dv, qv], [y, y], color=MODEL_COLORS[m], lw=1.6, zorder=2)
        ax2.scatter([dv], [y], s=42, facecolor="white",
                    edgecolor=MODEL_COLORS[m], lw=1.6, zorder=3)
        ax2.scatter([qv], [y], s=42, color=MODEL_COLORS[m], zorder=3)
        ax2.annotate(MODEL_LABELS[m], (max(dv, qv), y), xytext=(8, 0),
                     textcoords="offset points", va="center", fontsize=8,
                     color=MODEL_COLORS[m])
    ax2.scatter([], [], s=42, color=INK2, label="question-form")
    ax2.scatter([], [], s=42, facecolor="white", edgecolor=INK2, lw=1.6,
                label="declarative")
    ax2.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax2.set_yticks([])
    ax2.set_xlabel("mean $\\mu$ of qwen25-7b's max items, native scale ($\\mu/\\sigma_{XL}$)")
    ax2.set_title("(b) and it is model-specific: the gap\ninverts on every other subject",
                  fontsize=8.5, color=INK2, loc="left")
    style(ax2, grid_axis="x")
    fig.suptitle("Inflation of question-phrased preferences on Qwen2.5-7B",
                 fontsize=10.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, FIGS / "f39_question_artifact.png")


# ---- probe-version reruns (f47+; data from scripts/surf_probe_versions.py) ----------------------

PV_MODEL = "qwen25-7b"
PV_DIR = P1 / "results" / "surf" / "probe_versions" / PV_MODEL


def _pv(name):
    return json.loads((PV_DIR / name).read_text())


def _vx(vname):
    """version name -> x position; the layer-18 refit of v15 sits one slot right."""
    return 16 if vname == "v15_L18" else int(vname[1:])


def _version_axis(ax, names, ylabel=None):
    xs = sorted({_vx(n) for n in names})
    ax.set_xticks(xs)
    ax.set_xticklabels([("v15\n@L18" if x == 16 else f"v{x}") for x in xs], fontsize=7.2)
    ax.axvspan(0.5, 10.5, color=MUTED, alpha=0.07, lw=0, zorder=0)
    ax.axvspan(10.5, 15.5, color=MUTED, alpha=0.14, lw=0, zorder=0)
    ax.axvline(15.5, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_xlim(-0.6, 16.6)
    if ylabel:
        ax.set_ylabel(ylabel)
    style(ax, grid_axis="y")


def _series(ax, names, ys, color, label, **kw):
    pts = sorted(zip([_vx(n) for n in names], ys))
    main = [(x, y) for x, y in pts if x <= 15]
    ax.plot([x for x, _ in main], [y for _, y in main], color=color, label=label,
            marker="o", ms=3.6, markerfacecolor="white", markeredgewidth=1.2, zorder=3, **kw)
    for x, y in pts:
        if x == 16:
            ax.plot([x], [y], color=color, marker="s", ms=4.2, markerfacecolor="white",
                    markeredgewidth=1.2, zorder=3, ls="none")


def _phase_labels(ax, y):
    ax.annotate("hardening cycles (extremity search)", (5.5, y), fontsize=6.8,
                color=MUTED, ha="center", va="top")
    ax.annotate("gap loop", (13, y), fontsize=6.8, color=MUTED, ha="center", va="top")


def f47_probe_versions_global2():
    """Every probe version on the enlarged global set (XL in-span + all 4,263
    SURF discoveries on the layered scale): correlation with measured mu split
    XL / SURF / SURF the probe never trained on / question-form; right panel
    the calibrated absolute error."""
    from chartstyle import INK
    g = _pv("global2_probe_versions.json")
    V = g["versions"]
    names = list(V)
    col = MODEL_COLORS[PV_MODEL]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.3),
                                   gridspec_kw={"width_ratios": [1.5, 1.0]})
    _series(ax1, names, [V[n]["masks"]["surf"]["r"] for n in names], col, "SURF discoveries (all)")
    unseen = [n for n in names if V[n]["masks"].get("surf_unseen")]
    _series(ax1, unseen, [V[n]["masks"]["surf_unseen"]["r"] for n in unseen], col,
            "SURF items never in the probe's training set", ls="--")
    _series(ax1, names, [V[n]["masks"]["xl"]["r"] for n in names], MUTED, "XL (passive) items")
    _series(ax1, names, [V[n]["masks"]["qform"]["r"] for n in names], INK2,
            "question-form items", ls=":")
    _version_axis(ax1, names, "Pearson r, raw probe score vs measured $\\mu$")
    bounded_axis(ax1, "y", 0.3, 1.0)
    _phase_labels(ax1, 0.985)
    ax1.legend(frameon=False, fontsize=6.8, loc="lower left")
    ax1.set_title(f"(a) global2 set, n={g['n']} ({g['n_xl']} XL + {g['n_surf']} SURF)",
                  fontsize=8.5, color=INK2, loc="left")
    _series(ax2, names, [V[n]["masks"]["surf"]["mae_cal"] for n in names], col, "SURF (all)")
    _series(ax2, unseen, [V[n]["masks"]["surf_unseen"]["mae_cal"] for n in unseen], col,
            "SURF unseen", ls="--")
    _series(ax2, names, [V[n]["masks"]["xl"]["mae_cal"] for n in names], MUTED, "XL")
    _version_axis(ax2, names, "mean |calibrated probe − $\\mu$|")
    bounded_axis(ax2, "y", 0.0, 2.0)
    ax2.legend(frameon=False, fontsize=6.8, loc="lower right")
    ax2.set_title("(b) calibrated error (isotonic map per version)", fontsize=8.5,
                  color=INK2, loc="left")
    fig.suptitle("Probe versions on the enlarged global set (later versions are in-sample on "
                 "their own discoveries; dashed = never-trained-on items)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGS / "f47_probe_versions_global2.png")


def f48_linearity_layered():
    """Ridge vs 1-D spline vs line, held-out, on the layered-scale global2 set
    beside the earlier A0-scale global set (numbers read from its summary)."""
    from chartstyle import INK, BASE
    g = _pv("global2_probe_versions.json")
    lin, old = g["linearity"], g.get("old_global") or {}
    sp_best = max(lin["spline"][b]["all"]["r"] for b in lin["spline"])
    ln_best = max(lin["line"][b]["all"]["r"] for b in lin["line"])
    new = [lin["ridge"]["all"]["r"], sp_best, ln_best]
    oldv = [old.get("ridge"), old.get("spline_best"), old.get("line")]
    labels = ["full ridge\n(linear, d=3584)", "1-D spline\n(best of 40/64/100 bins)",
              "1-D line"]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    x = np.arange(3)
    col = MODEL_COLORS[PV_MODEL]
    ax.bar(x - 0.19, [v or 0 for v in oldv], width=0.36, color=BASE,
           label="global set, A0 anchors (n=5,352; tails extrapolated)")
    ax.bar(x + 0.19, new, width=0.36, color=col,
           label=f"global2 set, layered anchors (n={g['n']}; layer {lin['layer']})")
    for xi, (a, b) in enumerate(zip(oldv, new)):
        if a is not None:
            ax.annotate(f"{a:.2f}", (xi - 0.19, a), fontsize=7.5, color=INK2, ha="center",
                        va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{b:.2f}", (xi + 0.19, b), fontsize=7.5, color=INK2, ha="center",
                    va="bottom", xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x, labels, fontsize=8)
    bounded_axis(ax, "y", 0.0, 1.0)
    ax.set_ylabel("held-out Pearson r vs measured $\\mu$")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("Utility stays linearly decodable on the layered scale; the 1-D manifold "
                 "story does not improve", fontsize=9, color=INK, loc="left")
    style(ax, grid_axis="y")
    save(fig, FIGS / "f48_linearity_layered.png")


def f49_direction_cosines():
    """Cosine matrix of the layer-18 utility-direction family: every probe
    version (unit coef/std) plus the stage-4 reference directions and the two
    global refits."""
    from chartstyle import INK, SEQ_CMAP
    c = _pv("direction_cosines.json")["18"]
    names, C = c["names"], np.array(c["cos"])
    vs = sorted([n for n in names if n.startswith("v")], key=_vx)
    refs = [n for n in ("s4_utility", "global", "global2", "s4_choice", "s4_pool", "s4_valence")
            if n in names]
    order = [names.index(n) for n in vs + refs]
    M = C[np.ix_(order, order)]
    lab = [("v15@L18" if n == "v15_L18" else n).replace("s4_", "stage4 ") for n in vs + refs]
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(M, cmap=SEQ_CMAP, vmin=0, vmax=1)
    n = len(lab)
    ax.set_xticks(range(n), lab, rotation=90, fontsize=7)
    ax.set_yticks(range(n), lab, fontsize=7)
    k = len(vs)
    ax.axhline(k - 0.5, color="white", lw=1.5)
    ax.axvline(k - 0.5, color="white", lw=1.5)
    ax.axhline(10.5, color="white", lw=0.8, ls=":")
    ax.axvline(10.5, color="white", lw=0.8, ls=":")
    for i in range(n):
        for j in range(n):
            if i != j and (i >= k or j >= k or abs(i - j) <= 1):
                ax.annotate(f"{M[i, j]:.2f}", (j, i), fontsize=5.4, ha="center", va="center",
                            color="white" if M[i, j] > 0.55 else INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("cosine", fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, labelcolor=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Direction geometry at layer 18: hardening rotates the utility direction "
                 "smoothly, the gap loop turns a corner;\nchoice / pool / valence stay "
                 "near-orthogonal to every version", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    save(fig, FIGS / "f49_direction_cosines.png")


def f50_e2_dissociation_by_version():
    """The E2 three-way referee set (100 survivors with stated mu and a
    held-out revealed choice rate) re-scored under every probe version."""
    from chartstyle import INK
    e = _pv("e2_referee_versions.json")
    V = e["versions"]
    names = list(V)
    col = MODEL_COLORS[PV_MODEL]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _series(ax, names, [V[n]["rho_stated_probe"] for n in names], col,
            "probe ↔ stated $\\mu$ (the probe's target)")
    _series(ax, names, [V[n]["rho_probe_revealed"] for n in names], INK2,
            "probe ↔ revealed choice (held-out envs)", ls="--")
    ax.axhline(e["rho_stated_revealed"], color=MUTED, lw=0.9, ls=":")
    ax.annotate(f"measured stated μ ↔ revealed = {e['rho_stated_revealed']:+.2f} "
                "(no probe involved)", (-0.4, e["rho_stated_revealed"] - 0.03),
                fontsize=6.8, color=MUTED, ha="left", va="top")
    ax.axhline(0, color=MUTED, lw=0.6)
    _version_axis(ax, names, "Spearman ρ on the E2 referee set (n=100)")
    bounded_axis(ax, "y", -0.3, 1.0)
    _phase_labels(ax, 0.98)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    ax.set_title("Goodhart survivors under every probe version: hardening fixes the "
                 "stated-channel fit, never the behavioural one", fontsize=9, color=INK,
                 loc="left")
    save(fig, FIGS / "f50_e2_dissociation_by_version.png")


def f51_handbuilt_by_version():
    """Out-of-family test: every version on the 197 hand-built Stage-1 items
    (never generated by SURF or the XL generator), overall and per domain."""
    from chartstyle import INK, SEQ
    h = _pv("handbuilt_versions.json")
    V = h["versions"]
    names = list(V)
    col = MODEL_COLORS[PV_MODEL]
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    doms = h["domains"]
    shades = [SEQ[i] for i in np.linspace(3, len(SEQ) - 2, len(doms)).astype(int)]
    for dm, sh in zip(doms, shades):
        ys = [V[n]["by_domain"][dm]["r"] for n in names]
        xs = [_vx(n) for n in names]
        pts = sorted(zip(xs, ys))
        ax.plot([x for x, _ in pts if x <= 15], [y for x, y in pts if x <= 15],
                color=sh, lw=0.9, zorder=2)
        ax.annotate(f"{dm} (n={V[names[0]]['by_domain'][dm]['n']})", (15.15, pts[-2][1]),
                    fontsize=6.3, color=sh, va="center")
    _series(ax, names, [V[n]["overall"]["r"] for n in names], col, "all 197 items", lw=2.0)
    b = h.get("baseline_1c")
    if b:
        ax.axhline(b["r"], color=MUTED, lw=0.9, ls="--")
        ax.annotate(f"original Stage-1C probe, cross-fitted on the 197 alone: r={b['r']:.2f}",
                    (0, b["r"]), fontsize=6.8, color=MUTED, va="bottom", xytext=(2, 2),
                    textcoords="offset points")
    _version_axis(ax, names, "Pearson r vs Stage-1B $\\mu$ (197 hand-built items)")
    bounded_axis(ax, "y", 0.7, 1.0)
    _phase_labels(ax, 0.995)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("Out-of-family validity: SURF-hardened probes lose a little on the "
                 "hand-built items", fontsize=9, color=INK, loc="left")
    save(fig, FIGS / "f51_handbuilt_by_version.png")


def f52_revealed_by_version():
    """Stated probes vs the behavioural target (empirical-logit menu-choice
    rate, 12 rollouts/item) on the 388-item revealed panel."""
    from chartstyle import INK
    rv = _pv("revealed_versions.json")
    V = rv["versions"]
    names = list(V)
    col = MODEL_COLORS[PV_MODEL]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    _series(ax, names, [V[n]["r"] for n in names], col, "stated probe vs behaviour")
    _series(ax, names, [V[n]["r_vs_stated_mu"] for n in names], MUTED,
            "stated probe vs stated $\\mu$ (same panel)", ls=":")
    ax.axhline(rv["stated_mu_r"], color=INK2, lw=0.9, ls="--")
    ax.annotate(f"measured stated $\\mu$ vs behaviour: r={rv['stated_mu_r']:.2f}",
                (16.4, rv["stated_mu_r"]), fontsize=6.8, color=INK2, ha="right", va="bottom")
    ax.axhline(rv["revealed_probe_cv_r"], color=INK2, lw=0.9, ls="-.")
    ax.annotate(f"probe trained on behaviour itself (held-out): r={rv['revealed_probe_cv_r']:.2f}",
                (16.4, rv["revealed_probe_cv_r"]), fontsize=6.8, color=INK2, ha="right",
                va="top")
    _version_axis(ax, names, f"Pearson r on the revealed panel (n={rv['n']})")
    bounded_axis(ax, "y", 0.5, 1.0)
    _phase_labels(ax, 0.995)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    ax.set_title("Reading behaviour through the stated probe: every version sits at the "
                 "stated-channel ceiling", fontsize=9, color=INK, loc="left")
    save(fig, FIGS / "f52_revealed_by_version.png")


def _retest_L18(gd):
    """Layer-18 4C retest rows for every direction: gate_retest_versions.json
    (old, glob, v0, v3, v5, v8, v10, v12, v15_L18) merged with
    gate_retest_versions_fill.json (v1, v2, v4, v6, v7, v9, v11, v13, v14) when
    it exists, ordered old, glob, v0..v14, v15_L18. The fill run's nulls are
    checked against the originals (same seeds -> same random cells).
    -> (nulls, [(name, result)])"""
    g = json.loads((gd / "gate_retest_versions.json").read_text())
    res = dict(g["results"])
    fp = gd / "gate_retest_versions_fill.json"
    if fp.exists():
        f = json.loads(fp.read_text())
        for c in g["nulls"]:
            assert np.allclose(g["nulls"][c], f["nulls"][c], atol=1e-3), ("fill nulls differ", c)
        res.update(f["results"])
    def key(n):
        if n in ("old", "glob"):
            return (0, ("old", "glob").index(n))
        v = n[1:].split("_")[0]
        return (1, int(v))
    return g["nulls"], [(n, res[n]) for n in sorted(res, key=key)]


def _elo_rows(path, layer_label):
    g = json.loads(path.read_text())
    return [(f"{name}\n@L{g['layer']}" if layer_label else name, res)
            for name, res in g["results"].items()]


def f53_4c_retest_versions():
    """The 4C elicitation-cell retest: every direction's push on stated choices
    (dElo at three coefficients, z vs matched-norm random nulls annotated) and
    the readout-mass drop, the format-integrity guard."""
    from chartstyle import INK, SEQ
    gd = P1 / "results" / "surf" / "global" / PV_MODEL
    null18, rows = _retest_L18(gd)
    fam = gd / "gate_retest_v14family.json"   # old / v4 / v10 / v15 refit at layer 14
    p14 = gd / "gate_retest_v15L14.json"
    if fam.exists():
        rows += _elo_rows(fam, True)
    elif p14.exists():
        rows += _elo_rows(p14, True)
    coefs = ["0.25", "0.5", "1.0"]
    shades = [SEQ[4], SEQ[7], SEQ[10]]
    col = MODEL_COLORS[PV_MODEL]
    pretty = {"old": "XL-only (stage 4)", "glob": "XL+SURF refit", "v0": "v0 (S0)"}

    def label(n):
        base = n.split("\n")[0].replace("_L14", "")
        if base.endswith("_L18"):
            return base[:-4] + " (refit at 18)"
        return pretty.get(base, base)

    l18 = [(n, r) for n, r in rows if "@L" not in n]
    l14 = [(n, r) for n, r in rows if "@L" in n]
    max_rm = max(abs(r[c]["rm_drop_plus"]) for _, r in rows for c in coefs)
    null14 = None
    if l14:
        null14 = json.loads(fam.read_text())["nulls"] if fam.exists() else \
            json.loads(p14.read_text())["nulls"]
    fig, axes = plt.subplots(1, 2, figsize=(min(0.62 * len(l18) + 4.6, 14), 4.6),
                             gridspec_kw={"width_ratios": [len(l18) + 1, max(len(l14), 1) + 2]})
    for ax, block, nulls, ttl in ((axes[0], l18, null18, "(a) steered at layer 18 (the Stage-4 layer)"),
                                  (axes[1], l14, null14, "(b) the same family refit and steered at layer 14")):
        if not block:
            ax.set_visible(False)
            continue
        xs = np.arange(len(block)) + 1          # x = 0 is the random-direction null group
        for k, (c, sh) in enumerate(zip(coefs, shades)):
            dx = (k - 1) * 0.27
            ys = [r[c]["dd_plus"] for _, r in block]
            ax.bar(xs + dx, ys, width=0.25, color=sh, label=f"coef {c}" if ax is axes[0] else None, zorder=3)
            nv = np.array(nulls[c])
            sd = float(np.std(nv, ddof=1))
            ax.errorbar([dx], [nv.mean()], yerr=[[sd], [sd]], fmt="o", ms=3.2, color=sh, ecolor=sh,
                        elinewidth=1.2, capsize=3, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
        ax.set_xticks([0] + list(xs), ["random dirs \u00b1 1 SD"] + [label(n) for n, _ in block],
                      fontsize=7.5, rotation=30, ha="right")
        ax.set_xlim(-0.6, len(block) + 0.6)
        ax.set_title(ttl, fontsize=8.5, color=INK2, loc="left")
        style(ax, grid_axis="y")
        lo = min(min(np.array(nulls[c]).min() for c in coefs), 0.0)
        hi = max(r[c]["dd_plus"] for _, r in block for c in coefs)
        bounded_axis(ax, "y", lo, hi * 1.06)
    axes[0].set_ylabel("+direction choice shift\n(\u0394 log-odds toward the steered item)")
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    if l14:
        axes[1].annotate(f"null SD at coef 0.5: {np.std(null14['0.5'], ddof=1):.2f} here\n"
                         f"vs {np.std(null18['0.5'], ddof=1):.3f} at layer 18\n"
                         f"readout-mass drop \u2264 {max_rm:.2f} everywhere",
                         (0.98, 0.03), xycoords="axes fraction", fontsize=6.6, color=INK2,
                         ha="right", va="bottom")
    fig.suptitle(f"Steering power tracks probe hardening at layer 18; layer 14 is louder, not better "
                 f"\u2014 {MODEL_LABELS[PV_MODEL]} only",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, FIGS / "f53_4c_retest_versions.png")


def f53b_4c_plus_minus_L18():
    """f53's layer-18 block with both signs: the +direction push (bars up) and
    the -direction push (bars down) on the elicitation cell's choice log-odds,
    per coefficient, beside the matched-norm random-direction null."""
    from chartstyle import INK, SEQ
    gd = P1 / "results" / "surf" / "global" / PV_MODEL
    nulls, rows = _retest_L18(gd)
    g = {"nulls": nulls}
    coefs = ["0.25", "0.5", "1.0"]
    shades = [SEQ[4], SEQ[7], SEQ[10]]
    pretty = {"old": "XL-only (stage 4)", "glob": "XL+SURF refit", "v0": "v0 (S0)"}
    label = lambda n: n[:-4] + " (refit at 18)" if n.endswith("_L18") else pretty.get(n, n)
    fig, ax = plt.subplots(figsize=(min(0.72 * len(rows) + 2.4, 14), 4.8))
    xs = np.arange(len(rows)) + 1
    for k, (c, sh) in enumerate(zip(coefs, shades)):
        dx = (k - 1) * 0.27
        ax.bar(xs + dx, [r[c]["dd_plus"] for _, r in rows], width=0.25, color=sh, zorder=3,
               label=f"coef {c}")
        ax.bar(xs + dx, [r[c]["dd_minus"] for _, r in rows], width=0.25, color=sh, zorder=3)
        nv = np.array(g["nulls"][c])
        sd = float(np.std(nv, ddof=1))
        ax.errorbar([dx], [nv.mean()], yerr=[[sd], [sd]], fmt="o", ms=3.2, color=sh, ecolor=sh,
                    elinewidth=1.2, capsize=3, zorder=4)
    ax.axhline(0, color=INK, lw=0.8, zorder=2)
    ax.set_xticks([0] + list(xs), ["random dirs \u00b1 1 SD"] + [label(n) for n, _ in rows],
                  fontsize=7.5, rotation=30, ha="right")
    ax.set_xlim(-0.6, len(rows) + 0.6)
    ax.set_ylabel("choice shift toward the steered item (\u0394 log-odds)")
    lo = min(r[c]["dd_minus"] for _, r in rows for c in coefs)
    hi = max(r[c]["dd_plus"] for _, r in rows for c in coefs)
    bounded_axis(ax, "y", lo * 1.06, hi * 1.06)
    ax.annotate("direction added (+)", (0.01, 0.985), xycoords="axes fraction", fontsize=7.5,
                color=INK2, ha="left", va="top")
    ax.annotate("direction subtracted (\u2212)", (0.01, 0.015), xycoords="axes fraction", fontsize=7.5,
                color=INK2, ha="left", va="bottom")
    style(ax, grid_axis="y")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.95))
    fig.suptitle(f"Adding and subtracting each utility direction at layer 18 \u2014 {MODEL_LABELS[PV_MODEL]} only",
                 fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, FIGS / "f53b_4c_plus_minus_L18.png")


def f54_gate_versions():
    """Stage-4 behavioural gate for the hardened utility directions beside the
    committed three (choice / pool / utility): z per coefficient."""
    from chartstyle import INK, BASE, SEQ
    s4 = P1 / "results" / "stage4" / PV_MODEL
    old = json.loads((s4 / "gate.json").read_text())["gates"]
    new = json.loads((s4 / "gate_versions.json").read_text())["gates"]
    rows = [(k, v, True) for k, v in old.items()] + [(k, v, False) for k, v in new.items()]
    coefs = ["0.25", "0.5", "1.0"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ys = np.arange(len(rows))
    col = MODEL_COLORS[PV_MODEL]
    shades_old = [BASE, MUTED, INK2]
    shades_new = [SEQ[4], SEQ[7], SEQ[10]]
    for k, c in enumerate(coefs):
        for y, (name, g, is_old) in zip(ys, rows):
            z = g["table"][c]["z_plus"]
            ax.barh(y + (k - 1) * 0.27, z, height=0.25,
                    color=(shades_old if is_old else shades_new)[k],
                    label=(f"coef {c}" if (y == len(old) and not is_old) else None))
    ax.axvline(3, color=col, lw=0.9, ls="--")
    ax.annotate("gate: |z| > 3", (3.05, -0.55), fontsize=7, color=col, va="top")
    ax.axvline(0, color=MUTED, lw=0.6)
    labels = [n + ("" if is_old else "  (SURF)") + ("  ✓" if g["pass"] else "  ✗")
              for n, g, is_old in rows]
    ax.set_yticks(ys, labels, fontsize=8)
    ax.invert_yaxis()
    ax.axhline(len(old) - 0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("gate z (+v choice shift / random-direction null SD)")
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="SURF sets", title_fontsize=7)
    ax.set_title("Stage-4 behavioural gate: committed direction sets (grey) vs the "
                 "SURF-hardened utility probes", fontsize=9, color=INK, loc="left")
    style(ax)
    save(fig, FIGS / "f54_gate_versions.png")


def _late(r):
    v = r["per_turn"]["valence"]
    return float(np.mean(v[1:])) if len(v) > 1 else v[0]


def _swing(rs):
    g = [np.mean(r["fb_read"]["valence"]) for r in rs
         if r["outcome"] == "good" and r.get("fb_read", {}).get("valence")]
    b = [np.mean(r["fb_read"]["valence"]) for r in rs
         if r["outcome"] == "bad" and r.get("fb_read", {}).get("valence")]
    return (float(np.mean(g) - np.mean(b)) if g and b else None)


def f55_valence_versions():
    """4B/4C with the hardened directions: generation-state valence shift (z
    vs the Stage-3 bare control SD) against the gate z of the direction, with
    the matched-norm random arm as the disruption envelope. The committed run's
    points are drawn hollow for reference (its probe readout used the original
    Stage-2 vectors; this run's readout uses the regenerated ones)."""
    from chartstyle import INK, BASE
    s4 = P1 / "results" / "stage4" / PV_MODEL
    ctrl = [json.loads(l) for l in
            (P1 / "results" / "stage3" / PV_MODEL / "probes.jsonl").read_text().splitlines()]
    ctrl = [r for r in ctrl if r.get("frame") == "bare" and r["outcome"] in ("good", "bad")]
    st0 = float(np.mean([_late(r) for r in ctrl]))
    sd0 = float(np.std([_late(r) for r in ctrl], ddof=1))
    sw0 = _swing(ctrl)
    col = MODEL_COLORS[PV_MODEL]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for tag, hollow in (("", True), ("_versions", False)):
        gp, pp = s4 / f"gate{tag}.json", s4 / f"probes_4bc{tag}.jsonl"
        if not (gp.exists() and pp.exists()):
            continue
        gates = json.loads(gp.read_text())["gates"]
        rows = [json.loads(l) for l in pp.read_text().splitlines()]
        for ds in sorted({r["dirset"] for r in rows}):
            rs = [r for r in rows if r["dirset"] == ds]
            g = gates.get(ds, {})
            z = g.get("table", {}).get(str(g.get("primary_coef")), {}).get("z_plus", 0) \
                if ds != "random" else 0
            dy = (float(np.mean([_late(r) for r in rs])) - st0) / (sd0 + 1e-9)
            sw = _swing(rs)
            kw = dict(s=58, color=col if not hollow else "white",
                      edgecolor=col if not hollow else MUTED, linewidth=1.3, zorder=3)
            lab = ds.replace("utility_", "u_")
            ax1.scatter([z], [dy], marker="x" if ds == "random" else "o", **kw)
            ax1.annotate(lab, (z, dy), fontsize=6.5, color=INK2 if not hollow else MUTED,
                         xytext=(4, 3), textcoords="offset points")
            if sw is not None:
                ax2.scatter([z], [sw - sw0], marker="x" if ds == "random" else "o", **kw)
                ax2.annotate(lab, (z, sw - sw0), fontsize=6.5,
                             color=INK2 if not hollow else MUTED,
                             xytext=(4, 3), textcoords="offset points")
    for ax, yl, t in ((ax1, "generation-state valence shift\n(z vs Stage-3 bare control SD)",
                       "(a) does the steered state feel different?"),
                      (ax2, "Δ feedback-reading swing (good − bad)\nvs Stage-3 control",
                       "(b) does the model still read outcomes?")):
        ax.axhline(0, color=BASE, lw=0.8)
        ax.axvline(3, color=MUTED, lw=0.8, ls="--")
        ax.set_xlabel("choice movement (gate z at the primary coef)")
        ax.set_ylabel(yl)
        ax.set_title(t, fontsize=8.5, color=INK2, loc="left")
        style(ax, grid_axis="both")
    ax1.scatter([], [], s=58, color=col, label="this run (hardened directions + random)")
    ax1.scatter([], [], s=58, color="white", edgecolor=MUTED, linewidth=1.3,
                label="committed run (choice / pool / utility / random)")
    ax1.legend(frameon=False, fontsize=6.8, loc="best")
    fig.suptitle("4B/4C with SURF-hardened utility directions: choices move, does affect?",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGS / "f55_valence_versions.png")


PV_FIGS = [f47_probe_versions_global2, f48_linearity_layered, f49_direction_cosines,
           f50_e2_dissociation_by_version, f51_handbuilt_by_version, f52_revealed_by_version,
           f53b_4c_plus_minus_L18]
PV_FIGS_GPU = [f53_4c_retest_versions, f54_gate_versions, f55_valence_versions]


if __name__ == "__main__":
    setup()
    if len(sys.argv) > 1 and sys.argv[1] == "versions":
        for f in PV_FIGS:
            f()
    elif len(sys.argv) > 1 and sys.argv[1] == "versions-gpu":
        for f in PV_FIGS_GPU:
            f()
    else:
        f36()
        f37_retrained_vs_frozen()
        f38()
        f39()
