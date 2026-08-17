"""SURF probeloop figures (f36, f37), following CHARTS.md via root chartstyle.

f36: probe-hardening behavioral convergence — held-out menu choice rate of each
     cycle's top-20 probe-selected items, per model, with the pre-loop (E2 v0)
     baselines and each model's revealed-arm ceiling for reference.
f37: the prequential scissors — on each cycle's fresh discoveries, the current
     (just-retrained) probe's correlation vs the frozen v0 control. The widening
     gap is the probe hardening; v0's decay shows the search escaping its
     training support.

Usage: uv run python scripts/surf_figures.py   (CPU)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def f37():
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9), sharey=True)
    for ax, m in zip(axes, MODELS):
        cs = cycles(m)
        xs = [c["cycle"] for c in cs]
        cur = [c["per_probe"][f"v{c['cycle']}"]["raw"]["pearson"] for c in cs]
        v0 = [c["per_probe"]["v0"]["raw"]["pearson"] for c in cs]
        ax.plot(xs, cur, color=MODEL_COLORS[m], lw=1.8, marker="o", ms=4,
                markerfacecolor="white", markeredgewidth=1.4, zorder=3,
                label="current probe")
        ax.plot(xs, v0, color=MUTED, lw=1.4, ls="--", marker="o", ms=3.5,
                markerfacecolor="white", markeredgewidth=1.1, zorder=2,
                label="frozen v0")
        ax.set_title(MODEL_LABELS[m], fontsize=9, color=MODEL_COLORS[m])
        ax.set_xticks(xs)
        ax.set_xticklabels([f"c{x}" for x in xs])
        style(ax, grid_axis="y")
        bounded_axis(ax, "y", 0.2, 1.0)
    axes[0].set_ylabel("r vs measured $\\mu$,\nfresh discoveries")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.suptitle("Prequential scissors: each cycle's probe vs the frozen v0 control",
                 fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, FIGS / "f37_probeloop_scissors.png")


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


if __name__ == "__main__":
    setup()
    f36()
    f37()
    f38()
    f39()
