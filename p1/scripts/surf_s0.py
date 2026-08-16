"""SURF S0: caches + calibration checks for the scoring cascade.

CPU (run any time):
  uv run python scripts/surf_s0.py anchors        # cache pick_anchors() + per-model values
  uv run python scripts/surf_s0.py probe <model>  # calibrated ridge probe from acts_xl.pt

GPU (run in a free slot; gate-calib loads the 32B, the others the subject too):
  uv run python scripts/surf_s0.py gate-calib
  uv run python scripts/surf_s0.py t2-check <model>
  uv run python scripts/surf_s0.py t3-smoke <model>

The probe step fixes two gaps in the existing artifacts: probes.utility_probe
fits and discards its estimator, and stage1x's utility_dir.pt is a bare
L2-normalized direction (no intercept, no standardizer), so neither can score
a new item. Here: 2-fold cross-fit over the 3 cached working layers to pick
the layer and report honest held-out r, then a full-data refit persisted with
its (mean, std, intercept) so Tier1Probe gives calibrated mu estimates.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

from modelspec import ROSTER, STEER_FRACS

# config.num_hidden_layers per roster model, kept here so CPU steps never load
# weights or need the HF cache (values match p1/figures.py N_LAYERS).
N_LAYERS = {"llama31-8b": 32, "qwen25-7b": 28, "qwen3-4b": 36, "qwen25-32b": 64,
            "qwen25-05b": 24, "qwen25-15b": 28, "qwen25-3b": 36}


def s0_dir(model):
    d = P1 / "results" / "surf" / "s0" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_layers(model):
    n = N_LAYERS[model]
    return [round(f * n) for f in STEER_FRACS][2:]


def cmd_anchors(_=None):
    import stage1x
    anchors = stage1x.pick_anchors()
    save_json(P1 / "items_xl" / "anchors.json", anchors)
    print(f"cached {len(anchors)} anchors -> items_xl/anchors.json")
    for m in ROSTER:
        p = P1 / "results" / "stage1b" / m / "utilities.json"
        if not p.exists():
            continue
        ut = {r["id"]: [r["mu"], r["sigma2"]] for r in load_json(p)}
        vals = {a["id"]: ut[a["id"]] for a in anchors}
        save_json(s0_dir(m) / "anchor_values.json", vals)
        print(f"  {m}: anchor_values.json "
              f"(mu span [{min(v[0] for v in vals.values()):+.2f}, "
              f"{max(v[0] for v in vals.values()):+.2f}])")


def cmd_probe(model):
    from sklearn.linear_model import RidgeCV
    rows = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt").float()
    assert acts.shape[1] == len(rows), (acts.shape, len(rows))
    y = np.array([r["mu"] for r in rows])
    layers = work_layers(model)
    alphas = np.logspace(1, 6, 8)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(y))
    folds = [perm[: len(y) // 2], perm[len(y) // 2:]]

    held_r = []
    for lp in range(acts.shape[0]):
        X = acts[lp].numpy()
        pred = np.zeros(len(y))
        for f in range(2):
            te, tr = folds[f], folds[1 - f]
            m0, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            m = RidgeCV(alphas=alphas).fit((X[tr] - m0) / sd, y[tr])
            pred[te] = m.predict((X[te] - m0) / sd)
        held_r.append(pearson(list(pred), list(y)))
    lp = int(np.argmax(held_r))

    X = acts[lp].numpy()
    m0, sd = X.mean(0), X.std(0) + 1e-6
    m = RidgeCV(alphas=alphas).fit((X - m0) / sd, y)
    torch.save({"layer_pos": lp, "layer_global": layers[lp],
                "mean": m0, "std": sd, "coef": m.coef_, "intercept": float(m.intercept_),
                "alpha": float(m.alpha_), "cv_r": round(held_r[lp], 4), "n": len(y),
                "per_layer_cv_r": [round(r, 4) for r in held_r]},
               s0_dir(model) / "probe.pt")
    print(f"{model}: probe at layer {layers[lp]} (pos {lp}), "
          f"held-out r = {held_r[lp]:+.3f} (per layer: "
          + ", ".join(f"{layers[k]}:{r:+.3f}" for k, r in enumerate(held_r)) + ")")


def cmd_gate_calib(_=None):
    import surf_scores
    gen = {it["id"]: it for it in load_json(P1 / "items_xl" / "generated.json")}
    flags = load_json(P1 / "items_xl" / "qc_flags.json")
    bad_ids = [f["id"] for f in flags
               if f["flag"] in ("malformed", "wrong_domain", "valence_loaded")]
    flagged_all = {f["id"] for f in flags}
    import random
    rng = random.Random(0)
    bad = rng.sample(bad_ids, 30)
    clean = rng.sample([i for i in gen if i not in flagged_all], 30)
    handles = surf_scores.Handles(surf_scores.GENERATOR)
    gate = surf_scores.Tier0Gate(handles)
    res_clean = gate.gate([gen[i]["text"] for i in clean])
    res_bad = gate.gate([gen[i]["text"] for i in bad])
    clean_pass = sum(g["pass"] for g in res_clean) / len(res_clean)
    bad_fail = sum(not g["pass"] for g in res_bad) / len(res_bad)
    agree = (clean_pass + bad_fail) / 2
    out = {"clean_pass_rate": round(clean_pass, 3), "flagged_fail_rate": round(bad_fail, 3),
           "agreement": round(agree, 3),
           "clean": [{"id": i, **{k: g[k] for k in ("pass", "flags")}}
                     for i, g in zip(clean, res_clean)],
           "flagged": [{"id": i, **{k: g[k] for k in ("pass", "flags")}}
                       for i, g in zip(bad, res_bad)]}
    save_json(P1 / "results" / "surf" / "s0" / "gate_calib.json", out)
    print(f"gate-calib: clean pass {clean_pass:.2f}, flagged fail {bad_fail:.2f}, "
          f"agreement {agree:.2f} -> {'OK' if agree >= 0.8 else 'TUNE RUBRIC'}")


def _stratified(rows, n, seed=0):
    """n mu-spanning items: one uniform draw per mu-quantile bin."""
    rng = np.random.RandomState(seed)
    rows = sorted(rows, key=lambda r: r["mu"])
    edges = np.linspace(0, len(rows), n + 1).astype(int)
    return [rows[rng.randint(lo, hi)] for lo, hi in zip(edges[:-1], edges[1:])]


def cmd_t2_check(model):
    import surf_scores
    from surf import e1_config
    rows = [r for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
            if not r["validation"]]
    sample = _stratified(rows, 100)
    texts = [r["text"] for r in sample]
    handles = surf_scores.Handles(model)
    cfg = e1_config(model, "max", 0)
    fast = surf_scores.Tier2Fast(handles, model, cfg.reduced, "max").score(texts)
    full = surf_scores.Tier2Full(handles, model).score(texts)
    rho_full = spearman(fast, [f["mu"] for f in full])
    rho_xl = spearman(fast, [r["mu"] for r in sample])
    save_json(s0_dir(model) / "t2_check.json",
              {"reduced": cfg.reduced, "rho_fast_vs_full": round(rho_full, 4),
               "rho_fast_vs_xl": round(rho_xl, 4),
               "rows": [{"id": r["id"], "fast": round(fa, 4), "full_mu": round(fu["mu"], 4),
                         "xl_mu": r["mu"]}
                        for r, fa, fu in zip(sample, fast, full)]})
    print(f"{model}: t2-check rho(fast, full mu) = {rho_full:+.3f}, "
          f"rho(fast, 1X mu) = {rho_xl:+.3f} -> "
          f"{'PASS' if rho_full >= 0.9 else 'FAIL - widen reduced design to 2 orders'}")


def cmd_t3_smoke(model):
    import surf_scores
    rows = [r for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
            if not r["validation"]]
    sample = _stratified(rows, 8)
    handles = surf_scores.Handles(model)
    t3 = surf_scores.Tier3Revealed(handles, model)
    save_json(s0_dir(model) / "t3_anchor.json",
              {"item_id": t3.anchor["item_id"], "text": t3.anchor_text})
    rates = t3.score([r["text"] for r in sample])
    rho = spearman(rates, [r["mu"] for r in sample])
    save_json(s0_dir(model) / "t3_smoke.json",
              {"anchor": t3.anchor["item_id"], "rho_choice_vs_mu": round(rho, 4),
               "rows": [{"id": r["id"], "mu": r["mu"], "choice_rate": round(c, 3)}
                        for r, c in zip(sample, rates)]})
    print(f"{model}: t3-smoke rho(choice rate, mu) = {rho:+.3f} over 8 items "
          f"(anchor env {t3.anchor['item_id']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["anchors", "probe", "gate-calib", "t2-check", "t3-smoke"])
    ap.add_argument("model", nargs="?")
    a = ap.parse_args()
    {"anchors": cmd_anchors, "probe": cmd_probe, "gate-calib": cmd_gate_calib,
     "t2-check": cmd_t2_check, "t3-smoke": cmd_t3_smoke}[a.cmd](a.model)


if __name__ == "__main__":
    main()
