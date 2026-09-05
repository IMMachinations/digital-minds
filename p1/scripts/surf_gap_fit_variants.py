"""Probe-fit variants for the gap loop: what actually limits how far the probe
bends toward the adversarial (gap) items?  The XL gate in surf_probeloop.cmd_fit
is a report, not a constraint; the real limits are the ridge penalty (RidgeCV
picks alpha by CV error on the whole set) and the training mix (XL + hardening
cycles outnumber the gap sets).

Prequential protocol: every variant is trained on exactly what probe v(10+k)
saw (XL + dataset_c{9+k}) and scored on gap cycle k+1's UNSEEN discoveries
(over / under), plus the XL held-out r (2-fold) so the trade-off is visible.
Calibration is refit per variant on held-out predictions, as in cmd_fit.

Variants: ridgecv (= the production recipe), alpha fixed at 1/10 and 1/100 of
the CV choice, gap items up-weighted x5 / x20 (sample weights), no XL, gap-only
(+ XL for calibration support), and layer sweep.

Usage: uv run python scripts/surf_gap_fit_variants.py qwen25-7b --k 4
       (train through gap cycle 4 -> evaluate on gap5; needs cached acts_*.pt)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import surf_probeloop as pl
from surf_s0 import work_layers

ALPHAS = np.logspace(1, 6, 8)


def _fit(X, y, alpha=None, w=None, seed=0):
    """-> (model, alpha). alpha=None: RidgeCV on ALPHAS (weighted if w)."""
    from sklearn.linear_model import Ridge, RidgeCV
    m0, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - m0) / sd
    if alpha is None:
        m = RidgeCV(alphas=ALPHAS).fit(Z, y, sample_weight=w)
        alpha = float(m.alpha_)
    else:
        m = Ridge(alpha=alpha).fit(Z, y, sample_weight=w)
    return (m0, sd, m), alpha


def _pred(model, X):
    m0, sd, m = model
    return m.predict((X - m0) / sd)


def _heldout(X, y, alpha=None, w=None, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    folds = [perm[: len(y) // 2], perm[len(y) // 2:]]
    p = np.zeros(len(y))
    for f in range(2):
        te, tr = folds[f], folds[1 - f]
        model, _ = _fit(X[tr], y[tr], alpha=alpha, w=None if w is None else w[tr])
        p[te] = _pred(model, X[te])
    return p


def _calib(p_heldout, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_heldout, y)
    return {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--k", type=int, default=4, help="train through gap cycle k, test on k+1")
    a = ap.parse_args()
    m, k = a.model, a.k
    import torch
    d = pl.out_dir(m)
    xl_rows = load_json(P1 / "results" / "stage1x" / m / "utilities_xl.json")
    xl_acts = torch.load(P1 / "results" / "stage1x" / m / "acts_xl.pt", weights_only=False).float()
    xl_mu = np.array([r["mu"] for r in xl_rows])
    xl_texts = {r["text"].lower() for r in xl_rows}
    ds = [r for r in load_json(d / f"dataset_c{9 + k}.json") if r["text"].lower() not in xl_texts]
    ds_acts = pl._ds_acts(m, ds, d / f"acts_c{9 + k}.pt")
    ds_mu, _ = pl.winsorize([r["mu"] for r in ds])
    is_gap = np.array([r["source"].startswith("gpl") for r in ds])
    tests = {}
    for arm in ("over", "under"):
        rows = load_json(d / f"discoveries_gpl{k + 1}_{arm}.json")
        tests[arm] = (pl._ds_acts(m, rows, d / f"acts_gpl{k + 1}_{arm}.pt"), np.array([r["mu"] for r in rows]))
    layers = work_layers(m)
    lp_prod = int(pl._load_probe(m, 10 + k)["layer_pos"])

    def run(name, lp, use_xl=True, alpha=None, gap_w=1.0, surf_w=1.0):
        Xs = [ds_acts[lp].numpy()]
        ys = [ds_mu]
        ws = [np.where(is_gap, gap_w, surf_w)]
        if use_xl:
            Xs.insert(0, xl_acts[lp].numpy()); ys.insert(0, xl_mu); ws.insert(0, np.ones(len(xl_mu)))
        X, y, w = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ws)
        w = None if np.all(w == 1.0) else w
        ph = _heldout(X, y, alpha=alpha, w=w)
        cal = _calib(ph, y)
        model, a_used = _fit(X, y, alpha=alpha, w=w)
        n_xl = len(xl_mu) if use_xl else 0
        r_xl = pearson(list(ph[:n_xl]), list(y[:n_xl])) if use_xl else float("nan")
        # XL held-out for no-XL variants: predict XL directly (never trained on it)
        if not use_xl:
            r_xl = pearson(list(_pred(model, xl_acts[lp].numpy())), list(xl_mu))
        out = {"layer": layers[lp], "alpha": round(a_used, 1), "xl_heldout_r": round(r_xl, 3)}
        for arm, (acts, mu) in tests.items():
            pc = pl.apply_calib(cal, _pred(model, acts[lp].numpy()))
            g = pc - mu
            out[arm] = {"gap": round(float(g.mean()), 3), "abs_gap": round(float(np.abs(g).mean()), 3),
                        "r": round(pearson(list(pc), list(mu)), 3),
                        "rho": round(spearman(list(pc), list(mu)), 3)}
        return out

    results = {}
    base = run("ridgecv", lp_prod)
    results["ridgecv (production, layer %d)" % layers[lp_prod]] = base
    a0 = base["alpha"]
    results[f"alpha/10 ({a0 / 10:.0f})"] = run("a10", lp_prod, alpha=a0 / 10)
    results[f"alpha/100 ({a0 / 100:.0f})"] = run("a100", lp_prod, alpha=a0 / 100)
    results["gap items x5 weight"] = run("w5", lp_prod, gap_w=5.0)
    results["gap items x20 weight"] = run("w20", lp_prod, gap_w=20.0)
    results["no XL (SURF + gap only)"] = run("noxl", lp_prod, use_xl=False)
    results["no XL, alpha/10"] = run("noxl10", lp_prod, use_xl=False, alpha=a0 / 10)
    for lp in range(len(layers)):
        if lp != lp_prod:
            results[f"ridgecv, layer {layers[lp]}"] = run("layer", lp)

    lines = [f"gap-loop fit variants ({m}): trained through gap cycle {k} (XL n={len(xl_mu)} + SURF n={len(ds)}, "
             f"gap items {int(is_gap.sum())}); tested on gap{k + 1} over (n={len(tests['over'][1])}) / "
             f"under (n={len(tests['under'][1])}) — both UNSEEN",
             f"{'variant':36s} {'alpha':>8s} {'XL r':>6s} | {'over gap':>8s} {'|gap|':>6s} {'r':>5s} | "
             f"{'under gap':>9s} {'|gap|':>6s} {'r':>5s}"]
    for name, r in results.items():
        o, u = r["over"], r["under"]
        lines.append(f"{name:36s} {r['alpha']:8.0f} {r['xl_heldout_r']:6.3f} | {o['gap']:+8.2f} {o['abs_gap']:6.2f} "
                     f"{o['r']:5.2f} | {u['gap']:+9.2f} {u['abs_gap']:6.2f} {u['r']:5.2f}")
    out = P1 / "results" / "surf" / "probeloop" / m / f"fit_variants_gap{k + 1}.txt"
    out.write_text("\n".join(lines) + "\n")
    save_json(out.with_suffix(".json"), results)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
