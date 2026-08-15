"""Utility-manifold check: fit an OPEN spline through Stage-1 item activations
ordered by 1B utility mu, and test whether the spline coordinate predicts
held-out mu better than the linear ridge probe at the same layer.

The utility analog of the Stage-2 emotion ring: emotion quality was a closed
circumplex (periodic spline through emotion centroids ordered by angle);
utility is a line, so the spline is open (per=False) through mu-quantile-bin
centroids. Everything is 2-fold cross-fitted (basis, bins, spline, u->mu map
on train items only). Verdict rule mirrors rung 3: the spline earns its place
only if it beats the linear readout held-out.

Usage: uv run python scripts/utility_spline.py   (all three subjects, CPU-only)
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import splev, splprep

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json
from lib.valuation import pearson, spearman

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]
N_BINS = 12
K_FIT = 8
# (ambient_k, k_fit, n_bins) configs: the default mirrors the emotion pipeline
# numerically, but with ~98 train items 64 ambient components is nearly the
# full item subspace (64/97) - real reduction requires far smaller k here
CONFIGS = [(64, 8, 12), (16, 4, 12), (8, 3, 16), (24, 5, 20)]


def open_spline(centroids, smooth=None, k_fit=K_FIT):
    mu_c = centroids.mean(0)
    _, _, Vt = np.linalg.svd(centroids - mu_c, full_matrices=False)
    B = Vt[:k_fit]
    pts = (centroids - mu_c) @ B.T
    if smooth is None:
        smooth = pts.shape[0] * np.var(pts) * 0.5
    tck, _ = splprep(pts.T, s=smooth, per=False)
    return {"tck": tck, "mu": mu_c, "B": B}


def spline_u(X, sp, m=2000):
    Xc = (X - sp["mu"]) @ sp["B"].T
    u = np.linspace(0, 1, m)
    pts = np.stack(splev(u, sp["tck"]), axis=1)
    d = ((Xc[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    return u[d.argmin(1)]


def eval_model(model, ambient_k=64, k_fit=K_FIT, n_bins=N_BINS):
    from sklearn.linear_model import RidgeCV
    man = torch.load(P1 / "results" / "stage2" / model / "manifold.pt",
                     weights_only=False)
    L = man["layer"]
    X = torch.load(P1 / "results" / "stage1c" / model / "probe_acts.pt").float()[L].numpy()
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    mu = np.array([r["mu"] for r in ut])

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(mu))
    folds = [perm[:len(mu) // 2], perm[len(mu) // 2:]]
    pred_spline = np.zeros(len(mu))
    pred_ridge = np.zeros(len(mu))
    pred_line = np.zeros(len(mu))
    curvature_r = []
    for f in range(2):
        te, tr = folds[f], folds[1 - f]
        # 64-d basis on train items
        mean_tr = X[tr].mean(0)
        _, _, Vt = np.linalg.svd(X[tr] - mean_tr, full_matrices=False)
        B64 = Vt[:ambient_k]
        Ztr = (X[tr] - mean_tr) @ B64.T
        Zte = (X[te] - mean_tr) @ B64.T
        # mu-quantile bin centroids on train
        order = np.argsort(mu[tr])
        bins = np.array_split(order, n_bins)
        cent = np.stack([Ztr[b].mean(0) for b in bins])
        sp = open_spline(cent, k_fit=k_fit)
        u_tr, u_te = spline_u(Ztr, sp), spline_u(Zte, sp)
        a = np.polyfit(u_tr, mu[tr], 1)
        pred_spline[te] = np.polyval(a, u_te)
        # centroid-LINE control: same geometric pipeline, zero curvature -
        # separates "curvature hurts" from "the unsupervised pipeline hurts"
        _, _, Vc = np.linalg.svd(cent - cent.mean(0), full_matrices=False)
        axis = Vc[0]
        lu_tr, lu_te = Ztr @ axis, Zte @ axis
        al = np.polyfit(lu_tr, mu[tr], 1)
        pred_line[te] = np.polyval(al, lu_te)
        # linear ridge baseline at the same layer, standardized on train
        sd = X[tr].std(0) + 1e-6
        ridge = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X[tr] - mean_tr) / sd, mu[tr])
        pred_ridge[te] = ridge.predict((X[te] - mean_tr) / sd)
        # does the spline coordinate add anything beyond the linear direction?
        lin_tr = ridge.predict((X[tr] - mean_tr) / sd)
        curvature_r.append(abs(pearson(list(u_tr), list(lin_tr))))

    r_sp = pearson(list(pred_spline), list(mu))
    rho_sp = spearman(list(pred_spline), list(mu))
    r_ri = pearson(list(pred_ridge), list(mu))
    rho_ri = spearman(list(pred_ridge), list(mu))
    r_ln = pearson(list(pred_line), list(mu))
    verdict = "spline earns its place" if r_sp > r_ri else "LINEAR SUFFICES for utility too"
    lines = [f"{model} (layer {L}) ambient_k={ambient_k} k_fit={k_fit} bins={n_bins}: "
             "utility-spline vs linear ridge, 2-fold held-out",
             f"  spline-u  r = {r_sp:+.3f}  rho = {rho_sp:+.3f}",
             f"  ridge     r = {r_ri:+.3f}  rho = {rho_ri:+.3f}",
             f"  centroid-line control r = {r_ln:+.3f} (same pipeline, no curvature)",
             f"  |corr(u, linear readout)| on train: {np.mean(curvature_r):.3f} "
             "(near 1 = the spline is just a bent ruler along the linear direction)",
             f"  verdict: {verdict}"]
    with open(P1 / "results" / "stage2" / model / "utility_spline.txt", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return r_sp, r_ri


if __name__ == "__main__":
    for m in SUBJECTS:
        (P1 / "results" / "stage2" / m / "utility_spline.txt").write_text("")
        for ak, kf, nb in CONFIGS:
            eval_model(m, ambient_k=ak, k_fit=kf, n_bins=nb)
