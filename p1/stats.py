"""Generic statistics helpers for P1 (generalizes Day-1 prefs.color_cis)."""
import random


def clustered_ci(rows, group_of, stat_fn, n_boot=2000, seed=1):
    """Cluster bootstrap: resample groups (e.g. items) with replacement, weight
    each row by its groups' multiplicities, recompute stat_fn(rows, weights).

    rows: list of records; group_of(row) -> tuple of group keys the row touches;
    stat_fn(rows, weight_fn) -> float. Returns (observed, lo95, hi95)."""
    rng = random.Random(seed)
    groups = sorted({g for r in rows for g in group_of(r)})
    obs = stat_fn(rows, lambda r: 1)
    reps = []
    for _ in range(n_boot):
        w = {}
        for g in rng.choices(groups, k=len(groups)):
            w[g] = w.get(g, 0) + 1
        def weight(r, w=w):
            out = 1
            for g in group_of(r):
                out *= w.get(g, 0)
            return out
        reps.append(stat_fn(rows, weight))
    reps.sort()
    return obs, reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot) - 1]


def ols(y, X, names):
    """Standardized OLS loadings of y on X (+intercept): {name: beta}, r2."""
    import numpy as np
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    keep = [k for k in range(X.shape[1]) if X[:, k].std() > 0]
    Xs = (X[:, keep] - X[:, keep].mean(0)) / X[:, keep].std(0)
    ys = (y - y.mean()) / (y.std() or 1)
    A = np.column_stack([np.ones(len(y)), Xs])
    beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
    resid = ys - A @ beta
    r2 = 1 - resid.var() / (ys.var() or 1)
    return {names[k]: round(float(b), 3) for k, b in zip(keep, beta[1:])}, round(float(r2), 3)
