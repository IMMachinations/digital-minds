"""Statistics for Stage 1D revealed-preference data."""
import math

import numpy as np
from scipy.optimize import minimize


def conditional_logit(choices, ridge_gamma=0.01):
    """choices: [{menu_id, mus: [mu_A..mu_D], picked: 0-3|'opt'}] ->
    dict(beta, gamma, nll). P(i) ∝ exp(beta*mu_i); P(opt) ∝ exp(gamma).
    Ridge on gamma keeps the fit finite when opt-out is never chosen."""
    return _fit_clogit(choices, np.ones(len(choices)), ridge_gamma)


def _fit_clogit(choices, w, ridge_gamma=0.01):
    mus = np.array([c["mus"] for c in choices], float)
    picked = np.array([4 if c["picked"] == "opt" else c["picked"] for c in choices])

    def nll(x):
        b, g = x
        z = np.column_stack([b * mus, np.full(len(mus), g)])
        z -= z.max(1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(1, keepdims=True))
        return -(w * logp[np.arange(len(mus)), picked]).sum() / max(w.sum(), 1) \
            + ridge_gamma * g * g
    # multi-start: the likelihood can be near-flat in beta when gamma dominates
    # (heavy opt-out), stranding a single Nelder-Mead run in a local basin
    best = min((minimize(nll, x0, method="Nelder-Mead")
                for x0 in ([0.0, -2.0], [1.0, -2.0], [3.0, 0.0], [0.5, 1.0])),
               key=lambda o: o.fun)
    return {"beta": round(float(best.x[0]), 4), "gamma": round(float(best.x[1]), 4),
            "nll": round(float(best.fun), 5)}


def beta_ci(choices, n_boot=1000, seed=1):
    """Cluster bootstrap over menus for the conditional-logit beta."""
    import random
    rng = random.Random(seed)
    menus = sorted({c["menu_id"] for c in choices})
    reps = []
    for _ in range(n_boot):
        w = {}
        for m in rng.choices(menus, k=len(menus)):
            w[m] = w.get(m, 0) + 1
        wt = np.array([w.get(c["menu_id"], 0) for c in choices], float)
        reps.append(_fit_clogit(choices, wt)["beta"])
    reps.sort()
    return reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot) - 1]


def chosen_rates(choices, env_ids):
    """-> {env_id: (wins, exposures, rate)} over all parsed choices."""
    wins = {e: 0 for e in env_ids}
    expo = {e: 0 for e in env_ids}
    for c in choices:
        for k, e in enumerate(c["option_ids"]):
            expo[e] += 1
            if c["picked"] == k:
                wins[e] += 1
    return {e: (wins[e], expo[e], wins[e] / expo[e] if expo[e] else 0.0) for e in env_ids}


def persistence_summary(rows, max_turns=12):
    """rows: [{pool: 'pref'|'dispref', giveup_turn: int|None, end_turn: int}]."""
    out = {}
    for pool in ("pref", "dispref"):
        rs = [r for r in rows if r["pool"] == pool]
        survived = [r["giveup_turn"] if r["giveup_turn"] is not None else max_turns
                    for r in rs]
        censored = sum(1 for r in rs if r["giveup_turn"] is None)
        out[pool] = {"n": len(rs), "mean_survived": round(float(np.mean(survived)), 2)
                     if rs else None, "censored": censored,
                     "giveup_turns": sorted(r["giveup_turn"] for r in rs
                                            if r["giveup_turn"] is not None)}
    return out


def swap_summary(swaps):
    """swaps: [{delta_z, switched: bool}] -> rates by sign + logistic slope."""
    def rate(rs):
        return (sum(r["switched"] for r in rs) / len(rs), len(rs)) if rs else (None, 0)
    up = rate([s for s in swaps if s["delta_z"] > 0.15])
    down = rate([s for s in swaps if s["delta_z"] < -0.15])
    lat = rate([s for s in swaps if abs(s["delta_z"]) <= 0.15])
    x = np.array([s["delta_z"] for s in swaps])
    y = np.array([float(s["switched"]) for s in swaps])

    def nll(p):
        a, b = p
        z = a + b * x
        return float(np.mean(np.log1p(np.exp(-np.where(y > 0, z, -z))))
                     + 0.01 * (a * a + b * b))
    fit = minimize(nll, [0.0, 0.0], method="Nelder-Mead")
    return {"p_switch_up": up, "p_switch_down": down, "p_switch_lateral": lat,
            "logistic_slope": round(float(fit.x[1]), 3)}


def effort_summary(rows):
    """rows: [{share_high: float|None, dmu: float}] (share of the HIGHER-mu task)."""
    kept = [r for r in rows if r["share_high"] is not None]
    if len(kept) < 4:
        return {"n": len(kept), "mean_share_high": None}
    xs = [r["dmu"] for r in kept]
    ys = [r["share_high"] for r in kept]
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else None
    return {"n": len(kept), "mean_share_high": round(float(np.mean(ys)), 3),
            "slope_vs_dmu": round(slope, 4) if slope is not None else None}
