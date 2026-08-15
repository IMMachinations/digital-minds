"""Phase-G analysis: swap dose-response, merged effort/persistence, Stage-3
re-analysis at 60/cell. Run after the Phase-G chain completes.
Usage: uv run python scripts/phaseg_analysis.py
"""
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

import rollout_stats as rs
import stage3

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]


def swap2_events(model):
    p = P1 / "results" / "stage1d" / model / "rollouts_swap2.jsonl"
    out = []
    for r in map(json.loads, p.read_text().splitlines()):
        if r["meta"].get("switched") is not None:
            out.append({"delta_z": r["meta"]["delta_z"],
                        "switched": r["meta"]["switched"], "env": r["env_id"]})
    return out


def logistic_fit(events, n_boot=1000, seed=1):
    import random
    from scipy.optimize import minimize
    x = np.array([e["delta_z"] for e in events])
    y = np.array([float(e["switched"]) for e in events])

    def fit(xs, ys):
        def nll(p):
            z = p[0] + p[1] * xs
            return float(np.mean(np.log1p(np.exp(-np.where(ys > 0, z, -z))))
                         + 0.01 * (p[0] ** 2 + p[1] ** 2))
        return minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    a, b = fit(x, y)
    rng = random.Random(seed)
    envs = sorted({e["env"] for e in events})
    slopes = []
    for _ in range(n_boot):
        w = {}
        for e in rng.choices(envs, k=len(envs)):
            w[e] = w.get(e, 0) + 1
        idx = [k for k, ev in enumerate(events) for _ in range(w.get(ev["env"], 0))]
        if len(idx) > 10:
            slopes.append(fit(x[idx], y[idx])[1])
    slopes.sort()
    return {"intercept": round(float(a), 3), "slope": round(float(b), 3),
            "slope_ci": [round(slopes[int(0.025 * len(slopes))], 3),
                         round(slopes[int(0.975 * len(slopes)) - 1], 3)]}


def main():
    for m in SUBJECTS:
        lines = [f"{m}: Phase-G updates"]
        ev = swap2_events(m)
        fit = logistic_fit(ev)
        bins = {}
        for e in ev:
            key = round(e["delta_z"] * 2) / 2
            bins.setdefault(key, []).append(e["switched"])
        rate_up = np.mean([e["switched"] for e in ev if e["delta_z"] > 0.15])
        rate_dn = np.mean([e["switched"] for e in ev if e["delta_z"] < -0.15])
        lines.append(f"swap2 (assigned-task, n={len(ev)} parsed): "
                     f"P(switch|up) {rate_up:.2f} vs P(switch|down) {rate_dn:.2f}; "
                     f"logistic slope {fit['slope']} CI {fit['slope_ci']}")
        save_json(P1 / "results" / "stage1d" / m / "swap2.json",
                  {"events": ev, "fit": fit})

        eff_rows = []
        for fn in ("rollouts_effort.jsonl", "rollouts_effort2.jsonl"):
            p = P1 / "results" / "stage1d" / m / fn
            if p.exists():
                for r in map(json.loads, p.read_text().splitlines()):
                    sh = [s["share_high"] for s in r["meta"]["shares"]
                          if s["share_high"] is not None]
                    if sh:
                        eff_rows.append({"share_high": float(np.mean(sh)),
                                         "dmu": r["meta"]["dmu"]})
        es = rs.effort_summary(eff_rows)
        xs = [r["dmu"] for r in eff_rows]
        ys = [r["share_high"] for r in eff_rows]
        from lib.valuation import spearman
        lines.append(f"effort merged (n={len(eff_rows)}): mean share_high "
                     f"{es['mean_share_high']}, slope {es['slope_vs_dmu']}, "
                     f"spearman {spearman(xs, ys):+.3f}")
        save_json(P1 / "results" / "stage1d" / m / "effort_merged.json", eff_rows)

        per_rows = []
        for fn in ("rollouts_persist.jsonl", "rollouts_persist2.jsonl"):
            p = P1 / "results" / "stage1d" / m / fn
            if p.exists():
                for r in map(json.loads, p.read_text().splitlines()):
                    per_rows.append({"pool": r["meta"]["pool"],
                                     "giveup_turn": r["meta"]["giveup_turn"]})
        ps = rs.persistence_summary(per_rows)
        lines.append(f"persistence merged: {ps}")

        (P1 / "results" / "stage1d" / m / "phaseg.txt").write_text(
            "\n".join(lines) + "\n")
        print("\n".join(lines))
        stage3.cmd_analyze(m)


if __name__ == "__main__":
    main()
