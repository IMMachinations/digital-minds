"""Stage 1B driver: adaptive pairwise battery -> Thurstonian utilities per model.

Usage: uv run python stage1b.py <roster-name> [--budget 600] [--target 3500]
                                [--max-rounds 8] [--gate 0.7]

Outputs under results/stage1b/<name>/:
  pairs_raw.json   every (pair, template, order, p) record — resumable
  utilities.json   per item: mu, sigma2, n_pairs, mu_by_template, domain, tags
  summary.txt      gates (template/order consistency), confound audit, checks
Gate (spec): min pairwise per-template Spearman >= 0.7, else the model's
utilities are flagged and kept out of downstream pools.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.valuation import pearson, spearman
from lib.util import load_json, save_json

import adaptive
import harness
import items as items_mod
import stats
import thurstone
from pairs import PairBattery


def obs_from(recs, keep=lambda r: True):
    return [(r["i"], r["j"], r["p"]) for r in recs if keep(r)]


def fit_all(n, recs):
    return thurstone.fit(n, obs_from(recs))


def color_profile_check(its, mu, lines):
    """Battery-level Day-1 validation (qwen25-7b): per-color mean mu of the
    price-controlled objects vs the committed balanced_pref profile."""
    committed = load_json(_day1.DESIRES / "results" / "balanced_pref" / "balanced_pref.json")
    from lib.data import COLORS
    from lib.tasks import signed
    old = {c: sum(signed(r["ab_diff"], r, c) for r in committed if c in (r["color_a"], r["color_b"]))
           / sum(1 for r in committed if c in (r["color_a"], r["color_b"])) for c in COLORS}
    new = {}
    for c in COLORS:
        vals = [mu[k] for k, it in enumerate(its)
                if it["domain"] == "objects" and it["tags"].get("kind") == "balanced_tier"
                and it["tags"]["color"] == c]
        new[c] = sum(vals) / len(vals)
    rho = spearman([new[c] for c in COLORS], [old[c] for c in COLORS])
    lines.append("Day-1 color-profile check (balanced-tier objects, per-color mean mu vs committed):")
    lines.append("        " + "".join(f"{c:>8}" for c in COLORS))
    lines.append("  new  " + "".join(f"{new[c]:8.2f}" for c in COLORS))
    lines.append("  old  " + "".join(f"{old[c]:8.2f}" for c in COLORS))
    lines.append(f"  spearman(new, old) = {rho:+.2f}   (target >~ 0.6)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--target", type=int, default=3500)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--gate", type=float, default=0.7)
    args = ap.parse_args()

    its = items_mod.load_items()
    n = len(its)
    out = P1 / "results" / "stage1b" / args.model
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "pairs_raw.json"

    h = harness.load(args.model)
    battery = PairBattery(h, its)

    recs = load_json(raw_path) if raw_path.exists() else []
    measured = {(r["i"], r["j"]) for r in recs}
    if not recs:
        p0 = adaptive.round0(its)
        print(f"round 0: {len(p0)} pairs")
        recs += battery.run(p0)
        measured |= set(p0)
        save_json(raw_path, recs)

    fit = fit_all(n, recs)
    for rnd in range(1, args.max_rounds + 1):
        if len(measured) >= args.target:
            break
        new_pairs = adaptive.select(fit, measured, n, budget=args.budget, seed=rnd)
        if not new_pairs:
            break
        recs += battery.run(new_pairs)
        measured |= set(new_pairs)
        save_json(raw_path, recs)
        prev = fit
        fit = fit_all(n, recs)
        dmu = sum(abs(a - b) for a, b in zip(fit["mu"], prev["mu"])) / n
        print(f"round {rnd}: +{len(new_pairs)} pairs (total {len(measured)}), "
              f"mean|dmu|={dmu:.4f}, nll={fit['nll']}")
        if dmu < 0.05:
            print("converged")
            break

    mu, s2 = fit["mu"], fit["sigma2"]
    lines = [f"{args.model}: {n} items, {len(measured)} pairs, {len(recs)} observations"]

    # gates: per-template and per-order mu refits (sigma2 fixed from full fit)
    mu_t = [thurstone.refit_mu(n, obs_from(recs, lambda r, t=t: r["template"] == t), s2)
            for t in range(3)]
    tspear = [spearman(mu_t[a], mu_t[b]) for a, b in [(0, 1), (0, 2), (1, 2)]]
    mu_o = [thurstone.refit_mu(n, obs_from(recs, lambda r, o=o: r["order"] == o), s2)
            for o in (0, 1)]
    ospear = spearman(mu_o[0], mu_o[1])
    gate_ok = min(tspear) >= args.gate
    lines.append(f"template-consistency spearman (T0/T1, T0/T2, T1/T2): "
                 + " ".join(f"{r:+.3f}" for r in tspear)
                 + f"   GATE {'PASS' if gate_ok else 'FAIL'} (min >= {args.gate})")
    lines.append(f"order-consistency spearman: {ospear:+.3f}")

    # confound audit: pooled surface covariates, then per-domain incl. designed axes
    X, names = items_mod.covariate_matrix(its)
    betas, r2 = stats.ols(mu, X, names)
    lines.append(f"confound audit (pooled, standardized OLS): r2={r2}  {betas}")
    for dom in sorted({it["domain"] for it in its}):
        idx = [k for k, it in enumerate(its) if it["domain"] == dom]
        numeric = sorted({k for it in (its[i] for i in idx) for k, v in it["tags"].items()
                          if isinstance(v, (int, float))})
        Xd = [[its[i]["tags"].get(k2, 0) for k2 in numeric] for i in idx]
        bd, r2d = stats.ols([mu[i] for i in idx], Xd, numeric)
        lines.append(f"  {dom} (n={len(idx)}): r2={r2d}  {bd}")

    # per-domain utility summary
    for dom in sorted({it["domain"] for it in its}):
        vals = [mu[k] for k, it in enumerate(its) if it["domain"] == dom]
        lines.append(f"  mean mu {dom}: {sum(vals)/len(vals):+.2f}  "
                     f"[{min(vals):+.2f}, {max(vals):+.2f}]")

    if args.model == "qwen25-7b":
        color_profile_check(its, mu, lines)

    n_pairs_item = defaultdict(int)
    for i, j in measured:
        n_pairs_item[i] += 1
        n_pairs_item[j] += 1
    save_json(out / "utilities.json", [
        {"id": it["id"], "domain": it["domain"], "text": it["text"],
         "mu": round(mu[k], 4), "sigma2": round(s2[k], 4),
         "n_pairs": n_pairs_item[k],
         "mu_by_template": [round(m[k], 4) for m in mu_t],
         "gate_pass": gate_ok, "tags": it["tags"]}
        for k, it in enumerate(its)])
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
