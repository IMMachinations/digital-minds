"""Stage 1C driver: comparison-free valuation battery + convergence matrix.

Usage: uv run python stage1c.py <roster-name>

Runs SSR, utility probe, titration, logit rating, and BWS on the 1B item set,
then scores every method on the spec's three criteria: (i) correlation with 1B
utilities (probe is held-out by construction), (ii) paraphrase stability,
(iii) confound loadings below the method's own 1B correlation. Outputs under
results/stage1c/<name>/: scores.json, ssr_reactions.json, probe_acts.pt,
convergence.txt (the >=3-method convergence matrix).
Spec gate: at least one comparison-free method with r >= 0.5 vs 1B and
max |confound loading| < that correlation.
"""
import argparse
import sys
from pathlib import Path

import torch

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import bws
import harness
import items as items_mod
import logit_rating
import probes
import ssr
import stats
import titration


def stability(per_template):
    ts = list(per_template)
    rs = [pearson(ts[a], ts[b]) for a in range(len(ts)) for b in range(a + 1, len(ts))]
    return sum(rs) / len(rs) if rs else None


def mean_t(per_template):
    return [sum(col) / len(col) for col in zip(*per_template)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--ssr-samples", type=int, default=5)
    args = ap.parse_args()

    its = items_mod.load_items()
    ut = load_json(P1 / "results" / "stage1b" / args.model / "utilities.json")
    assert [r["id"] for r in ut] == [it["id"] for it in its], "item order mismatch vs 1B"
    y = [r["mu"] for r in ut]
    out = P1 / "results" / "stage1c" / args.model
    out.mkdir(parents=True, exist_ok=True)

    h = harness.load(args.model)
    per_template, scores = {}, {}

    per_template["rating"] = logit_rating.run(h, its)
    scores["rating"] = mean_t(per_template["rating"])
    print("rating done")

    per_template["titration"] = titration.run(h, its)
    scores["titration"] = mean_t(per_template["titration"])
    print("titration done")

    scores["bws"], used = bws.run(h, its)
    print(f"bws done ({used} usable tuples)")

    acts = probes.item_acts(h, its)
    torch.save(acts, out / "probe_acts.pt")
    preds, fold_layers, layer_scores = probes.utility_probe(acts, y)
    scores["probe"] = list(map(float, preds))
    print(f"probe done (fold layers {fold_layers})")

    texts = ssr.reactions(h, its, n_samples=args.ssr_samples)
    save_json(out / "ssr_reactions.json",
              [{"template": t, "id": its[k]["id"], "samples": texts[t][k]}
               for t in range(len(texts)) for k in range(len(its))])
    per_template["ssr"] = ssr.score_reactions(texts)
    scores["ssr"] = mean_t(per_template["ssr"])
    print("ssr done")

    # ---- convergence matrix -------------------------------------------------
    X, names = items_mod.covariate_matrix(its)
    lines = [f"{args.model}: 1C convergence matrix over {len(its)} items "
             f"(criteria: r vs 1B; paraphrase stability; confound loadings < r)"]
    comparison_free = ["ssr", "probe", "titration", "rating"]
    gate_hits = []
    for m in ["ssr", "probe", "titration", "rating", "bws"]:
        r = pearson(scores[m], y)
        rho = spearman(scores[m], y)
        stab = stability(per_template.get(m, []))
        betas, r2 = stats.ols(scores[m], X, names)
        max_load = max(abs(v) for v in betas.values())
        held = " (held-out)" if m == "probe" else ""
        ok = m in comparison_free and r >= 0.5 and max_load < r
        if ok:
            gate_hits.append(m)
        lines.append(f"  {m:10s} r={r:+.3f}{held}  rho={rho:+.3f}  "
                     f"stability={'n/a' if stab is None else f'{stab:+.3f}'}  "
                     f"confound r2={r2} max|load|={max_load:.3f}  {betas}"
                     + ("   GATE-OK" if ok else ""))
    lines.append(f"gate ({'PASS' if gate_hits else 'FAIL'}): comparison-free methods with "
                 f"r>=0.5 and max|confound load| < r: {gate_hits or 'none'}")
    lines.append(f"probe: fold layers {fold_layers}; train-CV r2 by layer (fold 0): {layer_scores}")

    methods = ["1B"] + ["ssr", "probe", "titration", "rating", "bws"]
    cols = {"1B": y, **scores}
    lines.append("method x method spearman:")
    lines.append("            " + "".join(f"{m:>10}" for m in methods))
    for a in methods:
        lines.append(f"  {a:10s}" + "".join(f"{spearman(cols[a], cols[b]):10.2f}" for b in methods))

    save_json(out / "scores.json", [
        {"id": it["id"], "domain": it["domain"], "mu_1b": y[k],
         **{m: round(float(scores[m][k]), 4) for m in scores},
         **{f"{m}_by_template": [round(float(v[k]), 4) for v in per_template[m]]
            for m in per_template}}
        for k, it in enumerate(its)])
    (out / "convergence.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
