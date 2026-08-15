"""G5: re-validate the cheap 1C methods (logit rating, BWS) against the
anchored XL utilities at n≈4k — the convergence matrix at 20x scale.

Usage: uv run python stage1c_xl.py <roster-name>
Outputs results/stage1c_xl/<model>/{scores.json, summary.txt}.
"""
import argparse
import json
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import bws
import harness
import logit_rating


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()
    rows = load_json(P1 / "results" / "stage1x" / args.model / "utilities_xl.json")
    flagged = {f["id"] for f in json.loads((P1 / "items_xl" / "qc_flags.json").read_text())}
    mu = [r["mu"] for r in rows]
    out = P1 / "results" / "stage1c_xl" / args.model
    out.mkdir(parents=True, exist_ok=True)
    h = harness.load(args.model)

    rating_t = logit_rating.run(h, rows)
    rating = [sum(col) / len(col) for col in zip(*rating_t)]
    print("rating done")
    mu_bws, used = bws.run(h, rows, rounds=4)
    print(f"bws done ({used} usable tuples)")

    lines = [f"{args.model}: 1C-at-XL-scale convergence (n={len(rows)})"]
    for name, v in (("rating", rating), ("bws", mu_bws)):
        r, rho = pearson(v, mu), spearman(v, mu)
        keep = [k for k, row in enumerate(rows) if row["id"] not in flagged]
        rq = pearson([v[k] for k in keep], [mu[k] for k in keep])
        lines.append(f"  {name:7s} r={r:+.3f}  rho={rho:+.3f}  "
                     f"(QC-filtered r={rq:+.3f}, n={len(keep)})")
    stab = [pearson(t, mu) for t in rating_t]
    lines.append("  rating per-template r: " + " ".join(f"{x:+.3f}" for x in stab))
    save_json(out / "scores.json", [
        {"id": r["id"], "mu": r["mu"], "rating": round(rating[k], 4),
         "bws": round(mu_bws[k], 4)} for k, r in enumerate(rows)])
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
