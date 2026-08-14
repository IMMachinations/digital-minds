"""Analysis for the valuation experiments (no GPU needed).

Run: python value_analysis.py --target {inherent,inherent_centered,repe,obj21}
  (after the matching valuation run has produced results/value_{target}/values.json)
  1. paired bootstrap of the matched-vs-mismatched steering contrast per domain/config
  2. baseline valuation ordering vs the A/B-comparison preferences of both modes
"""
import argparse
import random
from collections import defaultdict

from lib.data import COLORS
from lib.io import load_json
from lib.paths import RESULTS, results_dir
from lib.stats import mean_log10_by, pearson, spearman

DOMAINS = ["painting", "household", "real"]


def pref_means(mode):
    """Per-color mean signed A/B logit-diff, recomputed from preferences.json."""
    comps = load_json(RESULTS / mode / "preferences.json")
    out = {}
    for col in COLORS:
        ds = [c["diff"] if c["color_a"] == col else -c["diff"] for c in comps
              if col in (c["color_a"], c["color_b"])]
        out[col] = sum(ds) / len(ds)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["inherent", "inherent_centered", "repe", "obj21"],
                    default="inherent")
    args = ap.parse_args()

    rows = load_json(results_dir(f"value_{args.target}") / "values.json")
    configs = sorted({r["condition"].split("|")[1] for r in rows if "|" in r["condition"]})

    # per-item mean log10 value per condition
    ml = mean_log10_by(rows, lambda r: (r["condition"], r["domain"], r["item_color"], r["item"]))

    def item_deltas(domain, tag, items):
        """(matched, mismatched) per-item log10 deltas vs baseline. Cells where no sample parsed
        to a positive value (e.g. the model answered $0 every time) are skipped."""
        matched, mism = [], []
        for ic, it in items:
            base = ml[("base", domain, ic, it)]
            for sc in COLORS:
                v = ml.get((f"{sc}|{tag}", domain, ic, it))
                if v is not None:
                    (matched if sc == ic else mism).append(v - base)
        return matched, mism

    rng = random.Random(0)
    print("Matched-vs-mismatched steering contrast (delta log10 geomean vs base, bootstrap over items):")
    for tag in configs:
        print(f"\n=== {tag} ===")
        for domain in DOMAINS:
            items = sorted({(ic, it) for (c, d, ic, it) in ml if d == domain and c == "base"})
            ma, mi = item_deltas(domain, tag, items)
            diffs = []
            for _ in range(2000):
                bma, bmi = item_deltas(domain, tag, rng.choices(items, k=len(items)))
                diffs.append(sum(bma) / len(bma) - sum(bmi) / len(bmi))
            diffs.sort()
            print(f"{domain:>10}: matched {sum(ma)/len(ma):+.3f}  mismatched {sum(mi)/len(mi):+.3f}  "
                  f"contrast {sum(ma)/len(ma)-sum(mi)/len(mi):+.3f} "
                  f"95%CI [{diffs[50]:+.3f},{diffs[1949]:+.3f}]")

    print("\nBaseline valuations (geomean $ by item color) and correlation with A/B preferences:")
    prefs = {m: pref_means(m) for m in ["modifier", "inherent"]
             if (RESULTS / m / "preferences.json").exists()}
    for domain in DOMAINS:
        per_color = defaultdict(list)
        for (c, d, ic, it), v in ml.items():
            if c == "base" and d == domain:
                per_color[ic].append(v)
        lv = [sum(per_color[c]) / len(per_color[c]) for c in COLORS]
        print(f"\n{domain:>10}: " + "  ".join(f"{c}:{10 ** v:.3g}" for c, v in zip(COLORS, lv)))
        for m, p in prefs.items():
            ps = [p[c] for c in COLORS]
            print(f"           vs {m:>8} pref: pearson(log$) {pearson(lv, ps):+.2f}  "
                  f"spearman {spearman(lv, ps):+.2f}   (n=7 colors)")
