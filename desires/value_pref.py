"""Valuation-as-preference: does f(a) > f(b) predict a > b?

Run: python value_pref.py   (no GPU)

Merges per-item valuations (results/value_items/values.json, sampled dollar estimates of all
700 inherent items) with both pairwise preference measures over the same 420 cross-color
pairs: the A/B letter-logit diff (results/inherent/preferences.json) and the object-logprob
diff (results/inherent/objects.json). Reports sign agreement, correlations, and an item-level
win-rate-vs-value check. Note the pairs are cross-color by design, so agreement includes the
color-level price differences documented in FINDINGS.md.
"""
import argparse
import math
from collections import defaultdict

from lib.io import load_json
from lib.paths import results_dir
from lib.stats import mean_log10_by, pearson, spearman

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = results_dir("value_pref")
    # per-item mean log10 value over positive parsed samples
    vals = mean_log10_by(load_json(results_dir("value_items") / "values.json"),
                         lambda r: r["item"])

    lines = []
    for name, fname in [("A/B letter-logit", "preferences.json"), ("object-logprob", "objects.json")]:
        comps = load_json(results_dir("inherent") / fname)
        pairs = [(vals[c["item_a"]] - vals[c["item_b"]], c["diff"]) for c in comps
                 if c["item_a"] in vals and c["item_b"] in vals and vals[c["item_a"]] != vals[c["item_b"]]]
        dv, dp = zip(*pairs)
        agree = sum((a > 0) == (b > 0) for a, b in pairs) / len(pairs)
        ci = 1.96 * math.sqrt(agree * (1 - agree) / len(pairs))
        lines.append(f"{name} preference vs valuation ({len(pairs)} of {len(comps)} pairs usable):")
        lines.append(f"  sign agreement P[(v_a>v_b)==(a>b)]: {agree:.3f} +/- {ci:.3f}")
        lines.append(f"  corr(dlog10 value, pref diff): pearson {pearson(dv, dp):+.3f}  "
                     f"spearman {spearman(dv, dp):+.3f}")

        # item level: win rate across an item's comparisons vs its value
        wins, tot = defaultdict(int), defaultdict(int)
        for c in comps:
            w = c["item_a"] if c["diff"] > 0 else c["item_b"]
            l = c["item_b"] if c["diff"] > 0 else c["item_a"]
            wins[w] += 1
            tot[w] += 1
            tot[l] += 1
        items = [it for it in tot if it in vals and tot[it] >= 2]
        wr = [wins[it] / tot[it] for it in items]
        lv = [vals[it] for it in items]
        lines.append(f"  item level ({len(items)} items with >=2 comparisons): "
                     f"corr(win rate, log10 value): pearson {pearson(wr, lv):+.3f}  "
                     f"spearman {spearman(wr, lv):+.3f}\n")

    text = "\n".join(lines)
    (out / "analysis.txt").write_text(text + "\n")
    print(text)
