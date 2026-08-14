"""Analysis for the valuation experiment (no GPU needed).

Run: python value_analysis.py [modifier|inherent]   (after value.py <mode> full)
  1. paired bootstrap of the matched-vs-mismatched steering contrast per domain/config
  2. baseline valuation ordering vs the A/B-comparison preferences of both modes
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from data import COLORS

MODE = sys.argv[1] if len(sys.argv) > 1 else "inherent"
RES = Path(__file__).parent / "results"
rows = json.load(open(RES / f"value_{MODE}" / "values.json"))
DOMAINS = ["painting", "household", "real"]
CONFIGS = sorted({r["condition"].split("|")[1] for r in rows if "|" in r["condition"]})

# per-item mean log10 value per condition
ml = defaultdict(list)
for r in rows:
    if r["value"]:
        ml[(r["condition"], r["domain"], r["item_color"], r["item"])].append(math.log10(r["value"]))
ml = {k: sum(v) / len(v) for k, v in ml.items()}


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
for tag in CONFIGS:
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


def pref_means(mode):
    """Per-color mean signed A/B logit-diff, recomputed from preferences.json."""
    comps = json.load(open(RES / mode / "preferences.json"))
    out = {}
    for col in COLORS:
        ds = [c["diff"] if c["color_a"] == col else -c["diff"] for c in comps
              if col in (c["color_a"], c["color_b"])]
        out[col] = sum(ds) / len(ds)
    return out


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    rk = lambda v: [sorted(v).index(a) for a in v]
    return pearson([float(r) for r in rk(x)], [float(r) for r in rk(y)])


print("\nBaseline valuations (geomean $ by item color) and correlation with A/B preferences:")
prefs = {m: pref_means(m) for m in ["modifier", "inherent"] if (RES / m / "preferences.json").exists()}
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
