"""Price statistics of the original inherent items.

Run: python value_items.py inherent   (GPU; ~6 min)

Tests the hypothesis from VALUE.md: the centered prefer-vectors act as global price knobs
because each color's direction inherits the price statistics of that color's inherent items
(sapphires and night skies vs. vegetables). We value all 700 INHERENT items directly (baseline,
no steering), then correlate per-color average log-values — over the full 100-item pool and
over the top-K=20 extraction items the vectors were actually built from — with the centered
column effects measured in results/value_inherent_centered/values.json.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import value as V  # loads model; run with mode "inherent" so V.OUT matches the raw run
from data import COLORS, INHERENT
from value_data import TEMPLATE, SUFFIX

OUT = Path(__file__).parent / "results" / "value_items"
OUT.mkdir(parents=True, exist_ok=True)
RES = Path(__file__).parent / "results"
K = 20  # experiment.py's extraction K


def extraction_items(color, comps):
    """Winning-side items of the top-K comparisons `color` won most strongly (comps_for logic)."""
    mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
    mine.sort(key=lambda c: c["diff"] if c["color_a"] == color else -c["diff"], reverse=True)
    return [c["item_a"] if c["color_a"] == color else c["item_b"] for c in mine[:K]]


def column_effects(coef_tag):
    """Mean dlog10(per-item geomean) per steer color over painting+household rows."""
    rows = json.load(open(RES / "value_inherent_centered" / "values.json"))
    ml = defaultdict(list)
    for r in rows:
        if r["value"] and r["domain"] != "real":
            ml[(r["condition"], r["domain"], r["item_color"], r["item"])].append(math.log10(r["value"]))
    ml = {k: sum(v) / len(v) for k, v in ml.items()}
    out = {}
    for sc in COLORS:
        ds = [ml[(f"{sc}|{coef_tag}", d, ic, it)] - ml[("base", d, ic, it)]
              for (c, d, ic, it) in ml if c == "base" and (f"{sc}|{coef_tag}", d, ic, it) in ml]
        out[sc] = sum(ds) / len(ds)
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    rk = lambda v: [float(sorted(v).index(a)) for a in v]
    return pearson(rk(x), rk(y))


if __name__ == "__main__":
    items = [(c, it) for c in COLORS for it in INHERENT[c]]
    samples = V.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for _, it in items], seed=0)
    rows = [dict(color=c, item=it, sample_idx=si, text=t, value=V.parse_dollars(t))
            for (c, it), texts in zip(items, samples) for si, t in enumerate(texts)]
    (OUT / "values.json").write_text(json.dumps(rows, indent=1))

    # per-item mean log10 over positive parsed samples
    ml = defaultdict(list)
    for r in rows:
        if r["value"]:
            ml[(r["color"], r["item"])].append(math.log10(r["value"]))
    ml = {k: sum(v) / len(v) for k, v in ml.items()}

    comps = json.load(open(RES / "inherent" / "preferences.json"))
    pool = {c: [ml[(c, it)] for it in INHERENT[c] if (c, it) in ml] for c in COLORS}
    extr = {c: [ml[(c, it)] for it in extraction_items(c, comps) if (c, it) in ml] for c in COLORS}
    cols = {tag: column_effects(tag) for tag in ["L14x0.5", "L14x1.0"]}

    lines = [f"parsed items: {len(ml)}/700  (unparseable samples: "
             f"{sum(1 for r in rows if r['value'] is None)}/{len(rows)})",
             "\nper-color item price statistics (log10 $) vs centered column effects:",
             f"{'color':>8} {'pool mean':>10} {'pool geo$':>10} {'extr mean':>10} "
             f"{'extr geo$':>10} {'col@0.5':>8} {'col@1.0':>8}"]
    for c in COLORS:
        pm, em = sum(pool[c]) / len(pool[c]), sum(extr[c]) / len(extr[c])
        lines.append(f"{c:>8} {pm:>10.2f} {10 ** pm:>10.3g} {em:>10.2f} {10 ** em:>10.3g} "
                     f"{cols['L14x0.5'][c]:>+8.2f} {cols['L14x1.0'][c]:>+8.2f}")
    lines.append("\ncorrelations with column effect (n=7 colors, indicative only):")
    for tag in ["L14x0.5", "L14x1.0"]:
        ce = [cols[tag][c] for c in COLORS]
        for name, stat in [("pool", pool), ("extraction", extr)]:
            xs = [sum(stat[c]) / len(stat[c]) for c in COLORS]
            lines.append(f"  {name:>10} mean log10 vs {tag}: pearson {pearson(xs, ce):+.2f}  "
                         f"spearman {spearman(xs, ce):+.2f}")
    text = "\n".join(lines) + "\n"
    (OUT / "analysis.txt").write_text(text)
    print(text)
