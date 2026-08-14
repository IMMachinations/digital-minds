"""Price statistics of the original inherent items.

Run: python value_items.py   (GPU; ~6 min; after value_centered.py --mode inherent --stage full)

Tests the price-knob hypothesis: the centered prefer-vectors act as global price knobs
because each color's direction inherits the price statistics of that color's inherent items
(sapphires and night skies vs. vegetables). We value all 700 INHERENT items directly (baseline,
no steering), then correlate per-color average log-values — over the full 100-item pool and
over the top-K=20 extraction items the vectors were actually built from — with the centered
column effects measured in results/value_inherent_centered/values.json.
"""
import argparse
from lib.data import COLORS, INHERENT
from lib.io import load_json, save_json
from lib.harness import load
from lib.paths import results_dir
from lib.stats import column_effect, condition_ml, mean_log10_by, pearson, spearman
from lib.tiers import K, top_wins
from lib.value_data import SUFFIX, TEMPLATE
from lib.valuation import parse_dollars


def extraction_items(color, comps):
    """Winning-side items of the top-K comparisons `color` won most strongly."""
    return [c["item_a"] if c["color_a"] == color else c["item_b"]
            for c in top_wins(comps, color, K)]


def column_effects(coef_tag):
    """Mean dlog10(per-item geomean) per steer color over painting+household rows."""
    ml = condition_ml(load_json(results_dir("value_inherent_centered") / "values.json"))
    return {sc: column_effect(ml, f"{sc}|{coef_tag}") for sc in COLORS}


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = results_dir("value_items")
    h = load()
    items = [(c, it) for c in COLORS for it in INHERENT[c]]
    samples = h.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for _, it in items], seed=0)
    rows = [dict(color=c, item=it, sample_idx=si, text=t, value=parse_dollars(t))
            for (c, it), texts in zip(items, samples) for si, t in enumerate(texts)]
    save_json(out / "values.json", rows)

    # per-item mean log10 over positive parsed samples
    ml = mean_log10_by(rows, lambda r: (r["color"], r["item"]))

    comps = load_json(results_dir("inherent") / "preferences.json")
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
    (out / "analysis.txt").write_text(text)
    print(text)
