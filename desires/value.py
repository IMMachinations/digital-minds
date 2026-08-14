"""Do color-preference vectors shift the model's dollar valuation of single colored items?

Run: python value.py --mode {modifier,inherent} --stage {inspect,pilot,full}
  (after preferences.py; vectors come from results/{mode}/vectors.pt; results land in
  results/value_{mode}/)

No comparison in the prompt: each item is valued alone ("...its value at $"), we sample
5 completions at temperature, parse the leading dollar amount, and average — under no
steering and under steering with each color's prefer-vector (full cross: blue paintings
steered with the red vector, blue vector, ...).

Reproducibility note: torch.manual_seed is set per generate-batch, so outputs are stable only
for fixed item order, GEN_BATCH, and n_samples.
"""
import argparse

import torch

from lib.data import COLORS
from lib.harness import load
from lib.io import save_json
from lib.paths import results_dir
from lib.steering import scaled_vec
from lib.value_data import make_items
from lib.valuation import make_rows, parse_dollars, report, val_n_suffix

# Pilot findings (see FINDINGS.md; the original pilot grids are pruned from results/):
# coef 2 collapses generation into ": A:" babble (the vectors carry the extraction prompts'
# answer-a-letter content, which A/B logit-diffs cancelled but free generation exposes);
# coef 1 mode-collapses values to $1000. Coefs 0.25/0.5 stay fully parseable with a milder
# uniform depression, so the full cross runs both: 0.25 minimizes disruption/floor effects,
# 0.5 pushes harder.
STEER_CONFIGS = [(14, 0.25), (14, 0.5)]  # (layer, coef)
PILOT_LAYERS, PILOT_COEFS, PILOT_N = [14, 18], [1.0, 2.0], 5
DOMAINS = ["painting", "household", "real"]


def run(h, out, vecs, resid_norms, n_suf, items, conditions, out_name):
    """conditions: [(name, steer_color | None, (layer, coef) | None)]. Generates per condition
    over all items, parses, saves every sample row, returns rows.
    Seed formula (historical, do not change): condition index * 1000."""
    rows = []
    for ci, (name, color, cfg) in enumerate(conditions):
        steer = None
        if cfg is not None:
            layer, coef = cfg
            steer = (layer, scaled_vec(vecs[color][layer], coef, resid_norms[layer]))
        samples = h.generate([it["prompt"] for it in items], steer=steer, seed=ci * 1000,
                             n_suffix=n_suf)
        rows += make_rows(items, samples, name, cfg)
        print(f"condition {name} done ({ci + 1}/{len(conditions)})")
    save_json(out / out_name, rows)
    return rows


# ---- Step 0: inspect raw sampled completions on a few examples ----------------------------------
def inspect(h, out):
    items = [it for d in DOMAINS for it in [i for i in make_items() if i["domain"] == d][:2]]
    samples = h.generate([it["prompt"] for it in items], n_samples=3, seed=0)
    lines = []
    for it, texts in zip(items, samples):
        lines.append(it["prompt"].replace("\n", " | "))
        for t in texts:
            lines.append(f"  {t!r} -> {parse_dollars(t)}")
    text = "\n".join(lines) + "\n"
    (out / "inspect.txt").write_text(text)
    print(text)


# ---- Pilot: matched-color steering only, small grid, to pick (layer, coef) ----------------------
def pilot(h, out, vecs, resid_norms, n_suf):
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = []
    # baseline
    rows += run(h, out, vecs, resid_norms, n_suf, items, [("base", None, None)],
                "pilot_base.json")
    # matched steering: each color's items under that color's own vector, per grid point.
    # Seed formula (historical): layer * 100 + int(coef * 10).
    for L in PILOT_LAYERS:
        for cf in PILOT_COEFS:
            name = f"L{L}x{cf}"
            for c in COLORS:
                samples = h.generate([it["prompt"] for it in by_color[c]],
                                     steer=(L, scaled_vec(vecs[c][L], cf, resid_norms[L])),
                                     seed=L * 100 + int(cf * 10), n_suffix=n_suf)
                rows += make_rows(by_color[c], samples, name, (L, cf))
            print(f"pilot {name} done")
    save_json(out / "pilot.json", rows)
    conds = ["base"] + [f"L{L}x{cf}" for L in PILOT_LAYERS for cf in PILOT_COEFS]
    report(rows, conds)


# ---- Full: baseline + all 7 steer colors on every item (cross-color) ----------------------------
def full(h, out, vecs, resid_norms, n_suf):
    items = make_items()
    conditions = [("base", None, None)]
    for L, cf in STEER_CONFIGS:
        conditions += [(f"{c}|L{L}x{cf}", c, (L, cf)) for c in COLORS]
    rows = run(h, out, vecs, resid_norms, n_suf, items, conditions, "values.json")
    for L, cf in STEER_CONFIGS:
        tag = f"|L{L}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## config L{L} x {cf} ########")
        report(sub, ["base"] + COLORS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="inherent")
    ap.add_argument("--stage", choices=["inspect", "pilot", "full"], default="full")
    args = ap.parse_args()

    out = results_dir(f"value_{args.mode}")
    h = load()
    saved = torch.load(results_dir(args.mode) / "vectors.pt")
    n_suf = val_n_suffix(h.tok)
    inspect(h, out)
    if args.stage == "pilot":
        pilot(h, out, saved["vecs"], saved["resid_norms"], n_suf)
    elif args.stage == "full":
        full(h, out, saved["vecs"], saved["resid_norms"], n_suf)
