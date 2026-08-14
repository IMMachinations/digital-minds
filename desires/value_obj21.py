"""Valuation cross at L21 with the obj-centered vectors + random-vector control columns.

Run: python value_obj21.py   (GPU, ~60 min; after objects.py for BOTH modes and
value.py --mode inherent --stage full)

objects.py found centered A/B-free vectors give a genuine directional preference push,
cleanest at layer 21 coef 1 (where random-vector disruption dies). This runs the valuation
cross there with both modes' centered vectors_obj (modifier = their cleanest config;
inherent = continuity with prior valuation runs), plus two matched-norm random-vector
columns: rand_L21 (control for the new columns) and rand_L14 (retroactive control for the
earlier value_centered.py columns). Baseline rows reused from results/value_inherent/values.json.
Condition naming keeps value_analysis.py compatible: "{color}|objmod_L21x1.0" etc.; the rand
conditions carry no pipe so the bootstrap skips them.
"""
import argparse
import torch

from lib.data import COLORS
from lib.harness import load
from lib.io import save_json
from lib.paths import results_dir
from lib.steering import random_unit_matrix
from lib.stats import column_effect, condition_ml
from lib.value_data import make_items
from lib.valuation import base_rows, gen_rows, report, val_n_suffix

LAYER, COEF = 21, 1.0


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = results_dir("value_obj21")
    h = load()
    obj = {}
    for tag, mode in [("objmod", "modifier"), ("objinh", "inherent")]:
        saved = torch.load(results_dir(mode) / "vectors_obj.pt")
        obj[tag] = (saved["centered"], saved["resid_norms"])
    raw_norms = torch.load(results_dir("inherent") / "vectors.pt")["resid_norms"]
    rand_unit = random_unit_matrix(28, 3584)

    def vec_for(cond):
        if cond == "rand_L21":
            return 21, (rand_unit[21] * COEF * obj["objinh"][1][21]).to("cuda", torch.bfloat16)
        if cond == "rand_L14":
            return 14, (rand_unit[14] * COEF * raw_norms[14]).to("cuda", torch.bfloat16)
        color, tag = cond.split("|")[0], cond.split("|")[1].split("_")[0]
        vecs, norms = obj[tag]
        return LAYER, (vecs[color][LAYER] * COEF * norms[LAYER]).to("cuda", torch.bfloat16)

    items = make_items()
    n_suf = val_n_suffix(h.tok)
    conds = [f"{c}|objmod_L{LAYER}x{COEF}" for c in COLORS] \
          + [f"{c}|objinh_L{LAYER}x{COEF}" for c in COLORS] + ["rand_L21", "rand_L14"]
    rows = base_rows(items, results_dir("value_inherent") / "values.json")
    assert rows
    # Seed formula (historical): (condition index + 1) * 1000.
    for ci, cond in enumerate(conds):
        L, vec = vec_for(cond)
        rows += gen_rows(h, items, cond, (L, vec), (L, COEF), seed=(ci + 1) * 1000,
                         n_suffix=n_suf)
        print(f"condition {cond} done ({ci + 1}/{len(conds)})")
    save_json(out / "values.json", rows)

    for tag, rand_cond in [("objmod", "rand_L21"), ("objinh", "rand_L21")]:
        suffix = f"|{tag}_L{LAYER}x{COEF}"
        sub = [dict(r, condition=r["condition"].removesuffix(suffix).replace(rand_cond, "rand"))
               for r in rows if r["condition"] == "base" or r["condition"].endswith(suffix)
               or r["condition"] == rand_cond]
        print(f"\n######## {tag} centered, L{LAYER} x {COEF} (rand column = {rand_cond}) ########")
        report(sub, ["base"] + COLORS + ["rand"])

    print("\ncolumn effects (painting+household mean dlog10) vs random:")
    ml = condition_ml(rows)
    r21 = column_effect(ml, "rand_L21")
    r14 = column_effect(ml, "rand_L14")
    print(f"  rand_L21: {r21:+.3f}   rand_L14: {r14:+.3f} "
          "(compare rand_L14 with the value_centered columns in FINDINGS.md)")
    for tag in ["objmod", "objinh"]:
        cells = {c: column_effect(ml, f"{c}|{tag}_L{LAYER}x{COEF}") for c in COLORS}
        print(f"  {tag}: " + "  ".join(f"{c}:{v:+.2f}" for c, v in cells.items())
              + f"   (minus rand_L21: " + "  ".join(f"{c}:{v - r21:+.2f}" for c, v in cells.items()) + ")")
