"""Valuation cross at L21 with the obj-centered vectors + random-vector control columns.

Run: python value_controls.py inherent   (GPU, ~60 min)

The other session's objects.py found centered A/B-free vectors give a genuine directional
preference push, cleanest at layer 21 coef 1 (where random-vector disruption dies). This runs
the valuation cross there with both modes' centered vectors_obj (modifier = their cleanest
config; inherent = continuity with prior valuation runs), plus two matched-norm random-vector
columns: rand_L21 (control for the new columns) and rand_L14 (retroactive control for the
earlier value_centered.py columns). Baseline rows reused from results/value_inherent/values.json.
Condition naming keeps value_analysis.py compatible: "{color}|objmod_L21x1.0" etc.; the rand
conditions carry no pipe so the bootstrap skips them.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

import value as V
from data import COLORS
from value_data import make_items

RES = Path(__file__).parent / "results"
OUT = RES / "value_obj21"
OUT.mkdir(parents=True, exist_ok=True)

LAYER, COEF = 21, 1.0
OBJ = {}
for tag, mode in [("objmod", "modifier"), ("objinh", "inherent")]:
    saved = torch.load(RES / mode / "vectors_obj.pt")
    OBJ[tag] = (saved["centered"], saved["resid_norms"])
g = torch.Generator().manual_seed(0)
RAND_UNIT = torch.nn.functional.normalize(torch.randn(28, 3584, generator=g), dim=-1)


def vec_for(cond):
    if cond == "rand_L21":
        return 21, (RAND_UNIT[21] * COEF * OBJ["objinh"][1][21]).to("cuda", torch.bfloat16)
    if cond == "rand_L14":
        return 14, (RAND_UNIT[14] * COEF * V.RESID_NORMS[14]).to("cuda", torch.bfloat16)
    color, tag = cond.split("|")[0], cond.split("|")[1].split("_")[0]
    vecs, norms = OBJ[tag]
    return LAYER, (vecs[color][LAYER] * COEF * norms[LAYER]).to("cuda", torch.bfloat16)


def base_rows(items):
    keys = {(it["domain"], it["color"], it["item"]) for it in items}
    rows = json.load(open(V.OUT / "values.json"))
    return [r for r in rows if r["condition"] == "base"
            and (r["domain"], r["item_color"], r["item"]) in keys]


def col_effect(rows, cond):
    """Mean dlog10(per-item geomean) vs base over painting+household rows."""
    ml = defaultdict(list)
    for r in rows:
        if r["value"] and r["domain"] != "real":
            ml[(r["condition"], r["domain"], r["item_color"], r["item"])].append(math.log10(r["value"]))
    ml = {k: sum(v) / len(v) for k, v in ml.items()}
    ds = [ml[(cond, d, ic, it)] - ml[("base", d, ic, it)]
          for (c, d, ic, it) in ml if c == "base" and (cond, d, ic, it) in ml]
    return sum(ds) / len(ds) if ds else float("nan")


if __name__ == "__main__":
    items = make_items()
    conds = [f"{c}|objmod_L{LAYER}x{COEF}" for c in COLORS] \
          + [f"{c}|objinh_L{LAYER}x{COEF}" for c in COLORS] + ["rand_L21", "rand_L14"]
    rows = base_rows(items)
    assert rows
    for ci, cond in enumerate(conds):
        L, vec = vec_for(cond)
        samples = V.generate([it["prompt"] for it in items], steer=(L, vec), seed=(ci + 1) * 1000)
        rows += [dict(domain=it["domain"], item_color=it["color"], item=it["item"], condition=cond,
                      layer=L, coef=COEF, sample_idx=si, text=t, value=V.parse_dollars(t))
                 for it, texts in zip(items, samples) for si, t in enumerate(texts)]
        print(f"condition {cond} done ({ci + 1}/{len(conds)})")
    (OUT / "values.json").write_text(json.dumps(rows, indent=1))

    for tag, rand_cond in [("objmod", "rand_L21"), ("objinh", "rand_L21")]:
        suffix = f"|{tag}_L{LAYER}x{COEF}"
        sub = [dict(r, condition=r["condition"].removesuffix(suffix).replace(rand_cond, "rand"))
               for r in rows if r["condition"] == "base" or r["condition"].endswith(suffix)
               or r["condition"] == rand_cond]
        print(f"\n######## {tag} centered, L{LAYER} x {COEF} (rand column = {rand_cond}) ########")
        V.report(sub, ["base"] + COLORS + ["rand"])

    print("\ncolumn effects (painting+household mean dlog10) vs random:")
    r21 = col_effect(rows, "rand_L21")
    r14 = col_effect(rows, "rand_L14")
    print(f"  rand_L21: {r21:+.3f}   rand_L14: {r14:+.3f} "
          "(compare rand_L14 with the value_centered columns in VALUE.md)")
    for tag in ["objmod", "objinh"]:
        cells = {c: col_effect(rows, f"{c}|{tag}_L{LAYER}x{COEF}") for c in COLORS}
        print(f"  {tag}: " + "  ".join(f"{c}:{v:+.2f}" for c, v in cells.items())
              + f"   (minus rand_L21: " + "  ".join(f"{c}:{v - r21:+.2f}" for c, v in cells.items()) + ")")
