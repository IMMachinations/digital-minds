"""Valuation steering with mean-centered prefer-vectors: does the null go away?

Run: python value_centered.py [modifier|inherent] [pilot|full]   (after value.py <mode> full)

value.py showed the raw prefer-vectors (normalized mean activations) depress valuations
uniformly and carry no color-differential effect: their shared non-color component dominates.
Here each color's vector is centered against the across-color mean per layer and renormalized,
so steering pushes only the color-differential direction. Results in
results/value_{mode}_centered/ so value_analysis.py works with mode "<mode>_centered".
"""
import json
import sys
from pathlib import Path

import torch

import value as V  # loads model + raw vectors; V.STAGE is our sys.argv[2] too
from data import COLORS
from value_data import make_items

OUT = Path(__file__).parent / "results" / f"value_{V.E.MODE}_centered"
OUT.mkdir(parents=True, exist_ok=True)

_raw = torch.stack([V.VECS[c] for c in COLORS])          # [7 colors, 28 layers, d_model]
_cent = _raw - _raw.mean(0, keepdim=True)
_frac = _cent.norm(dim=-1) / _raw.norm(dim=-1)           # size of the differential component
CVECS = {c: (_cent / _cent.norm(dim=-1, keepdim=True))[i] for i, c in enumerate(COLORS)}
print(f"differential fraction ||v - mean||/||v||: layer 14 mean {_frac[:, 14].mean():.3f}, "
      f"all layers {_frac.mean():.3f}")
# NB: renormalizing to unit length means coef x resid_norm now pushes the pure differential
# direction at full strength — a much larger differential push than the raw vectors gave.

PILOT_COEFS = [0.5, 1.0, 2.0]
LAYER = 14
FULL_COEFS = [0.5, 1.0]  # confirmed/updated by the pilot


def csteer_vec(color, layer, coef):
    return (CVECS[color][layer] * coef * V.RESID_NORMS[layer]).to("cuda", torch.bfloat16)


def gen_rows(items, name, color, cfg, seed):
    """Generate for `items` under one condition and return sample rows (value.py row format)."""
    steer = None if cfg is None else (cfg[0], csteer_vec(color, cfg[0], cfg[1]))
    samples = V.generate([it["prompt"] for it in items], steer=steer, seed=seed)
    return [dict(domain=it["domain"], item_color=it["color"], item=it["item"], condition=name,
                 layer=cfg[0] if cfg else None, coef=cfg[1] if cfg else None, sample_idx=si,
                 text=t, value=V.parse_dollars(t))
            for it, texts in zip(items, samples) for si, t in enumerate(texts)]


def base_rows(items):
    """Baseline rows reused from the raw-vector run (unsteered => identical condition)."""
    keys = {(it["domain"], it["color"], it["item"]) for it in items}
    rows = json.load(open(V.OUT / "values.json"))
    return [r for r in rows if r["condition"] == "base"
            and (r["domain"], r["item_color"], r["item"]) in keys]


# ---- Pilot: matched-color steering only, coef sweep at one layer ---------------------------------
def pilot():
    items = make_items(n_per_color=V.PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = base_rows(items)
    assert rows, "run `python value.py <mode> full` first (baseline rows are reused from it)"
    for cf in PILOT_COEFS:
        name = f"L{LAYER}x{cf}"
        for c in COLORS:
            rows += gen_rows(by_color[c], name, c, (LAYER, cf), seed=LAYER * 100 + int(cf * 100))
        print(f"pilot {name} done")
    (OUT / "pilot.json").write_text(json.dumps(rows, indent=1))
    V.report(rows, ["base"] + [f"L{LAYER}x{cf}" for cf in PILOT_COEFS])
    print("\nsample steered texts:")
    for cf in PILOT_COEFS:
        for r in [r for r in rows if r["condition"] == f"L{LAYER}x{cf}"][:3]:
            print(f"  [x{cf}][{r['domain']}/{r['item_color']}] {r['text']!r} -> {r['value']}")


# ---- Full: baseline + all 7 steer colors on every item (cross-color) -----------------------------
def full():
    items = make_items()
    rows = base_rows(items)
    assert rows
    for ci, (cf, c) in enumerate((cf, c) for cf in FULL_COEFS for c in COLORS):
        rows += gen_rows(items, f"{c}|L{LAYER}x{cf}", c, (LAYER, cf), seed=(ci + 1) * 1000)
        print(f"condition {c}|L{LAYER}x{cf} done ({ci + 1}/{len(FULL_COEFS) * len(COLORS)})")
    (OUT / "values.json").write_text(json.dumps(rows, indent=1))
    for cf in FULL_COEFS:
        tag = f"|L{LAYER}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## centered L{LAYER} x {cf} ########")
        V.report(sub, ["base"] + COLORS)


if __name__ == "__main__":
    if V.STAGE == "pilot":
        pilot()
    else:
        full()
