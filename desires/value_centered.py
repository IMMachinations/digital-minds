"""Valuation steering with mean-centered prefer-vectors: does the null go away?

Run: python value_centered.py --mode {modifier,inherent} --stage {pilot,full}
  (after value.py --mode <mode> --stage full; baseline rows are reused from it)

value.py showed the raw prefer-vectors (normalized mean activations) depress valuations
uniformly and carry no color-differential effect: their shared non-color component dominates.
Here each color's vector is centered against the across-color mean per layer and renormalized,
so steering pushes only the color-differential direction. Results in
results/value_{mode}_centered/ so value_analysis.py works with --target <mode>_centered.
"""
import argparse

import torch

from lib.data import COLORS
from lib.harness import load
from lib.io import save_json
from lib.paths import results_dir
from lib.steering import scaled_vec
from lib.value_data import make_items
from lib.valuation import base_rows, gen_rows, report, val_n_suffix

PILOT_COEFS = [0.5, 1.0, 2.0]
LAYER = 14
FULL_COEFS = [0.5, 1.0]  # confirmed/updated by the pilot
PILOT_N = 5


def center(vecs):
    raw = torch.stack([vecs[c] for c in COLORS])          # [7 colors, 28 layers, d_model]
    cent = raw - raw.mean(0, keepdim=True)
    frac = cent.norm(dim=-1) / raw.norm(dim=-1)           # size of the differential component
    cvecs = {c: (cent / cent.norm(dim=-1, keepdim=True))[i] for i, c in enumerate(COLORS)}
    print(f"differential fraction ||v - mean||/||v||: layer 14 mean {frac[:, 14].mean():.3f}, "
          f"all layers {frac.mean():.3f}")
    # NB: renormalizing to unit length means coef x resid_norm now pushes the pure differential
    # direction at full strength — a much larger differential push than the raw vectors gave.
    return cvecs


# ---- Pilot: matched-color steering only, coef sweep at one layer ---------------------------------
def pilot(h, out, base_path, cvecs, resid_norms, n_suf):
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = base_rows(items, base_path)
    assert rows, "run `python value.py --mode <mode> --stage full` first (baseline rows are reused)"
    # Seed formula (historical): layer * 100 + int(coef * 100).
    for cf in PILOT_COEFS:
        name = f"L{LAYER}x{cf}"
        for c in COLORS:
            steer = (LAYER, scaled_vec(cvecs[c][LAYER], cf, resid_norms[LAYER]))
            rows += gen_rows(h, by_color[c], name, steer, (LAYER, cf),
                             seed=LAYER * 100 + int(cf * 100), n_suffix=n_suf)
        print(f"pilot {name} done")
    save_json(out / "pilot.json", rows)
    report(rows, ["base"] + [f"L{LAYER}x{cf}" for cf in PILOT_COEFS])
    print("\nsample steered texts:")
    for cf in PILOT_COEFS:
        for r in [r for r in rows if r["condition"] == f"L{LAYER}x{cf}"][:3]:
            print(f"  [x{cf}][{r['domain']}/{r['item_color']}] {r['text']!r} -> {r['value']}")


# ---- Full: baseline + all 7 steer colors on every item (cross-color) -----------------------------
def full(h, out, base_path, cvecs, resid_norms, n_suf):
    items = make_items()
    rows = base_rows(items, base_path)
    assert rows
    # Seed formula (historical): (condition index + 1) * 1000, conditions in coef-major order.
    for ci, (cf, c) in enumerate((cf, c) for cf in FULL_COEFS for c in COLORS):
        steer = (LAYER, scaled_vec(cvecs[c][LAYER], cf, resid_norms[LAYER]))
        rows += gen_rows(h, items, f"{c}|L{LAYER}x{cf}", steer, (LAYER, cf),
                         seed=(ci + 1) * 1000, n_suffix=n_suf)
        print(f"condition {c}|L{LAYER}x{cf} done ({ci + 1}/{len(FULL_COEFS) * len(COLORS)})")
    save_json(out / "values.json", rows)
    for cf in FULL_COEFS:
        tag = f"|L{LAYER}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## centered L{LAYER} x {cf} ########")
        report(sub, ["base"] + COLORS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="inherent")
    ap.add_argument("--stage", choices=["pilot", "full"], default="full")
    args = ap.parse_args()

    out = results_dir(f"value_{args.mode}_centered")
    base_path = results_dir(f"value_{args.mode}") / "values.json"
    h = load()
    saved = torch.load(results_dir(args.mode) / "vectors.pt")
    cvecs = center(saved["vecs"])
    n_suf = val_n_suffix(h.tok)
    if args.stage == "pilot":
        pilot(h, out, base_path, cvecs, saved["resid_norms"], n_suf)
    else:
        full(h, out, base_path, cvecs, saved["resid_norms"], n_suf)
