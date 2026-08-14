"""Corrected A/B check for the RepE vectors: tier + random controls.

Run: python repe_controls.py   (GPU, ~5 min; after repe.py and preferences.py --mode inherent)

The original RepE "sanity check" steered only each color's 20 *worst* pairs — the design
cross.py showed is confounded: random vectors compress existing diffs toward zero, so
worst-pair gains prove nothing. Here: steer each color's worst / neutral / best 20 pairs (by
baseline prefer diff) with the RepE vector vs a matched-norm random unit vector, L14,
coefs {0.5, 1.0}. A genuine preference direction moves *neutral* pairs beyond random.
"""
import argparse
import torch

from lib.abtask import ab_scores, variant_ids
from lib.data import COLORS, SUFFIX
from lib.harness import count_suffix_tokens, load
from lib.io import load_json, save_json
from lib.paths import results_dir
from lib.steering import random_unit_per_color, scaled_vec
from lib.tiers import TIERS, tier_slice

LAYER = 14
COEFS = [0.5, 1.0]


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    h = load()
    saved = torch.load(results_dir("repe") / "vectors.pt")
    rvecs, resid_norms = saved["vecs"], saved["resid_norms"]
    rand = random_unit_per_color(rvecs[COLORS[0]].shape, COLORS)
    sources = {"repe": rvecs, "rand": rand}
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")

    comps = load_json(results_dir("inherent") / "preferences.json")
    n_suf = count_suffix_tokens(h.tok, comps[0]["prompt"],
                                comps[0]["prompt"].removesuffix(SUFFIX))

    rows = []
    for tier, (lo, hi) in TIERS.items():
        for color in COLORS:
            pairs = tier_slice(comps, color, lo, hi)
            prompts = [c["prompt"] for c in pairs]
            sign = torch.tensor([1.0 if c["color_a"] == color else -1.0 for c in pairs])
            base = torch.tensor([c["diff"] for c in pairs]) * sign
            for src, vecs in sources.items():
                for cf in COEFS:
                    vec = scaled_vec(vecs[color][LAYER], cf, resid_norms[LAYER])
                    sa, sb, _, _ = ab_scores(
                        h.last_logits(prompts, steer=(LAYER, vec), n_suffix=n_suf), a_ids, b_ids)
                    delta = ((sa - sb) * sign - base).mean().item()
                    rows.append(dict(tier=tier, color=color, source=src, coef=cf,
                                     base=base.mean().item(), delta=delta))
        print(f"tier {tier} done")
    save_json(results_dir("repe") / "controls.json", rows)

    print(f"\nMean delta signed logit-diff toward steered color at L{LAYER} (7-color mean; "
          "base = unsteered tier mean):")
    print(f"{'tier':>8} {'base':>7} | " + "  ".join(f"{s} c{cf}" for s in sources for cf in COEFS))
    for tier in TIERS:
        sel = [r for r in rows if r["tier"] == tier]
        base = sum(r["base"] for r in sel) / len(sel)
        cells = [sum(r["delta"] for r in sel if (r["source"], r["coef"]) == (s, cf)) / 7
                 for s in sources for cf in COEFS]
        print(f"{tier:>8} {base:>7.2f} | " + "  ".join(f"{v:+8.2f}" for v in cells))
