"""Corrected A/B check for the RepE vectors: tier + random controls.

Run: python repe_controls.py inherent   (GPU, ~5 min)

VALUE.md's original RepE "sanity check" steered only each color's 20 *worst* pairs — the
design cross.py showed is confounded: random vectors compress existing diffs toward zero, so
worst-pair gains prove nothing. Here: steer each color's worst / neutral / best 20 pairs (by
baseline prefer diff) with the RepE vector vs a matched-norm random unit vector, L14,
coefs {0.5, 1.0}. A genuine preference direction moves *neutral* pairs beyond random.
"""
import json
from pathlib import Path

import torch

import experiment as E  # loads model; mode from argv[1] (use inherent)
from data import COLORS

RES = Path(__file__).parent / "results"
saved = torch.load(RES / "repe" / "vectors.pt")
RVECS, RESID_NORMS = saved["vecs"], saved["resid_norms"]
LAYER = 14
COEFS = [0.5, 1.0]
g = torch.Generator().manual_seed(0)
RAND = {c: torch.nn.functional.normalize(torch.randn(RVECS[c].shape, generator=g), dim=-1)
        for c in COLORS}
SOURCES = {"repe": RVECS, "rand": RAND}

comps = json.load(open(RES / "inherent" / "preferences.json"))


def tier_pairs(color, tier):
    mine = sorted((c for c in comps if color in (c["color_a"], c["color_b"])),
                  key=lambda c: c["diff"] if c["color_a"] == color else -c["diff"])
    n = len(mine)
    idx = {"worst": range(20), "neutral": range(n // 2 - 10, n // 2 + 10),
           "best": range(n - 20, n)}[tier]
    return [mine[i] for i in idx]


rows = []
for tier in ["worst", "neutral", "best"]:
    for color in COLORS:
        pairs = tier_pairs(color, tier)
        prompts = [c["prompt"] for c in pairs]
        sign = torch.tensor([1.0 if c["color_a"] == color else -1.0 for c in pairs])
        base = torch.tensor([c["diff"] for c in pairs]) * sign
        for src, vecs in SOURCES.items():
            for cf in COEFS:
                vec = (vecs[color][LAYER] * cf * RESID_NORMS[LAYER]).to("cuda", torch.bfloat16)
                sa, sb, _, _ = E.scores(E.last_logits(prompts, steer=(LAYER, vec)))
                delta = ((sa - sb) * sign - base).mean().item()
                rows.append(dict(tier=tier, color=color, source=src, coef=cf,
                                 base=base.mean().item(), delta=delta))
    print(f"tier {tier} done")
(RES / "repe" / "controls.json").write_text(json.dumps(rows, indent=1))

print(f"\nMean delta signed logit-diff toward steered color at L{LAYER} (7-color mean; "
      "base = unsteered tier mean):")
print(f"{'tier':>8} {'base':>7} | " + "  ".join(f"{s} c{cf}" for s in SOURCES for cf in COEFS))
for tier in ["worst", "neutral", "best"]:
    sel = [r for r in rows if r["tier"] == tier]
    base = sum(r["base"] for r in sel) / len(sel)
    cells = [sum(r["delta"] for r in sel if (r["source"], r["coef"]) == (s, cf)) / 7
             for s in SOURCES for cf in COEFS]
    print(f"{tier:>8} {base:>7.2f} | " + "  ".join(f"{v:+8.2f}" for v in cells))
