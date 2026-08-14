"""Cross-mode vector transfer + steering across example tiers (worst/neutral/best).

Run: python cross.py [modifier|inherent] — steers this mode's comparisons with both its own
vectors ("same") and the other mode's vectors ("cross"), on three tiers of examples ranked by
baseline prefer diff. Saves results/{mode}/cross.json.
"""
import json
import os

import torch

import experiment as E  # loads model; picks up mode from sys.argv[1]
import flip as F
from data import COLORS

OTHER = {"modifier": "inherent", "inherent": "modifier"}[E.MODE]
g = torch.Generator().manual_seed(0)  # random-direction control at matched norm
RAND = {c: torch.nn.functional.normalize(
    torch.randn(F.VECS[c].shape, generator=g), dim=-1) for c in COLORS}
SOURCES = {"same": F.VECS, "cross": torch.load(E.OUT.parent / OTHER / "vectors.pt")["vecs"],
           "rand": RAND}
TIERS = {"worst": (0, 20), "neutral": (50, 70), "best": (100, 120)}  # of 120 pairs per color
FRAMINGS = {k: F.FRAMINGS[k] for k in ("prefer", "worse")}
COEFS = [float(x) for x in os.environ.get("CROSS_COEFS", "1.0,2.0").split(",")]
TAG = os.environ.get("CROSS_TAG", "")  # e.g. "_lo" to write cross_lo.json


def tier_comps(color, lo, hi):
    mine = [c for c in F.COMPS if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: E.signed(c["diff"], c, color))[lo:hi]


rows = []
for fname, suffix in FRAMINGS.items():
    n_suf = len(E.tok(F.reframe(F.COMPS[0], suffix)).input_ids) - \
        len(E.tok(F.reframe(F.COMPS[0], "")).input_ids)
    for tier, (lo, hi) in TIERS.items():
        for color in COLORS:
            comps = tier_comps(color, lo, hi)
            wp = [F.reframe(c, suffix) for c in comps]
            base = F.sides(E.last_logits(wp), comps, color)
            for src, vecs in SOURCES.items():
                for L in E.STEER_LAYERS:
                    for cf in COEFS:
                        vec = (vecs[color][L] * cf * F.RESID_NORMS[L]).to("cuda", torch.bfloat16)
                        st = F.sides(E.last_logits(wp, steer=(L, vec), n_suffix=n_suf),
                                     comps, color)
                        rows.append(dict(
                            framing=fname, tier=tier, source=src, color=color, layer=L, coef=cf,
                            base=base["own"] - base["opp"], steered=st["own"] - st["opp"],
                            delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                            d_own=st["own"] - base["own"], d_opp=st["opp"] - base["opp"]))
        print(f"done {fname}/{tier}")
(E.OUT / f"cross{TAG}.json").write_text(json.dumps(rows, indent=1))


def cell(rs):
    return sum(r["delta"] for r in rs) / len(rs)

for fname in FRAMINGS:
    for cf in COEFS:
        print(f"\n{E.MODE} / {fname} framing, coef {cf} — mean Δdiff toward steered color "
              "(cols: vector source x example tier):")
        cols = [(s, t) for s in SOURCES for t in TIERS]
        print("       " + "".join(f"{s[:1]}:{t:>8}"[:11].rjust(11) for s, t in cols))
        for L in E.STEER_LAYERS:
            sel = lambda s, t: [r for r in rows if (r["framing"], r["coef"], r["layer"],
                                                    r["source"], r["tier"]) == (fname, cf, L, s, t)]
            print(f"L{L:>4}  " + "".join(f"{cell(sel(s, t)):11.2f}" for s, t in cols))
        bases = {t: sum(r["base"] for r in rows if (r["framing"], r["tier"]) == (fname, t))
                 / sum(1 for r in rows if (r["framing"], r["tier"]) == (fname, t)) for t in TIERS}
        print("  unsteered base per tier: " +
              ", ".join(f"{t}={b:+.2f}" for t, b in bases.items()))
