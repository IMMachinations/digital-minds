"""Cross-mode vector transfer + steering across example tiers (worst/neutral/best).

Run: python cross.py --mode {modifier,inherent} [--coefs 1.0,2.0] [--tag ""]
  (after preferences.py has run for BOTH modes)

Steers this mode's comparisons with its own vectors ("same"), the other mode's vectors
("cross"), and matched-norm random vectors ("rand"), on three tiers of examples ranked by
baseline prefer diff. This is the control that showed the original steering effect was
non-specific: random vectors reproduce it, and neutral pairs never move.
Saves results/{mode}/cross{tag}.json (use --coefs 0.25,0.5 --tag _lo for the low-coef grid).
"""
import argparse

import torch

from lib.abtask import ABTask, FRAMINGS, reframe
from lib.data import COLORS
from lib.harness import STEER_LAYERS, load
from lib.io import load_json, save_json
from lib.paths import results_dir
from lib.steering import random_unit_per_color, scaled_vec
from lib.tiers import TIERS, tier_slice


def main(args):
    out = results_dir(args.mode)
    other = {"modifier": "inherent", "inherent": "modifier"}[args.mode]
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")
    saved = torch.load(out / "vectors.pt")
    vecs, resid_norms = saved["vecs"], saved["resid_norms"]
    coefs = [float(x) for x in args.coefs.split(",")]

    rand = random_unit_per_color(vecs[COLORS[0]].shape, COLORS)
    sources = {"same": vecs,
               "cross": torch.load(results_dir(other) / "vectors.pt")["vecs"],
               "rand": rand}
    framings = {k: FRAMINGS[k] for k in ("prefer", "worse")}

    rows = []
    for fname, suffix in framings.items():
        n_suf = task.framing_n_suffix(suffix)
        for tier, (lo, hi) in TIERS.items():
            for color in COLORS:
                tier_cs = tier_slice(comps, color, lo, hi)
                wp = [reframe(c, suffix) for c in tier_cs]
                base = task.sides(h.last_logits(wp), tier_cs, color)
                for src, vv in sources.items():
                    for L in STEER_LAYERS:
                        for cf in coefs:
                            vec = scaled_vec(vv[color][L], cf, resid_norms[L])
                            st = task.sides(h.last_logits(wp, steer=(L, vec), n_suffix=n_suf),
                                            tier_cs, color)
                            rows.append(dict(
                                framing=fname, tier=tier, source=src, color=color, layer=L,
                                coef=cf,
                                base=base["own"] - base["opp"], steered=st["own"] - st["opp"],
                                delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                                d_own=st["own"] - base["own"], d_opp=st["opp"] - base["opp"]))
            print(f"done {fname}/{tier}")
    save_json(out / f"cross{args.tag}.json", rows)

    def cell(rs):
        return sum(r["delta"] for r in rs) / len(rs)

    for fname in framings:
        for cf in coefs:
            print(f"\n{args.mode} / {fname} framing, coef {cf} — mean Δdiff toward steered "
                  "color (cols: vector source x example tier):")
            cols = [(s, t) for s in sources for t in TIERS]
            print("       " + "".join(f"{s[:1]}:{t:>8}"[:11].rjust(11) for s, t in cols))
            for L in STEER_LAYERS:
                sel = lambda s, t: [r for r in rows if (r["framing"], r["coef"], r["layer"],
                                                        r["source"], r["tier"]) == (fname, cf, L, s, t)]
                print(f"L{L:>4}  " + "".join(f"{cell(sel(s, t)):11.2f}" for s, t in cols))
            bases = {t: sum(r["base"] for r in rows if (r["framing"], r["tier"]) == (fname, t))
                     / sum(1 for r in rows if (r["framing"], r["tier"]) == (fname, t)) for t in TIERS}
            print("  unsteered base per tier: " +
                  ", ".join(f"{t}={b:+.2f}" for t, b in bases.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    ap.add_argument("--coefs", default="1.0,2.0")
    ap.add_argument("--tag", default="", help='result-file suffix, e.g. "_lo" -> cross_lo.json')
    main(ap.parse_args())
