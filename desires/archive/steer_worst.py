"""[ARTIFACT] Steer each color's worst comparisons with its own prefer-vector.

Run: python -m archive.steer_worst --mode {modifier,inherent}   (from desires/, after
preferences.py)

This was Step 3 of the original experiment. Its positive deltas were later shown by cross.py
to be non-specific: matched-norm random vectors reproduce them at every magnitude, because any
large injection compresses the existing A-B readout toward zero and only worst pairs were
steered. Kept to reproduce results/{mode}/steering.json — see FINDINGS.md.
"""
import argparse

import torch

from lib.abtask import ABTask
from lib.data import COLORS
from lib.harness import COEFS, STEER_LAYERS, load
from lib.io import load_json, save_json
from lib.paths import results_dir
from lib.steering import scaled_vec
from lib.tiers import signed, worst_k


def steer(h, task, out, comps, vecs, resid_norms):
    rows = []
    for color in COLORS:
        worst = worst_k(comps, color)
        prompts = [c["prompt"] for c in worst]
        base = torch.tensor([signed(c["diff"], c, color) for c in worst])
        for L in STEER_LAYERS:
            for cf in COEFS:
                vec = scaled_vec(vecs[color][L], cf, resid_norms[L])
                sa, sb, _, _ = task.scores(h.last_logits(prompts, steer=(L, vec),
                                                         n_suffix=task.n_suffix))
                new = torch.tensor([signed((xa - xb).item(), c, color)
                                    for c, xa, xb in zip(worst, sa, sb)])
                rows.append(dict(color=color, layer=L, coef=cf, base=base.mean().item(),
                                 steered=new.mean().item(), delta=(new - base).mean().item()))
        print(f"steered {color}")
    save_json(out / "steering.json", rows)

    print("\nMean delta in signed logit-diff toward steered color (rows=layer, cols=coef):")
    print("       " + "".join(f"{cf:>8}" for cf in COEFS))
    for L in STEER_LAYERS:
        deltas = [torch.tensor([r["delta"] for r in rows if (r["layer"], r["coef"]) == (L, cf)]).mean()
                  for cf in COEFS]
        print(f"L{L:>4}  " + "".join(f"{d:8.2f}" for d in deltas))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    args = ap.parse_args()

    print("NOTE: this steering design is superseded — cross.py showed the effect is "
          "non-specific disruption (see FINDINGS.md). Running for the historical record.")
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")
    saved = torch.load(out / "vectors.pt")
    steer(h, task, out, comps, saved["vecs"], saved["resid_norms"])
