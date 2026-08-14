"""A/B-free preference measurement and steering vectors from object logits.

Run: python objects.py --mode {modifier,inherent}   (after preferences.py, for the
old-vs-new correlation)

Prompts present the two items inline with no A/B labels; preference is the teacher-forced
log-prob of each item as the continuation of "... Model: I prefer". New vectors (raw +
centered) are extracted from the new suffix span and validated with tier + random-vector
controls. Outputs: objects.json, vectors_obj.pt, objects_steer.json.
"""
import argparse

import torch

from lib.data import COLORS
from lib.harness import STEER_LAYERS, load
from lib.io import load_json, save_json
from lib.objtask import build_obj_comps, cont_logprob
from lib.paths import results_dir
from lib.steering import random_unit_per_color, scaled_vec
from lib.tiers import TIERS, signed, tier_slice


def measure(h, out, comps, n_suf):
    prompts = [c["prompt"] for c in comps for _ in range(2)]
    conts = [" " + c[k] for c in comps for k in ("item_a", "item_b")]
    m, s = cont_logprob(h, prompts, conts, n_suf)
    for j, c in enumerate(comps):
        c["lp_a"], c["lp_b"] = m[2 * j].item(), m[2 * j + 1].item()
        c["lp_a_sum"], c["lp_b_sum"] = s[2 * j].item(), s[2 * j + 1].item()
        c["diff"] = c["lp_a"] - c["lp_b"]  # mean log-prob diff, the primary score
    save_json(out / "objects.json", comps)

    old = load_json(out / "preferences.json")
    r = torch.corrcoef(torch.stack([torch.tensor([c["diff"] for c in comps]),
                                    torch.tensor([c["diff"] for c in old])]))[0, 1]
    print(f"\ncorr(object-logprob diff, old A/B logit diff) over 420 pairs = {r:+.3f}")
    print("\nPer-color mean signed preference (mean log-prob units):")
    for col in COLORS:
        v = torch.tensor([signed(c["diff"], c, col) for c in comps
                          if col in (c["color_a"], c["color_b"])]).mean()
        print(f"  {col:>8}: {v:+.4f}")


def extract(h, out, comps, n_suf):
    means = {}
    all_norms = []
    for color in COLORS:
        top = tier_slice(comps, color, 100, 120)  # strongest wins by the new measure
        acts, norms = h.suffix_acts([c["prompt"] for c in top], n_suffix=n_suf)
        means[color] = acts.mean(1)
        all_norms.append(norms)
    mu = torch.stack(list(means.values())).mean(0)
    vecs = {c: m / m.norm(dim=-1, keepdim=True) for c, m in means.items()}
    cent = {c: torch.nn.functional.normalize(m - mu, dim=-1) for c, m in means.items()}
    resid_norms = torch.stack(all_norms).mean(0)
    torch.save({"vecs": vecs, "centered": cent, "resid_norms": resid_norms},
               out / "vectors_obj.pt")
    return vecs, cent, resid_norms


def steer(h, out, comps, n_suf, mode, vecs, cent, resid_norms):
    rand = random_unit_per_color(vecs[COLORS[0]].shape, COLORS)
    sources = {"same": vecs, "cent": cent, "rand": rand}
    rows = []
    for tier, (lo, hi) in TIERS.items():
        for color in COLORS:
            tier_cs = tier_slice(comps, color, lo, hi)
            ps = [c["prompt"] for c in tier_cs for _ in range(2)]
            cs = [" " + c[k] for c in tier_cs for k in ("item_a", "item_b")]
            def signed_diff(m):
                return torch.tensor([signed((m[2 * j] - m[2 * j + 1]).item(), c, color)
                                     for j, c in enumerate(tier_cs)]).mean().item()
            base = signed_diff(cont_logprob(h, ps, cs, n_suf)[0])
            for src, vv in sources.items():
                for L in STEER_LAYERS:
                    for cf in [1.0, 2.0]:
                        vec = scaled_vec(vv[color][L], cf, resid_norms[L])
                        new = signed_diff(cont_logprob(h, ps, cs, n_suf, steer=(L, vec))[0])
                        rows.append(dict(tier=tier, source=src, color=color, layer=L, coef=cf,
                                         base=base, steered=new, delta=new - base))
        print(f"steered {tier}")
    save_json(out / "objects_steer.json", rows)

    for cf in [1.0, 2.0]:
        print(f"\n{mode} objects, coef {cf} — mean Δ(mean-logprob diff) toward steered color:")
        cols = [(s, t) for s in sources for t in TIERS]
        print("       " + "".join(f"{s[:1]}:{t:>8}"[:11].rjust(11) for s, t in cols))
        for L in STEER_LAYERS:
            vals = []
            for s, t in cols:
                rs = [r["delta"] for r in rows if (r["coef"], r["layer"], r["source"],
                                                   r["tier"]) == (cf, L, s, t)]
                vals.append(sum(rs) / len(rs))
            print(f"L{L:>4}  " + "".join(f"{v:11.3f}" for v in vals))
    bases = {t: sum(r["base"] for r in rows if r["tier"] == t) /
             max(1, sum(1 for r in rows if r["tier"] == t)) for t in TIERS}
    print("  unsteered base per tier: " + ", ".join(f"{t}={b:+.3f}" for t, b in bases.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    args = ap.parse_args()

    out = results_dir(args.mode)
    h = load()
    comps, n_suf = build_obj_comps(h, args.mode)
    # spot-check: the last k tokens of prompt+cont must decode back to the continuation
    for c in comps[:3]:
        ids = h.tok(c["prompt"] + " " + c["item_a"]).input_ids
        k = len(ids) - len(h.tok(c["prompt"]).input_ids)
        assert h.tok.decode(ids[-k:]) == " " + c["item_a"], c
    measure(h, out, comps, n_suf)
    steer(h, out, comps, n_suf, args.mode, *extract(h, out, comps, n_suf))
