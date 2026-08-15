"""RepE-style stated-preference steering vectors.

Subcommands (run `python repe.py <cmd> --help`):
  run       Extract the vectors and run the valuation stages.
            --stage {extract,pilot,full}; --sanity additionally runs the RETRACTED
            worst-pairs A/B check (see below).
            -> results/repe/vectors.pt, results/value_repe/
            (after `value.py cross --stage full`)
  controls  The corrected A/B check: tier + random-vector controls. (~5 min)
            -> results/repe/controls.json

Instead of averaging activations over prompts a color *won* (prefs.py measure), take paired
prompts that differ only in the stated preference — "you prefer blue" vs "you prefer red" —
and average the activation difference over the shared tokens that follow the statement.
vec(c) = mean over templates of [acts(c) - mean over other colors of acts(o)], unit-normalized
per layer.

The original worst-pairs "sanity check" was retracted: prefs.py cross showed that design is
confounded (random vectors compress existing diffs toward zero, so worst-pair gains prove
nothing). `run --sanity` reproduces it for the historical record only
(results/archive/repe_sanity.json); `controls` is the corrected check.
"""
import argparse

import torch

from lib.data import COLORS, SUFFIX as AB_SUFFIX
from lib.harness import count_suffix_tokens, load, random_unit_per_color, scaled_vec
from lib.tasks import TIERS, ab_scores, tier_slice, variant_ids, worst_k
from lib.util import load_json, results_dir, save_json
from lib.value_data import make_items
from lib.valuation import base_rows, gen_rows, report, val_n_suffix

STATEMENTS = [
    "System: You prefer the color {color} above every other color.",
    "System: Your favorite color is {color}.",
    "System: You love {color} things more than anything else.",
    "System: Of all the colors, you like {color} the most.",
]
SHARED = "\nModel: Understood. That preference will guide all of my choices."

LAYER = 14
SANITY_COEFS = [0.5, 1.0]
PILOT_COEFS = [0.5, 1.0, 2.0]
# Pilot: coef 2 degrades generation into warning/danger babble; 0.5 fluent, 1.0 mostly
# coherent (occasional "red flag, do not buy" style artifacts). Matched-only deltas noisy.
FULL_COEFS = [0.5, 1.0]
PILOT_N = 5


# ==== run: extraction + valuation stages =========================================================

def extract(h, out, raw_vecs):
    per_template = []
    all_norms = []
    for st in STATEMENTS:
        full_p = st.format(color=COLORS[0]) + SHARED
        n_suf = count_suffix_tokens(h.tok, full_p, st.format(color=COLORS[0]))
        acts, norms = h.suffix_acts([st.format(color=c) + SHARED for c in COLORS], n_suf)
        per_template.append(acts)          # [n_layers, 7, d_model]
        all_norms.append(norms)
    A = torch.stack(per_template)          # [n_templates, n_layers, 7, d_model]
    vecs = {}
    for i, c in enumerate(COLORS):
        others = [j for j in range(len(COLORS)) if j != i]
        diff = (A[:, :, i, :] - A[:, :, others, :].mean(2)).mean(0)   # [n_layers, d_model]
        vecs[c] = diff / diff.norm(dim=-1, keepdim=True)
    resid_norms = torch.stack(all_norms).mean(0)
    torch.save({"vecs": vecs, "resid_norms": resid_norms}, out / "vectors.pt")

    vs = torch.stack([vecs[c][LAYER] for c in COLORS])
    cos = vs @ vs.T
    off = cos[~torch.eye(7, dtype=bool)]
    print(f"RepE vectors: mean pairwise cosine at L{LAYER}: {off.mean():+.3f}")
    craw = torch.stack([raw_vecs[c] for c in COLORS])
    ccent = craw - craw.mean(0, keepdim=True)
    ccent = ccent / ccent.norm(dim=-1, keepdim=True)
    match = torch.tensor([float(vecs[c][LAYER] @ ccent[i][LAYER]) for i, c in enumerate(COLORS)])
    print("cosine(RepE, centered-inherent) per color at L14: "
          + "  ".join(f"{c}:{m:+.2f}" for c, m in zip(COLORS, match)))
    return vecs, resid_norms


def sanity(h, vecs, resid_norms):
    """RETRACTED worst-pairs-only A/B check (see module docstring). Writes the historical
    sanity rows to results/archive/repe_sanity.json; the corrected check is `controls`."""
    print("\nWARNING: this sanity check's design is retracted (see FINDINGS.md); its positive "
          "deltas are reproduced by random vectors. Running it for the historical record only.")
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    comps = load_json(results_dir("inherent") / "preferences.json")
    n_suf = count_suffix_tokens(h.tok, comps[0]["prompt"],
                                comps[0]["prompt"].removesuffix(AB_SUFFIX))
    print(f"\nA/B sanity check at L{LAYER} (mean delta signed logit-diff on 20 worst):")
    rows = []
    for color in COLORS:
        worst = worst_k(comps, color)
        prompts = [c["prompt"] for c in worst]
        sign = torch.tensor([1.0 if c["color_a"] == color else -1.0 for c in worst])
        base = torch.tensor([c["diff"] for c in worst]) * sign
        for cf in SANITY_COEFS:
            sa, sb, _, _ = ab_scores(
                h.last_logits(prompts, steer=(LAYER, scaled_vec(vecs[color][LAYER], cf,
                                                                resid_norms[LAYER])),
                              n_suffix=n_suf),
                a_ids, b_ids)
            delta = ((sa - sb) * sign - base).mean().item()
            rows.append(dict(color=color, coef=cf, base=base.mean().item(), delta=delta))
    save_json(results_dir("archive") / "repe_sanity.json", rows)
    for cf in SANITY_COEFS:
        ds = [r["delta"] for r in rows if r["coef"] == cf]
        per = "  ".join(f"{r['color']}:{r['delta']:+.1f}" for r in rows if r["coef"] == cf)
        print(f"  coef {cf}: mean {sum(ds) / len(ds):+.2f}   {per}")


def pilot(h, vout, base_path, vecs, resid_norms, n_suf):
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = base_rows(items, base_path)
    # Seed formula (historical): layer * 100 + int(coef * 100).
    for cf in PILOT_COEFS:
        name = f"L{LAYER}x{cf}"
        for c in COLORS:
            steer = (LAYER, scaled_vec(vecs[c][LAYER], cf, resid_norms[LAYER]))
            rows += gen_rows(h, by_color[c], name, steer, (LAYER, cf),
                             seed=LAYER * 100 + int(cf * 100), n_suffix=n_suf)
        print(f"pilot {name} done")
    save_json(vout / "pilot.json", rows)
    report(rows, ["base"] + [f"L{LAYER}x{cf}" for cf in PILOT_COEFS])
    print("\nsample steered texts:")
    for cf in PILOT_COEFS:
        for r in [r for r in rows if r["condition"] == f"L{LAYER}x{cf}"][:3]:
            print(f"  [x{cf}][{r['domain']}/{r['item_color']}] {r['text']!r} -> {r['value']}")


def full(h, vout, base_path, vecs, resid_norms, n_suf):
    items = make_items()
    rows = base_rows(items, base_path)
    # Seed formula (historical): (condition index + 1) * 1000, conditions in coef-major order.
    for ci, (cf, c) in enumerate((cf, c) for cf in FULL_COEFS for c in COLORS):
        steer = (LAYER, scaled_vec(vecs[c][LAYER], cf, resid_norms[LAYER]))
        rows += gen_rows(h, items, f"{c}|L{LAYER}x{cf}", steer, (LAYER, cf),
                         seed=(ci + 1) * 1000, n_suffix=n_suf)
        print(f"condition {c}|L{LAYER}x{cf} done ({ci + 1}/{len(FULL_COEFS) * len(COLORS)})")
    save_json(vout / "values.json", rows)
    for cf in FULL_COEFS:
        tag = f"|L{LAYER}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## repe L{LAYER} x {cf} ########")
        report(sub, ["base"] + COLORS)


def cmd_run(args):
    out = results_dir("repe")
    vout = results_dir("value_repe")
    base_path = results_dir("value_inherent") / "values.json"
    h = load()
    raw = torch.load(results_dir("inherent") / "vectors.pt")["vecs"]
    vecs, resid_norms = extract(h, out, raw)
    if args.sanity:
        sanity(h, vecs, resid_norms)
    n_suf = val_n_suffix(h.tok)
    if args.stage == "pilot":
        pilot(h, vout, base_path, vecs, resid_norms, n_suf)
    elif args.stage == "full":
        full(h, vout, base_path, vecs, resid_norms, n_suf)


# ==== controls: the corrected A/B check ==========================================================

CTRL_COEFS = [0.5, 1.0]


def cmd_controls(args):
    h = load()
    saved = torch.load(results_dir("repe") / "vectors.pt")
    rvecs, resid_norms = saved["vecs"], saved["resid_norms"]
    rand = random_unit_per_color(rvecs[COLORS[0]].shape, COLORS)
    sources = {"repe": rvecs, "rand": rand}
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")

    comps = load_json(results_dir("inherent") / "preferences.json")
    n_suf = count_suffix_tokens(h.tok, comps[0]["prompt"],
                                comps[0]["prompt"].removesuffix(AB_SUFFIX))

    rows = []
    for tier, (lo, hi) in TIERS.items():
        for color in COLORS:
            pairs = tier_slice(comps, color, lo, hi)
            prompts = [c["prompt"] for c in pairs]
            sign = torch.tensor([1.0 if c["color_a"] == color else -1.0 for c in pairs])
            base = torch.tensor([c["diff"] for c in pairs]) * sign
            for src, vecs in sources.items():
                for cf in CTRL_COEFS:
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
    print(f"{'tier':>8} {'base':>7} | " + "  ".join(f"{s} c{cf}" for s in sources for cf in CTRL_COEFS))
    for tier in TIERS:
        sel = [r for r in rows if r["tier"] == tier]
        base = sum(r["base"] for r in sel) / len(sel)
        cells = [sum(r["delta"] for r in sel if (r["source"], r["coef"]) == (s, cf)) / 7
                 for s in sources for cf in CTRL_COEFS]
        print(f"{tier:>8} {base:>7.2f} | " + "  ".join(f"{v:+8.2f}" for v in cells))


# ==== CLI ========================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="extract vectors + valuation stages")
    p.add_argument("--stage", choices=["extract", "pilot", "full"], default="full")
    p.add_argument("--sanity", action="store_true",
                   help="also run the retracted worst-pairs A/B check (historical record only)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("controls", help="corrected A/B check: tier + random controls")
    p.set_defaults(fn=cmd_controls)

    args = ap.parse_args()
    args.fn(args)
