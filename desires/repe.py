"""RepE-style stated-preference steering vectors.

Run: python repe.py inherent [sanity|pilot|full]   (after value.py inherent full)

Instead of averaging activations over prompts a color *won* (experiment.py), take paired
prompts that differ only in the stated preference — "you prefer blue" vs "you prefer red" —
and average the activation difference over the shared tokens that follow the statement.
vec(c) = mean over templates of [acts(c) - mean over other colors of acts(o)], unit-normalized
per layer. Saved to results/repe/vectors.pt (same schema as experiment.py's vectors.pt).

Stages: sanity = extract + A/B-task steering check only (cheap, no generation);
pilot / full = the valuation stages of value_centered.py with these vectors
(baseline rows reused from results/value_inherent/values.json; results in results/value_repe/).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

import value as V
from data import COLORS
from value_data import make_items

E = V.E
OUT = Path(__file__).parent / "results" / "repe"
OUT.mkdir(parents=True, exist_ok=True)
VOUT = Path(__file__).parent / "results" / "value_repe"
VOUT.mkdir(parents=True, exist_ok=True)

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


@torch.no_grad()
def capture(prompts, n_suffix):
    """Mean residual activation over the last n_suffix tokens: [n_layers, N, d_model], plus
    per-layer mean token norm. (E.suffix_acts has its span baked to the prefer framing.)"""
    acts, norms = [[] for _ in E.LAYERS], [[] for _ in E.LAYERS]
    def cap(j):
        def hook(m, i, o):
            s = E.resid(o)[:, -n_suffix:, :].float()
            acts[j].append(s.mean(1).cpu())
            norms[j].append(s.norm(dim=-1).mean().cpu())
        return hook
    handles = [L.register_forward_hook(cap(j)) for j, L in enumerate(E.LAYERS)]
    try:
        E.model(**E.encode(prompts))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([torch.cat(a) for a in acts]), torch.tensor([torch.stack(n).mean() for n in norms])


def extract():
    per_template = []
    all_norms = []
    for st in STATEMENTS:
        full = st.format(color=COLORS[0]) + SHARED
        n_suf = len(E.tok(full).input_ids) - len(E.tok(st.format(color=COLORS[0])).input_ids)
        acts, norms = capture([st.format(color=c) + SHARED for c in COLORS], n_suf)
        per_template.append(acts)          # [n_layers, 7, d_model]
        all_norms.append(norms)
    A = torch.stack(per_template)          # [n_templates, n_layers, 7, d_model]
    vecs = {}
    for i, c in enumerate(COLORS):
        others = [j for j in range(len(COLORS)) if j != i]
        diff = (A[:, :, i, :] - A[:, :, others, :].mean(2)).mean(0)   # [n_layers, d_model]
        vecs[c] = diff / diff.norm(dim=-1, keepdim=True)
    resid_norms = torch.stack(all_norms).mean(0)
    torch.save({"vecs": vecs, "resid_norms": resid_norms}, OUT / "vectors.pt")

    vs = torch.stack([vecs[c][LAYER] for c in COLORS])
    cos = vs @ vs.T
    off = cos[~torch.eye(7, dtype=bool)]
    print(f"RepE vectors: mean pairwise cosine at L{LAYER}: {off.mean():+.3f}")
    craw = torch.stack([V.VECS[c] for c in COLORS])
    ccent = craw - craw.mean(0, keepdim=True)
    ccent = ccent / ccent.norm(dim=-1, keepdim=True)
    match = torch.tensor([float(vecs[c][LAYER] @ ccent[i][LAYER]) for i, c in enumerate(COLORS)])
    print("cosine(RepE, centered-inherent) per color at L14: "
          + "  ".join(f"{c}:{m:+.2f}" for c, m in zip(COLORS, match)))
    return vecs, resid_norms


def steer_vec(vecs, resid_norms, color, coef):
    return (vecs[color][LAYER] * coef * resid_norms[LAYER]).to("cuda", torch.bfloat16)


def sanity(vecs, resid_norms):
    """Steer the original A/B prefer task (each color's 20 worst comparisons) — if the RepE
    vectors can't move even the forced choice, generation isn't worth running."""
    comps = json.load(open(Path(__file__).parent / "results" / "inherent" / "preferences.json"))
    print(f"\nA/B sanity check at L{LAYER} (mean delta signed logit-diff on 20 worst):")
    rows = []
    for color in COLORS:
        mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
        worst = sorted(mine, key=lambda c: c["diff"] if c["color_a"] == color else -c["diff"])[:20]
        prompts = [c["prompt"] for c in worst]
        sign = torch.tensor([1.0 if c["color_a"] == color else -1.0 for c in worst])
        base = torch.tensor([c["diff"] for c in worst]) * sign
        for cf in SANITY_COEFS:
            sa, sb, _, _ = E.scores(E.last_logits(prompts, steer=(LAYER, steer_vec(vecs, resid_norms, color, cf))))
            delta = ((sa - sb) * sign - base).mean().item()
            rows.append(dict(color=color, coef=cf, base=base.mean().item(), delta=delta))
    (OUT / "sanity.json").write_text(json.dumps(rows, indent=1))
    for cf in SANITY_COEFS:
        ds = [r["delta"] for r in rows if r["coef"] == cf]
        per = "  ".join(f"{r['color']}:{r['delta']:+.1f}" for r in rows if r["coef"] == cf)
        print(f"  coef {cf}: mean {sum(ds) / len(ds):+.2f}   {per}")


def gen_rows(vecs, resid_norms, items, name, color, coef, seed):
    steer = (LAYER, steer_vec(vecs, resid_norms, color, coef))
    samples = V.generate([it["prompt"] for it in items], steer=steer, seed=seed)
    return [dict(domain=it["domain"], item_color=it["color"], item=it["item"], condition=name,
                 layer=LAYER, coef=coef, sample_idx=si, text=t, value=V.parse_dollars(t))
            for it, texts in zip(items, samples) for si, t in enumerate(texts)]


def base_rows(items):
    keys = {(it["domain"], it["color"], it["item"]) for it in items}
    rows = json.load(open(V.OUT / "values.json"))
    return [r for r in rows if r["condition"] == "base"
            and (r["domain"], r["item_color"], r["item"]) in keys]


def pilot(vecs, resid_norms):
    items = make_items(n_per_color=V.PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = base_rows(items)
    for cf in PILOT_COEFS:
        name = f"L{LAYER}x{cf}"
        for c in COLORS:
            rows += gen_rows(vecs, resid_norms, by_color[c], name, c, cf, seed=LAYER * 100 + int(cf * 100))
        print(f"pilot {name} done")
    (VOUT / "pilot.json").write_text(json.dumps(rows, indent=1))
    V.report(rows, ["base"] + [f"L{LAYER}x{cf}" for cf in PILOT_COEFS])
    print("\nsample steered texts:")
    for cf in PILOT_COEFS:
        for r in [r for r in rows if r["condition"] == f"L{LAYER}x{cf}"][:3]:
            print(f"  [x{cf}][{r['domain']}/{r['item_color']}] {r['text']!r} -> {r['value']}")


def full(vecs, resid_norms):
    items = make_items()
    rows = base_rows(items)
    for ci, (cf, c) in enumerate((cf, c) for cf in FULL_COEFS for c in COLORS):
        rows += gen_rows(vecs, resid_norms, items, f"{c}|L{LAYER}x{cf}", c, cf, seed=(ci + 1) * 1000)
        print(f"condition {c}|L{LAYER}x{cf} done ({ci + 1}/{len(FULL_COEFS) * len(COLORS)})")
    (VOUT / "values.json").write_text(json.dumps(rows, indent=1))
    for cf in FULL_COEFS:
        tag = f"|L{LAYER}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## repe L{LAYER} x {cf} ########")
        V.report(sub, ["base"] + COLORS)


if __name__ == "__main__":
    vecs, resid_norms = extract()
    sanity(vecs, resid_norms)
    if V.STAGE == "pilot":
        pilot(vecs, resid_norms)
    elif V.STAGE == "full":
        full(vecs, resid_norms)
