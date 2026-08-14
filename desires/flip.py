"""Validate the prefer-vectors on negated framings ("worse", "avoid") over the same item pairs.

Run: python flip.py [modifier|inherent]  (after experiment.py has produced results/{mode}/)
If the vectors encode preference, baselines should anti-correlate with the prefer framing and
steering deltas should be negative (less likely to name the color as worse/avoided).
"""
import json

import torch

import experiment as E  # loads model; picks up mode from sys.argv[1]
from data import COLORS, TEMPLATES

FRAMINGS = {
    "worse": "Which of the two items is worse? Answer with one letter.\nModel: The worse one is",
    "avoid": "If you had to avoid one of these, which would it be? "
             "Answer with one letter.\nModel: I would avoid",
}

COMPS = json.loads((E.OUT / "preferences.json").read_text())  # same 420 pairs, with prefer diffs
saved = torch.load(E.OUT / "vectors.pt")
VECS, RESID_NORMS = saved["vecs"], saved["resid_norms"]


def reframe(c, suffix):
    return TEMPLATES[c["template"]].format(a=c["item_a"], b=c["item_b"], suffix=suffix)


def worst_by_prefer(color):
    mine = [c for c in COMPS if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: E.signed(c["diff"], c, color))[:E.K]


def run_framing(name, suffix):
    n_suf = len(E.tok(reframe(COMPS[0], suffix)).input_ids) - \
        len(E.tok(reframe(COMPS[0], "")).input_ids)
    prompts = [reframe(c, suffix) for c in COMPS]

    lines = [f"== {name} =="]
    for p, row in zip(prompts[:3], E.last_logits(prompts[:3])):
        top = row.topk(10)
        lines += [p.replace("\n", " | "),
                  "  " + ", ".join(f"{E.tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top))]
    print("\n".join(lines))
    with open(E.OUT / "flip_inspect.txt", "a") as f:
        f.write("\n".join(lines) + "\n")

    sa, sb, _, _ = E.scores(E.last_logits(prompts))
    diffs = (sa - sb).tolist()
    prefer = torch.tensor([c["diff"] for c in COMPS])
    r = torch.corrcoef(torch.stack([prefer, torch.tensor(diffs)]))[0, 1].item()
    print(f"\n{name}: corr(prefer diff, {name} diff) over 420 pairs = {r:+.3f}")
    print(f"{'color':>8}  prefer   {name}")
    for col in COLORS:
        sel = [(c, d) for c, d in zip(COMPS, diffs) if col in (c["color_a"], c["color_b"])]
        mp = torch.tensor([E.signed(c["diff"], c, col) for c, _ in sel]).mean()
        mf = torch.tensor([E.signed(d, c, col) for c, d in sel]).mean()
        print(f"{col:>8}  {mp:+.3f}  {mf:+.3f}")

    rows = []
    for color in COLORS:
        worst = worst_by_prefer(color)
        wp = [reframe(c, suffix) for c in worst]
        sa, sb, _, _ = E.scores(E.last_logits(wp))
        base = torch.tensor([E.signed((xa - xb).item(), c, color)
                             for c, xa, xb in zip(worst, sa, sb)])
        for L in E.STEER_LAYERS:
            for cf in E.COEFS:
                vec = (VECS[color][L] * cf * RESID_NORMS[L]).to("cuda", torch.bfloat16)
                sa, sb, _, _ = E.scores(E.last_logits(wp, steer=(L, vec), n_suffix=n_suf))
                new = torch.tensor([E.signed((xa - xb).item(), c, color)
                                    for c, xa, xb in zip(worst, sa, sb)])
                rows.append(dict(color=color, layer=L, coef=cf, base=base.mean().item(),
                                 steered=new.mean().item(), delta=(new - base).mean().item()))
    (E.OUT / f"flip_{name}.json").write_text(json.dumps(rows, indent=1))

    print(f"\n{name}: mean delta toward steered color after adding its PREFER vector "
          "(negative = consistent preference):")
    print("       " + "".join(f"{cf:>8}" for cf in E.COEFS))
    for L in E.STEER_LAYERS:
        ds = [torch.tensor([r["delta"] for r in rows if (r["layer"], r["coef"]) == (L, cf)]).mean()
              for cf in E.COEFS]
        print(f"L{L:>4}  " + "".join(f"{d:8.2f}" for d in ds))


if __name__ == "__main__":
    for name, suffix in FRAMINGS.items():
        run_framing(name, suffix)
