"""Validate the prefer-vectors on negated framings ("worse", "avoid") over the same item pairs.

Run: python flip.py [modifier|inherent]  (after experiment.py has produced results/{mode}/)
If the vectors encode preference, baselines should anti-correlate with the prefer framing and
steering deltas should be negative (less likely to name the color as worse/avoided).
"""
import json

import torch

import experiment as E  # loads model; picks up mode from sys.argv[1]
from data import COLORS, SUFFIX, TEMPLATES

FRAMINGS = {
    "prefer": SUFFIX,  # decomposition baseline; deltas should reproduce steering.json
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


def sides(logits, comps, color):
    """Mean letter score for `color`'s side (own) and the other side (opp), as raw logits and
    as log-probs (calibrated against the whole vocab)."""
    out = {}
    for tag, lg in [("", logits), ("_lp", logits.log_softmax(-1))]:
        a = torch.logsumexp(lg[:, list(E.A_IDS.values())], -1)
        b = torch.logsumexp(lg[:, list(E.B_IDS.values())], -1)
        on_a = torch.tensor([c["color_a"] == color for c in comps])
        out["own" + tag] = torch.where(on_a, a, b).mean().item()
        out["opp" + tag] = torch.where(on_a, b, a).mean().item()
    return out


def run_framing(name, suffix):
    n_suf = len(E.tok(reframe(COMPS[0], suffix)).input_ids) - \
        len(E.tok(reframe(COMPS[0], "")).input_ids)
    prompts = [reframe(c, suffix) for c in COMPS]

    if name != "prefer":  # inspection + baseline flip only make sense for the new framings
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
        base = sides(E.last_logits(wp), worst, color)
        for L in E.STEER_LAYERS:
            for cf in E.COEFS:
                vec = (VECS[color][L] * cf * RESID_NORMS[L]).to("cuda", torch.bfloat16)
                st = sides(E.last_logits(wp, steer=(L, vec), n_suffix=n_suf), worst, color)
                rows.append(dict(color=color, layer=L, coef=cf,
                                 base=base["own"] - base["opp"], steered=st["own"] - st["opp"],
                                 delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                                 **{"base_" + k: v for k, v in base.items()},
                                 **{"steered_" + k: v for k, v in st.items()}))
    (E.OUT / f"flip_{name}.json").write_text(json.dumps(rows, indent=1))

    def avg(rs, key):
        return sum(r["steered_" + key] - r["base_" + key] for r in rs) / len(rs)

    print(f"\n{name}: mean steering deltas (per coef: Ddiff = Down - Dopp, raw letter logits; "
          "Dopp_lp = opposite letter log-prob):")
    print("      " + "".join(f" | c={cf}: Ddiff  Down  Dopp  Dopp_lp" for cf in E.COEFS))
    for L in E.STEER_LAYERS:
        cells = []
        for cf in E.COEFS:
            rs = [r for r in rows if (r["layer"], r["coef"]) == (L, cf)]
            d = sum(r["delta"] for r in rs) / len(rs)
            cells.append(f"{d:9.2f} {avg(rs, 'own'):5.2f} {avg(rs, 'opp'):5.2f}"
                         f" {avg(rs, 'opp_lp'):8.2f}")
        print(f"L{L:>4} " + " | ".join(cells))


if __name__ == "__main__":
    for name, suffix in FRAMINGS.items():
        run_framing(name, suffix)
