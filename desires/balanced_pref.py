"""Price-controlled color-preference measurement on the balanced tiers.

Run: python balanced_pref.py inherent   (GPU, ~4 min)

Every prior preference measurement compared items whose price differed systematically by color
(blue = sapphires, green = vegetables), and value_pref.py showed valuation weakly predicts
preference — so how much color preference survives when price is held constant? Pairs are
drawn within a value tier (frozen, unflagged items from results/balanced_tiers/
balanced_tiers.json), balanced over all 42 ordered color pairs, and measured two ways on the
same item/template draws: the A/B letter-logit diff (experiment.py's measure) and the
object-logprob diff (objects.py's A/B-free measure).
"""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch

import objects as O  # cont_logprob, BODIES, OBJ_SUFFIX; imports experiment (loads model)
from data import COLORS, TEMPLATES, SUFFIX

E = O.E
RES = Path(__file__).parent / "results"
OUT = RES / "balanced_pref"
OUT.mkdir(parents=True, exist_ok=True)

N_PER_PAIR = 5
POOL = defaultdict(list)
for r in json.load(open(RES / "balanced_tiers" / "balanced_tiers.json")):
    if not r["flagged"]:
        POOL[(r["color"], r["tier"])].append(r["item"])

rng = random.Random(0)
comps = []
for tier in (1, 2, 3):
    for ca in COLORS:
        for cb in COLORS:
            if ca == cb:
                continue
            for _ in range(N_PER_PAIR):
                ia, ib = rng.choice(POOL[(ca, tier)]), rng.choice(POOL[(cb, tier)])
                comps.append(dict(tier=tier, color_a=ca, color_b=cb, item_a=ia, item_b=ib,
                                  template=rng.randrange(len(TEMPLATES))))

# measure 1: A/B letter logits
ab_prompts = [TEMPLATES[c["template"]].format(a=c["item_a"], b=c["item_b"], suffix=SUFFIX)
              for c in comps]
sa, sb, _, _ = E.scores(E.last_logits(ab_prompts))
for c, xa, xb in zip(comps, sa, sb):
    c["ab_diff"] = (xa - xb).item()
print("A/B letter measure done")

# measure 2: object logprobs (teacher-forced), same draws
obj_prompts = [O.BODIES[c["template"]].format(a=c["item_a"], b=c["item_b"]) + O.OBJ_SUFFIX
               for c in comps for _ in range(2)]
conts = [" " + c[k] for c in comps for k in ("item_a", "item_b")]
m, _ = O.cont_logprob(obj_prompts, conts)
for j, c in enumerate(comps):
    c["obj_diff"] = (m[2 * j] - m[2 * j + 1]).item()
print("object-logprob measure done")
(OUT / "balanced_pref.json").write_text(json.dumps(comps, indent=1))


def per_color(rows, key):
    out = {}
    for col in COLORS:
        ds = [E.signed(c[key], c, col) for c in rows if col in (c["color_a"], c["color_b"])]
        out[col] = sum(ds) / len(ds)
    return out


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    rk = lambda v: [float(sorted(v).index(a)) for a in v]
    return pearson(rk(x), rk(y))


# uncontrolled per-color preferences for comparison
uncontrolled = {
    "ab": per_color(json.load(open(RES / "inherent" / "preferences.json")), "diff"),
    "obj": per_color(json.load(open(RES / "inherent" / "objects.json")), "diff"),
}

lines = [f"{len(comps)} comparisons: 3 tiers x 42 ordered color pairs x {N_PER_PAIR}, "
         "unflagged balanced-tier items\n"]
for key, name, unc in [("ab_diff", "A/B letter-logit", uncontrolled["ab"]),
                       ("obj_diff", "object-logprob", uncontrolled["obj"])]:
    lines.append(f"== {name} ==")
    lines.append("per-color mean signed preference (rows = tier):")
    lines.append("        " + "".join(f"{c:>8}" for c in COLORS))
    for tier in (1, 2, 3):
        pc = per_color([c for c in comps if c["tier"] == tier], key)
        lines.append(f"  T{tier}   " + "".join(f"{pc[c]:8.2f}" for c in COLORS))
    pc_all = per_color(comps, key)
    lines.append("  all  " + "".join(f"{pc_all[c]:8.2f}" for c in COLORS))
    lines.append("  uncontrolled (original inherent items):")
    lines.append("       " + "".join(f"{unc[c]:8.2f}" for c in COLORS))
    x = [pc_all[c] for c in COLORS]
    y = [unc[c] for c in COLORS]
    lines.append(f"  corr(controlled, uncontrolled) over 7 colors: "
                 f"pearson {pearson(x, y):+.2f}  spearman {spearman(x, y):+.2f}")
    lines.append(f"  mean |per-color pref|: controlled {sum(abs(v) for v in x) / 7:.3f}  "
                 f"vs uncontrolled {sum(abs(v) for v in y) / 7:.3f}")
    # tier consistency: correlation of per-color prefs between tiers
    t = {tier: per_color([c for c in comps if c["tier"] == tier], key) for tier in (1, 2, 3)}
    for a, b in [(1, 2), (1, 3), (2, 3)]:
        lines.append(f"  corr(T{a}, T{b}): pearson "
                     f"{pearson([t[a][c] for c in COLORS], [t[b][c] for c in COLORS]):+.2f}")
    lines.append("")

text = "\n".join(lines)
(OUT / "analysis.txt").write_text(text + "\n")
print(text)
