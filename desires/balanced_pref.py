"""Price-controlled color-preference measurement on the balanced tiers.

Run: python balanced_pref.py   (GPU, ~4 min; after balanced_tiers.py, preferences.py
--mode inherent, and objects.py --mode inherent)

Every prior preference measurement compared items whose price differed systematically by color
(blue = sapphires, green = vegetables), and value_pref.py showed valuation weakly predicts
preference — so how much color preference survives when price is held constant? Pairs are
drawn within a value tier (frozen, unflagged items from results/balanced_tiers/
balanced_tiers.json), balanced over all 42 ordered color pairs, and measured two ways on the
same item/template draws: the A/B letter-logit diff (preferences.py's measure) and the
object-logprob diff (objects.py's A/B-free measure).
"""
import argparse
import random
from collections import defaultdict

from lib.abtask import ab_scores, variant_ids
from lib.data import COLORS, SUFFIX, TEMPLATES
from lib.harness import count_suffix_tokens, load
from lib.io import load_json, save_json
from lib.objtask import BODIES, OBJ_SUFFIX, cont_logprob
from lib.paths import results_dir
from lib.stats import pearson, spearman
from lib.tiers import signed

N_PER_PAIR = 10


def per_color(rows, key):
    out = {}
    for col in COLORS:
        ds = [signed(c[key], c, col) for c in rows if col in (c["color_a"], c["color_b"])]
        out[col] = sum(ds) / len(ds)
    return out


def color_cis(rows, key, pool, n_boot=2000, seed=1):
    """Per-color mean signed diff with a 95% CI from an item-clustered bootstrap: each
    replicate resamples the item pool of every (color, tier) cell with replacement and
    weights each comparison by the product of its two items' multiplicities — so the CI
    reflects item idiosyncrasy, the dominant noise source, not just comparison sampling."""
    brng = random.Random(seed)
    pools = {k: sorted(set(v)) for k, v in pool.items()}
    per_color_reps = {c: [] for c in COLORS}
    for _ in range(n_boot):
        w = defaultdict(int)
        for k, items in pools.items():
            for it in brng.choices(items, k=len(items)):
                w[it] += 1
        for col in COLORS:
            num = den = 0.0
            for c in rows:
                if col in (c["color_a"], c["color_b"]):
                    wt = w[c["item_a"]] * w[c["item_b"]]
                    if wt:
                        num += wt * signed(c[key], c, col)
                        den += wt
            per_color_reps[col].append(num / den if den else 0.0)
    out = {}
    for col in COLORS:
        reps = sorted(per_color_reps[col])
        obs = per_color(rows, key)[col]
        out[col] = (obs, reps[int(0.025 * n_boot)], reps[int(0.975 * n_boot) - 1])
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = results_dir("balanced_pref")
    h = load()
    pool = defaultdict(list)
    for r in load_json(results_dir("balanced_tiers") / "balanced_tiers.json"):
        if not r["flagged"]:
            pool[(r["color"], r["tier"])].append(r["item"])

    rng = random.Random(0)
    comps = []
    for tier in (1, 2, 3):
        for ca in COLORS:
            for cb in COLORS:
                if ca == cb:
                    continue
                for _ in range(N_PER_PAIR):
                    ia, ib = rng.choice(pool[(ca, tier)]), rng.choice(pool[(cb, tier)])
                    comps.append(dict(tier=tier, color_a=ca, color_b=cb, item_a=ia, item_b=ib,
                                      template=rng.randrange(len(TEMPLATES))))

    # measure 1: A/B letter logits
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    ab_prompts = [TEMPLATES[c["template"]].format(a=c["item_a"], b=c["item_b"], suffix=SUFFIX)
                  for c in comps]
    sa, sb, _, _ = ab_scores(h.last_logits(ab_prompts), a_ids, b_ids)
    for c, xa, xb in zip(comps, sa, sb):
        c["ab_diff"] = (xa - xb).item()
    print("A/B letter measure done")

    # measure 2: object logprobs (teacher-forced), same draws
    obj_prompts = [BODIES[c["template"]].format(a=c["item_a"], b=c["item_b"]) + OBJ_SUFFIX
                   for c in comps for _ in range(2)]
    conts = [" " + c[k] for c in comps for k in ("item_a", "item_b")]
    n_suf = count_suffix_tokens(h.tok, obj_prompts[0], obj_prompts[0].removesuffix(OBJ_SUFFIX))
    m, _ = cont_logprob(h, obj_prompts, conts, n_suf)
    for j, c in enumerate(comps):
        c["obj_diff"] = (m[2 * j] - m[2 * j + 1]).item()
    print("object-logprob measure done")
    save_json(out / "balanced_pref.json", comps)

    # uncontrolled per-color preferences for comparison
    uncontrolled = {
        "ab": per_color(load_json(results_dir("inherent") / "preferences.json"), "diff"),
        "obj": per_color(load_json(results_dir("inherent") / "objects.json"), "diff"),
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
        lines.append("  per-color 95% CI (item-clustered bootstrap):")
        cis = color_cis(comps, key, pool)
        for col in COLORS:
            obs, lo, hi = cis[col]
            star = " *" if lo > 0 or hi < 0 else ""
            lines.append(f"    {col:>8}: {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
        lines.append("  blue per tier:")
        for tier in (1, 2, 3):
            obs, lo, hi = color_cis([c for c in comps if c["tier"] == tier], key, pool)["blue"]
            star = " *" if lo > 0 or hi < 0 else ""
            lines.append(f"      T{tier}: {obs:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
        lines.append("")

    text = "\n".join(lines)
    (out / "analysis.txt").write_text(text + "\n")
    print(text)
