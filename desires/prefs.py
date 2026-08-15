"""A/B color-preference experiments: measurement, steering vectors, and their controls.

Subcommands (run `python prefs.py <cmd> --help`):
  measure   A/B letter-logit preferences over 420 pairs; extract prefer-vectors.
            -> results/{mode}/{inspect.txt,preferences.json,vectors.pt}   (~8 min GPU)
  flip      Re-measure under negated framings ("worse", "avoid") and steer with the
            unchanged prefer-vectors. Baseline flip stands; steering rows superseded.
            -> results/{mode}/flip_*.json
  cross     Tier / cross-mode / random-vector steering controls — the experiment that
            showed the original steering effect was non-specific disruption.
            -> results/{mode}/cross{tag}.json
  objects   A/B-free object-logprob measure; raw + centered vectors; controlled steering.
            -> results/{mode}/{objects.json,vectors_obj.pt,objects_steer.json}
  balanced  Price-controlled preference on the balanced tiers — the headline result.
            -> results/balanced_pref/   (needs `value.py tiers` + measure/objects inherent)

See FINDINGS.md for the full narrative, including which results were retracted and why.
"""
import argparse
import random
from collections import defaultdict

import torch

from lib.data import COLORS, SUFFIX, TEMPLATES
from lib.harness import COEFS, STEER_LAYERS, count_suffix_tokens, load, \
    random_unit_per_color, scaled_vec
from lib.tasks import ABTask, BODIES, FRAMINGS, OBJ_SUFFIX, TIERS, ab_scores, build_obj_comps, \
    cont_logprob, reframe, signed, tier_slice, top_wins, variant_ids, worst_k
from lib.util import load_json, results_dir, save_json
from lib.valuation import pearson, spearman


# ==== measure: A/B preferences + prefer-vectors ==================================================

def inspect(h, task, out, n=5):
    lines = []
    lg = h.last_logits([c["prompt"] for c in task.comps[:n]])
    for c, row in zip(task.comps[:n], lg):
        top = row.topk(20)
        lines.append(c["prompt"].replace("\n", " | "))
        lines.append("  " + ", ".join(f"{h.tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top)))
    text = "\n".join(lines) + f"\n\nA variants used: {task.a_ids}\nB variants used: {task.b_ids}\n"
    (out / "inspect.txt").write_text(text)
    print(text)


def measure(h, task, out):
    comps = task.comps
    lg = h.last_logits([c["prompt"] for c in comps])
    sa, sb, va, vb = task.scores(lg)
    for c, xa, xb, ra, rb in zip(comps, sa, sb, va, vb):
        c["score_a"], c["score_b"], c["diff"] = xa.item(), xb.item(), (xa - xb).item()
        c["logits_a"] = dict(zip(task.a_ids, ra.tolist()))
        c["logits_b"] = dict(zip(task.b_ids, rb.tolist()))
    save_json(out / "preferences.json", comps)

    print("\nMean diff (row color as A vs col color as B; + = row preferred):")
    print("        " + "".join(f"{c:>8}" for c in COLORS))
    for ca in COLORS:
        row = [torch.tensor([c["diff"] for c in comps if (c["color_a"], c["color_b"]) == (ca, cb)]).mean()
               if ca != cb else float("nan") for cb in COLORS]
        print(f"{ca:>8}" + "".join(f"{v:8.2f}" for v in row))
    print("\nPer-color mean signed preference:")
    for col in COLORS:
        m = torch.tensor([signed(c["diff"], c, col) for c in comps
                          if col in (c["color_a"], c["color_b"])]).mean()
        print(f"  {col:>8}: {m:+.3f}")


def extract(h, task, out):
    comps = task.comps
    if "diff" not in comps[0]:  # measure() didn't run this invocation — reuse the saved scores
        comps = load_json(out / "preferences.json")
    vecs, all_norms = {}, []
    for color in COLORS:
        acts, norms = h.suffix_acts([c["prompt"] for c in top_wins(comps, color)], task.n_suffix)
        mean = acts.mean(1)                                   # [n_layers, d_model]
        vecs[color] = mean / mean.norm(dim=-1, keepdim=True)  # unit vector per layer
        all_norms.append(norms)
    resid_norms = torch.stack(all_norms).mean(0)              # typical residual norm per layer
    torch.save({"vecs": vecs, "resid_norms": resid_norms}, out / "vectors.pt")


def cmd_measure(args):
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    stages = args.stages.split(",")
    if "inspect" in stages:
        inspect(h, task, out)
    if "measure" in stages:
        measure(h, task, out)
    if "extract" in stages:
        extract(h, task, out)


# ==== flip: negated-framing validation ===========================================================

def run_framing(h, task, out, comps, vecs, resid_norms, name, suffix):
    n_suf = task.framing_n_suffix(suffix)
    prompts = [reframe(c, suffix) for c in comps]

    if name != "prefer":  # inspection + baseline flip only make sense for the new framings
        lines = [f"== {name} =="]
        for p, row in zip(prompts[:3], h.last_logits(prompts[:3])):
            top = row.topk(10)
            lines += [p.replace("\n", " | "),
                      "  " + ", ".join(f"{h.tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top))]
        print("\n".join(lines))
        with open(out / "flip_inspect.txt", "a") as f:
            f.write("\n".join(lines) + "\n")

        sa, sb, _, _ = task.scores(h.last_logits(prompts))
        diffs = (sa - sb).tolist()
        prefer = torch.tensor([c["diff"] for c in comps])
        r = torch.corrcoef(torch.stack([prefer, torch.tensor(diffs)]))[0, 1].item()
        print(f"\n{name}: corr(prefer diff, {name} diff) over 420 pairs = {r:+.3f}")
        print(f"{'color':>8}  prefer   {name}")
        for col in COLORS:
            sel = [(c, d) for c, d in zip(comps, diffs) if col in (c["color_a"], c["color_b"])]
            mp = torch.tensor([signed(c["diff"], c, col) for c, _ in sel]).mean()
            mf = torch.tensor([signed(d, c, col) for c, d in sel]).mean()
            print(f"{col:>8}  {mp:+.3f}  {mf:+.3f}")

    rows = []
    for color in COLORS:
        worst = worst_k(comps, color)
        wp = [reframe(c, suffix) for c in worst]
        base = task.sides(h.last_logits(wp), worst, color)
        for L in STEER_LAYERS:
            for cf in COEFS:
                vec = scaled_vec(vecs[color][L], cf, resid_norms[L])
                st = task.sides(h.last_logits(wp, steer=(L, vec), n_suffix=n_suf), worst, color)
                rows.append(dict(color=color, layer=L, coef=cf,
                                 base=base["own"] - base["opp"], steered=st["own"] - st["opp"],
                                 delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                                 **{"base_" + k: v for k, v in base.items()},
                                 **{"steered_" + k: v for k, v in st.items()}))
    save_json(out / f"flip_{name}.json", rows)

    def avg(rs, key):
        return sum(r["steered_" + key] - r["base_" + key] for r in rs) / len(rs)

    print(f"\n{name}: mean steering deltas (per coef: Ddiff = Down - Dopp, raw letter logits; "
          "Dopp_lp = opposite letter log-prob):")
    print("      " + "".join(f" | c={cf}: Ddiff  Down  Dopp  Dopp_lp" for cf in COEFS))
    for L in STEER_LAYERS:
        cells = []
        for cf in COEFS:
            rs = [r for r in rows if (r["layer"], r["coef"]) == (L, cf)]
            d = sum(r["delta"] for r in rs) / len(rs)
            cells.append(f"{d:9.2f} {avg(rs, 'own'):5.2f} {avg(rs, 'opp'):5.2f}"
                         f" {avg(rs, 'opp_lp'):8.2f}")
        print(f"L{L:>4} " + " | ".join(cells))


def cmd_flip(args):
    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")  # same 420 pairs, with prefer diffs
    saved = torch.load(out / "vectors.pt")
    (out / "flip_inspect.txt").write_text("")  # truncate: reruns should not append forever
    for name, suffix in FRAMINGS.items():
        run_framing(h, task, out, comps, saved["vecs"], saved["resid_norms"], name, suffix)


# ==== cross: tier / cross-mode / random controls =================================================

def cmd_cross(args):
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


# ==== objects: A/B-free measurement + centered vectors ===========================================

def obj_measure(h, out, comps, n_suf):
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


def obj_extract(h, out, comps, n_suf):
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


def obj_steer(h, out, comps, n_suf, mode, vecs, cent, resid_norms):
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


def cmd_objects(args):
    out = results_dir(args.mode)
    h = load()
    comps, n_suf = build_obj_comps(h, args.mode)
    # spot-check: the last k tokens of prompt+cont must decode back to the continuation
    for c in comps[:3]:
        ids = h.tok(c["prompt"] + " " + c["item_a"]).input_ids
        k = len(ids) - len(h.tok(c["prompt"]).input_ids)
        assert h.tok.decode(ids[-k:]) == " " + c["item_a"], c
    obj_measure(h, out, comps, n_suf)
    obj_steer(h, out, comps, n_suf, args.mode, *obj_extract(h, out, comps, n_suf))


# ==== balanced: price-controlled preference ======================================================

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


def cmd_balanced(args):
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


# ==== CLI ========================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("measure", help="A/B preferences + prefer-vectors")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    p.add_argument("--stages", default="inspect,measure,extract")
    p.set_defaults(fn=cmd_measure)

    p = sub.add_parser("flip", help="negated-framing validation")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    p.set_defaults(fn=cmd_flip)

    p = sub.add_parser("cross", help="tier/cross-mode/random steering controls")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    p.add_argument("--coefs", default="1.0,2.0")
    p.add_argument("--tag", default="", help='result-file suffix, e.g. "_lo" -> cross_lo.json')
    p.set_defaults(fn=cmd_cross)

    p = sub.add_parser("objects", help="A/B-free object-logprob measure + centered vectors")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    p.set_defaults(fn=cmd_objects)

    p = sub.add_parser("balanced", help="price-controlled preference (headline result)")
    p.set_defaults(fn=cmd_balanced)

    args = ap.parse_args()
    args.fn(args)
