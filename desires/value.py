"""Valuation experiments: do color-preference vectors shift dollar valuations?

Subcommands (run `python value.py <cmd> --help`):
  cross      Valuation cross with the raw prefer-vectors (uniform depression; null contrast).
             --stage {inspect,pilot,full} -> results/value_{mode}/   (after `prefs.py measure`)
  centered   Same cross with mean-centered vectors -> per-color global price knobs.
             --stage {pilot,full} -> results/value_{mode}_centered/  (after `cross --stage full`)
  items      Value all 700 inherent items unsteered; price-statistics hypothesis. (~6 min)
             -> results/value_items/   (after `centered --stage full`)
  obj21      Cross at L21 with obj-centered vectors + random control columns. (~60 min)
             -> results/value_obj21/   (after `prefs.py objects` both modes + `cross full`)
  calibrate  Model dollar estimates vs real-world prices. (~3 min) -> results/calibration/
  tiers      Curate + verify the color x price-balanced item set. (~8 min)
             -> results/balanced_tiers/

Prompted as "...I would estimate its value at $", 5 sampled completions per item, dollar
amounts parsed and aggregated as geomeans. See FINDINGS.md Part II for results.

Reproducibility note: torch.manual_seed is set per generate-batch, so outputs are stable only
for fixed item order, GEN_BATCH, and n_samples.
"""
import argparse
import math

import torch

from lib.data import COLORS, INHERENT
from lib.harness import load, random_unit_matrix, scaled_vec
from lib.tasks import K, top_wins
from lib.util import load_json, results_dir, save_json
from lib.value_data import SUFFIX, TEMPLATE, make_items
from lib.valuation import base_rows, column_effect, condition_ml, gen_rows, make_rows, \
    mean_log10_by, parse_dollars, pearson, report, spearman, val_n_suffix

DOMAINS = ["painting", "household", "real"]
PILOT_N = 5


# ==== cross: raw prefer-vectors ==================================================================

# Pilot findings (see FINDINGS.md; the original pilot grids are pruned from results/):
# coef 2 collapses generation into ": A:" babble (the vectors carry the extraction prompts'
# answer-a-letter content, which A/B logit-diffs cancelled but free generation exposes);
# coef 1 mode-collapses values to $1000. Coefs 0.25/0.5 stay fully parseable with a milder
# uniform depression, so the full cross runs both: 0.25 minimizes disruption/floor effects,
# 0.5 pushes harder.
STEER_CONFIGS = [(14, 0.25), (14, 0.5)]  # (layer, coef)
PILOT_LAYERS, PILOT_COEFS = [14, 18], [1.0, 2.0]


def run(h, out, vecs, resid_norms, n_suf, items, conditions, out_name):
    """conditions: [(name, steer_color | None, (layer, coef) | None)]. Generates per condition
    over all items, parses, saves every sample row, returns rows.
    Seed formula (historical, do not change): condition index * 1000."""
    rows = []
    for ci, (name, color, cfg) in enumerate(conditions):
        steer = None
        if cfg is not None:
            layer, coef = cfg
            steer = (layer, scaled_vec(vecs[color][layer], coef, resid_norms[layer]))
        samples = h.generate([it["prompt"] for it in items], steer=steer, seed=ci * 1000,
                             n_suffix=n_suf)
        rows += make_rows(items, samples, name, cfg)
        print(f"condition {name} done ({ci + 1}/{len(conditions)})")
    save_json(out / out_name, rows)
    return rows


def inspect(h, out):
    items = [it for d in DOMAINS for it in [i for i in make_items() if i["domain"] == d][:2]]
    samples = h.generate([it["prompt"] for it in items], n_samples=3, seed=0)
    lines = []
    for it, texts in zip(items, samples):
        lines.append(it["prompt"].replace("\n", " | "))
        for t in texts:
            lines.append(f"  {t!r} -> {parse_dollars(t)}")
    text = "\n".join(lines) + "\n"
    (out / "inspect.txt").write_text(text)
    print(text)


def raw_pilot(h, out, vecs, resid_norms, n_suf):
    """Matched-color steering only, small grid, to pick (layer, coef)."""
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = []
    # baseline
    rows += run(h, out, vecs, resid_norms, n_suf, items, [("base", None, None)],
                "pilot_base.json")
    # matched steering: each color's items under that color's own vector, per grid point.
    # Seed formula (historical): layer * 100 + int(coef * 10).
    for L in PILOT_LAYERS:
        for cf in PILOT_COEFS:
            name = f"L{L}x{cf}"
            for c in COLORS:
                samples = h.generate([it["prompt"] for it in by_color[c]],
                                     steer=(L, scaled_vec(vecs[c][L], cf, resid_norms[L])),
                                     seed=L * 100 + int(cf * 10), n_suffix=n_suf)
                rows += make_rows(by_color[c], samples, name, (L, cf))
            print(f"pilot {name} done")
    save_json(out / "pilot.json", rows)
    conds = ["base"] + [f"L{L}x{cf}" for L in PILOT_LAYERS for cf in PILOT_COEFS]
    report(rows, conds)


def raw_full(h, out, vecs, resid_norms, n_suf):
    """Baseline + all 7 steer colors on every item (cross-color)."""
    items = make_items()
    conditions = [("base", None, None)]
    for L, cf in STEER_CONFIGS:
        conditions += [(f"{c}|L{L}x{cf}", c, (L, cf)) for c in COLORS]
    rows = run(h, out, vecs, resid_norms, n_suf, items, conditions, "values.json")
    for L, cf in STEER_CONFIGS:
        tag = f"|L{L}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## config L{L} x {cf} ########")
        report(sub, ["base"] + COLORS)


def cmd_cross(args):
    out = results_dir(f"value_{args.mode}")
    h = load()
    saved = torch.load(results_dir(args.mode) / "vectors.pt")
    n_suf = val_n_suffix(h.tok)
    inspect(h, out)
    if args.stage == "pilot":
        raw_pilot(h, out, saved["vecs"], saved["resid_norms"], n_suf)
    elif args.stage == "full":
        raw_full(h, out, saved["vecs"], saved["resid_norms"], n_suf)


# ==== centered: mean-centered prefer-vectors =====================================================

CENT_PILOT_COEFS = [0.5, 1.0, 2.0]
CENT_LAYER = 14
CENT_FULL_COEFS = [0.5, 1.0]  # confirmed/updated by the pilot


def center(vecs):
    raw = torch.stack([vecs[c] for c in COLORS])          # [7 colors, 28 layers, d_model]
    cent = raw - raw.mean(0, keepdim=True)
    frac = cent.norm(dim=-1) / raw.norm(dim=-1)           # size of the differential component
    cvecs = {c: (cent / cent.norm(dim=-1, keepdim=True))[i] for i, c in enumerate(COLORS)}
    print(f"differential fraction ||v - mean||/||v||: layer 14 mean {frac[:, 14].mean():.3f}, "
          f"all layers {frac.mean():.3f}")
    # NB: renormalizing to unit length means coef x resid_norm now pushes the pure differential
    # direction at full strength — a much larger differential push than the raw vectors gave.
    return cvecs


def cent_pilot(h, out, base_path, cvecs, resid_norms, n_suf):
    """Matched-color steering only, coef sweep at one layer."""
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = base_rows(items, base_path)
    assert rows, "run `python value.py cross --stage full` first (baseline rows are reused)"
    # Seed formula (historical): layer * 100 + int(coef * 100).
    for cf in CENT_PILOT_COEFS:
        name = f"L{CENT_LAYER}x{cf}"
        for c in COLORS:
            steer = (CENT_LAYER, scaled_vec(cvecs[c][CENT_LAYER], cf, resid_norms[CENT_LAYER]))
            rows += gen_rows(h, by_color[c], name, steer, (CENT_LAYER, cf),
                             seed=CENT_LAYER * 100 + int(cf * 100), n_suffix=n_suf)
        print(f"pilot {name} done")
    save_json(out / "pilot.json", rows)
    report(rows, ["base"] + [f"L{CENT_LAYER}x{cf}" for cf in CENT_PILOT_COEFS])
    print("\nsample steered texts:")
    for cf in CENT_PILOT_COEFS:
        for r in [r for r in rows if r["condition"] == f"L{CENT_LAYER}x{cf}"][:3]:
            print(f"  [x{cf}][{r['domain']}/{r['item_color']}] {r['text']!r} -> {r['value']}")


def cent_full(h, out, base_path, cvecs, resid_norms, n_suf):
    """Baseline + all 7 steer colors on every item (cross-color)."""
    items = make_items()
    rows = base_rows(items, base_path)
    assert rows
    # Seed formula (historical): (condition index + 1) * 1000, conditions in coef-major order.
    for ci, (cf, c) in enumerate((cf, c) for cf in CENT_FULL_COEFS for c in COLORS):
        steer = (CENT_LAYER, scaled_vec(cvecs[c][CENT_LAYER], cf, resid_norms[CENT_LAYER]))
        rows += gen_rows(h, items, f"{c}|L{CENT_LAYER}x{cf}", steer, (CENT_LAYER, cf),
                         seed=(ci + 1) * 1000, n_suffix=n_suf)
        print(f"condition {c}|L{CENT_LAYER}x{cf} done "
              f"({ci + 1}/{len(CENT_FULL_COEFS) * len(COLORS)})")
    save_json(out / "values.json", rows)
    for cf in CENT_FULL_COEFS:
        tag = f"|L{CENT_LAYER}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## centered L{CENT_LAYER} x {cf} ########")
        report(sub, ["base"] + COLORS)


def cmd_centered(args):
    out = results_dir(f"value_{args.mode}_centered")
    base_path = results_dir(f"value_{args.mode}") / "values.json"
    h = load()
    saved = torch.load(results_dir(args.mode) / "vectors.pt")
    cvecs = center(saved["vecs"])
    n_suf = val_n_suffix(h.tok)
    if args.stage == "pilot":
        cent_pilot(h, out, base_path, cvecs, saved["resid_norms"], n_suf)
    else:
        cent_full(h, out, base_path, cvecs, saved["resid_norms"], n_suf)


# ==== items: price statistics of the inherent pool ===============================================

def extraction_items(color, comps):
    """Winning-side items of the top-K comparisons `color` won most strongly."""
    return [c["item_a"] if c["color_a"] == color else c["item_b"]
            for c in top_wins(comps, color, K)]


def column_effects(coef_tag):
    """Mean dlog10(per-item geomean) per steer color over painting+household rows."""
    ml = condition_ml(load_json(results_dir("value_inherent_centered") / "values.json"))
    return {sc: column_effect(ml, f"{sc}|{coef_tag}") for sc in COLORS}


def cmd_items(args):
    out = results_dir("value_items")
    h = load()
    items = [(c, it) for c in COLORS for it in INHERENT[c]]
    samples = h.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for _, it in items], seed=0)
    rows = [dict(color=c, item=it, sample_idx=si, text=t, value=parse_dollars(t))
            for (c, it), texts in zip(items, samples) for si, t in enumerate(texts)]
    save_json(out / "values.json", rows)

    # per-item mean log10 over positive parsed samples
    ml = mean_log10_by(rows, lambda r: (r["color"], r["item"]))

    comps = load_json(results_dir("inherent") / "preferences.json")
    pool = {c: [ml[(c, it)] for it in INHERENT[c] if (c, it) in ml] for c in COLORS}
    extr = {c: [ml[(c, it)] for it in extraction_items(c, comps) if (c, it) in ml] for c in COLORS}
    cols = {tag: column_effects(tag) for tag in ["L14x0.5", "L14x1.0"]}

    lines = [f"parsed items: {len(ml)}/700  (unparseable samples: "
             f"{sum(1 for r in rows if r['value'] is None)}/{len(rows)})",
             "\nper-color item price statistics (log10 $) vs centered column effects:",
             f"{'color':>8} {'pool mean':>10} {'pool geo$':>10} {'extr mean':>10} "
             f"{'extr geo$':>10} {'col@0.5':>8} {'col@1.0':>8}"]
    for c in COLORS:
        pm, em = sum(pool[c]) / len(pool[c]), sum(extr[c]) / len(extr[c])
        lines.append(f"{c:>8} {pm:>10.2f} {10 ** pm:>10.3g} {em:>10.2f} {10 ** em:>10.3g} "
                     f"{cols['L14x0.5'][c]:>+8.2f} {cols['L14x1.0'][c]:>+8.2f}")
    lines.append("\ncorrelations with column effect (n=7 colors, indicative only):")
    for tag in ["L14x0.5", "L14x1.0"]:
        ce = [cols[tag][c] for c in COLORS]
        for name, stat in [("pool", pool), ("extraction", extr)]:
            xs = [sum(stat[c]) / len(stat[c]) for c in COLORS]
            lines.append(f"  {name:>10} mean log10 vs {tag}: pearson {pearson(xs, ce):+.2f}  "
                         f"spearman {spearman(xs, ce):+.2f}")
    text = "\n".join(lines) + "\n"
    (out / "analysis.txt").write_text(text)
    print(text)


# ==== obj21: L21 obj-centered cross with random controls =========================================

OBJ21_LAYER, OBJ21_COEF = 21, 1.0


def cmd_obj21(args):
    out = results_dir("value_obj21")
    h = load()
    obj = {}
    for tag, mode in [("objmod", "modifier"), ("objinh", "inherent")]:
        saved = torch.load(results_dir(mode) / "vectors_obj.pt")
        obj[tag] = (saved["centered"], saved["resid_norms"])
    raw_norms = torch.load(results_dir("inherent") / "vectors.pt")["resid_norms"]
    rand_unit = random_unit_matrix(28, 3584)

    def vec_for(cond):
        if cond == "rand_L21":
            return 21, (rand_unit[21] * OBJ21_COEF * obj["objinh"][1][21]).to("cuda", torch.bfloat16)
        if cond == "rand_L14":
            return 14, (rand_unit[14] * OBJ21_COEF * raw_norms[14]).to("cuda", torch.bfloat16)
        color, tag = cond.split("|")[0], cond.split("|")[1].split("_")[0]
        vecs, norms = obj[tag]
        return OBJ21_LAYER, (vecs[color][OBJ21_LAYER] * OBJ21_COEF * norms[OBJ21_LAYER]).to(
            "cuda", torch.bfloat16)

    items = make_items()
    n_suf = val_n_suffix(h.tok)
    conds = [f"{c}|objmod_L{OBJ21_LAYER}x{OBJ21_COEF}" for c in COLORS] \
          + [f"{c}|objinh_L{OBJ21_LAYER}x{OBJ21_COEF}" for c in COLORS] + ["rand_L21", "rand_L14"]
    rows = base_rows(items, results_dir("value_inherent") / "values.json")
    assert rows
    # Seed formula (historical): (condition index + 1) * 1000.
    for ci, cond in enumerate(conds):
        L, vec = vec_for(cond)
        rows += gen_rows(h, items, cond, (L, vec), (L, OBJ21_COEF), seed=(ci + 1) * 1000,
                         n_suffix=n_suf)
        print(f"condition {cond} done ({ci + 1}/{len(conds)})")
    save_json(out / "values.json", rows)

    for tag, rand_cond in [("objmod", "rand_L21"), ("objinh", "rand_L21")]:
        suffix = f"|{tag}_L{OBJ21_LAYER}x{OBJ21_COEF}"
        sub = [dict(r, condition=r["condition"].removesuffix(suffix).replace(rand_cond, "rand"))
               for r in rows if r["condition"] == "base" or r["condition"].endswith(suffix)
               or r["condition"] == rand_cond]
        print(f"\n######## {tag} centered, L{OBJ21_LAYER} x {OBJ21_COEF} "
              f"(rand column = {rand_cond}) ########")
        report(sub, ["base"] + COLORS + ["rand"])

    print("\ncolumn effects (painting+household mean dlog10) vs random:")
    ml = condition_ml(rows)
    r21 = column_effect(ml, "rand_L21")
    r14 = column_effect(ml, "rand_L14")
    print(f"  rand_L21: {r21:+.3f}   rand_L14: {r14:+.3f} "
          "(compare rand_L14 with the value_centered columns in FINDINGS.md)")
    for tag in ["objmod", "objinh"]:
        cells = {c: column_effect(ml, f"{c}|{tag}_L{OBJ21_LAYER}x{OBJ21_COEF}") for c in COLORS}
        print(f"  {tag}: " + "  ".join(f"{c}:{v:+.2f}" for c, v in cells.items())
              + f"   (minus rand_L21: " + "  ".join(f"{c}:{v - r21:+.2f}" for c, v in cells.items()) + ")")


# ==== calibrate: model dollars vs real prices ====================================================

CAL_ITEMS = [  # (item, approx real US price $)
    ("a banana", 0.25), ("a first-class postage stamp", 0.75), ("a loaf of white bread", 2.5),
    ("a dozen large eggs", 4), ("a gallon of whole milk", 4), ("a pound of ground beef", 5.5),
    ("a Big Mac", 5.5), ("a Starbucks grande latte", 5.5), ("a 12-pack of Coca-Cola cans", 7),
    ("a Frisbee", 10), ("a movie theater ticket", 12), ("an umbrella", 15),
    ("a paperback novel", 15), ("a standard basketball", 25), ("a yoga mat", 25),
    ("an electric kettle", 30), ("a two-slice toaster", 30), ("a pair of Levi's 501 jeans", 60),
    ("a sleeping bag", 60), ("a two-person camping tent", 80), ("an IKEA Billy bookcase", 90),
    ("a microwave oven", 100), ("an Instant Pot", 100),
    ("a pair of Nike Air Force 1 sneakers", 110), ("a DeWalt cordless drill", 130),
    ("a chainsaw", 200), ("a window air conditioner unit", 250),
    ("a pair of AirPods Pro", 250), ("a Nintendo Switch", 300), ("a gas push lawn mower", 300),
    ("a Dyson V8 cordless vacuum", 350), ("a GoPro HERO12 camera", 400),
    ("a KitchenAid stand mixer", 400), ("a PlayStation 5", 500),
    ("an entry-level Canon DSLR camera", 550), ("a 65-inch Samsung 4K TV", 700),
    ("a washing machine", 700), ("a queen-size mattress", 700), ("an iPhone 15", 800),
    ("a snow blower", 800), ("a Fender Player Stratocaster electric guitar", 850),
    ("a MacBook Air", 1100), ("a mid-range Trek mountain bike", 1200),
    ("a Peloton exercise bike", 1500), ("a French-door refrigerator", 1800),
    ("an ounce of gold", 2700), ("a John Deere riding lawn mower", 3000),
    ("a 1-carat diamond engagement ring", 5000), ("a Yamaha upright piano", 5000),
    ("a Rolex Submariner", 10000), ("a Hermes Birkin bag", 12000),
    ("a used 2015 Honda Civic", 12000), ("a new 20-foot pontoon boat", 30000),
    ("a new Toyota Camry", 28000), ("a Tesla Model 3", 40000), ("a new Ford F-150", 45000),
    ("a new Steinway grand piano", 100000),
]


def cmd_calibrate(args):
    out = results_dir("calibration")
    h = load()
    samples = h.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for it, _ in CAL_ITEMS], seed=0)
    rows = [dict(item=it, ref=ref, sample_idx=si, text=t, value=parse_dollars(t))
            for (it, ref), texts in zip(CAL_ITEMS, samples) for si, t in enumerate(texts)]
    save_json(out / "values.json", rows)

    est = {}
    for it, ref in CAL_ITEMS:
        vs = [math.log10(r["value"]) for r in rows if r["item"] == it and r["value"]]
        if vs:
            est[it] = sum(vs) / len(vs)
    refs = {it: math.log10(ref) for it, ref in CAL_ITEMS if it in est}
    x = [refs[it] for it in est]   # log10 real
    y = [est[it] for it in est]    # log10 model

    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)
    inter = my - slope * mx
    rk = lambda v: [float(sorted(v).index(a)) for a in v]
    errs = sorted(((est[it] - refs[it], it) for it in est), key=lambda e: e[0])
    med = sorted(abs(e) for e, _ in errs)[len(errs) // 2]

    lines = [f"calibration over {n} items (log10 $): pearson {pearson(x, y):+.3f}  "
             f"spearman {pearson(rk(x), rk(y)):+.3f}",
             f"OLS: log10(model) = {slope:.2f} * log10(real) + {inter:+.2f}   "
             f"median |dlog10| = {med:.2f}",
             "\nmost underestimated:"]
    lines += [f"  {it}: model ${10 ** est[it]:,.0f} vs real ${10 ** refs[it]:,.0f} ({e:+.2f})"
              for e, it in errs[:5]]
    lines.append("most overestimated:")
    lines += [f"  {it}: model ${10 ** est[it]:,.0f} vs real ${10 ** refs[it]:,.0f} ({e:+.2f})"
              for e, it in errs[-5:][::-1]]
    text = "\n".join(lines)
    (out / "analysis.txt").write_text(text + "\n")
    print(text)


# ==== tiers: the color x price-balanced item set =================================================
# 7 colors x 3 value tiers x 12 inherently- or iconically-colored items, each with a hand price
# estimate, verified against the model; items >0.5 log10 from their tier's model-median center
# get flagged for replacement. Tier targets: T1 ~ $2, T2 ~ $50, T3 ~ $1500 (log-spaced).
# Caveats: mid/high tiers lean on iconically-colored branded goods; indigo overlaps denim/violet.

TIER_TARGET = {1: 2, 2: 50, 3: 1500}

ITEMS = {  # (color, tier): [(item, my estimated price $)]
    ("red", 1): [("a red apple", 1), ("a can of Coca-Cola", 1), ("a red bell pepper", 1.5),
                 ("a bottle of ketchup", 3.5), ("a pint of strawberries", 4), ("a candy cane", 0.5),
                 ("a vine-ripened tomato", 0.8), ("a red rose stem", 3), ("a can of tomato soup", 1.5),
                 ("a bag of red lentils", 3), ("a cherry popsicle", 1), ("a red party balloon", 1)],
    ("red", 2): [("a Swiss Army knife", 40), ("a bouquet of a dozen red roses", 60),
                 ("a fire extinguisher", 50), ("a Manchester United home jersey", 90),
                 ("a Santa Claus costume", 50), ("a Radio Flyer wagon", 100),
                 ("a red flannel shirt", 40), ("a red umbrella", 20), ("a red toolbox", 40),
                 ("a hot sauce gift set", 30), ("a red throw blanket", 35), ("a red bicycle helmet", 50)],
    ("red", 3): [("a ruby ring", 2000), ("a garnet and gold necklace", 800),
                 ("an antique ruby brooch", 1500), ("a vintage Coca-Cola vending machine", 3000),
                 ("a red Vespa scooter", 5000), ("a signed Michael Jordan Bulls jersey", 2000),
                 ("a red leather recliner", 900), ("a red electric guitar", 700), ("a red kayak", 800),
                 ("a red gaming PC", 1500), ("a red moped", 1500), ("a ruby pendant necklace", 1200)],
    ("orange", 1): [("an orange", 0.5), ("a bunch of carrots", 1.5), ("a can of Fanta", 1),
                    ("a bag of Cheetos", 4), ("a pumpkin", 5), ("a persimmon", 1),
                    ("a sweet potato", 1), ("a bag of baby carrots", 1.5), ("a can of orange soda", 1),
                    ("a mango", 1.5), ("a box of mac and cheese", 1.5), ("a bag of candy corn", 2)],
    ("orange", 2): [("a hunting safety vest", 40), ("a Spalding basketball", 30),
                    ("a bottle of Aperol", 30), ("a framed monarch butterfly specimen", 60),
                    ("a life jacket", 40), ("a construction traffic cone set", 25),
                    ("an over-door basketball hoop", 30), ("an orange raincoat", 40),
                    ("a bottle of Grand Marnier", 45), ("an orange Nerf blaster set", 30),
                    ("a Halloween decoration set", 40), ("an orange beach towel set", 25)],
    ("orange", 3): [("a padparadscha sapphire ring", 2500), ("a citrine gemstone ring", 600),
                    ("a vintage neon Fanta sign", 800), ("a Harley-Davidson leather jacket", 600),
                    ("a basketball signed by LeBron James", 3000), ("a Loewe orange leather handbag", 2500),
                    ("an orange KitchenAid stand mixer", 450), ("a Stihl chainsaw", 400),
                    ("an orange kayak", 800), ("a Gretsch orange hollow-body guitar", 2000),
                    ("a citrine pendant and earring set", 700), ("a mid-century orange lounge chair", 900)],
    ("yellow", 1): [("a bunch of bananas", 1.5), ("a lemon", 0.5), ("a stick of butter", 1),
                    ("a box of Cheerios", 4), ("a yellow onion", 1), ("a can of pineapple chunks", 2),
                    ("a bag of popcorn kernels", 2), ("an ear of sweet corn", 0.75), ("a yellow squash", 1),
                    ("a box of yellow cake mix", 2), ("a can of corn", 1.2), ("a jar of mustard", 2.5)],
    ("yellow", 2): [("a yellow rain slicker", 40), ("a jar of manuka honey", 90),
                    ("a bottle of limoncello", 25), ("a beach umbrella", 40),
                    ("a construction hard hat", 25), ("a LEGO Creator set", 60),
                    ("a yellow umbrella", 20), ("a yellow bath towel set", 30),
                    ("a Pittsburgh Steelers cap", 30), ("a yellow tool chest", 60),
                    ("a yellow desk lamp", 35), ("a set of twenty rubber ducks", 25)],
    ("yellow", 3): [("a gold chain necklace", 1500), ("an ounce of gold", 2700),
                    ("100 grams of saffron", 800), ("a butterscotch Fender Telecaster", 900),
                    ("a gold bracelet", 1200), ("a yellow gold wedding band", 600),
                    ("a gold pendant necklace", 800), ("a yellow topaz ring", 700),
                    ("a yellow DeWalt power tool set", 600), ("a quarter-ounce gold coin", 700),
                    ("a yellow stand-up paddleboard", 900), ("a TV yellow Les Paul guitar", 1500)],
    ("green", 1): [("a head of lettuce", 1.5), ("a bunch of green grapes", 3), ("an avocado", 1.5),
                   ("a bunch of broccoli", 2), ("a can of Sprite", 1), ("a cucumber", 1),
                   ("a green bell pepper", 1), ("a bunch of celery", 1.5), ("a kiwi fruit", 0.75),
                   ("a jar of pickles", 3), ("a bag of frozen peas", 1.5), ("a bunch of fresh basil", 2)],
    ("green", 2): [("a jade bead bracelet", 50), ("a garden hose and sprinkler set", 40),
                   ("a potted monstera plant", 40), ("a bottle of Chartreuse", 70),
                   ("a Boston Celtics jersey", 90), ("two dozen Titleist golf balls", 90),
                   ("a set of gardening tools", 35), ("a green rain jacket", 45),
                   ("a bottle of Midori", 25), ("a green yoga mat set", 30),
                   ("three potted aloe vera plants", 30), ("a green Stanley thermos", 35)],
    ("green", 3): [("an emerald ring", 2500), ("a grade-A jade pendant", 1500),
                   ("a John Deere lawn tractor", 3000), ("an antique billiards table", 3000),
                   ("a thirty-year-old bonsai juniper", 1000), ("a set of vintage jade mahjong tiles", 1500),
                   ("a jade bangle", 1000), ("a Big Green Egg kamado grill", 1200),
                   ("an emerald pendant", 1500), ("a green Coleman canoe", 900),
                   ("a malachite jewelry box", 700), ("a peridot and gold necklace", 700)],
    ("blue", 1): [("a pint of blueberries", 4), ("a can of Pepsi", 1), ("a blue raspberry slushie", 2),
                  ("a bottle of Dawn dish soap", 3), ("a pack of blue ballpoint pens", 3),
                  ("a box of Oreos", 4), ("a blueberry muffin", 3), ("a can of blue Powerade", 2),
                  ("a blueberry yogurt cup", 1.5), ("a blue rubber ball", 2),
                  ("a pack of blue Post-it notes", 3), ("a blue spiral notebook", 3)],
    ("blue", 2): [("a pair of Levi's 501 jeans", 60), ("a Los Angeles Dodgers cap", 35),
                  ("a bottle of Bombay Sapphire gin", 30), ("a chambray button-down shirt", 50),
                  ("a Levi's denim trucker jacket", 80), ("a pair of navy Sperry boat shoes", 90),
                  ("a blue Columbia fleece", 45), ("a pair of blue Adidas Gazelles", 90),
                  ("a blue enamel teapot", 30), ("a Duke Blue Devils jersey", 80),
                  ("a Le Creuset mug set", 60), ("a French blue apron", 25)],
    ("blue", 3): [("a sapphire ring", 2500), ("a blue topaz and diamond pendant", 1000),
                  ("a Tiffany & Co. silver bracelet", 1500), ("an aquamarine ring", 1500),
                  ("an antique Delft porcelain vase", 1200), ("a Blue Note first-pressing jazz LP", 1000),
                  ("a blue topaz ring", 600), ("a Levi's Vintage Clothing jacket", 400),
                  ("a blue Rickenbacker guitar", 2000), ("a lapis inlay chess set", 800),
                  ("a Wedgwood dinner service", 1200), ("an opal and sapphire bracelet", 1500)],
    ("indigo", 1): [("a spool of indigo embroidery thread", 3), ("a box of blackberries", 4), ("a plum", 0.7),
                    ("a bag of blue corn tortilla chips", 4), ("a handful of ripe damson plums", 2),
                    ("a packet of morning glory seeds", 2), ("a bunch of black grapes", 2),
                    ("a can of acai berry drink", 3), ("a packet of cornflower seeds", 2),
                    ("a navy blue bandana", 3), ("a pair of navy shoelaces", 2),
                    ("a handful of juniper berries", 2)],
    ("indigo", 2): [("an indigo-dyed canvas tote bag", 30), ("a pair of Wrangler dark-wash jeans", 40),
                    ("a denim bucket hat", 30), ("a denim work shirt", 50),
                    ("a denim apron", 35), ("a kit of indigo fabric dye", 20),
                    ("a pair of Levi's 511 dark-wash jeans", 60), ("a navy hoodie", 35),
                    ("a denim overshirt", 45), ("a navy beanie and scarf set", 30),
                    ("a dark-wash denim vest", 40), ("a navy duffel bag", 45)],
    ("indigo", 3): [("a black opal pendant", 1500), ("an antique Japanese boro textile", 1500),
                    ("a lapis lazuli and silver necklace", 800), ("a vintage indigo-dyed kimono", 800),
                    ("an iolite gemstone necklace", 600), ("a vintage Evisu denim jacket", 500),
                    ("a navy Schott peacoat", 400), ("an RRL denim chore coat", 500),
                    ("a sapphire pendant", 1200), ("a pair of dark opal earrings", 1000),
                    ("a vintage US Navy deck jacket", 400), ("a small navy Persian rug", 800)],
    ("violet", 1): [("a bunch of Concord grapes", 3), ("an eggplant", 1.5), ("a can of grape soda", 1),
                    ("a head of purple cabbage", 2), ("a sprig of fresh lavender", 1), ("a turnip", 1),
                    ("a taro root", 1.5), ("a purple sweet potato", 1.5), ("a box of lavender tea bags", 3),
                    ("a grape lollipop", 0.5), ("a small bunch of violets", 2),
                    ("a potted purple hyacinth", 4)],
    ("violet", 2): [("a bottle of lavender essential oil", 20), ("an amethyst bead bracelet", 35),
                    ("a Los Angeles Lakers jersey", 90), ("a bottle of Crown Royal", 35),
                    ("a lilac bush sapling", 40), ("a dried-lavender wreath", 35),
                    ("a lavender candle set", 25), ("a small amethyst pendant", 40),
                    ("a bottle of creme de violette", 25), ("a purple bath bomb gift set", 20),
                    ("a Minnesota Vikings cap", 30), ("a potted orchid", 30)],
    ("violet", 3): [("a Tahitian peacock pearl necklace", 2000), ("an amethyst and gold necklace", 900),
                    ("an antique purple velvet settee", 1500), ("a Victorian amethyst brooch", 1200),
                    ("a rare purple orchid specimen", 500), ("a lavender jadeite carving", 1500),
                    ("a fine amethyst ring", 700), ("a purple velvet armchair", 800),
                    ("a purple Fender Jazz Bass", 1200), ("a pair of amethyst geode bookends", 500),
                    ("a Baccarat purple crystal vase", 800), ("a purple leather designer handbag", 1200)],
}
for c in COLORS:
    for t in (1, 2, 3):
        assert len(ITEMS[(c, t)]) == 12, (c, t)


def cmd_tiers(args):
    out = results_dir("balanced_tiers")
    h = load()
    flat = [(c, t, it, est) for (c, t), lst in ITEMS.items() for it, est in lst]
    samples = h.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for _, _, it, _ in flat], seed=0)
    rows = [dict(color=c, tier=t, item=it, my_estimate=est, sample_idx=si, text=txt,
                 value=parse_dollars(txt))
            for (c, t, it, est), texts in zip(flat, samples) for si, txt in enumerate(texts)]
    save_json(out / "values.json", rows)

    geo = mean_log10_by(rows, lambda r: (r["color"], r["tier"], r["item"]))

    # tier centers = median model log-value within tier; flag items >0.5 log10 away
    centers = {}
    for t in (1, 2, 3):
        vs = sorted(v for (c, tt, it), v in geo.items() if tt == t)
        centers[t] = vs[len(vs) // 2]
    summary, flagged = [], []
    for c, t, it, est in flat:
        g = geo.get((c, t, it))
        row = dict(color=c, tier=t, item=it, my_estimate=est,
                   model_geomean=round(10 ** g, 2) if g is not None else None,
                   dlog_vs_center=round(g - centers[t], 2) if g is not None else None)
        row["flagged"] = g is None or abs(g - centers[t]) > 0.5
        summary.append(row)
        if row["flagged"]:
            flagged.append(row)
    save_json(out / "balanced_tiers.json", summary)

    print("tier centers (model median): " +
          "  ".join(f"T{t}: ${10 ** centers[t]:,.0f} (target ${TIER_TARGET[t]})" for t in (1, 2, 3)))
    print(f"\nper color x tier model geomean $ (n=6 items):")
    print("        " + "".join(f"{f'T{t}':>10}" for t in (1, 2, 3)))
    for c in COLORS:
        cells = []
        for t in (1, 2, 3):
            vs = [v for (cc, tt, it), v in geo.items() if (cc, tt) == (c, t)]
            cells.append(10 ** (sum(vs) / len(vs)))
        print(f"{c:>8}" + "".join(f"{v:>10.3g}" for v in cells))
    for t in (1, 2, 3):
        per = [sum(v for (cc, tt, _), v in geo.items() if (cc, tt) == (c, t)) /
               max(1, sum(1 for (cc, tt, _) in geo if (cc, tt) == (c, t))) for c in COLORS]
        print(f"T{t} cross-color spread: {max(per) - min(per):.2f} log10")
    print(f"\nflagged ({len(flagged)}):")
    for r in flagged:
        print(f"  [{r['color']} T{r['tier']}] {r['item']}: model ${r['model_geomean']} "
              f"(est ${r['my_estimate']}, {r['dlog_vs_center']:+} vs center)")


# ==== CLI ========================================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("cross", help="valuation cross with raw prefer-vectors")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="inherent")
    p.add_argument("--stage", choices=["inspect", "pilot", "full"], default="full")
    p.set_defaults(fn=cmd_cross)

    p = sub.add_parser("centered", help="valuation cross with mean-centered vectors")
    p.add_argument("--mode", choices=["modifier", "inherent"], default="inherent")
    p.add_argument("--stage", choices=["pilot", "full"], default="full")
    p.set_defaults(fn=cmd_centered)

    p = sub.add_parser("items", help="value all 700 inherent items; price statistics")
    p.set_defaults(fn=cmd_items)

    p = sub.add_parser("obj21", help="L21 obj-centered cross + random controls (~60 min)")
    p.set_defaults(fn=cmd_obj21)

    p = sub.add_parser("calibrate", help="model dollars vs real prices")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("tiers", help="curate + verify the balanced item set")
    p.set_defaults(fn=cmd_tiers)

    args = ap.parse_args()
    args.fn(args)
