"""Color-balanced item tiers: decouple color from price.

Run: python balanced_tiers.py inherent   (GPU, ~8 min)

Every prior result confounds color with the price of the things that carry it (blue = sapphires
and Levi's, green = vegetables). This file curates 7 colors x 3 value tiers x 6 inherently- or
iconically-colored items, each with MY OWN price estimate (approximate US dollars, recorded
below), then verifies against the model's estimates (same valuation prompt, 5 samples, no
steering). Items >0.5 log10 from their tier's model-median center get flagged for replacement.
Tier targets: T1 ~ $2, T2 ~ $50, T3 ~ $1500 (log-spaced). Caveats: mid/high tiers lean on
iconically-colored branded goods (a Coca-Cola can, Louboutin soles); indigo overlaps denim/
violet at the edges.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import value as V
from data import COLORS
from value_data import TEMPLATE, SUFFIX

OUT = Path(__file__).parent / "results" / "balanced_tiers"
OUT.mkdir(parents=True, exist_ok=True)

TIER_TARGET = {1: 2, 2: 50, 3: 1500}

ITEMS = {  # (color, tier): [(item, my estimated price $)]
    ("red", 1): [("a red apple", 1), ("a can of Coca-Cola", 1), ("a red bell pepper", 1.5),
                 ("a bottle of ketchup", 3.5), ("a pint of strawberries", 4), ("a candy cane", 0.5)],
    ("red", 2): [("a Swiss Army knife", 40), ("a bouquet of a dozen red roses", 60),
                 ("a fire extinguisher", 50), ("a Manchester United home jersey", 90),
                 ("a Santa Claus costume", 50), ("a Radio Flyer wagon", 100)],
    ("red", 3): [("a ruby ring", 2000), ("a garnet and gold necklace", 800),
                 ("an antique ruby brooch", 1500), ("a vintage Coca-Cola vending machine", 3000),
                 ("a red Vespa scooter", 5000), ("a signed Michael Jordan Bulls jersey", 2000)],
    ("orange", 1): [("an orange", 0.5), ("a bunch of carrots", 1.5), ("a can of Fanta", 1),
                    ("a bag of Cheetos", 4), ("a pumpkin", 5), ("a persimmon", 1)],
    ("orange", 2): [("a hunting safety vest", 40), ("a Spalding basketball", 30),
                    ("a bottle of Aperol", 30), ("a framed monarch butterfly specimen", 60),
                    ("a life jacket", 40), ("a construction traffic cone set", 25)],
    ("orange", 3): [("a padparadscha sapphire ring", 2500), ("a citrine gemstone ring", 600),
                    ("a vintage neon Fanta sign", 800), ("a Harley-Davidson leather jacket", 600),
                    ("a basketball signed by LeBron James", 3000), ("a Loewe orange leather handbag", 2500)],
    ("yellow", 1): [("a bunch of bananas", 1.5), ("a lemon", 0.5), ("a stick of butter", 1),
                    ("a box of Cheerios", 4), ("a yellow onion", 1), ("a can of pineapple chunks", 2)],
    ("yellow", 2): [("a yellow rain slicker", 40), ("a jar of manuka honey", 90),
                    ("a bottle of limoncello", 25), ("a beach umbrella", 40),
                    ("a construction hard hat", 25), ("a LEGO Creator set", 60)],
    ("yellow", 3): [("a gold chain necklace", 1500), ("an ounce of gold", 2700),
                    ("100 grams of saffron", 800), ("a butterscotch Fender Telecaster", 900),
                    ("a gold bracelet", 1200), ("a yellow gold wedding band", 600)],
    ("green", 1): [("a head of lettuce", 1.5), ("a bunch of green grapes", 3), ("an avocado", 1.5),
                   ("a bunch of broccoli", 2), ("a can of Sprite", 1), ("a cucumber", 1)],
    ("green", 2): [("a jade bead bracelet", 50), ("a garden hose and sprinkler set", 40),
                   ("a potted monstera plant", 40), ("a bottle of Chartreuse", 70),
                   ("a Boston Celtics jersey", 90), ("two dozen Titleist golf balls", 90)],
    ("green", 3): [("an emerald ring", 2500), ("a grade-A jade pendant", 1500),
                   ("a John Deere lawn tractor", 3000), ("an antique billiards table", 3000),
                   ("a thirty-year-old bonsai juniper", 1000), ("a set of vintage jade mahjong tiles", 1500)],
    ("blue", 1): [("a pint of blueberries", 4), ("a can of Pepsi", 1), ("a blue raspberry slushie", 2),
                  ("a bottle of Dawn dish soap", 3), ("a pack of blue ballpoint pens", 3),
                  ("a box of Oreos", 4)],
    ("blue", 2): [("a pair of Levi's 501 jeans", 60), ("a Los Angeles Dodgers cap", 35),
                  ("a bottle of Bombay Sapphire gin", 30), ("a chambray button-down shirt", 50),
                  ("a Levi's denim trucker jacket", 80), ("a pair of navy Sperry boat shoes", 90)],
    ("blue", 3): [("a sapphire ring", 2500), ("a blue topaz and diamond pendant", 1000),
                  ("a Tiffany & Co. silver bracelet", 1500), ("an aquamarine ring", 1500),
                  ("an antique Delft porcelain vase", 1200), ("a Blue Note first-pressing jazz LP", 1000)],
    ("indigo", 1): [("a spool of indigo embroidery thread", 3), ("a box of blackberries", 4), ("a plum", 0.7),
                    ("a bag of blue corn tortilla chips", 4), ("a handful of ripe damson plums", 2),
                    ("a packet of morning glory seeds", 2)],
    ("indigo", 2): [("an indigo-dyed canvas tote bag", 30), ("a pair of Wrangler dark-wash jeans", 40),
                    ("a denim bucket hat", 30), ("a denim work shirt", 50),
                    ("a denim apron", 35), ("a kit of indigo fabric dye", 20)],
    ("indigo", 3): [("a black opal pendant", 1500), ("an antique Japanese boro textile", 1500),
                    ("a lapis lazuli and silver necklace", 800), ("a vintage indigo-dyed kimono", 800),
                    ("an iolite gemstone necklace", 600), ("a vintage Evisu denim jacket", 500)],
    ("violet", 1): [("a bunch of Concord grapes", 3), ("an eggplant", 1.5), ("a can of grape soda", 1),
                    ("a head of purple cabbage", 2), ("a sprig of fresh lavender", 1), ("a turnip", 1)],
    ("violet", 2): [("a bottle of lavender essential oil", 20), ("an amethyst bead bracelet", 35),
                    ("a Los Angeles Lakers jersey", 90), ("a bottle of Crown Royal", 35),
                    ("a lilac bush sapling", 40), ("a dried-lavender wreath", 35)],
    ("violet", 3): [("a Tahitian peacock pearl necklace", 2000), ("an amethyst and gold necklace", 900),
                    ("an antique purple velvet settee", 1500), ("a Victorian amethyst brooch", 1200),
                    ("a rare purple orchid specimen", 500), ("a lavender jadeite carving", 1500)],
}
for c in COLORS:
    for t in (1, 2, 3):
        assert len(ITEMS[(c, t)]) == 6, (c, t)

if __name__ == "__main__":
    flat = [(c, t, it, est) for (c, t), lst in ITEMS.items() for it, est in lst]
    samples = V.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for _, _, it, _ in flat], seed=0)
    rows = [dict(color=c, tier=t, item=it, my_estimate=est, sample_idx=si, text=txt,
                 value=V.parse_dollars(txt))
            for (c, t, it, est), texts in zip(flat, samples) for si, txt in enumerate(texts)]
    (OUT / "values.json").write_text(json.dumps(rows, indent=1))

    ml = defaultdict(list)
    for r in rows:
        if r["value"]:
            ml[(r["color"], r["tier"], r["item"])].append(math.log10(r["value"]))
    geo = {k: sum(v) / len(v) for k, v in ml.items()}

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
    (OUT / "balanced_tiers.json").write_text(json.dumps(summary, indent=1))

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
