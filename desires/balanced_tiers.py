"""Color-balanced item tiers: decouple color from price.

Run: python balanced_tiers.py   (GPU, ~8 min)

Every prior result confounds color with the price of the things that carry it (blue = sapphires
and Levi's, green = vegetables). This file curates 7 colors x 3 value tiers x 12 inherently- or
iconically-colored items, each with a hand price estimate (approximate US dollars, recorded
below), then verifies against the model's estimates (same valuation prompt, 5 samples, no
steering). Items >0.5 log10 from their tier's model-median center get flagged for replacement.
Tier targets: T1 ~ $2, T2 ~ $50, T3 ~ $1500 (log-spaced). Caveats: mid/high tiers lean on
iconically-colored branded goods (a Coca-Cola can, Louboutin soles); indigo overlaps denim/
violet at the edges.
"""
import argparse
import math

from lib.data import COLORS
from lib.harness import load
from lib.io import save_json
from lib.paths import results_dir
from lib.stats import mean_log10_by
from lib.value_data import SUFFIX, TEMPLATE
from lib.valuation import parse_dollars

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

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
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
