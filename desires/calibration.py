"""Model dollar estimates vs real-world prices.

Run: python calibration.py   (GPU, ~3 min)

~50 items with well-known prices, valued with the same prompt as every other experiment here
(5 samples, no steering). Reference prices are approximate 2025-ish US prices from model
knowledge, stated to one significant-ish figure — the point is order-of-magnitude calibration,
not exact retail tracking.
"""
import argparse
import math

from lib.harness import load
from lib.io import save_json
from lib.paths import results_dir
from lib.stats import pearson
from lib.value_data import SUFFIX, TEMPLATE
from lib.valuation import parse_dollars

ITEMS = [  # (item, approx real US price $)
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

if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    out = results_dir("calibration")
    h = load()
    samples = h.generate([TEMPLATE.format(item=it, suffix=SUFFIX) for it, _ in ITEMS], seed=0)
    rows = [dict(item=it, ref=ref, sample_idx=si, text=t, value=parse_dollars(t))
            for (it, ref), texts in zip(ITEMS, samples) for si, t in enumerate(texts)]
    save_json(out / "values.json", rows)

    est = {}
    for it, ref in ITEMS:
        vs = [math.log10(r["value"]) for r in rows if r["item"] == it and r["value"]]
        if vs:
            est[it] = sum(vs) / len(vs)
    refs = {it: math.log10(ref) for it, ref in ITEMS if it in est}
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
