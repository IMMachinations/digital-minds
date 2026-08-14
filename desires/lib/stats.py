"""Pure-python statistics shared across analyses. spearman keeps the original min-rank tie
handling (sorted(v).index(a)) — do not swap in average ranks, committed numbers depend on it."""
import math
from collections import defaultdict


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    rk = lambda v: [float(sorted(v).index(a)) for a in v]
    return pearson(rk(x), rk(y))


def mean_log10_by(rows, key_fn, skip=None):
    """Per-key mean log10 of positive parsed values ("geomean via mean log10")."""
    ml = defaultdict(list)
    for r in rows:
        if r["value"] and (skip is None or not skip(r)):
            ml[key_fn(r)].append(math.log10(r["value"]))
    return {k: sum(v) / len(v) for k, v in ml.items()}


def condition_ml(rows):
    """The valuation cross's standard aggregation: per (condition, domain, item_color, item)
    mean log10 over painting+household rows."""
    return mean_log10_by(rows, lambda r: (r["condition"], r["domain"], r["item_color"], r["item"]),
                         skip=lambda r: r["domain"] == "real")


def column_effect(ml, cond):
    """Mean dlog10(per-item geomean) of `cond` vs base over the keys both share."""
    ds = [ml[(cond, d, ic, it)] - ml[("base", d, ic, it)]
          for (c, d, ic, it) in ml if c == "base" and (cond, d, ic, it) in ml]
    return sum(ds) / len(ds) if ds else float("nan")


def tier_deltas(rows, src, cf, L):
    """Per-tier mean steering delta (7-color mean) from an objects_steer.json-style row list."""
    return {t: sum(r["delta"] for r in rows
                   if (r["source"], r["coef"], r["layer"], r["tier"]) == (src, cf, L, t)) / 7
            for t in ["worst", "neutral", "best"]}


def tier_components(d):
    """(uniform, antisymmetric) decomposition of a per-tier delta dict."""
    return sum(d.values()) / 3, (d["worst"] - d["best"]) / 2
