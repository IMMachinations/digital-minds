"""Valuation machinery (dollar parsing, sample rows, per-domain reporting) and the shared
pure-python statistics. spearman keeps the original min-rank tie handling
(sorted(v).index(a)) — do not swap in average ranks, committed numbers depend on it."""
import math
import re
from collections import defaultdict

from .data import COLORS
from .harness import count_suffix_tokens
from .util import load_json
from .value_data import SUFFIX, TEMPLATE

DOMAINS = ["painting", "household", "real"]

# The prompt ends with "$", so a continuation should start with a number. \b keeps "b" from
# matching the b of "bucks"; suffixes multiply (200k, 1.5 million).
_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn|k|m|b)?\b"
_LEAD = re.compile(r"^\s*" + _NUM, re.I)
_RANGE = re.compile(r"^\s*(?:-|–|—|to)\s*\$?\s*" + _NUM, re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def parse_dollars(text):
    """Leading dollar amount, or None. Ranges ('50 to $100') return the midpoint — the model
    stated an interval and its midpoint is the natural point estimate; taking the low end would
    bias steered-vs-baseline deltas wherever steering changes how often ranges appear."""
    m = _LEAD.match(text)
    if not m:
        return None
    def val(g):
        return float(g.group(1).replace(",", "")) * _MULT.get((g.group(2) or "").lower(), 1)
    r = _RANGE.match(text[m.end():])
    if r:
        lo, hi = val(m), val(r)
        if not m.group(2) and r.group(2):  # "20-30 million": the suffix distributes to both ends
            lo *= _MULT.get(r.group(2).lower(), 1)
        return (lo + hi) / 2
    return val(m)


def val_n_suffix(tok):
    """Steered span length of the valuation prompt (identical across items by design)."""
    full = TEMPLATE.format(item="x", suffix=SUFFIX)
    n = count_suffix_tokens(tok, full, TEMPLATE.format(item="x", suffix=""))
    assert tok.decode(tok(full).input_ids[-n:]).endswith("$")
    return n


def make_rows(items, samples, condition, cfg):
    """Sample rows in the standard schema; cfg=(layer, coef) or None for baseline."""
    return [dict(domain=it["domain"], item_color=it["color"], item=it["item"],
                 condition=condition, layer=cfg[0] if cfg else None,
                 coef=cfg[1] if cfg else None, sample_idx=si, text=t, value=parse_dollars(t))
            for it, texts in zip(items, samples) for si, t in enumerate(texts)]


def gen_rows(h, items, condition, steer, cfg, seed, n_suffix):
    """Generate one condition over `items` and return its sample rows. steer=(layer, vec) or
    None; cfg=(layer, coef) or None is what the rows record."""
    samples = h.generate([it["prompt"] for it in items], steer=steer, seed=seed,
                         n_suffix=n_suffix)
    return make_rows(items, samples, condition, cfg)


def base_rows(items, values_path):
    """Baseline rows reused from an earlier run (unsteered => identical condition)."""
    keys = {(it["domain"], it["color"], it["item"]) for it in items}
    rows = load_json(values_path)
    return [r for r in rows if r["condition"] == "base"
            and (r["domain"], r["item_color"], r["item"]) in keys]


# ---- aggregation / reporting --------------------------------------------------------------------
def val_stats(rows):
    """(plain mean, median, geometric mean, n_parsed, n_total) over parsed positive values."""
    import torch
    vals = torch.tensor([r["value"] for r in rows if r["value"]])
    if not len(vals):
        return float("nan"), float("nan"), float("nan"), 0, len(rows)
    return (vals.mean().item(), vals.median().item(),
            (10 ** vals.log10().mean()).item(), len(vals), len(rows))


def print_matrix(title, cell, conds, fmt="{:>10.3g}"):
    """cell(item_color, cond) -> float; rows = item color, cols = conditions."""
    print(f"\n{title}")
    print("        " + "".join(f"{c:>10}" for c in conds))
    for ic in COLORS:
        print(f"{ic:>8}" + "".join(fmt.format(cell(ic, c)) for c in conds))


def report(rows, conds):
    """Per-domain matrices of mean / geomean dollar value and deltas vs baseline; matched vs
    mismatched contrast on delta log10(geomean)."""
    def sel(domain, ic, cond):
        return [r for r in rows if (r["domain"], r["item_color"], r["condition"]) == (domain, ic, cond)]
    for domain in DOMAINS:
        if not any(r["domain"] == domain for r in rows):
            continue
        print(f"\n===== {domain} =====")
        S = {(ic, c): val_stats(sel(domain, ic, c)) for ic in COLORS for c in conds}
        print_matrix("mean $", lambda ic, c: S[ic, c][0], conds)
        print_matrix("median $", lambda ic, c: S[ic, c][1], conds)
        print_matrix("geomean $", lambda ic, c: S[ic, c][2], conds)
        if len(conds) > 1:
            dlog = {(ic, c): math.log10(S[ic, c][2] / S[ic, "base"][2])
                    if S[ic, c][2] > 0 and S[ic, "base"][2] > 0 else float("nan")
                    for ic in COLORS for c in conds[1:]}
            print_matrix("delta log10(geomean) vs base", lambda ic, c: dlog.get((ic, c), 0.0),
                         conds[1:], fmt="{:>10.3f}")
            steer_cols = [c for c in conds[1:] if c in COLORS]
            if steer_cols:
                matched = [dlog[ic, ic] for ic in steer_cols]
                mism = [dlog[ic, c] for ic in COLORS for c in steer_cols if ic != c]
                print(f"\nmatched (diag) mean dlog10: {sum(matched) / len(matched):+.3f}   "
                      f"mismatched mean dlog10: {sum(mism) / len(mism):+.3f}")
        bad = sum(1 for r in rows if r["domain"] == domain and r["value"] is None)
        print(f"unparseable: {bad}/{sum(1 for r in rows if r['domain'] == domain)}")


# ---- statistics ---------------------------------------------------------------------------------
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
