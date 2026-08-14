"""Signed preference scores and tier/rank selection over comparison lists."""

K = 20  # extraction/steering set size used throughout
TIERS = {"worst": (0, 20), "neutral": (50, 70), "best": (100, 120)}  # of 120 pairs per color


def signed(diff, comp, color):
    """Logit diff oriented toward `color` (positive = model favors that color's side)."""
    return diff if comp["color_a"] == color else -diff


def sort_signed(comps, color, key="diff"):
    """This color's comparisons, ascending by signed score (worst first)."""
    mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: signed(c[key], c, color))


def tier_slice(comps, color, lo, hi, key="diff"):
    return sort_signed(comps, color, key)[lo:hi]


def worst_k(comps, color, k=K, key="diff"):
    return sort_signed(comps, color, key)[:k]


def top_wins(comps, color, k=K, key="diff"):
    """The k comparisons `color` won most strongly."""
    mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: signed(c[key], c, color), reverse=True)[:k]
