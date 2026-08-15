"""Adaptive pair selection for the 1B battery (Utility-Engineering style).

Round 0 seeds connectivity: every item gets random partners, mostly within its
domain plus a couple cross-domain (cross-domain pairs put all domains on one
utility scale). Later rounds pick the pairs with the highest Fisher information
under the current Thurstonian fit — near-tied, high-uncertainty pairs — with a
per-item cap so the selector cannot pile onto two ambiguous items.
"""
import math
import random


def round0(items, seed=0, within=6, cross=2):
    rng = random.Random(seed)
    by_dom = {}
    for idx, it in enumerate(items):
        by_dom.setdefault(it["domain"], []).append(idx)
    pairs = set()
    for idx, it in enumerate(items):
        same = [k for k in by_dom[it["domain"]] if k != idx]
        other = [k for k in range(len(items)) if items[k]["domain"] != it["domain"]]
        for k in rng.sample(same, min(within, len(same))) + rng.sample(other, cross):
            pairs.add((min(idx, k), max(idx, k)))
    return sorted(pairs)


def fisher_info(mu, s2, i, j):
    v = s2[i] + s2[j]
    z = (mu[i] - mu[j]) / math.sqrt(v)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    P = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    P = min(max(P, 1e-9), 1 - 1e-9)
    return phi * phi / (P * (1 - P) * v)


def select(fit, measured, n_items, budget=600, cap=6, seed=0):
    """Top-`budget` unmeasured pairs by Fisher information, <= `cap` new pairs
    per item per round."""
    mu, s2 = fit["mu"], fit["sigma2"]
    cands = [(i, j) for i in range(n_items) for j in range(i + 1, n_items)
             if (i, j) not in measured]
    rng = random.Random(seed)
    if len(cands) > 30000:
        cands = rng.sample(cands, 30000)
    cands.sort(key=lambda p: fisher_info(mu, s2, *p), reverse=True)
    picked, used = [], {}
    for i, j in cands:
        if used.get(i, 0) >= cap or used.get(j, 0) >= cap:
            continue
        picked.append((i, j))
        used[i] = used.get(i, 0) + 1
        used[j] = used.get(j, 0) + 1
        if len(picked) >= budget:
            break
    return picked
