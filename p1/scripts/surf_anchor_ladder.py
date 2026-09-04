"""Layered anchor ladder for anchored Tier-2 measurement.

The 12 central anchors (A0, items_xl/anchors.json, pinned at the model's own
stage-1B (mu, sigma2)) span only mu ~[-1.3, +2.4]; SURF discoveries sit at
3..9 / -1.5..-3.5, where most A0 readouts are saturated and mu is extrapolated.
Wider rungs let saturated items be re-read against anchors near their own
level (surf_scores.Tier2Layered escalates only when needed):

  A1  2 per side from the 1B battery itself, |mu| in [2.6, 3.5] — values are
      already on the anchored scale (no measurement needed).
  A2  target |mu| 4-5.5, A3 target |mu| 5.5-8: candidates from XL and the
      probeloop discoveries, valued by a mini-battery (all pairs among A0
      extremes + A1 + candidates, 3 templates x 2 orders) fit with A0+A1
      pinned — so A2/A3 land on the same scale by chaining.

Writes items_xl/anchors_layered.json (flat list, A0 first, with rung/side) and
results/surf/s0/<model>/anchor_values_layered.json (id -> [mu, sigma2]), plus
the battery records for provenance.

Usage: uv run python scripts/surf_anchor_ladder.py qwen25-7b [--dry]
"""
import argparse
import itertools
import math
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

import thurstone

TARGETS = {2: (3.5, 5.0), 3: (5.0, 10.0)}  # rung -> |PRIOR mu| band to draw candidates from
N_CAND = 5                                  # candidates per side per prior band
N_KEEP = 2                                  # anchors kept per side per rung
A1_BAND = (2.6, 3.5)
RUNG_EDGES = [3.0, 4.0]                     # FITTED |mu| edges: rung 2 = [3.0, 4.0), rung 3 = >= 4.0
S2_MAX = 4.5                                # fitted sigma2 cap (positive tail items are all s2 3-4:
                                            # the model's preference among them is inconsistent)


def a1_pick(model):
    """2 per side from the 1B battery in A1_BAND, preferring low sigma2 and low
    1E frame-instability (the pick_anchors criteria)."""
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    stab = {}
    p = P1 / "results" / "stage1e" / model / "frame_utilities.json"
    if p.exists():
        stab = {r["id"]: r["stability_std"] for r in load_json(p)}
    out = []
    for side in (+1, -1):
        cand = [r for r in ut if A1_BAND[0] <= side * r["mu"] <= A1_BAND[1]]
        cand.sort(key=lambda r: (stab.get(r["id"], 0.5), r["sigma2"]))
        out += [{"id": r["id"], "text": r["text"], "rung": 1, "side": side,
                 "mu_1b": r["mu"], "s2_1b": r["sigma2"]} for r in cand[:N_KEEP]]
    return out


def tail_candidates(model, exclude):
    """N_CAND per side per rung from XL + probeloop discoveries, nearest the
    band centre (prior mu is the A0-extrapolated value; only used to pick)."""
    rows = [{"id": r["id"], "text": r["text"], "mu": r["mu"]}
            for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")]
    d = P1 / "results" / "surf" / "probeloop" / model
    for f in sorted(d.glob("discoveries_plc*.json")):
        rows += [{"id": f"surf:{r['text'][:40]}", "text": r["text"], "mu": r["mu"]}
                 for r in load_json(f)]
    seen, uniq = set(), []
    for r in rows:
        k = r["text"].lower()
        if k in seen or k in exclude:
            continue
        seen.add(k)
        uniq.append(r)
    out = []
    for rung, (lo, hi) in TARGETS.items():
        for side in (+1, -1):
            band = [r for r in uniq if lo <= side * r["mu"] <= hi]
            band.sort(key=lambda r: abs(abs(r["mu"]) - (lo + hi) / 2))
            out += [{"id": r["id"], "text": r["text"], "rung": rung, "side": side,
                     "mu_prior": r["mu"]} for r in band[:N_CAND]]
    return out


def battery(model, items, dry=False):
    """All-pairs mini-battery -> records {i, j, template, order, p}."""
    pairs = list(itertools.combinations(range(len(items)), 2))
    if dry:
        rng = np.random.RandomState(0)
        mu = np.array([it.get("mu_1b", it.get("mu_prior", 0.0)) for it in items])
        recs = []
        for i, j in pairs:
            for t in range(3):
                for o in (0, 1):
                    z = (mu[i] - mu[j]) / np.sqrt(2) + rng.normal(0, 0.3)
                    recs.append({"i": i, "j": j, "template": t, "order": o,
                                 "p": float(0.5 * (1 + math.erf(z / math.sqrt(2))))})
        return recs
    import harness
    from pairs import PairBattery
    h = harness.load(model)
    return PairBattery(h, items).run(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dry", action="store_true", help="synthetic battery (CPU smoke test)")
    ap.add_argument("--reuse", action="store_true", help="refit from the saved ladder_battery.json")
    a = ap.parse_args()
    model = a.model
    a0 = load_json(P1 / "items_xl" / "anchors.json")
    a0_vals = load_json(P1 / "results" / "surf" / "s0" / model / "anchor_values.json")
    a0_items = [{"id": x["id"], "text": x["text"], "rung": 0, "side": 0} for x in a0]
    a1 = a1_pick(model)
    ex = {x["text"].lower() for x in a0_items + a1}
    cands = tail_candidates(model, ex)

    # mini-battery: A0 extremes + A1 pinned, candidates free
    lo0 = min(a0, key=lambda x: a0_vals[x["id"]][0])
    hi0 = max(a0, key=lambda x: a0_vals[x["id"]][0])
    pinned = [{"id": x["id"], "text": x["text"], "mu_1b": a0_vals[x["id"]][0],
               "s2_1b": a0_vals[x["id"]][1]} for x in (lo0, hi0)] + a1
    items = pinned + cands
    s0_dir = P1 / "results" / "surf" / "s0" / model
    if a.reuse:
        saved = load_json(s0_dir / "ladder_battery.json")
        items, recs = saved["items"], saved["records"]
        pinned, cands = [x for x in items if "mu_1b" in x], [x for x in items if "mu_1b" not in x]
        for x in items:
            x.pop("mu_fit", None); x.pop("s2_fit", None)
    else:
        recs = battery(model, items, dry=a.dry)
    obs = [(r["i"], r["j"], r["p"]) for r in recs]
    fit = thurstone.fit_anchored(len(items), obs, list(range(len(pinned))),
                                 [x["mu_1b"] for x in pinned], [x["s2_1b"] for x in pinned])
    for k, it in enumerate(items):
        it["mu_fit"], it["s2_fit"] = round(fit["mu"][k], 4), round(fit["sigma2"][k], 4)

    # keep N_KEEP per side per rung: inside the band by fitted mu, lowest sigma2
    kept = []
    lines = [f"anchor ladder ({model}): battery {len(items)} items, {len(recs)} readouts, "
             f"nll {fit['nll']:.4f}"]
    for x in pinned:
        lines.append(f"  pinned {x.get('mu_1b'):+.2f} (fit {x['mu_fit']:+.2f})  {x['text'][:60]}")
    for c in sorted(cands, key=lambda c: c["mu_fit"]):
        lines.append(f"  cand prior {c['mu_prior']:+.2f} -> fit {c['mu_fit']:+.2f} "
                     f"s2 {c['s2_fit']:.2f}  {c['text'][:60]}")
    # rungs are assigned by FITTED mu (the prior is A0-extrapolated and unreliable)
    for c in cands:
        m = abs(c["mu_fit"])
        c["rung"] = 2 if RUNG_EDGES[0] <= m < RUNG_EDGES[1] else (3 if m >= RUNG_EDGES[1] else None)
        c["side"] = 1 if c["mu_fit"] > 0 else -1
    for rung in (2, 3):
        for side in (+1, -1):
            ok = [c for c in cands if c["rung"] == rung and c["side"] == side and c["s2_fit"] <= S2_MAX]
            ok.sort(key=lambda c: c["s2_fit"])
            kept += ok[:N_KEEP]
    kept.sort(key=lambda c: (c["rung"], -c["side"]))
    n_r = {r: {s: sum(c["rung"] == r and c["side"] == s for c in kept) for s in (1, -1)} for r in (2, 3)}
    lines.append(f"kept per rung/side: {n_r}  (fitted-mu edges {RUNG_EDGES}, s2 <= {S2_MAX})")
    flat = a0_items + [{"id": x["id"], "text": x["text"], "rung": 1, "side": x["side"]} for x in a1] \
        + [{"id": c["id"], "text": c["text"], "rung": c["rung"], "side": c["side"]} for c in kept]
    vals = {**{x["id"]: a0_vals[x["id"]] for x in a0_items},
            **{x["id"]: [x["mu_1b"], x["s2_1b"]] for x in a1},
            **{c["id"]: [c["mu_fit"], c["s2_fit"]] for c in kept}}
    for x in flat:
        lines.append(f"  rung {x['rung']} side {x['side']:+d} mu {vals[x['id']][0]:+.2f} "
                     f"s2 {vals[x['id']][1]:.2f}  {x['text'][:60]}")
    out_items = P1 / "items_xl" / ("anchors_layered_dry.json" if a.dry else "anchors_layered.json")
    s0 = P1 / "results" / "surf" / "s0" / model
    out_vals = s0 / ("anchor_values_layered_dry.json" if a.dry else "anchor_values_layered.json")
    save_json(out_items, {"model": model, "items": flat})
    save_json(out_vals, vals)
    save_json(s0 / ("ladder_battery_dry.json" if a.dry else "ladder_battery.json"),
              {"items": items, "records": recs, "nll": fit["nll"]})
    (s0 / ("ladder_summary_dry.txt" if a.dry else "ladder_summary.txt")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
