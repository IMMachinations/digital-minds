"""Reliability of the anchored Tier-2 mu, and the noise of reduced designs.

Readouts are deterministic logits, so re-running the same prompts reproduces
mu exactly; reliability has to come from splitting the DESIGN (anchors,
orders, templates). One layered readout pass (surf_scores.Tier2Layered, the
72-readout A0 design + escalation to wider rungs for saturated items) on a
mu-stratified sample, then anchored fits on record subsets:

  a0_full        A0 anchors only, all templates/orders   (= Tier2Full, the protocol so far)
  layered_full   everything (reference mu)
  anchor_even / anchor_odd, order0 / order1, tmpl0/1/2   (split halves)
  mid_layered    templates {0} x 2 orders x 12 A0 (+ escalation)  = the gap-loop in-loop design
  mid_a0, mid6x2x1, mid6x1x1                              (cheaper alternatives)

Each design is fitted with free sigma2 and with sigma2 fixed at 1 (reduced
designs barely identify sigma2). Reported vs the reference: pearson, spearman,
residual SD — overall, in-span (rung 0), and per rung; split-half r with the
Spearman-Brown step-up; whether the per-item half-disagreement predicts the
mid-design error (decides gap_use_delta); the tail shift layered - A0-only;
and the calibrated probe's error SD on the same items. summary.json["se0_mid"]
= in-span residual SD of mid_layered (free sigma2) vs the reference — the
gap-loop noise floor.

Usage: uv run python scripts/surf_mu_reliability.py qwen25-7b [--n 150] [--reuse]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import surf_scores
from surf_s0 import _stratified
from surf_probeloop import _ds_acts, _load_probe, apply_probe, apply_calib as _calib, out_dir as pl_dir

SPAN = 2.5   # |mu| beyond which the A0 anchors [-1.3, 2.4] no longer bracket an item


def rel_dir(model):
    d = P1 / "results" / "surf" / "reliability" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_items(model, n, seed=0):
    rows = [{"text": r["text"], "mu_prior": r["mu"], "source": "xl"}
            for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
            if not r.get("validation")]
    for f in sorted(pl_dir(model).glob("discoveries_plc*.json")):
        rows += [{"text": r["text"], "mu_prior": r["mu"], "source": f.stem.replace("discoveries_", "")}
                 for r in load_json(f)]
    seen, uniq = set(), []
    for r in rows:
        k = r["text"].lower()
        if k not in seen:
            seen.add(k)
            uniq.append({**r, "mu": r["mu_prior"]})
    return _stratified(uniq, n, seed=seed)


def designs(a0, six):
    a0s, sixs = set(a0), set(six)
    return {
        "a0_full": lambda r: r["anchor"] in a0s,
        "layered_full": lambda r: True,
        "anchor_even": lambda r: r["anchor"] % 2 == 0,
        "anchor_odd": lambda r: r["anchor"] % 2 == 1,
        "order0": lambda r: r["order"] == 0,
        "order1": lambda r: r["order"] == 1,
        "tmpl0": lambda r: r["t"] == 0,
        "tmpl1": lambda r: r["t"] == 1,
        "tmpl2": lambda r: r["t"] == 2,
        "mid_layered": lambda r: r["t"] == 0,
        "mid_a0": lambda r: r["t"] == 0 and r["anchor"] in a0s,
        "mid6x2x1": lambda r: r["t"] == 0 and r["anchor"] in sixs,
        "mid6x1x1": lambda r: r["t"] == 0 and r["order"] == 0 and r["anchor"] in sixs,
    }


def _stats(pred, ref, mask=None):
    pred, ref = np.asarray(pred), np.asarray(ref)
    if mask is not None:
        pred, ref = pred[mask], ref[mask]
    if len(ref) < 4:
        return {"n": int(len(ref))}
    return {"n": int(len(ref)), "pearson": round(pearson(list(pred), list(ref)), 3),
            "spearman": round(spearman(list(pred), list(ref)), 3),
            "resid_sd": round(float(np.std(pred - ref)), 3),
            "resid_mean": round(float(np.mean(pred - ref)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--reuse", action="store_true", help="reuse records.json (no GPU pass)")
    a = ap.parse_args()
    model, d = a.model, rel_dir(a.model)

    if a.reuse and (d / "records.json").exists():
        blob = load_json(d / "records.json")
        items, recs, fitted = blob["items"], blob["records"], blob["fitted"]
        lad = surf_scores.load_anchor_ladder(model)
        anchor_vals, rungs = lad[1], lad[2]
    else:
        items = sample_items(model, a.n)
        handles = surf_scores.Handles(model)
        t2 = surf_scores.Tier2Layered(handles, model, design=surf_scores.FULL_DESIGN)
        anchor_vals, rungs = t2.anchor_vals, t2.rungs
        fitted, recs = t2.score_with_records([it["text"] for it in items])
        save_json(d / "records.json", {"items": items, "records": recs, "fitted": fitted})
    n = len(items)
    a0 = rungs[0]
    six = [a0[i] for i in surf_scores._spaced(len(a0), 6)]
    rung = np.array([f["rung"] for f in fitted])
    sat = np.array([max(f["sat_hi"], f["sat_lo"]) for f in fitted])

    mus = {}
    for name, keep in designs(a0, six).items():
        for s2 in (None, 1.0):
            mu, _ = surf_scores.fit_records(recs, n, anchor_vals, keep=keep, fix_s2=s2)
            mus[(name, s2)] = np.array(mu)
    ref = mus[("layered_full", None)]
    in_span = rung == 0
    tail = ~in_span

    out = {"n": n, "n_readouts": len(recs), "n_in_span": int(in_span.sum()),
           "rung_counts": {int(r): int((rung == r).sum()) for r in sorted(set(rung.tolist()))},
           "sat_after_escalation": {"mean": round(float(sat.mean()), 3),
                                    "frac_tail_below_.5": round(float((sat[tail] < .5).mean()), 3)
                                    if tail.any() else None},
           "designs": {}}
    lines = [f"mu reliability ({model}): n={n} items, {len(recs)} readouts; in-span (rung 0) "
             f"{int(in_span.sum())}, escalated {int(tail.sum())} "
             f"(rungs {out['rung_counts']}); mean residual saturation {sat.mean():.2f}",
             "design            s2     overall r/rho/sd      in-span r/rho/sd      tail r/rho/sd"]
    for name in designs(a0, six):
        for s2 in (None, 1.0):
            m = mus[(name, s2)]
            rep = {"overall": _stats(m, ref), "in_span": _stats(m, ref, in_span),
                   "tail": _stats(m, ref, tail),
                   "by_rung": {int(r): _stats(m, ref, rung == r) for r in sorted(set(rung.tolist()))}}
            out["designs"][f"{name}|s2={'free' if s2 is None else s2}"] = rep
            f = lambda s: (f"{s['pearson']:+.3f}/{s['spearman']:+.3f}/{s['resid_sd']:.2f}"
                           if "pearson" in s else "   n<4   ")
            lines.append(f"{name:17s} {'free' if s2 is None else '1.0 ':5s} "
                         f"{f(rep['overall']):22s} {f(rep['in_span']):22s} {f(rep['tail'])}")

    # split-half reliability (Spearman-Brown to the full design)
    def sb(a, b, mask=None):
        x, y = mus[a], mus[b]
        if mask is not None:
            x, y = x[mask], y[mask]
        r = pearson(list(x), list(y))
        return {"r_half": round(r, 3), "r_full_sb": round(2 * r / (1 + r), 3)}
    out["split_half"] = {}
    lines.append("split-half (Spearman-Brown to full):")
    for label, (pa, pb) in {"anchors": ("anchor_even", "anchor_odd"),
                            "orders": ("order0", "order1"),
                            "templates_0v1": ("tmpl0", "tmpl1"),
                            "templates_0v2": ("tmpl0", "tmpl2")}.items():
        for s2 in (None, 1.0):
            key = f"{label}|s2={'free' if s2 is None else s2}"
            out["split_half"][key] = {"overall": sb((pa, s2), (pb, s2)),
                                      "in_span": sb((pa, s2), (pb, s2), in_span),
                                      "tail": sb((pa, s2), (pb, s2), tail) if tail.sum() > 3 else None}
            o, i = out["split_half"][key]["overall"], out["split_half"][key]["in_span"]
            lines.append(f"  {key:24s} overall r_half {o['r_half']:+.3f} -> {o['r_full_sb']:+.3f} | "
                         f"in-span {i['r_half']:+.3f} -> {i['r_full_sb']:+.3f}")

    # does the per-item half-disagreement predict the mid-design error?
    delta = np.abs(mus[("anchor_even", None)] - mus[("anchor_odd", None)])
    mid_err = np.abs(mus[("mid_layered", None)] - ref)
    out["delta_predicts_error"] = {
        "corr_all": round(pearson(list(delta), list(mid_err)), 3),
        "corr_in_span": round(pearson(list(delta[in_span]), list(mid_err[in_span])), 3)
        if in_span.sum() > 3 else None,
        "spearman_all": round(spearman(list(delta), list(mid_err)), 3)}
    lines.append(f"delta (|even-odd|, free s2) vs |mid - ref|: r {out['delta_predicts_error']['corr_all']:+.3f} "
                 f"(in-span {out['delta_predicts_error']['corr_in_span']}), "
                 f"rho {out['delta_predicts_error']['spearman_all']:+.3f}")

    # tail shift: what escalation changed
    shift = ref - mus[("a0_full", None)]
    bins = [(-99, -SPAN), (-SPAN, -1.32), (-1.32, 2.37), (2.37, SPAN), (SPAN, 5), (5, 99)]
    a0mu = mus[("a0_full", None)]
    out["tail_shift"] = {}
    lines.append("layered - A0-only mu by A0 mu bin:")
    for lo, hi in bins:
        m = (a0mu >= lo) & (a0mu < hi)
        if m.sum():
            out["tail_shift"][f"[{lo},{hi})"] = {"n": int(m.sum()), "mean": round(float(shift[m].mean()), 3),
                                                  "sd": round(float(shift[m].std()), 3)}
            lines.append(f"  [{lo:>4},{hi:>4})  n={int(m.sum()):3d}  shift {shift[m].mean():+.2f} "
                         f"(sd {shift[m].std():.2f})")

    # calibrated probe error on the same items
    v = max(int(p.stem.split("_v")[1]) for p in pl_dir(model).glob("probe_v*.pt"))
    acts = _ds_acts(model, items, d / "acts.pt")
    pr = _load_probe(model, v)
    pc = _calib(load_json(pl_dir(model) / f"calib_v{v}.json"), apply_probe(pr, acts))
    out["probe"] = {"version": v, "overall": _stats(pc, ref), "in_span": _stats(pc, ref, in_span),
                    "tail": _stats(pc, ref, tail),
                    "vs_a0_only": _stats(pc, a0mu)}
    lines.append(f"calibrated probe v{v} vs layered mu: r {out['probe']['overall'].get('pearson')} "
                 f"resid sd {out['probe']['overall'].get('resid_sd')} "
                 f"(in-span {out['probe']['in_span'].get('resid_sd')}, tail {out['probe']['tail'].get('resid_sd')}); "
                 f"vs A0-only mu: r {out['probe']['vs_a0_only'].get('pearson')} sd {out['probe']['vs_a0_only'].get('resid_sd')}")

    se0 = out["designs"]["mid_layered|s2=free"]["in_span"].get("resid_sd")
    out["se0_mid"] = se0
    out["gates"] = {"mid_r_ge_.9": bool(out["designs"]["mid_layered|s2=free"]["in_span"].get("pearson", 0) >= .9),
                    "split_half_anchors_ge_.8": bool(out["split_half"]["anchors|s2=free"]["in_span"]["r_half"] >= .8),
                    "escalation_desaturates": bool((out["sat_after_escalation"]["frac_tail_below_.5"] or 0) >= .9)}
    out["mu_a0_full"] = [round(float(x), 4) for x in a0mu]        # for figures.f43
    out["mu_layered_full"] = [round(float(x), 4) for x in ref]
    lines.append(f"se0_mid (in-span resid sd of mid_layered, free s2) = {se0}; gates {out['gates']}")
    save_json(d / "summary.json", out)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
