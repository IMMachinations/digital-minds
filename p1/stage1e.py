"""Stage 1E driver: cross-environment preference consistency.

Usage: uv run python stage1e.py <roster-name>     (all four roster models)
       uv run python stage1e.py cross             (cross-model aggregation)

Fixed 64-item subset (stratified by domain and consensus 1B utility) and a
fixed seeded pair set, identical across frames AND models. Per frame:
Thurstonian refit -> mu; outputs the frame x frame Spearman matrix, per-item
stability (std of frame-z-scored mu), per-domain stability, and robust/flipper
item lists for Stages 3-4.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import spearman

import frames
import harness
import items as items_mod
import thurstone

MODELS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
N_PER_DOMAIN = {"activities": 16, "objects": 12, "topics": 12, "selfstates": 12, "others": 12}


def consensus_z(its):
    zs = []
    for m in MODELS:
        ut = {r["id"]: r["mu"] for r in load_json(P1 / "results" / "stage1b" / m / "utilities.json")}
        v = np.array([ut[it["id"]] for it in its])
        zs.append((v - v.mean()) / v.std())
    return np.mean(zs, axis=0)


def pick_subset():
    """64 items, per-domain, evenly spaced along consensus utility (deterministic)."""
    its = items_mod.load_items()
    cz = consensus_z(its)
    subset = []
    for dom, n in N_PER_DOMAIN.items():
        rows = sorted(((cz[k], it) for k, it in enumerate(its) if it["domain"] == dom),
                      key=lambda r: r[0])
        idx = np.linspace(0, len(rows) - 1, n).round().astype(int)
        subset += [rows[i][1] for i in idx]
    assert len(subset) == 64
    return subset


def make_pairs(subset, seed=0, within=4, cross=2):
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for k, it in enumerate(subset):
        by_dom[it["domain"]].append(k)
    pairs = set()
    for k, it in enumerate(subset):
        same = [x for x in by_dom[it["domain"]] if x != k]
        other = [x for x in range(len(subset)) if subset[x]["domain"] != it["domain"]]
        for x in rng.sample(same, min(within, len(same))) + rng.sample(other, cross):
            pairs.add((min(k, x), max(k, x)))
    return sorted(pairs)


def run_model(model):
    subset = pick_subset()
    texts = [it["text"] for it in subset]
    pair_list = make_pairs(subset)
    out = P1 / "results" / "stage1e" / model
    out.mkdir(parents=True, exist_ok=True)
    h = harness.load(model)
    raw, mus = {}, {}
    for frame in frames.FRAME_NAMES:
        recs = frames.run_frame(h, texts, pair_list, frame)
        raw[frame] = recs
        fit = thurstone.fit(len(subset), [(r["i"], r["j"], r["p"]) for r in recs])
        mus[frame] = fit["mu"]
        print(f"{frame}: fitted ({len(pair_list)} pairs, nll={fit['nll']})")
    save_json(out / "pairs_raw.json", raw)
    save_json(out / "frame_utilities.json",
              [{"id": it["id"], "domain": it["domain"], "text": it["text"],
                **{f: round(mus[f][k], 4) for f in frames.FRAME_NAMES}}
               for k, it in enumerate(subset)])
    summarize(model)


def summarize(model):
    out = P1 / "results" / "stage1e" / model
    rows = load_json(out / "frame_utilities.json")
    F = frames.FRAME_NAMES
    mus = {f: [r[f] for r in rows] for f in F}
    lines = [f"{model}: Stage 1E cross-environment consistency (64 items, 5 frames)"]
    lines.append("frame x frame spearman:")
    lines.append("           " + "".join(f"{f:>9}" for f in F))
    M = {}
    for a in F:
        M[a] = {b: spearman(mus[a], mus[b]) for b in F}
        lines.append(f"  {a:8s}" + "".join(f"{M[a][b]:9.2f}" for b in F))
    off = [M[a][b] for ai, a in enumerate(F) for b in F[ai + 1:]]
    lines.append(f"mean off-diagonal rho: {np.mean(off):+.3f}")
    shift = sorted(((1 - M['bare'][f], f) for f in F if f != "bare"), reverse=True)
    lines.append("frame shift vs bare (1 - rho): "
                 + "  ".join(f"{f} {d:.2f}" for d, f in shift))

    z = {f: (np.array(mus[f]) - np.mean(mus[f])) / np.std(mus[f]) for f in F}
    stab = np.std(np.stack([z[f] for f in F]), axis=0)
    for r, s in zip(rows, stab):
        r["stability_std"] = round(float(s), 3)
    save_json(out / "frame_utilities.json", rows)
    by_dom = defaultdict(list)
    for r, s in zip(rows, stab):
        by_dom[r["domain"]].append(s)
    lines.append("per-domain mean instability (std of frame-z): "
                 + "  ".join(f"{d} {np.mean(v):.2f}" for d, v in sorted(by_dom.items())))
    order = np.argsort(stab)
    lines.append("most stable (robust preferences, feed Stage 3/4):")
    for k in order[:8]:
        lines.append(f"  {stab[k]:.2f}  [{rows[k]['domain']}] {rows[k]['text'][:64]}")
    lines.append("least stable (frame-dependent, feed instability analysis):")
    for k in order[-8:]:
        zs = " ".join(f"{f}:{z[f][k]:+.1f}" for f in F)
        lines.append(f"  {stab[k]:.2f}  [{rows[k]['domain']}] {rows[k]['text'][:48]}  ({zs})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cross():
    lines = ["Stage 1E cross-model summary", ""]
    stabs = {}
    for m in MODELS:
        p = P1 / "results" / "stage1e" / m / "frame_utilities.json"
        if not p.exists():
            continue
        rows = load_json(p)
        stabs[m] = {r["id"]: r["stability_std"] for r in rows}
        s = (P1 / "results" / "stage1e" / m / "summary.txt").read_text().splitlines()
        lines += [l for l in s if "mean off-diagonal" in l or "frame shift" in l
                  or "per-domain" in l]
        lines[-3] = f"---- {m}\n" + lines[-3]
    ms = sorted(stabs)
    lines.append("")
    lines.append("do models agree on WHICH items are frame-unstable? (spearman of stability):")
    ids = sorted(set.intersection(*(set(v) for v in stabs.values())))
    for ai, a in enumerate(ms):
        for b in ms[ai + 1:]:
            rho = spearman([stabs[a][i] for i in ids], [stabs[b][i] for i in ids])
            lines.append(f"  {a:12s} x {b:12s} {rho:+.3f}")
    (P1 / "results" / "stage1e" / "cross_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    args = ap.parse_args()
    if args.cmd == "cross":
        cross()
    else:
        run_model(args.cmd)


if __name__ == "__main__":
    main()
