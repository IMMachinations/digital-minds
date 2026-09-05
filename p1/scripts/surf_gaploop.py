"""Gap-objective probe loop ("gaploop"): search for items where the CALIBRATED
probe and the measured mu disagree, retrain, repeat.

Continues the probeloop's probe numbering from BASE_V. Gap cycle k searches
against probe v(BASE_V+k-1) / calib v(BASE_V+k-1) with fitness = z-scored
signed gap (surf_scores.GapScorer; mid-design layered Tier-2 mu in-loop,
layered 72-readout mu_full on buffer entrants), two arms:
  over   probe rates the item HIGHER than measured mu
  under  probe rates it LOWER
then evaluates the fresh discoveries under every honest probe (v <= the one
searched against): signed / absolute calibrated gap vs mu_full and vs mu_comp
(a re-readout on templates != 0, i.e. disjoint from the in-loop mid design, to
strip selection noise), split in-span / escalated, plus a Tier-3 referee on the
top-20 |z| items reporting whether revealed choice sides with the probe or with
stated mu. Harvest overlays the layered re-measurement of the old tails
(scripts/surf_remeasure.py) before fitting v(BASE_V+k).

Usage:
  uv run python scripts/surf_gaploop.py search  qwen25-7b --k 1 --arm over
  uv run python scripts/surf_gaploop.py eval    qwen25-7b --k 1 --arm over
  uv run python scripts/surf_gaploop.py harvest qwen25-7b --k 1     # -> dataset_c{BASE_V+k-1}
  uv run python scripts/surf_gaploop.py fit     qwen25-7b --k 1     # -> probe_v{BASE_V+k}
  uv run python scripts/surf_gaploop.py cycle   qwen25-7b --k 3
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

import surf_probeloop as pl

BASE_V = 10
ARMS = ("over", "under")
N_PLC = 9          # probeloop cycles that feed the harvest
SPAN_RUNG = 0      # rung 0 = measured within the A0 span


def exp(k):
    return f"gpl{k}"


def probe_v(k):
    return BASE_V + k - 1


def _reliability(model):
    p = P1 / "results" / "surf" / "reliability" / model / "summary.json"
    return load_json(p) if p.exists() else {}


def _run_dir(model, k, arm):
    import surf
    return surf.SURF_ROOT / exp(k) / model / f"{arm}-s0"


# ---- search -------------------------------------------------------------------------------------

def cmd_search(model, k, arm, T=15, n_cand=192, patience=3):
    import surf
    import surf_scores
    rel = _reliability(model)
    dpe = (rel.get("delta_predicts_error") or {}).get("corr_in_span")
    cfg = surf.RunConfig(
        experiment=exp(k), model=model, direction=arm, fitness="gap",
        allowed_tiers=["t0", "t1", "t2"], pool_file="items/surf_attributes_item.json",
        pool_kind="item", pool_init="", seed=0, T=T, n_cand=n_cand, patience=patience,
        probe_path=str(pl.probe_path(model, probe_v(k))),
        calib_path=str(pl.out_dir(model) / f"calib_v{probe_v(k)}.json"),
        gap_se0=float(rel.get("se0_mid") or 0.0), gap_use_delta=(dpe or 0.0) >= 0.2,
        t2_layered=True, generator=surf_scores.GENERATOR)
    rd = cfg.out_dir()
    st, _ = surf._load_state(rd) if rd.exists() else (None, -1)
    if st and st.get("stopped"):
        print(f"{cfg.run_id}: already terminated")
        return
    surf.run(cfg, surf_scores.build(cfg))


# ---- eval ---------------------------------------------------------------------------------------

def _detail(model, k, arm):
    """gap_detail.jsonl rows by lowercase text (last write wins)."""
    p = _run_dir(model, k, arm) / "gap_detail.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["text"].lower()] = r
    return out


def _provenance(model, k, arm):
    """rung_full / sat_full of buffer entrants from the run's iter jsonls."""
    out = {}
    for p in sorted(_run_dir(model, k, arm).glob("iter_*.jsonl")):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("mu_full") is not None:
                out[r["text"].lower()] = {"rung": r.get("rung_full"), "sat": r.get("sat_full")}
    return out


def _gap_report(pc, mu, mask=None):
    pc, mu = np.asarray(pc), np.asarray(mu)
    if mask is not None:
        pc, mu = pc[mask], mu[mask]
    if len(mu) < 4:
        return {"n": int(len(mu))}
    g = pc - mu
    return {"n": int(len(mu)), "gap_mean": round(float(g.mean()), 3),
            "abs_gap_mean": round(float(np.abs(g).mean()), 3),
            "frac_abs_gap_gt2": round(float((np.abs(g) > 2).mean()), 3),
            "pearson": round(pearson(list(pc), list(mu)), 3),
            "spearman": round(spearman(list(pc), list(mu)), 3)}


def cmd_eval(model, k, arm, n_rolls=12):
    import surf_scores
    from surf_e2_referee import heldout_env_ids
    d = pl.out_dir(model)
    fresh = pl.cmd_harvest(model, cycle=-1, exps=[exp(k)], direction=arm)
    (d / "dataset_c-1.json").unlink()  # side artifact of the exps override
    assert fresh, f"no measured discoveries in {exp(k)} ({arm})"
    prov, det = _provenance(model, k, arm), _detail(model, k, arm)
    for r in fresh:
        key = r["text"].lower()
        r.update(prov.get(key, {}))
        if key in det:
            r["z_search"] = det[key]["z"]
            r["mu_mid"] = det[key]["mu_mid"]
            r["probe_cal_search"] = det[key]["probe_cal"]
    save_json(d / f"discoveries_{exp(k)}_{arm}.json", fresh)
    acts = pl._ds_acts(model, fresh, d / f"acts_{exp(k)}_{arm}.pt")
    texts = [r["text"] for r in fresh]
    mu = np.array([r["mu"] for r in fresh])
    rung = np.array([r.get("rung") if r.get("rung") is not None else -1 for r in fresh])
    in_span = rung == SPAN_RUNG

    # complementary mu: templates != 0 (disjoint from the in-loop mid design)
    handles = surf_scores.Handles(model)
    t2 = surf_scores.Tier2Layered(handles, model, design=surf_scores.FULL_DESIGN)
    fitted, recs = t2.score_with_records(texts)
    mu_comp, _ = surf_scores.fit_records(recs, len(texts), t2.anchor_vals, keep=lambda r: r["t"] != 0)
    mu_comp = np.array(mu_comp)
    for r, f, mc in zip(fresh, fitted, mu_comp):
        r["mu_comp"] = round(float(mc), 4)
        r["mu_relayered"] = round(f["mu"], 4)
    save_json(d / f"discoveries_{exp(k)}_{arm}.json", fresh)

    row = {"cycle": k, "arm": arm, "n_new": len(fresh),
           "rung_counts": {int(r): int((rung == r).sum()) for r in sorted(set(rung.tolist()))},
           "probe_searched": probe_v(k), "per_probe": {}}
    for v in range(0, probe_v(k) + 1):
        pr = pl._load_probe(model, v)
        raw = pl.apply_probe(pr, acts)
        cpath = d / f"calib_v{v}.json"
        pc = pl.apply_calib(load_json(cpath), raw) if cpath.exists() else raw
        row["per_probe"][f"v{v}"] = {
            "vs_mu_full": _gap_report(pc, mu), "vs_mu_comp": _gap_report(pc, mu_comp),
            "in_span": _gap_report(pc, mu, in_span), "escalated": _gap_report(pc, mu, ~in_span),
            "raw_pearson": round(pearson(list(raw), list(mu)), 3)}
        if v == probe_v(k):
            pc_cur = pc

    # referee: top-20 by search z (largest = most in the arm's direction)
    z = np.array([r.get("z_search", 0.0) for r in fresh])
    top = list(np.argsort(-z)[:20])
    t3 = surf_scores.Tier3Revealed(handles, model, direction="max", n_rolls=n_rolls,
                                   anchor_ids=heldout_env_ids(model))
    rates = t3.score([texts[i] for i in top])
    row["t3_top20"] = [{"text": texts[i], "rate": round(c, 3), "z": round(float(z[i]), 3),
                        "probe_cal": round(float(pc_cur[i]), 3), "mu": round(float(mu[i]), 3),
                        "rung": int(rung[i])} for i, c in zip(top, rates)]
    row["t3_top20_mean"] = round(float(np.mean(rates)), 3)
    def _rho(a, b):  # constant rates (e.g. nothing ever chosen) -> undefined
        return round(spearman(list(a), list(b)), 3) if len(set(a)) > 1 and len(set(b)) > 1 else None
    row["t3_rate_vs_probe"] = _rho(rates, [float(pc_cur[i]) for i in top])
    row["t3_rate_vs_mu"] = _rho(rates, [float(mu[i]) for i in top])

    cpath = d / f"gap_cycles_{arm}.json"
    cycles = load_json(cpath) if cpath.exists() else []
    cycles = sorted([c for c in cycles if c["cycle"] != k] + [row], key=lambda c: c["cycle"])
    save_json(cpath, cycles)
    write_summary(model)
    print((d / "summary_gap.txt").read_text())


def write_summary(model):
    d = pl.out_dir(model)
    lines = [f"gaploop cycles ({model}); gap = calibrated probe - mu (mu units); rows = every "
             "probe that never saw the cycle's discoveries"]
    for arm in ARMS:
        cpath = d / f"gap_cycles_{arm}.json"
        if not cpath.exists():
            continue
        for c in load_json(cpath):
            lines.append(f"[{arm}] cycle {c['cycle']} (searched v{c['probe_searched']}): "
                         f"n_new={c['n_new']} rungs={c['rung_counts']} | referee top20 rate "
                         f"{c['t3_top20_mean']}, rho(rate,probe)={c['t3_rate_vs_probe']} "
                         f"rho(rate,mu)={c['t3_rate_vs_mu']}")
            for vname, rep in c["per_probe"].items():
                f, cm, s, e = rep["vs_mu_full"], rep["vs_mu_comp"], rep["in_span"], rep["escalated"]
                fmt = lambda r: (f"gap {r['gap_mean']:+.2f} |gap| {r['abs_gap_mean']:.2f} r {r['pearson']:+.2f}"
                                 if "gap_mean" in r else f"n={r['n']}")
                lines.append(f"  {vname}: full[{fmt(f)}] comp[{fmt(cm)}] | in-span[{fmt(s)}] "
                             f"escalated[{fmt(e)}]")
    (d / "summary_gap.txt").write_text("\n".join(lines) + "\n")


# ---- harvest / fit ------------------------------------------------------------------------------

def cmd_harvest(model, k):
    d = pl.out_dir(model)
    exps = pl.BASE_EXPS + [f"plc{j}" for j in range(1, N_PLC + 1)] + [exp(j) for j in range(1, k + 1)]
    rows = pl.cmd_harvest(model, cycle=probe_v(k), exps=exps)
    over = {}
    rp = d / "remeasure_layered.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                over[r["text"].lower()] = r
    n_over = 0
    for r in rows:
        o = over.get(r["text"].lower())
        if o is not None and not r["source"].startswith("gpl"):
            r["mu_orig"], r["mu"] = r["mu"], o["mu_layered"]
            r["rung"], r["sat"] = o["rung"], max(o["sat_hi"], o["sat_lo"])
            n_over += 1
    path = d / f"dataset_c{probe_v(k)}.json"
    save_json(path, rows)
    print(f"harvest gap{k}: {len(rows)} rows -> {path.name}; {n_over} tail mu overlaid with the "
          f"layered re-measurement ({len(over)} available)")
    return rows


def cmd_fit(model, k):
    pl.cmd_fit(model, probe_v(k) + 1)


def cmd_cycle(model, k_max, k_min=1):
    d = pl.out_dir(model)
    for k in range(k_min, k_max + 1):
        assert pl.probe_path(model, probe_v(k)).exists(), f"probe v{probe_v(k)} missing"
        for arm in ARMS:
            cmd_search(model, k, arm)
            cpath = d / f"gap_cycles_{arm}.json"
            done = load_json(cpath) if cpath.exists() else []
            if not any(c["cycle"] == k for c in done):
                cmd_eval(model, k, arm)
        if not (d / f"dataset_c{probe_v(k)}.json").exists():
            cmd_harvest(model, k)
        if not pl.probe_path(model, probe_v(k) + 1).exists():
            cmd_fit(model, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["search", "eval", "harvest", "fit", "cycle"])
    ap.add_argument("model")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--arm", default="over", choices=ARMS)
    ap.add_argument("--T", type=int, default=15)
    ap.add_argument("--n-cand", type=int, default=192)
    ap.add_argument("--patience", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "search":
        cmd_search(a.model, a.k, a.arm, T=a.T, n_cand=a.n_cand, patience=a.patience)
    elif a.cmd == "eval":
        cmd_eval(a.model, a.k, a.arm)
    elif a.cmd == "harvest":
        cmd_harvest(a.model, a.k)
    elif a.cmd == "fit":
        cmd_fit(a.model, a.k)
    else:
        cmd_cycle(a.model, a.k)


if __name__ == "__main__":
    main()
