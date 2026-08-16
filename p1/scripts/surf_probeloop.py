"""Probe-hardening loop: calibrate -> retrain -> re-SURF -> repeat.

v0 = the S0 probe (XL-only, results/surf/s0/<model>/probe.pt). Cycle k >= 1:
  harvest (CPU)   dataset_c{k-1}.json — every SURF discovery with a measured
                  72-readout anchored mu (mu_full on buffer entrants in the
                  iter jsonls, plus confirm files), deduped, XL items excluded.
  fit (GPU, min)  probe_v{k}.pt retrained on XL + dataset, same RidgeCV recipe
                  as surf_s0.cmd_probe; isotonic calibration calib_v{k}.json
                  fit on cross-fitted held-out XL predictions (never in-sample).
                  Cycle 1 also writes calib_v0.json and the v0 train-vs-
                  generated comparison (the plan's steps 1-2).
  search (GPU)    plc{k}: an e2p-style SURF run with fitness = probe_v{k}.
  eval (GPU, min) Goodhart metrics on plc{k}'s FRESH discoveries under every
                  probe version (prequential: v{k} never trained on them),
                  plus a held-out Tier-3 referee on the top-20; appends to
                  cycles.json.

Question-form items are KEPT in the target (measured mu is the defined ground
truth); every metric is reported split by question_form so stated-channel
artifact-learning stays visible. mu targets are winsorized to +/-8 (anchor-
strain extrapolations shouldn't get ridge leverage). Spearman is invariant to
the monotone calibration, so f's value shows up in Pearson and mean |f(k)-mu|.

Usage:
  uv run python scripts/surf_probeloop.py harvest <model> [--cycle 0]
  uv run python scripts/surf_probeloop.py fit <model> --cycle 1
  uv run python scripts/surf_probeloop.py search <model> --cycle 1
  uv run python scripts/surf_probeloop.py eval <model> --cycle 1
  uv run python scripts/surf_probeloop.py cycle <model> --k 3
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

from surf_s0 import work_layers

WINSOR = 8.0
BASE_EXPS = ["e1", "e2p", "e2r"]


def out_dir(model):
    d = P1 / "results" / "surf" / "probeloop" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_question(text, attrs=None):
    return text.rstrip().endswith("?") or "it_question_form" in (attrs or [])


def winsorize(mu, lim=WINSOR):
    v = np.clip(np.asarray(mu, float), -lim, lim)
    return v, int((np.abs(np.asarray(mu, float)) > lim).sum())


def apply_probe(d, acts):
    """d: loaded probe dict; acts: [3, N, dim] working-layer slice -> preds [N]."""
    X = acts[d["layer_pos"]].numpy()
    return ((X - np.asarray(d["mean"])) / np.asarray(d["std"])) @ np.asarray(d["coef"]) \
        + float(d["intercept"])


def apply_calib(calib, k):
    return np.interp(np.asarray(k, float), calib["x"], calib["y"])


def probe_path(model, v):
    if v == 0:
        return P1 / "results" / "surf" / "s0" / model / "probe.pt"
    return out_dir(model) / f"probe_v{v}.pt"


def _load_probe(model, v):
    import torch
    return torch.load(probe_path(model, v), weights_only=False)


def heldout_preds(acts, y, seed=0):
    """The surf_s0.cmd_probe protocol: per working layer, 2-fold cross-fit
    RidgeCV -> (held-out preds per layer, per-layer held-out pearson)."""
    from sklearn.linear_model import RidgeCV
    alphas = np.logspace(1, 6, 8)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    folds = [perm[: len(y) // 2], perm[len(y) // 2:]]
    preds, rs = [], []
    for lp in range(acts.shape[0]):
        X = acts[lp].numpy()
        p = np.zeros(len(y))
        for f in range(2):
            te, tr = folds[f], folds[1 - f]
            m0, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            m = RidgeCV(alphas=alphas).fit((X[tr] - m0) / sd, y[tr])
            p[te] = m.predict((X[te] - m0) / sd)
        preds.append(p)
        rs.append(pearson(list(p), list(y)))
    return preds, rs


def _corr_report(pred, mu, qf):
    pred, mu, qf = np.asarray(pred), np.asarray(mu), np.asarray(qf, bool)
    out = {"n": len(mu), "pearson": round(pearson(list(pred), list(mu)), 3),
           "spearman": round(spearman(list(pred), list(mu)), 3),
           "overest_mean": round(float((pred - mu).mean()), 3),
           "mae": round(float(np.abs(pred - mu).mean()), 3)}
    for name, m in (("qform", qf), ("declarative", ~qf)):
        if 3 <= m.sum() < len(mu):
            out[f"pearson_{name}"] = round(pearson(list(pred[m]), list(mu[m])), 3)
            out[f"n_{name}"] = int(m.sum())
    return out


# ---- harvest ------------------------------------------------------------------------------------

def cmd_harvest(model, cycle=0, exps=None):
    import surf
    exps = exps if exps is not None else BASE_EXPS + [f"plc{j}" for j in range(1, cycle + 1)]
    got = {}
    for exp in exps:
        root = surf.SURF_ROOT / exp / model
        if not root.exists():
            continue
        for rd in sorted(d for d in root.iterdir() if d.is_dir() and d.name != "confirm"):
            for p in sorted(rd.glob("iter_*.jsonl")):
                for line in p.read_text().splitlines():
                    r = json.loads(line)
                    if r.get("mu_full") is not None:
                        got[r["text"].lower()] = {"text": r["text"], "mu": r["mu_full"],
                                                  "attrs": r.get("attrs"), "source": exp}
        cpath = root / "confirm" / "confirmed.json"
        if cpath.exists():
            for r in load_json(cpath):
                if "mu" in r:  # frame confirms carry one_minus_rho_full instead
                    got[r["text"].lower()] = {"text": r["text"], "mu": r["mu"],
                                              "attrs": r.get("attrs"),
                                              "source": exp + "-confirm"}
    rows = [{**v, "question_form": is_question(v["text"], v["attrs"])}
            for v in got.values()]
    path = out_dir(model) / f"dataset_c{cycle}.json"
    save_json(path, rows)
    nq = sum(r["question_form"] for r in rows)
    print(f"harvest c{cycle}: {len(rows)} unique measured items "
          f"({nq} question-form) -> {path.name}")
    return rows


# ---- fit ----------------------------------------------------------------------------------------

def _ds_acts(model, ds, cache):
    import torch
    if cache.exists():
        acts = torch.load(cache, weights_only=False).float()
        if acts.shape[1] == len(ds):
            return acts
    import harness
    import probes
    h = harness.load(model)
    full = probes.item_acts(h, ds)
    acts = full[work_layers(model)].clone()
    torch.save(acts.half(), cache)
    return acts.float()


def cmd_fit(model, cycle):
    import torch
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import RidgeCV

    d = out_dir(model)
    xl_rows = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    xl_acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt",
                         weights_only=False).float()
    xl_mu = np.array([r["mu"] for r in xl_rows])
    xl_texts = {r["text"].lower() for r in xl_rows}
    ds = [r for r in load_json(d / f"dataset_c{cycle - 1}.json")
          if r["text"].lower() not in xl_texts]
    ds_acts = _ds_acts(model, ds, d / f"acts_c{cycle - 1}.pt")
    ds_mu, n_clip = winsorize([r["mu"] for r in ds])
    qf = np.array([False] * len(xl_rows) + [r["question_form"] for r in ds])
    is_surf = np.array([False] * len(xl_rows) + [True] * len(ds))

    lines = [f"probeloop fit cycle {cycle} ({model}): XL n={len(xl_rows)} + "
             f"SURF n={len(ds)} ({n_clip} winsorized at +/-{WINSOR:g})"]

    # cycle 1 bootstrap: v0's calibration + the train-vs-generated comparison
    if cycle == 1 and not (d / "calib_v0.json").exists():
        v0 = _load_probe(model, 0)
        p_xl, _ = heldout_preds(xl_acts, xl_mu)
        k_xl = p_xl[v0["layer_pos"]]
        iso0 = IsotonicRegression(out_of_bounds="clip").fit(k_xl, xl_mu)
        save_json(d / "calib_v0.json", {"x": iso0.X_thresholds_.tolist(),
                                        "y": iso0.y_thresholds_.tolist()})
        calib0 = load_json(d / "calib_v0.json")
        k_ds = apply_probe(v0, ds_acts)
        lines.append("v0 on XL (held-out): "
                     + json.dumps(_corr_report(k_xl, xl_mu, [False] * len(xl_mu))))
        lines.append("v0 on XL, calibrated: "
                     + json.dumps(_corr_report(apply_calib(calib0, k_xl), xl_mu,
                                               [False] * len(xl_mu))))
        lines.append("v0 on SURF discoveries (OOD): "
                     + json.dumps(_corr_report(k_ds, ds_mu,
                                               [r["question_form"] for r in ds])))
        lines.append("v0 on SURF, calibrated: "
                     + json.dumps(_corr_report(apply_calib(calib0, k_ds), ds_mu,
                                               [r["question_form"] for r in ds])))

    # retrain on the combined set
    acts = torch.cat([xl_acts, ds_acts], dim=1)
    y = np.concatenate([xl_mu, ds_mu])
    preds, rs = heldout_preds(acts, y)
    lp = int(np.argmax(rs))
    X = acts[lp].numpy()
    m0, sd = X.mean(0), X.std(0) + 1e-6
    m = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X - m0) / sd, y)
    prev = _load_probe(model, cycle - 1)
    v_prev_r = prev.get("cv_r_xl_only", prev["cv_r"])  # compare XL-subset to XL-subset
    xl_mask = ~is_surf
    r_xl_new = pearson(list(preds[lp][xl_mask]), list(y[xl_mask]))
    torch.save({"layer_pos": lp, "layer_global": work_layers(model)[lp],
                "mean": m0, "std": sd, "coef": m.coef_, "intercept": float(m.intercept_),
                "alpha": float(m.alpha_), "cv_r": round(rs[lp], 4),
                "cv_r_xl_only": round(r_xl_new, 4), "n": len(y), "n_surf": int(len(ds)),
                "n_winsorized": n_clip}, d / f"probe_v{cycle}.pt")
    iso = IsotonicRegression(out_of_bounds="clip").fit(preds[lp], y)
    save_json(d / f"calib_v{cycle}.json", {"x": iso.X_thresholds_.tolist(),
                                           "y": iso.y_thresholds_.tolist()})
    lines.append(f"v{cycle}: layer {work_layers(model)[lp]}, combined held-out "
                 f"r {rs[lp]:+.3f}; XL-subset held-out r {r_xl_new:+.3f} "
                 f"(gate: >= {v_prev_r - 0.02:+.3f} -> "
                 f"{'PASS' if r_xl_new >= v_prev_r - 0.02 else 'FAIL'})")
    lines.append(f"v{cycle} on SURF subset (in-sample cross-fit): "
                 + json.dumps(_corr_report(preds[lp][is_surf], y[is_surf],
                                           qf[is_surf])))
    (d / f"fit_v{cycle}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- search -------------------------------------------------------------------------------------

def cmd_search(model, cycle):
    import surf
    import surf_scores
    init = surf.SURF_ROOT / "tags" / model / "pool_weights_max.json"
    cfg = surf.RunConfig(
        experiment=f"plc{cycle}", model=model, direction="max", fitness="t1_probe",
        allowed_tiers=["t0", "t1", "t2"], pool_file="items/surf_attributes_item.json",
        pool_kind="item", pool_init=str(init) if init.exists() else "", seed=0, T=15,
        probe_path=str(probe_path(model, cycle)))
    surf.run(cfg, surf_scores.build(cfg))


# ---- eval ---------------------------------------------------------------------------------------

def cmd_eval(model, cycle):
    d = out_dir(model)
    fresh = cmd_harvest(model, cycle=-1, exps=[f"plc{cycle}"])
    assert fresh, f"no measured discoveries in plc{cycle}"
    (d / f"dataset_c-1.json").unlink()  # side artifact of the exps override
    save_json(d / f"discoveries_plc{cycle}.json", fresh)
    acts = _ds_acts(model, fresh, d / f"acts_plc{cycle}.pt")
    mu, _ = winsorize([r["mu"] for r in fresh], lim=1e9)
    qf = [r["question_form"] for r in fresh]

    row = {"cycle": cycle, "n_new": len(fresh), "per_probe": {}}
    for v in range(cycle + 1):
        p = _load_probe(model, v)
        k = apply_probe(p, acts)
        rep = {"raw": _corr_report(k, mu, qf)}
        cpath = d / f"calib_v{v}.json"
        if cpath.exists():
            rep["calibrated"] = _corr_report(apply_calib(load_json(cpath), k), mu, qf)
        row["per_probe"][f"v{v}"] = rep

    # held-out behavioral referee on the top-20 by the current probe
    import surf_scores
    from surf_e2_referee import heldout_env_ids
    k_cur = apply_probe(_load_probe(model, cycle), acts)
    top = [fresh[i] for i in np.argsort(-k_cur)[:20]]
    t3 = surf_scores.Tier3Revealed(surf_scores.Handles(model), model, n_rolls=12,
                                   anchor_ids=heldout_env_ids(model))
    rates = t3.score([r["text"] for r in top])
    row["t3_top20_mean"] = round(float(np.mean(rates)), 3)
    row["t3_top20"] = [{"text": r["text"], "rate": round(c, 3)}
                       for r, c in zip(top, rates)]

    cpath = d / "cycles.json"
    cycles = load_json(cpath) if cpath.exists() else []
    cycles = [c for c in cycles if c["cycle"] != cycle] + [row]
    save_json(cpath, sorted(cycles, key=lambda c: c["cycle"]))

    lines = [f"probeloop cycles ({model}); prequential column = each probe on the "
             "LATER cycle's fresh discoveries"]
    for c in sorted(cycles, key=lambda c: c["cycle"]):
        lines.append(f"cycle {c['cycle']}: n_new={c['n_new']} "
                     f"t3(top20)={c['t3_top20_mean']}")
        for vname, rep in c["per_probe"].items():
            r = rep["raw"]
            cal = rep.get("calibrated", {})
            lines.append(f"  {vname}: r={r['pearson']:+.3f} rho={r['spearman']:+.3f} "
                         f"overest={r['overest_mean']:+.2f}"
                         + (f" | calibrated r={cal['pearson']:+.3f} "
                            f"mae={cal['mae']:.2f}" if cal else "")
                         + (f" | qform r={r['pearson_qform']:+.3f}"
                            if "pearson_qform" in r else ""))
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- orchestration ------------------------------------------------------------------------------

def cmd_cycle(model, k):
    import surf
    for c in range(1, k + 1):
        if not (out_dir(model) / f"dataset_c{c - 1}.json").exists():
            cmd_harvest(model, cycle=c - 1)
        if not probe_path(model, c).exists():
            cmd_fit(model, c)
        run_dir = surf.SURF_ROOT / f"plc{c}" / model / "max-s0"
        st, _ = surf._load_state(run_dir) if run_dir.exists() else (None, -1)
        if not (st and st.get("stopped")):
            cmd_search(model, c)
        done = (out_dir(model) / "cycles.json")
        rows = load_json(done) if done.exists() else []
        if not any(r["cycle"] == c for r in rows):
            cmd_eval(model, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["harvest", "fit", "search", "eval", "cycle"])
    ap.add_argument("model")
    ap.add_argument("--cycle", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "harvest":
        cmd_harvest(a.model, a.cycle)
    elif a.cmd == "cycle":
        cmd_cycle(a.model, a.k)
    else:
        assert a.cycle >= 1, "--cycle must be >= 1"
        {"fit": cmd_fit, "search": cmd_search, "eval": cmd_eval}[a.cmd](a.model, a.cycle)


if __name__ == "__main__":
    main()
