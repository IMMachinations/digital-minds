"""Probe-version reruns (qwen25-7b): every probe-dependent downstream test
re-scored under the S0 probe (v0), the hardening probes (v1-v10), the gap-loop
probes (v11-v15) and the layer-18 refit of v15 (`v15_L18`, written by
`surf_probeloop.py fit --cycle 15 --layer 18 --tag _L18`).

Nothing here overwrites an earlier result: every output lands under
results/surf/probe_versions/<model>/ (acts caches included; gitignored).

  acts <model>       GPU once: activation caches for the E2 referee set (100),
                     the 197 hand-built items and the 388-item revealed panel.
  global2 <model>    CPU. The enlarged global set = XL (stage1x, in-span A0 mu)
                     + dataset_c14 (4,263 SURF discoveries, layered-anchor mu;
                     acts cached by the probeloop). Every probe version scored
                     split XL / SURF / SURF-unseen (sources the probe never
                     trained on) / source group / question-form; a fresh
                     held-out global probe (probe_global2.pt) and its layer-18
                     steering direction (utility_dir_global2.pt); the
                     spline-vs-ridge-vs-line verdict on the layered scale.
  e2 <model>         CPU (after acts). The E2 three-way referee set re-scored:
                     stated<->probe and probe<->revealed Spearman per version,
                     arm means, probe-hack / blind-spot counts.
  handbuilt <model>  CPU (after acts). Stage-1C on the 197 hand-built items:
                     every version vs the Stage-1B utilities, overall and per
                     domain — the out-of-family test.
  revealed <model>   CPU (after acts). Stated probes vs the behavioural
                     (empirical-logit menu-choice) target on the revealed panel.
  cosines <model>    CPU. Direction geometry across versions at layer 18 (and
                     the layer-14 family for v15 native).
  extra-dirs <model> CPU. Unit steering directions for stage4 (utility_v3,
                     utility_v10, utility_v15L18 at layer 18) -> extra_dirs.pt.
  all-cpu <model>    global2 + e2 + handbuilt + revealed + cosines + extra-dirs.

Usage: uv run python scripts/surf_probe_versions.py <cmd> <model>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import surf_probeloop as pl
from surf_probeloop import (apply_calib, apply_probe, calib_path, heldout_preds,
                            probe_path, winsorize)
from surf_s0 import work_layers

VERSIONS = list(range(0, 16)) + ["15_L18"]
BASE_SOURCES = {"e1", "e1-confirm", "e2p", "e2p-confirm", "e2r", "e2r-confirm"}
SOURCE_GROUPS = {"pre-loop": lambda s: s in BASE_SOURCES,
                 "plc1-3": lambda s: s in {f"plc{k}" for k in range(1, 4)},
                 "plc4-9": lambda s: s in {f"plc{k}" for k in range(4, 10)},
                 "gpl1-5": lambda s: s in {f"gpl{k}" for k in range(1, 6)}}


def out_dir(model):
    d = P1 / "results" / "surf" / "probe_versions" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def vname(v):
    return f"v{v}"


def versions(model):
    return [v for v in VERSIONS if probe_path(model, v).exists()]


def seen_sources(v):
    """Sources a probe version trained on. v_k (k<=10) = dataset_c{k-1}:
    pre-loop + plc1..plc(k-1); v_(10+k) = dataset_c{10+k-1}: + plc1..9 +
    gpl1..gplk. v0 = XL only."""
    k = int(str(v).split("_")[0])
    if k == 0:
        return set()
    seen = set(BASE_SOURCES)
    if k <= 10:
        seen |= {f"plc{j}" for j in range(1, k)}
    else:
        seen |= {f"plc{j}" for j in range(1, 10)} | {f"gpl{j}" for j in range(1, k - 10 + 1)}
    return seen


def _calib(model, v):
    p = calib_path(model, v)
    return load_json(p) if p.exists() else None


def _stats(pred, mu, mask, pred_cal=None):
    mask = np.asarray(mask, bool)
    if mask.sum() < 3:
        return None
    p, m = np.asarray(pred)[mask], np.asarray(mu)[mask]
    out = {"n": int(mask.sum()), "r": round(pearson(list(p), list(m)), 4),
           "rho": round(spearman(list(p), list(m)), 4)}
    if pred_cal is not None:
        pc = np.asarray(pred_cal)[mask]
        out["mae_cal"] = round(float(np.abs(pc - m).mean()), 4)
        out["bias_cal"] = round(float((pc - m).mean()), 4)
    return out


class _Acts:
    """One model handle shared across the acts caches."""
    def __init__(self, model):
        self.model, self.h = model, None

    def get(self, rows, cache):
        import torch
        if cache.exists():
            acts = torch.load(cache, weights_only=False).float()
            if acts.shape[1] == len(rows):
                return acts
        if self.h is None:
            import harness
            self.h = harness.load(self.model)
        import probes
        full = probes.item_acts(self.h, rows)
        acts = full[work_layers(self.model)].clone()
        torch.save(acts.half(), cache)
        return acts.float()


# ---- item sets ----------------------------------------------------------------------------------

def e2_rows(model):
    return load_json(P1 / "results" / "surf" / "e2" / model / "referee.json")["rows"]


def handbuilt_rows(model):
    import items as items_mod
    its = items_mod.load_items()
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    assert [r["id"] for r in ut] == [it["id"] for it in its], "1B row order != load_items()"
    for it, u in zip(its, ut):
        it["mu"] = u["mu"]
    return its


def panel_rows(model):
    from surf_revealed_probe import _labels
    items, rate, y = _labels(model)
    return items, rate, y


def cmd_acts(model):
    A = _Acts(model)
    d = out_dir(model)
    A.get(e2_rows(model), d / "acts_e2ref.pt")
    A.get(handbuilt_rows(model), d / "acts_handbuilt.pt")
    items, _, _ = panel_rows(model)
    A.get(items, d / "acts_panel_v2.pt")
    print("acts caches ready:", sorted(p.name for p in d.glob("acts_*.pt")))


# ---- global2 ------------------------------------------------------------------------------------

def global2_set(model):
    """-> rows (xl + dataset_c14 minus XL texts), acts [3, N, d]."""
    import torch
    xl_rows = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    xl_acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt",
                         weights_only=False).float()
    xl_texts = {r["text"].lower() for r in xl_rows}
    ds = [r for r in load_json(pl.out_dir(model) / "dataset_c14.json")
          if r["text"].lower() not in xl_texts]
    ds_acts = pl._ds_acts(model, ds, pl.out_dir(model) / "acts_c14.pt")
    rows = [{"text": r["text"], "mu": r["mu"], "source": "xl", "question_form": False}
            for r in xl_rows] + ds
    return rows, torch.cat([xl_acts, ds_acts], dim=1)


def _parse_old_global(model):
    p = P1 / "results" / "surf" / "global" / model / "analyze_summary.txt"
    if not p.exists():
        return None
    txt = p.read_text()
    out = {}
    m = re.search(r"ridge \(held-out\): all: r=([+-]\d\.\d+)", txt)
    out["ridge"] = float(m.group(1)) if m else None
    sp = [float(x) for x in re.findall(r"bins=\d+: spline all: r=([+-]\d\.\d+)", txt)]
    out["spline_best"] = max(sp) if sp else None
    ln = [float(x) for x in re.findall(r"line all: r=([+-]\d\.\d+)", txt)]
    out["line"] = max(ln) if ln else None
    m = re.search(r"fresh global probe \(held-out, layer (\d+)\): all: r=([+-]\d\.\d+)", txt)
    out["fresh_r"] = float(m.group(2)) if m else None
    return out


def cmd_global2(model):
    import torch
    from sklearn.linear_model import RidgeCV
    import utility_spline as usp

    d = out_dir(model)
    rows, acts = global2_set(model)
    mu_raw = np.array([r["mu"] for r in rows])
    mu, n_clip = winsorize(mu_raw)
    src = np.array([r["source"] for r in rows])
    qf = np.array([r["question_form"] for r in rows], bool)
    is_surf = src != "xl"
    masks = {"all": np.ones(len(rows), bool), "xl": ~is_surf, "surf": is_surf,
             "qform": qf, "declarative": is_surf & ~qf}
    for g, fn in SOURCE_GROUPS.items():
        masks[g] = np.array([fn(s) for s in src], bool)
    W = work_layers(model)
    lines = [f"{model} global2 (layered-scale SURF + in-span XL): n={len(rows)} "
             f"({int((~is_surf).sum())} XL + {int(is_surf.sum())} SURF; {n_clip} winsorized); "
             f"sources: " + ", ".join(f"{g}={int(m.sum())}" for g, m in masks.items()
                                      if g in SOURCE_GROUPS)]
    out = {"n": len(rows), "n_xl": int((~is_surf).sum()), "n_surf": int(is_surf.sum()),
           "versions": {}}
    for v in versions(model):
        pr = pl._load_probe(model, v)
        k = apply_probe(pr, acts)
        cal = _calib(model, v)
        kc = apply_calib(cal, k) if cal else None
        seen = seen_sources(v)
        unseen = is_surf & np.array([s not in seen for s in src], bool)
        rec = {"layer": int(pr["layer_global"]), "seen": sorted(seen), "masks": {}}
        for name, m in list(masks.items()) + [("surf_unseen", unseen)]:
            rec["masks"][name] = _stats(k, mu_raw, m, kc)
        out["versions"][vname(v)] = rec
        f = lambda n: rec["masks"][n]  # noqa: E731
        lines.append(f"  {vname(v):8s} L{rec['layer']}: all r={f('all')['r']:+.3f} | "
                     f"XL r={f('xl')['r']:+.3f} | SURF r={f('surf')['r']:+.3f} "
                     f"rho={f('surf')['rho']:+.3f} mae={f('surf')['mae_cal']:.3f} | "
                     f"SURF-unseen r={(f('surf_unseen') or {}).get('r', float('nan')):+.3f} "
                     f"(n={(f('surf_unseen') or {}).get('n', 0)}) | "
                     f"qform r={f('qform')['r']:+.3f} | "
                     + " ".join(f"{g}={f(g)['r']:+.3f}" for g in SOURCE_GROUPS if f(g)))

    # fresh held-out global probe (argmax layer) + a pinned layer-18 steering direction
    preds, rs = heldout_preds(acts, mu)
    lp = int(np.argmax(rs))
    fresh = {"layer": W[lp], "per_layer_r": [round(r, 4) for r in rs], "masks": {}}
    for name, m in masks.items():
        fresh["masks"][name] = _stats(preds[lp], mu_raw, m)
    out["fresh"] = fresh
    lines.append(f"  fresh global2 probe (held-out, layer {W[lp]}; per layer "
                 + ", ".join(f"{W[i]}:{r:+.3f}" for i, r in enumerate(rs)) + "): "
                 + "; ".join(f"{n} r={fresh['masks'][n]['r']:+.3f}" for n in
                             ("all", "xl", "surf", "qform")))
    for L_pin, fname in ((W[lp], "probe_global2.pt"), (W[1], "utility_dir_global2.pt")):
        lpp = W.index(L_pin)
        X = acts[lpp].numpy()
        m0, sd = X.mean(0), X.std(0) + 1e-6
        m = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X - m0) / sd, mu)
        if fname == "probe_global2.pt":
            torch.save({"layer_pos": lpp, "layer_global": L_pin, "mean": m0, "std": sd,
                        "coef": m.coef_, "intercept": float(m.intercept_),
                        "alpha": float(m.alpha_), "cv_r": round(rs[lpp], 4), "n": len(rows)},
                       d / fname)
        else:
            wdir = m.coef_ / sd
            torch.save({"layer": L_pin, "dir": torch.tensor(wdir / np.linalg.norm(wdir)),
                        "cv_r": round(rs[lpp], 4), "n": len(rows)}, d / fname)

    # spline-vs-ridge-vs-line at the fresh probe's layer (surf_global.cmd_analyze protocol)
    X = acts[lp].numpy()
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(mu))
    folds = [perm[: len(mu) // 2], perm[len(mu) // 2:]]
    lin = {"layer": W[lp], "ridge": None, "spline": {}, "line": {}}
    for n_bins in (40, 64, 100):
        pred_sp, pred_ln, pred_ri = np.zeros(len(mu)), np.zeros(len(mu)), np.zeros(len(mu))
        for f in range(2):
            te, tr = folds[f], folds[1 - f]
            m0f = X[tr].mean(0)
            _, _, Vt = np.linalg.svd(X[tr] - m0f, full_matrices=False)
            B64 = Vt[:64]
            Ztr, Zte = (X[tr] - m0f) @ B64.T, (X[te] - m0f) @ B64.T
            order = np.argsort(mu[tr])
            cent = np.stack([Ztr[b].mean(0) for b in np.array_split(order, n_bins)])
            sp = usp.open_spline(cent, k_fit=8)
            u_tr, u_te = usp.spline_u(Ztr, sp), usp.spline_u(Zte, sp)
            pred_sp[te] = np.polyval(np.polyfit(u_tr, mu[tr], 1), u_te)
            _, _, Vc = np.linalg.svd(cent - cent.mean(0), full_matrices=False)
            pred_ln[te] = np.polyval(np.polyfit(Ztr @ Vc[0], mu[tr], 1), Zte @ Vc[0])
            if n_bins == 40:
                sdf = X[tr].std(0) + 1e-6
                ri = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X[tr] - m0f) / sdf, mu[tr])
                pred_ri[te] = ri.predict((X[te] - m0f) / sdf)
        if n_bins == 40:
            lin["ridge"] = {"all": _stats(pred_ri, mu_raw, masks["all"]),
                            "surf": _stats(pred_ri, mu_raw, masks["surf"])}
        lin["spline"][str(n_bins)] = {"all": _stats(pred_sp, mu_raw, masks["all"]),
                                      "surf": _stats(pred_sp, mu_raw, masks["surf"])}
        lin["line"][str(n_bins)] = {"all": _stats(pred_ln, mu_raw, masks["all"]),
                                    "surf": _stats(pred_ln, mu_raw, masks["surf"])}
        lines.append(f"  linearity bins={n_bins}: ridge all r={lin['ridge']['all']['r']:+.3f} "
                     f"| spline all r={lin['spline'][str(n_bins)]['all']['r']:+.3f} "
                     f"SURF r={lin['spline'][str(n_bins)]['surf']['r']:+.3f} "
                     f"| line all r={lin['line'][str(n_bins)]['all']['r']:+.3f}")
    out["linearity"] = lin
    out["old_global"] = _parse_old_global(model)
    save_json(d / "global2_probe_versions.json", out)
    (d / "global2_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- E2 referee ---------------------------------------------------------------------------------

def cmd_e2(model):
    d = out_dir(model)
    rows = e2_rows(model)
    acts = _Acts(model).get(rows, d / "acts_e2ref.pt")
    mu_t2 = np.array([r["mu_t2"] for r in rows])
    t3 = np.array([r["t3_rate"] for r in rows])
    stored = np.array([r["probe_mu"] for r in rows])
    arm = {a: np.array([a in r["sources"] for r in rows], bool) for a in ("e2p", "e2r")}
    out = {"n": len(rows),
           "rho_stated_revealed": round(spearman(list(mu_t2), list(t3)), 4),
           "arm_t3_mean": {a: round(float(t3[m].mean()), 4) for a, m in arm.items()},
           "arm_mu_mean": {a: round(float(mu_t2[m].mean()), 4) for a, m in arm.items()},
           "versions": {}}
    lines = [f"{model} E2 referee set re-scored (n={len(rows)}); "
             f"spearman(stated, revealed) = {out['rho_stated_revealed']:+.3f} (version-free)"]
    tq = np.percentile(t3, [50, 75])
    for v in versions(model):
        pr = pl._load_probe(model, v)
        k = apply_probe(pr, acts)
        cal = _calib(model, v)
        kc = apply_calib(cal, k) if cal else k
        pq = np.percentile(k, [50, 75])
        hack = int(((k >= pq[1]) & (t3 <= tq[0])).sum())
        blind = int(((t3 >= tq[1]) & (k <= pq[0])).sum())
        rec = {"layer": int(pr["layer_global"]),
               "rho_stated_probe": round(spearman(list(mu_t2), list(k)), 4),
               "r_stated_probe": round(pearson(list(mu_t2), list(k)), 4),
               "rho_probe_revealed": round(spearman(list(k), list(t3)), 4),
               "r_probe_revealed": round(pearson(list(k), list(t3)), 4),
               "mae_cal_vs_stated": round(float(np.abs(kc - mu_t2).mean()), 4),
               "arm_probe_mean": {a: round(float(k[m].mean()), 4) for a, m in arm.items()},
               "n_hack": hack, "n_blind": blind}
        rec["p_minus_r_probe"] = round(rec["arm_probe_mean"]["e2p"]
                                       - rec["arm_probe_mean"]["e2r"], 4)
        if v == 0:
            rec["sanity_r_vs_stored_probe_mu"] = round(pearson(list(k), list(stored)), 4)
        out["versions"][vname(v)] = rec
        lines.append(f"  {vname(v):8s}: rho(stated,probe)={rec['rho_stated_probe']:+.3f} "
                     f"rho(probe,revealed)={rec['rho_probe_revealed']:+.3f} "
                     f"mae_cal={rec['mae_cal_vs_stated']:.3f} "
                     f"P-R(probe)={rec['p_minus_r_probe']:+.3f} hack={hack} blind={blind}"
                     + (f"  [v0 vs stored probe_mu r={rec['sanity_r_vs_stored_probe_mu']:+.4f}]"
                        if v == 0 else ""))
    save_json(d / "e2_referee_versions.json", out)
    (d / "e2_referee_versions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- hand-built 197 (Stage 1C out-of-family) ----------------------------------------------------

def _baseline_1c(model):
    p = P1 / "results" / "stage1c" / model / "convergence.txt"
    if not p.exists():
        return None
    m = re.search(r"probe\s+r=([+-]\d\.\d+) \(held-out\)\s+rho=([+-]\d\.\d+)", p.read_text())
    return {"r": float(m.group(1)), "rho": float(m.group(2))} if m else None


def cmd_handbuilt(model):
    d = out_dir(model)
    its = handbuilt_rows(model)
    acts = _Acts(model).get(its, d / "acts_handbuilt.pt")
    mu = np.array([it["mu"] for it in its])
    dom = np.array([it["domain"] for it in its])
    doms = sorted(set(dom))
    out = {"n": len(its), "baseline_1c": _baseline_1c(model), "domains": doms, "versions": {}}
    lines = [f"{model} hand-built 197 items vs Stage-1B mu (out-of-family); "
             f"original cross-fitted 1C probe: {out['baseline_1c']}"]
    for v in versions(model):
        pr = pl._load_probe(model, v)
        k = apply_probe(pr, acts)
        cal = _calib(model, v)
        kc = apply_calib(cal, k) if cal else None
        rec = {"layer": int(pr["layer_global"]),
               "overall": _stats(k, mu, np.ones(len(mu), bool), kc),
               "by_domain": {dm: _stats(k, mu, dom == dm, kc) for dm in doms}}
        out["versions"][vname(v)] = rec
        lines.append(f"  {vname(v):8s} L{rec['layer']}: r={rec['overall']['r']:+.3f} "
                     f"rho={rec['overall']['rho']:+.3f} mae_cal={rec['overall']['mae_cal']:.3f} | "
                     + " ".join(f"{dm[:4]}={rec['by_domain'][dm]['r']:+.3f}" for dm in doms))
    save_json(d / "handbuilt_versions.json", out)
    (d / "handbuilt_versions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- revealed panel -----------------------------------------------------------------------------

def cmd_revealed(model):
    import torch
    d = out_dir(model)
    items, rate, y = panel_rows(model)
    acts = _Acts(model).get(items, d / "acts_panel_v2.pt")
    mu = np.array([r["mu"] for r in items])
    rp = torch.load(P1 / "results" / "surf" / "revealed" / model / "revealed_probe.pt",
                    weights_only=False)
    out = {"n": len(items), "stated_mu_r": round(pearson(list(mu), list(y)), 4),
           "stated_mu_rho": round(spearman(list(mu), list(y)), 4),
           "revealed_probe_cv_r": float(rp["cv_r"]), "versions": {}}
    lines = [f"{model} revealed panel (n={len(items)}, empirical-logit choice-rate target): "
             f"stated mu r={out['stated_mu_r']:+.3f} rho={out['stated_mu_rho']:+.3f}; "
             f"behaviour-trained probe held-out r={out['revealed_probe_cv_r']:+.3f}"]
    for v in versions(model):
        pr = pl._load_probe(model, v)
        k = apply_probe(pr, acts)
        rec = {"layer": int(pr["layer_global"]),
               "r": round(pearson(list(k), list(y)), 4),
               "rho": round(spearman(list(k), list(y)), 4),
               "r_rate": round(pearson(list(k), list(rate)), 4),
               "r_vs_stated_mu": round(pearson(list(k), list(mu)), 4)}
        out["versions"][vname(v)] = rec
        lines.append(f"  {vname(v):8s} L{rec['layer']}: vs behaviour r={rec['r']:+.3f} "
                     f"rho={rec['rho']:+.3f} | vs stated mu r={rec['r_vs_stated_mu']:+.3f}")
    save_json(d / "revealed_versions.json", out)
    (d / "revealed_versions.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- direction geometry -------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-12)


def probe_dir(pr):
    return unit(np.asarray(pr["coef"]) / np.asarray(pr["std"]))


def direction_family(model):
    """-> {layer: [(name, unit vec)]} for every version at its own layer plus
    the stage4 / global reference directions at each working layer."""
    import torch
    fam = {}
    D = torch.load(P1 / "results" / "stage4" / model / "directions.pt", weights_only=False)
    for L, dd in D["dirs"].items():
        fam.setdefault(int(L), [])
        for kind in ("utility", "choice", "pool", "valence"):
            fam[int(L)].append((f"s4_{kind}", unit(dd[kind])))
    gp = P1 / "results" / "surf" / "global" / model / "utility_dir_global.pt"
    if gp.exists():
        g = torch.load(gp, weights_only=False)
        fam.setdefault(int(g["layer"]), []).append(("global", unit(g["dir"].numpy())))
    g2p = out_dir(model) / "utility_dir_global2.pt"
    if g2p.exists():
        g2 = torch.load(g2p, weights_only=False)
        fam.setdefault(int(g2["layer"]), []).append(("global2", unit(g2["dir"].numpy())))
    for v in versions(model):
        pr = pl._load_probe(model, v)
        fam.setdefault(int(pr["layer_global"]), []).append((vname(v), probe_dir(pr)))
    return fam


def cmd_cosines(model):
    d = out_dir(model)
    fam = direction_family(model)
    out, lines = {}, [f"{model} direction cosines (unit coef/std per probe version)"]
    for L in sorted(fam):
        names = [n for n, _ in fam[L]]
        M = np.stack([v for _, v in fam[L]])
        C = M @ M.T
        out[str(L)] = {"names": names, "cos": np.round(C, 4).tolist()}
        lines.append(f"  layer {L}: {len(names)} directions")
        vs = [n for n in names if n.startswith("v")]
        ref = [n for n in names if not n.startswith("v")]
        for r in ref:
            i = names.index(r)
            lines.append(f"    cos({r}, ·): " + " ".join(f"{n}={C[i, names.index(n)]:+.2f}"
                                                        for n in vs))
        if len(vs) > 1:
            lines.append("    consecutive versions: "
                         + " ".join(f"{a}->{b}={C[names.index(a), names.index(b)]:+.3f}"
                                    for a, b in zip(vs[:-1], vs[1:])))
    save_json(d / "direction_cosines.json", out)
    (d / "direction_cosines.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_extra_dirs(model):
    import torch
    W = work_layers(model)
    L = W[1]
    dirs = {}
    for name, v in (("utility_v3", 3), ("utility_v10", 10), ("utility_v15L18", "15_L18")):
        pr = pl._load_probe(model, v)
        assert int(pr["layer_global"]) == L, (name, pr["layer_global"], L)
        dirs[name] = probe_dir(pr).astype(np.float32)
    torch.save({"dirs": {L: dirs}, "source": "surf_probe_versions.cmd_extra_dirs"},
               out_dir(model) / "extra_dirs.pt")
    print(f"extra_dirs.pt: layer {L}, " + ", ".join(dirs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acts", "global2", "e2", "handbuilt", "revealed",
                                    "cosines", "extra-dirs", "all-cpu"])
    ap.add_argument("model")
    a = ap.parse_args()
    fns = {"acts": cmd_acts, "global2": cmd_global2, "e2": cmd_e2,
           "handbuilt": cmd_handbuilt, "revealed": cmd_revealed, "cosines": cmd_cosines,
           "extra-dirs": cmd_extra_dirs}
    if a.cmd == "all-cpu":
        for c in ("global2", "e2", "handbuilt", "revealed", "cosines", "extra-dirs"):
            fns[c](a.model)
    else:
        fns[a.cmd](a.model)


if __name__ == "__main__":
    main()
