"""Global re-integration of SURF discoveries + probe-dependent downstream retests.

Merges every SURF-discovered measured item into the XL item set for one model:
  measure (GPU)  re-measure all unique SURF items with the EXACT stage1x design
                 (12 anchors x 2 templates x 2 orders = 48 readouts), sharded
                 and resume-safe — so XL and SURF rows sit on one protocol.
  fit (CPU)      one global anchored Thurstonian fit over XL shard obs + SURF
                 shard obs (anchors pinned at 1B values; with pinned anchors
                 the fit is separable per item, so this is exact, and the XL
                 subset must reproduce utilities_xl mu — asserted r > 0.999).
                 -> utilities_global.json
  analyze (GPU-min) the probe-dependent downstream retests on the enlarged set:
                 1C-style probe-utility convergence for v0 (S0 probe) and v3
                 (hardened probeloop probe) plus a fresh global refit; the
                 stage1x spline-vs-ridge-vs-line linearity verdict re-run with
                 adversarial extremes included; utility_dir_global.pt for the
                 4C steering retest. All correlations split XL/SURF and by
                 question_form (the stated-channel artifact lives in the target).

Usage: uv run python scripts/surf_global.py {measure,fit,analyze} <model>
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

from surf_probeloop import (apply_probe, cmd_harvest, heldout_preds, is_question,
                            probe_path, winsorize, _load_probe)
from surf_s0 import work_layers

SHARD = 500
N_ANCHORS = 12


def out_dir(model):
    d = P1 / "results" / "surf" / "global" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def surf_items(model):
    """All unique measured SURF discoveries, minus anything already in the XL
    protocol (XL free items, originals, anchors), with stable surf_g ids."""
    import items as items_mod
    ds_path = P1 / "results" / "surf" / "probeloop" / model / "dataset_c3.json"
    if not ds_path.exists():
        cmd_harvest(model, cycle=3)
    known = {r["text"].lower()
             for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")}
    known |= {it["text"].lower() for it in items_mod.load_items()}
    known |= {a["text"].lower() for a in load_json(P1 / "items_xl" / "anchors.json")}
    rows = [r for r in load_json(ds_path) if r["text"].lower() not in known]
    rows.sort(key=lambda r: r["text"].lower())
    return [{"id": f"surf_g_{k:04d}", "text": r["text"], "attrs": r.get("attrs"),
             "source": r["source"], "question_form": r["question_form"],
             "mu_prior": r["mu"]} for k, r in enumerate(rows)]


def cmd_measure(model):
    import torch
    import harness
    from lib.data import SUFFIX
    from lib.tasks import ab_scores, variant_ids
    from pairs import TEMPLATES_GENERIC

    items = surf_items(model)
    save_json(out_dir(model) / "surf_items.json", items)
    anchors = load_json(P1 / "items_xl" / "anchors.json")
    h = harness.load(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    n_shards = (len(items) + SHARD - 1) // SHARD
    for sh in range(n_shards):
        path = out_dir(model) / f"shard_{sh:02d}.jsonl"
        if path.exists():
            continue
        chunk = items[sh * SHARD:(sh + 1) * SHARD]
        prompts, meta = [], []
        for it in chunk:
            for ai, a in enumerate(anchors):
                for t in range(2):
                    for order in (0, 1):
                        x, y = (it["text"], a["text"]) if order == 0 else (a["text"], it["text"])
                        prompts.append(TEMPLATES_GENERIC[t].format(a=x, b=y, suffix=SUFFIX))
                        meta.append((it["id"], ai, t, order))
        logits = h.last_logits(prompts)
        sa, sb, _, _ = ab_scores(logits, a_ids, b_ids)
        with open(path, "w") as f:
            for (iid, ai, t, order), xa, xb in zip(meta, sa, sb):
                d = (xa - xb).item() if order == 0 else (xb - xa).item()
                p = float(torch.sigmoid(torch.tensor(d)))
                f.write(json.dumps({"item": iid, "anchor": ai, "t": t,
                                    "order": order, "p": round(p, 5)}) + "\n")
        print(f"shard {sh + 1}/{n_shards} done ({len(chunk)} items)")


def _shard_obs(paths, idx, n_free):
    obs = []
    for p in paths:
        for row in map(json.loads, p.read_text().splitlines()):
            obs.append((idx[row["item"]], n_free + row["anchor"], row["p"]))
    return obs


def cmd_fit(model):
    import thurstone
    xl_rows = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    surf_rows = load_json(out_dir(model) / "surf_items.json")
    anchors = load_json(P1 / "items_xl" / "anchors.json")
    vals = load_json(P1 / "results" / "surf" / "s0" / model / "anchor_values.json")
    free = xl_rows + surf_rows
    idx = {r["id"]: k for k, r in enumerate(free)}
    n = len(free)
    obs = _shard_obs(sorted((P1 / "results" / "stage1x" / model).glob("shard_*.jsonl")),
                     idx, n)
    obs += _shard_obs(sorted(out_dir(model).glob("shard_*.jsonl")), idx, n)
    fit = thurstone.fit_anchored(
        n + N_ANCHORS, obs, list(range(n, n + N_ANCHORS)),
        [vals[a["id"]][0] for a in anchors], [vals[a["id"]][1] for a in anchors])
    out = []
    for k, r in enumerate(free):
        src = "xl" if k < len(xl_rows) else r["source"]
        out.append({"id": r["id"], "text": r["text"], "mu": round(fit["mu"][k], 4),
                    "sigma2": round(fit["sigma2"][k], 4), "source": src,
                    "question_form": bool(r.get("question_form",
                                                is_question(r["text"], None)))})
    save_json(out_dir(model) / "utilities_global.json", out)
    # separability sanity: pinned anchors -> XL mu must reproduce utilities_xl
    r_rep = pearson([o["mu"] for o in out[:len(xl_rows)]],
                    [r["mu"] for r in xl_rows])
    mus = [o["mu"] for o in out]
    lines = [f"{model} global anchored fit: n={n} items "
             f"({len(xl_rows)} XL + {len(surf_rows)} SURF), {len(obs)} readouts, "
             f"nll={fit['nll']}",
             f"XL-subset reproduction vs utilities_xl: r={r_rep:+.5f} "
             f"(separability check {'PASS' if r_rep > 0.999 else 'FAIL'})",
             f"global mu span [{min(mus):+.2f}, {max(mus):+.2f}]; SURF subset "
             f"[{min(mus[len(xl_rows):]):+.2f}, {max(mus[len(xl_rows):]):+.2f}]"]
    assert r_rep > 0.999, r_rep
    (out_dir(model) / "fit_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_analyze(model):
    import torch
    from sklearn.linear_model import RidgeCV
    import utility_spline as usp

    d = out_dir(model)
    rows = load_json(d / "utilities_global.json")
    n_xl = sum(r["source"] == "xl" for r in rows)
    mu_raw = np.array([r["mu"] for r in rows])
    mu, n_clip = winsorize(mu_raw)
    qf = np.array([r["question_form"] for r in rows], bool)
    is_surf = np.array([r["source"] != "xl" for r in rows], bool)

    xl_acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt",
                         weights_only=False).float()
    cache = d / "acts_global_surf.pt"
    if cache.exists():
        s_acts = torch.load(cache, weights_only=False).float()
    else:
        import harness
        import probes
        h = harness.load(model)
        full = probes.item_acts(h, [r for r in rows if r["source"] != "xl"])
        s_acts = full[work_layers(model)].clone()
        torch.save(s_acts.half(), cache)
        s_acts = s_acts.float()
    acts = torch.cat([xl_acts, s_acts], dim=1)

    lines = [f"{model} global downstream retests (n={len(rows)}: {n_xl} XL + "
             f"{len(rows) - n_xl} SURF; {n_clip} targets winsorized)"]

    def rep(pred, mask, label):
        r = pearson(list(np.asarray(pred)[mask]), list(mu_raw[mask]))
        rho = spearman(list(np.asarray(pred)[mask]), list(mu_raw[mask]))
        return f"{label}: r={r:+.3f} rho={rho:+.3f} (n={int(mask.sum())})"

    # 1C-style probe convergence: v0, hardened v3, fresh global refit
    for name, v in (("v0 (S0 probe)", 0), ("v3 (hardened)", 3)):
        p = _load_probe(model, v)
        k = apply_probe(p, acts)
        lines.append(f"probe {name} on global set: "
                     + "; ".join([rep(k, np.ones(len(rows), bool), "all"),
                                  rep(k, ~is_surf, "XL"), rep(k, is_surf, "SURF"),
                                  rep(k, qf, "qform"), rep(k, ~qf, "declarative")]))
    preds, rs = heldout_preds(acts, mu)
    lp = int(np.argmax(rs))
    lines.append(f"fresh global probe (held-out, layer {work_layers(model)[lp]}): "
                 + "; ".join([rep(preds[lp], np.ones(len(rows), bool), "all"),
                              rep(preds[lp], ~is_surf, "XL"),
                              rep(preds[lp], is_surf, "SURF"),
                              rep(preds[lp], qf, "qform")]))
    X = acts[lp].numpy()
    m0, sd = X.mean(0), X.std(0) + 1e-6
    m = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X - m0) / sd, mu)
    torch.save({"layer_pos": lp, "layer_global": work_layers(model)[lp], "mean": m0,
                "std": sd, "coef": m.coef_, "intercept": float(m.intercept_),
                "alpha": float(m.alpha_), "cv_r": round(rs[lp], 4), "n": len(rows)},
               d / "probe_global.pt")
    wdir = m.coef_ / sd
    torch.save({"layer": work_layers(model)[lp],
                "dir": torch.tensor(wdir / np.linalg.norm(wdir))},
               d / "utility_dir_global.pt")

    # spline-vs-ridge-vs-line at the probe layer (stage1x.cmd_analyze protocol)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(mu))
    folds = [perm[: len(mu) // 2], perm[len(mu) // 2:]]
    for n_bins in (40, 64, 100):
        pred_sp, pred_ln = np.zeros(len(mu)), np.zeros(len(mu))
        pred_ri = np.zeros(len(mu))
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
            lines.append("ridge (held-out): "
                         + "; ".join([rep(pred_ri, np.ones(len(mu), bool), "all"),
                                      rep(pred_ri, is_surf, "SURF")]))
        lines.append(f"bins={n_bins}: spline "
                     + rep(pred_sp, np.ones(len(mu), bool), "all")
                     + " | " + rep(pred_sp, is_surf, "SURF")
                     + f" ;; line " + rep(pred_ln, np.ones(len(mu), bool), "all"))
    (d / "analyze_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["measure", "fit", "analyze"])
    ap.add_argument("model")
    a = ap.parse_args()
    {"measure": cmd_measure, "fit": cmd_fit, "analyze": cmd_analyze}[a.cmd](a.model)


if __name__ == "__main__":
    main()
