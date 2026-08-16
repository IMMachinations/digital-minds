"""Stage 1-XL driver: anchored mu re-measurement of the 20x item set and the
re-powered utility-geometry analyses.

Usage: uv run python stage1x.py run <model>       anchored elicitation + fit + gates
       uv run python stage1x.py analyze <model>   acts, spline-vs-ridge at scale,
                                                  ridge refit, confound audit
       uv run python stage1x.py cross

Protocol: every item (generated + 185 validation originals) is compared against
12 fixed anchors (mu-spanning, low-variance, 1E-frame-stable originals), both
orders x 2 templates = 48 letter-logit readouts per item; anchored Thurstonian
fit pins the anchors at the model's own 1B (mu, sigma2). Gate (a): the 185
re-measured originals must recover their full-battery mu at r >= 0.85.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.data import SUFFIX
from lib.tasks import ab_scores, variant_ids
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import harness
import items as items_mod
import stats
import thurstone
from pairs import TEMPLATES_GENERIC

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
N_ANCHORS = 12
SHARD = 500


def out_dir(model):
    d = P1 / "results" / "stage1x" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def pick_anchors():
    """12 originals: 2 per consensus-z sextile, preferring low mean 1B sigma2
    and low 1E stability_std (frame-stable), spread over domains."""
    its = items_mod.load_items()
    sys.path.insert(0, str(P1 / "scripts"))
    from build_xl import consensus_z
    cz = consensus_z(its)
    s2 = {}
    stab = {}
    for m in SUBJECTS:
        for r in load_json(P1 / "results" / "stage1b" / m / "utilities.json"):
            s2.setdefault(r["id"], []).append(r["sigma2"])
        p = P1 / "results" / "stage1e" / m / "frame_utilities.json"
        if p.exists():
            for r in load_json(p):
                stab.setdefault(r["id"], []).append(r["stability_std"])
    rows = [{"id": it["id"], "text": it["text"], "z": float(cz[k]),
             "s2": float(np.mean(s2[it["id"]])),
             "stab": float(np.mean(stab.get(it["id"], [0.5])))}
            for k, it in enumerate(its)]
    rows.sort(key=lambda r: r["z"])
    sext = np.array_split(rows, 6)
    anchors = []
    for s in sext:
        cand = sorted(s, key=lambda r: (r["stab"], r["s2"]))
        anchors += cand[:2]
    return anchors


def items_xl():
    gen = json.loads((P1 / "items_xl" / "generated.json").read_text())
    anchors = pick_anchors()
    a_ids = {a["id"] for a in anchors}
    orig = [it for it in items_mod.load_items() if it["id"] not in a_ids]
    return gen + orig, anchors  # free items (generated + validation originals), anchors


def cmd_run(model):
    free, anchors = items_xl()
    ut1b = {r["id"]: (r["mu"], r["sigma2"])
            for r in load_json(P1 / "results" / "stage1b" / model / "utilities.json")}
    h = harness.load(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")

    n_shards = (len(free) + SHARD - 1) // SHARD
    for sh in range(n_shards):
        path = out_dir(model) / f"shard_{sh:02d}.jsonl"
        if path.exists():
            continue
        chunk = free[sh * SHARD:(sh + 1) * SHARD]
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

    # fit
    idx = {it["id"]: k for k, it in enumerate(free)}
    n_free = len(free)
    anchor_idx = list(range(n_free, n_free + N_ANCHORS))
    obs = []
    for sh in range(n_shards):
        for row in map(json.loads, (out_dir(model) / f"shard_{sh:02d}.jsonl").read_text().splitlines()):
            obs.append((idx[row["item"]], n_free + row["anchor"], row["p"]))
    fit = thurstone.fit_anchored(
        n_free + N_ANCHORS, obs, anchor_idx,
        [ut1b[a["id"]][0] for a in anchors], [ut1b[a["id"]][1] for a in anchors])
    rows = [{"id": it["id"], "domain": it["domain"], "text": it["text"],
             "mu": round(fit["mu"][k], 4), "sigma2": round(fit["sigma2"][k], 4),
             "tags": it["tags"], "validation": it["id"] in ut1b}
            for k, it in enumerate(free)]
    save_json(out_dir(model) / "utilities_xl.json", rows)

    val = [(r["mu"], ut1b[r["id"]][0]) for r in rows if r["validation"]]
    r_val = pearson([a for a, _ in val], [b for _, b in val])
    rho_val = spearman([a for a, _ in val], [b for _, b in val])
    lines = [f"{model}: Stage 1-XL anchored elicitation "
             f"({len(rows)} items, {len(obs)} readouts, nll={fit['nll']})",
             f"GATE (a): validation originals (n={len(val)}) anchored-mu vs 1B mu: "
             f"r={r_val:+.3f} rho={rho_val:+.3f} -> "
             f"{'PASS' if r_val >= 0.85 else 'FAIL'} (>=0.85)"]
    gen_rows = [r for r in rows if not r["validation"]]
    X, names = items_mod.covariate_matrix(gen_rows)
    betas, r2 = stats.ols([r["mu"] for r in gen_rows], X, names)
    lines.append(f"confound audit (generated, n={len(gen_rows)}): r2={r2}  {betas}")
    for dom in sorted({r["domain"] for r in rows}):
        v = [r["mu"] for r in gen_rows if r["domain"] == dom]
        if v:
            lines.append(f"  {dom}: n={len(v)} mu mean {np.mean(v):+.2f} "
                         f"[{np.min(v):+.2f}, {np.max(v):+.2f}]")
    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_analyze(model):
    import probes
    from sklearn.linear_model import RidgeCV
    sys.path.insert(0, str(P1 / "scripts"))
    import utility_spline as usp

    rows = load_json(out_dir(model) / "utilities_xl.json")
    mu = np.array([r["mu"] for r in rows])
    h = harness.load(model)
    w = list(h.spec.steer_layers[2:])
    acts_path = out_dir(model) / "acts_xl.pt"
    if acts_path.exists():
        acts = torch.load(acts_path).float()
    else:
        full = probes.item_acts(h, rows)
        acts = full[w].clone()
        torch.save(acts.half(), acts_path)
        acts = acts.float()
    L_mid = 1  # middle of the 3 working layers
    X = acts[L_mid].numpy()

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(mu))
    folds = [perm[:len(mu) // 2], perm[len(mu) // 2:]]
    lines = [f"{model}: Stage 1-XL utility geometry at scale "
             f"(n={len(mu)}, layer {w[L_mid]})"]
    for n_bins in (40, 64, 100):
        pred_sp = np.zeros(len(mu))
        pred_ln = np.zeros(len(mu))
        pred_ri = np.zeros(len(mu))
        for f in range(2):
            te, tr = folds[f], folds[1 - f]
            m0 = X[tr].mean(0)
            _, _, Vt = np.linalg.svd(X[tr] - m0, full_matrices=False)
            B64 = Vt[:64]
            Ztr, Zte = (X[tr] - m0) @ B64.T, (X[te] - m0) @ B64.T
            order = np.argsort(mu[tr])
            cent = np.stack([Ztr[b].mean(0)
                             for b in np.array_split(order, n_bins)])
            sp = usp.open_spline(cent, k_fit=8)
            u_tr, u_te = usp.spline_u(Ztr, sp), usp.spline_u(Zte, sp)
            a = np.polyfit(u_tr, mu[tr], 1)
            pred_sp[te] = np.polyval(a, u_te)
            _, _, Vc = np.linalg.svd(cent - cent.mean(0), full_matrices=False)
            lu_tr, lu_te = Ztr @ Vc[0], Zte @ Vc[0]
            pred_ln[te] = np.polyval(np.polyfit(lu_tr, mu[tr], 1), lu_te)
            if n_bins == 40:
                sd = X[tr].std(0) + 1e-6
                ridge = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X[tr] - m0) / sd, mu[tr])
                pred_ri[te] = ridge.predict((X[te] - m0) / sd)
                if f == 0:
                    wdir = ridge.coef_ / sd
                    wdir = wdir / np.linalg.norm(wdir)
                    torch.save({"layer": w[L_mid], "dir": torch.tensor(wdir)},
                               out_dir(model) / "utility_dir.pt")
        if n_bins == 40:
            lines.append(f"  ridge (held-out): r = {pearson(list(pred_ri), list(mu)):+.3f}")
        lines.append(f"  bins={n_bins}: spline r = {pearson(list(pred_sp), list(mu)):+.3f}  "
                     f"line r = {pearson(list(pred_ln), list(mu)):+.3f}")
    (out_dir(model) / "utility_spline_xl.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_cross(_=None):
    lines = ["Stage 1-XL cross-model summary", ""]
    for m in SUBJECTS:
        for f in ("summary.txt", "utility_spline_xl.txt"):
            p = out_dir(m) / f
            if p.exists():
                lines += p.read_text().splitlines() + [""]
    (P1 / "results" / "stage1x" / "cross_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model", nargs="?")
    args = ap.parse_args()
    {"run": cmd_run, "analyze": cmd_analyze, "cross": cmd_cross}[args.cmd](args.model)


if __name__ == "__main__":
    main()
