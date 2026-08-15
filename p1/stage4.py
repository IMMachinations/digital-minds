"""Stage 4 driver: causal steering experiments (spec 4A-4D).

Usage: uv run python stage4.py <cmd> [model] [--smoke]
  dirs <m>     extract directions (choice probe, pool contrast, utility@3 layers),
               geometry table, frame resid norms -> directions.pt / dirs_report.json
  4a <m>       emotion-vector steering during anchored elicitation (+ geodesic on
               qwen25-7b) -> 4a cells + dmu tables
  gate <m>     behavioral gate: preference/utility directions must move choices
  4bc <m>      steered Stage-3-style rollouts for gate-passing direction-sets
               (+ shared random arm) + steered probe pass + NLL pass
  4d <m>       cross-frame transfer (bare directions in the agentic frame)
  judge        one 32B pass over all models' coherence queues
  analyze <m>  contrasts + summary.txt        cross   cross-model summary

Design contracts (see steering.py): steer only the free item's span in
elicitation (never anchors/suffix); hybrid hook for rollouts; Stage-3 bare
rollouts are the unsteered control arm; behavior-before-affect enforced;
within-batch mixed arms; w[1] pre-registered primary layer; readout-mass and
coherence guards with pre-registered demotion rules.
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
import judge as judge_mod
import rollout as ro
import steering as st
import stage1x
from pairs import TEMPLATES_GENERIC
from stage2 import EMOTIONS, out_dir as s2_dir

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]
RATER = "qwen25-32b"
COEFS = [0.25, 0.5, 1.0]
NORMS = load_json(P1 / "items" / "emotion_norms.json")


def out_dir(model):
    d = P1 / "results" / "stage4" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_layers(h):
    return list(h.spec.steer_layers[2:])


# ---- shared elicitation-cell machinery ----------------------------------------------------------

def pick_items24(model):
    """24 mu-spanning validation originals (4/sextile), excluding the 12 anchors."""
    anchors = stage1x.pick_anchors()
    a_ids = {a["id"] for a in anchors}
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    import items as items_mod
    texts = {it["id"]: it["text"] for it in items_mod.load_items()}
    rows = sorted((r for r in ut if r["id"] not in a_ids), key=lambda r: r["mu"])
    sext = np.array_split(rows, 6)
    out = []
    for s in sext:
        picked = sorted(s, key=lambda r: r["sigma2"])[:4]
        out += [{"id": r["id"], "text": texts[r["id"]], "mu": r["mu"],
                 "sigma2": r["sigma2"]} for r in picked]
    assert len(out) == 24
    return out, [a for i, a in enumerate(anchors) if i % 2 == 0]  # 6 alternate anchors


def build_cell_prompts(items, anchors, steer_anchor=False):
    """-> (prompts, spans, meta) for 24 items x 6 anchors x 2 templates x 2
    orders; span = the FREE ITEM's span (or the anchor's, for the localization
    control)."""
    prompts, spans, meta = [], [], []
    for it in items:
        for ai, a in enumerate(anchors):
            for t in range(2):
                for order in (0, 1):
                    x, y = (it["text"], a["text"]) if order == 0 else (a["text"], it["text"])
                    prompt, sa, sb = st.ab_prompt_spans(TEMPLATES_GENERIC[t], x, y, SUFFIX)
                    item_span = sa if order == 0 else sb
                    anchor_span = sb if order == 0 else sa
                    prompts.append(prompt)
                    spans.append(anchor_span if steer_anchor else item_span)
                    meta.append((it["id"], ai, t, order))
    return prompts, spans, meta


def run_cell(h, prompts, spans, meta, layer, vec, a_ids, b_ids,
             control_logits=None, sanity=None):
    """-> (records, readout_mass, kl_vs_control) for one steering cell."""
    logits = st.last_logits_span_steer(h, prompts, spans, layer, vec, sanity=sanity)
    sa, sb, _, _ = ab_scores(logits, a_ids, b_ids)
    recs = []
    for (iid, ai, t, order), xa, xb in zip(meta, sa, sb):
        d = (xa - xb).item() if order == 0 else (xb - xa).item()
        recs.append({"item": iid, "anchor": ai, "t": t, "order": order,
                     "p": float(torch.sigmoid(torch.tensor(d)))})
    rm = st.readout_mass(logits, a_ids, b_ids)
    kl = None
    if control_logits is not None:
        lp = logits.log_softmax(-1)
        lc = control_logits.log_softmax(-1)
        kl = float((lc.exp() * (lc - lp)).sum(-1).mean())
    return recs, rm, kl, logits


def fit_cell(model, items, anchors, recs):
    """Anchored refit -> per-item mu dict."""
    import thurstone
    ut1b = {r["id"]: (r["mu"], r["sigma2"])
            for r in load_json(P1 / "results" / "stage1b" / model / "utilities.json")}
    idx = {it["id"]: k for k, it in enumerate(items)}
    n = len(items)
    obs = [(idx[r["item"]], n + r["anchor"], r["p"]) for r in recs]
    fit = thurstone.fit_anchored(
        n + len(anchors), obs, list(range(n, n + len(anchors))),
        [ut1b[a["id"]][0] for a in anchors], [ut1b[a["id"]][1] for a in anchors],
        steps=1500)
    return {it["id"]: fit["mu"][k] for k, it in enumerate(items)}


def pick_emotions20():
    ranked = sorted(EMOTIONS, key=lambda e: NORMS[e]["valence"])
    required = ["hostile", "miserable", "desperate", "angry", "afraid", "gloomy",
                "bored", "weary", "indifferent", "calm", "content", "hopeful",
                "playful", "happy", "joyful", "excited", "loving", "blissful",
                "euphoric", "grateful"]
    return [e for e in ranked if e in required]


# ---- dirs ---------------------------------------------------------------------------------------

def cmd_dirs(model, smoke=False):
    import random
    from sklearn.linear_model import RidgeCV
    h = harness.load(model)
    w = work_layers(h)
    rows = load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
    acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt").float()
    mu = np.array([r["mu"] for r in rows])

    # (i) option-span choice probe from reconstructed stage1x prompts
    anchors = stage1x.pick_anchors()
    rng = random.Random(0)
    order_ids = np.argsort(mu)
    dec = np.array_split(order_ids, 10)
    sample_idx = [i for d in dec for i in rng.sample(list(d), min(len(d), 100 if not smoke else 8))]
    shard_p = {}
    for sh in sorted((P1 / "results" / "stage1x" / model).glob("shard_*.jsonl")):
        for r in map(json.loads, sh.read_text().splitlines()):
            shard_p[(r["item"], r["anchor"], r["t"], r["order"])] = r["p"]
    prompts, spans_item, spans_anchor, labels, weights = [], [], [], [], []
    lo_a, hi_a = 2, 9  # one low-mu, one high-mu anchor (sextiles 1 and 5)
    for i in sample_idx:
        it = rows[i]
        for ai in (lo_a, hi_a):
            for order in (0, 1):
                key = (it["id"], ai, 0, order)
                if key not in shard_p:
                    continue
                p = shard_p[key]
                if abs(p - 0.5) < 0.025:
                    continue
                a = anchors[ai]
                x, y = (it["text"], a["text"]) if order == 0 else (a["text"], it["text"])
                prompt, sa, sb = st.ab_prompt_spans(TEMPLATES_GENERIC[0], x, y, SUFFIX)
                item_span = sa if order == 0 else sb
                anch_span = sb if order == 0 else sa
                prompts.append(prompt)
                spans_item.append(item_span)
                spans_anchor.append(anch_span)
                labels.append(p > 0.5)
                weights.append(abs(p - 0.5))
    A_item = st.span_acts(h, prompts, spans_item, w)
    A_anch = st.span_acts(h, prompts, spans_anchor, w)
    X_by_layer = {}
    for s, L in enumerate(w):
        X = np.concatenate([A_item[s].numpy(), A_anch[s].numpy()])
        y = np.array(labels + [not l for l in labels])
        wt = np.array(weights + weights)
        X_by_layer[L] = (X, y, wt)

    dirs = {}
    report = {"model": model, "n_probe_examples": len(labels) * 2, "layers": {}}
    for L in w:
        X, y, wt = X_by_layer[L]
        v_choice, auc = st.fit_logistic_direction(X, y, wt)
        # (ii) pool contrast
        q1, q3 = np.quantile(mu, [0.25, 0.75])
        Al = acts[w.index(L)].numpy()
        v_pool = st.unit(Al[mu >= q3].mean(0) - Al[mu <= q1].mean(0))
        act_rows = [k for k, r in enumerate(rows) if r["domain"] == "activities"]
        am = mu[act_rows]
        Aa = Al[act_rows]
        v_pool_act = st.unit(Aa[am >= np.quantile(am, 0.75)].mean(0)
                             - Aa[am <= np.quantile(am, 0.25)].mean(0))
        # (iii) utility ridge at this layer
        sd = Al.std(0) + 1e-6
        ridge = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((Al - Al.mean(0)) / sd, mu)
        v_util = st.unit(ridge.coef_ / sd)
        if pearson(list(Al @ v_util), list(mu)) < 0:
            v_util = -v_util
        # valence PC1 for geometry
        den = torch.load(s2_dir(model) / "vectors.pt")["den"].float()[L].numpy()
        _, _, Vt = np.linalg.svd(den - den.mean(0), full_matrices=False)
        val = np.array([NORMS[e]["valence"] for e in EMOTIONS])
        sgn = 1.0 if pearson(list((den - den.mean(0)) @ Vt[0]), list(val)) >= 0 else -1.0
        v_val = Vt[0] * sgn
        rand = [st.unit(harness.random_unit_matrix(h.spec.n_layers, h.spec.d_model,
                                                   seed=sd_)[L].numpy())
                for sd_ in range(3)]
        dirs[L] = {"choice": v_choice, "pool": v_pool, "utility": v_util,
                   "valence": v_val, "random": rand}
        report["layers"][str(L)] = {
            "choice_auc": round(auc, 4),
            "cos_choice_pool": round(float(v_choice @ v_pool), 3),
            "cos_choice_utility": round(float(v_choice @ v_util), 3),
            "cos_pool_utility": round(float(v_pool @ v_util), 3),
            "cos_utility_valence": round(float(v_util @ v_val), 3),
            "cos_pool_valence": round(float(v_pool @ v_val), 3),
            "cos_pool_activities_only": round(float(v_pool @ v_pool_act), 3)}

    # frame-matched resid norms
    items24, anchors6 = pick_items24(model)
    ep, _, _ = build_cell_prompts(items24[:12], anchors6[:3])
    norms_completion = st.frame_resid_norms(h, ep, w)
    chat_prompts = [ro.build_prompt(h, [
        {"role": "system", "content": ro.SYSTEM},
        {"role": "user", "content": f"Session brief: {t}"}]) for t in
        [it["text"] for it in items24] * 9][:200]
    norms_chat = st.frame_resid_norms(h, chat_prompts, w)
    torch.save({"dirs": {L: {k: (np.stack(v) if k == "random" else v)
                             for k, v in d.items()} for L, d in dirs.items()},
                "resid_norm": {"completion": norms_completion, "chat": norms_chat},
                "work_layers": w},
               out_dir(model) / "directions.pt")
    save_json(out_dir(model) / "dirs_report.json", report)
    print(json.dumps(report, indent=1))


# ---- 4A -----------------------------------------------------------------------------------------

def _cell_path(model, name):
    d = out_dir(model) / "4a"
    d.mkdir(exist_ok=True)
    return d / f"{name}.json"


def cmd_4a(model, smoke=False):
    h = harness.load(model)
    w = work_layers(h)
    L = w[1]
    D = torch.load(out_dir(model) / "directions.pt", weights_only=False)
    rn = D["resid_norm"]["completion"]
    den_all = torch.load(s2_dir(model) / "vectors.pt")["den"].float()
    items24, anchors6 = pick_items24(model)
    if smoke:
        items24 = items24[:6]
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = build_cell_prompts(items24, anchors6)
    prompts_anch, spans_anch, _ = build_cell_prompts(items24, anchors6, steer_anchor=True)

    # control pass (kept in memory for KL) + control fit
    ctrl_path = _cell_path(model, "control")
    recs, rm0, _, ctrl_logits = run_cell(h, prompts, spans, meta, L, None,
                                         a_ids, b_ids, sanity=items24[0]["text"])
    mu_ctrl = fit_cell(model, items24, anchors6, recs)
    save_json(ctrl_path, {"recs": recs, "readout_mass": rm0, "mu": mu_ctrl})

    emotions = pick_emotions20()[:4] if smoke else pick_emotions20()
    cells = [(e, c, L) for e in emotions for c in (COEFS[:1] if smoke else COEFS)]
    cells += [(e, 0.5, Lx) for e in ["blissful", "hostile", "content", "miserable"]
              for Lx in (w[0], w[2]) if not smoke]
    for sd in range(3):
        for c in (COEFS[:1] if smoke else COEFS):
            cells.append((f"random{sd}", c, L))
    results = {}
    coh_queue = []
    for e, c, Lx in cells:
        name = f"{e}_{c}_{Lx}"
        pth = _cell_path(model, name)
        if pth.exists():
            results[name] = load_json(pth)
            continue
        if e.startswith("random"):
            v_unit = torch.tensor(D["dirs"][Lx]["random"][int(e[-1])])
        else:
            v_unit = torch.nn.functional.normalize(
                den_all[Lx, EMOTIONS.index(e)], dim=-1)
        vec = harness.scaled_vec(v_unit, c, rn[Lx])
        recs, rm, kl, _ = run_cell(h, prompts, spans, meta, Lx, vec, a_ids, b_ids,
                                   control_logits=ctrl_logits)
        mu_s = fit_cell(model, items24, anchors6, recs)
        dmu = {iid: mu_s[iid] - mu_ctrl[iid] for iid in mu_s}
        cell = {"emotion": e, "coef": c, "layer": Lx,
                "dmu_mean": round(float(np.mean(list(dmu.values()))), 4),
                "dmu": {k: round(v, 4) for k, v in dmu.items()},
                "readout_mass": rm, "readout_drop": round(rm0 - rm, 3),
                "kl": kl}
        save_json(pth, cell)
        results[name] = cell
        print(f"4a {name}: dmu={cell['dmu_mean']:+.3f} rm_drop={cell['readout_drop']:+.2f}")
        # coherence side-gens (skip randoms)
        if not e.startswith("random"):
            gen_prompts = [f"Describe your approach to {it['text']}." for it in items24[:12]]
            outs = ro.gen_turns(
                h, [ro.build_prompt(h, [{"role": "user", "content": gp}])
                    for gp in gen_prompts],
                seed=1, max_new=120, batch=12,
                steer=(Lx, lambda sl, enc, v=vec: (
                    torch.zeros(enc["input_ids"].shape, dtype=torch.bool,
                                device="cuda").index_fill_(
                        1, torch.tensor([enc["input_ids"].shape[1] - 1],
                                        device="cuda"), True),
                    v.expand(enc["input_ids"].shape[0], 1, -1))))
            for o in outs:
                coh_queue.append({"kind": "coherence", "cell": name, "text": o[:600],
                                  "prompt": 'Rate this text: {"coherent": 0|1|2}. '
                                  "0=broken/degenerate, 1=flawed but readable, "
                                  f"2=normal.\n---\n{o[:600]}\n---"})

    # anchor-span localization control
    for e in ["blissful", "hostile"]:
        name = f"anchorspan_{e}_0.5_{L}"
        pth = _cell_path(model, name)
        if not pth.exists():
            v_unit = torch.nn.functional.normalize(den_all[L, EMOTIONS.index(e)], dim=-1)
            vec = harness.scaled_vec(v_unit, 0.5, rn[L])
            recs, rm, kl, _ = run_cell(h, prompts_anch, spans_anch, meta, L, vec,
                                       a_ids, b_ids, control_logits=ctrl_logits)
            mu_s = fit_cell(model, items24, anchors6, recs)
            dmu = {iid: mu_s[iid] - mu_ctrl[iid] for iid in mu_s}
            save_json(pth, {"emotion": f"anchorspan_{e}", "coef": 0.5, "layer": L,
                            "dmu_mean": round(float(np.mean(list(dmu.values()))), 4),
                            "readout_mass": rm, "kl": kl})

    # geodesic mini (qwen25-7b only)
    if model == "qwen25-7b" and not smoke:
        man = torch.load(s2_dir(model) / "manifold.pt", weights_only=False)
        sp = man["spline"]
        from scipy.interpolate import splev
        acts_d = torch.load(s2_dir(model) / "acts.pt")
        A2 = acts_d["acts"].float()[L].numpy()
        labs = acts_d["labels"]
        import manifold as mf
        for e in ["blissful", "hostile"]:
            name = f"geo_{e}_0.5_{L}"
            pth = _cell_path(model, name)
            if pth.exists():
                continue
            cent = A2[[k for k, l in enumerate(labs) if l == e]].mean(0)
            c64 = mf.to_basis(cent[None], man["mu64"], man["Vt64"])[0]
            c8 = (c64 - sp["mu"]) @ sp["B"].T
            uu = np.linspace(0, 1, 2000)
            pts = np.stack(splev(uu, sp["tck"]), axis=1)
            u0 = uu[((pts - c8) ** 2).sum(1).argmin()]
            du = 0.02 * (1 if NORMS[e]["valence"] > 5 else -1)
            d8 = np.stack(splev([min(u0 + du, 1.0)], sp["tck"]), axis=1)[0] - \
                np.stack(splev([u0], sp["tck"]), axis=1)[0]
            dD = (d8 @ sp["B"]) @ man["Vt64"]
            vec = harness.scaled_vec(torch.tensor(st.unit(dD)), 0.5, rn[L])
            recs, rm, kl, _ = run_cell(h, prompts, spans, meta, L, vec, a_ids, b_ids,
                                       control_logits=ctrl_logits)
            mu_s = fit_cell(model, items24, anchors6, recs)
            dmu = {iid: mu_s[iid] - mu_ctrl[iid] for iid in mu_s}
            save_json(pth, {"emotion": f"geo_{e}", "coef": 0.5, "layer": L,
                            "dmu_mean": round(float(np.mean(list(dmu.values()))), 4),
                            "readout_mass": rm, "kl": kl})
    if coh_queue:
        with open(out_dir(model) / "coherence_queue.jsonl", "a") as f:
            for t in coh_queue:
                f.write(json.dumps(t) + "\n")
    print("4a done")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    import stage4_rollouts as s4r
    fns = {"dirs": cmd_dirs, "4a": cmd_4a, "gate": s4r.cmd_gate,
           "4bc": s4r.cmd_4bc, "4d": s4r.cmd_4d, "judge": s4r.cmd_judge,
           "analyze": s4r.cmd_analyze, "cross": s4r.cmd_cross}
    fn = fns[args.cmd]
    if args.cmd in ("judge", "cross"):
        fn()
    else:
        fn(args.model, smoke=args.smoke)


if __name__ == "__main__":
    main()
