"""Revealed-target probe (queue item 5): train a probe against BEHAVIOR, not
stated mu — the channel where the original E2 Goodhart lived — then SURF
against it and referee behaviorally.

  label (GPU)   menu-then-do choice rates for a ~400-item stratified panel
                (global-mu deciles, XL + SURF mixed) vs the IN-LOOP anchor env
                (12 rollouts/item; the 3 held-out referee envs stay untouched
                for eval). Sharded, resume-safe. Target = empirical logit
                log((c+.5)/(n-c+.5)) to unsaturate the rate.
  fit (GPU-min) RidgeCV on working-layer acts -> revealed_probe.pt; reports
                held-out r vs the behavioral target, plus how well the stated
                probes (v0/v3) predict behavior on the same panel (the
                stated/revealed convergence at the probe level).
  search (GPU)  SURF run `rvp1` with fitness = the revealed probe.
  eval (GPU)    held-out 3-env Tier-3 referee on the top-20 discoveries;
                compare against plc3 (stated-hardened search: 0.821).

Usage: uv run python scripts/surf_revealed_probe.py {label,fit,search,eval} <model>
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

from surf_probeloop import apply_probe, heldout_preds, _load_probe
from surf_s0 import work_layers

N_PANEL = 400
N_ROLLS = 12
SHARD = 50


def out_dir(model):
    d = P1 / "results" / "surf" / "revealed" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel(model):
    """Stratified ~400-item panel over global mu deciles, XL+SURF mixed."""
    rows = load_json(P1 / "results" / "surf" / "global" / model / "utilities_global.json")
    rng = np.random.RandomState(7)
    rows = sorted(rows, key=lambda r: r["mu"])
    edges = np.linspace(0, len(rows), 41).astype(int)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        binr = rows[lo:hi]
        xl = [r for r in binr if r["source"] == "xl"]
        sf = [r for r in binr if r["source"] != "xl"]
        take = [xl[i] for i in rng.choice(len(xl), min(6, len(xl)), replace=False)] \
            + [sf[i] for i in rng.choice(len(sf), min(4, len(sf)), replace=False)]
        out += take
    return out[:N_PANEL]


def cmd_label(model):
    import surf_scores
    items = panel(model)
    save_json(out_dir(model) / "panel.json", items)
    handles = surf_scores.Handles(model)
    t3 = surf_scores.Tier3Revealed(handles, model, n_rolls=N_ROLLS)  # in-loop anchor env
    n_shards = (len(items) + SHARD - 1) // SHARD
    for sh in range(n_shards):
        path = out_dir(model) / f"labels_{sh:02d}.json"
        if path.exists():
            continue
        chunk = items[sh * SHARD:(sh + 1) * SHARD]
        rates = t3.score([r["text"] for r in chunk])
        save_json(path, [{"id": r["id"], "rate": round(c, 4)}
                         for r, c in zip(chunk, rates)])
        print(f"labels shard {sh + 1}/{n_shards} done")


def _labels(model):
    items = load_json(out_dir(model) / "panel.json")
    rate = {}
    for p in sorted(out_dir(model).glob("labels_*.json")):
        for r in load_json(p):
            rate[r["id"]] = r["rate"]
    items = [r for r in items if r["id"] in rate]
    c = np.array([rate[r["id"]] * N_ROLLS for r in items])
    y = np.log((c + 0.5) / (N_ROLLS - c + 0.5))  # empirical logit of choice rate
    return items, np.array([rate[r["id"]] for r in items]), y


def _panel_acts(model, items):
    import torch
    cache = out_dir(model) / "acts_panel.pt"
    if cache.exists():
        acts = torch.load(cache, weights_only=False).float()
        if acts.shape[1] == len(items):
            return acts
    import harness
    import probes
    h = harness.load(model)
    acts = probes.item_acts(h, items)[work_layers(model)].clone()
    torch.save(acts.half(), cache)
    return acts.float()


def cmd_fit(model):
    import torch
    from sklearn.linear_model import RidgeCV
    items, rate, y = _labels(model)
    acts = _panel_acts(model, items)
    preds, rs = heldout_preds(acts, y)
    lp = int(np.argmax(rs))
    X = acts[lp].numpy()
    m0, sd = X.mean(0), X.std(0) + 1e-6
    m = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X - m0) / sd, y)
    torch.save({"layer_pos": lp, "layer_global": work_layers(model)[lp], "mean": m0,
                "std": sd, "coef": m.coef_, "intercept": float(m.intercept_),
                "alpha": float(m.alpha_), "cv_r": round(rs[lp], 4), "n": len(items),
                "target": "empirical_logit_choice_rate"},
               out_dir(model) / "revealed_probe.pt")
    mu = np.array([r["mu"] for r in items])
    lines = [f"{model} revealed-target probe: n={len(items)} labeled "
             f"(rate mean {rate.mean():.3f}, sd {rate.std():.3f})",
             f"held-out r vs behavioral target: {rs[lp]:+.3f} at layer "
             f"{work_layers(model)[lp]} (per layer: "
             + ", ".join(f"{work_layers(model)[k]}:{r:+.3f}" for k, r in enumerate(rs)) + ")",
             f"stated mu vs behavioral target on panel: r={pearson(list(mu), list(y)):+.3f} "
             f"rho={spearman(list(mu), list(y)):+.3f}"]
    for name, v in (("v0", 0), ("v3", 3)):
        k = apply_probe(_load_probe(model, v), acts)
        lines.append(f"stated probe {name} vs behavioral target: "
                     f"r={pearson(list(k), list(y)):+.3f} "
                     f"rho={spearman(list(k), list(y)):+.3f}")
    (out_dir(model) / "fit_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_search(model):
    import surf
    import surf_scores
    init = surf.SURF_ROOT / "tags" / model / "pool_weights_max.json"
    cfg = surf.RunConfig(
        experiment="rvp1", model=model, direction="max", fitness="t1_probe",
        allowed_tiers=["t0", "t1", "t2"], pool_file="items/surf_attributes_item.json",
        pool_kind="item", pool_init=str(init) if init.exists() else "", seed=0, T=15,
        probe_path=str(out_dir(model) / "revealed_probe.pt"))
    surf.run(cfg, surf_scores.build(cfg))


def cmd_eval(model):
    import surf
    import surf_scores
    from surf_e2_referee import heldout_env_ids
    st, _ = surf._load_state(surf.SURF_ROOT / "rvp1" / model / "max-s0")
    top = sorted(st["buffer"], key=lambda e: -e["score"])[:20]
    t3 = surf_scores.Tier3Revealed(surf_scores.Handles(model), model, n_rolls=12,
                                   anchor_ids=heldout_env_ids(model))
    rates = t3.score([e["text"] for e in top])
    rows = [{"text": e["text"], "attrs": e["attrs"], "score": e["score"],
             "mu_full": e["mu_full"], "t3_rate": round(c, 3)}
            for e, c in zip(top, rates)]
    save_json(out_dir(model) / "eval.json", rows)
    lines = [f"{model} revealed-probe search, top-20 held-out referee: "
             f"mean t3 {np.mean(rates):.3f} (stated-hardened plc3 was 0.821; "
             "E2 stated-probe arm was 0.512)"]
    for r in rows[:8]:
        lines.append(f"  {r['t3_rate']:.2f} mu={r['mu_full']}  {r['text']!r}")
    (out_dir(model) / "eval_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "fit", "search", "eval"])
    ap.add_argument("model")
    a = ap.parse_args()
    {"label": cmd_label, "fit": cmd_fit, "search": cmd_search,
     "eval": cmd_eval}[a.cmd](a.model)


if __name__ == "__main__":
    main()
