"""Elo (Bradley-Terry) fits: the logit-link twin of the Thurstonian utilities,
for (a) a robustness line (Elo vs mu rank agreement on the 1B battery) and
(b) expressing 4A steering effects in Elo points, directly comparable to the
emotions paper's blissful +212 / hostile -303 benchmarks.

Elo convention: P(i beats j) = 1/(1+10^-(Ri-Rj)/400), so an Elo difference is
(400/ln 10) x the log-odds = 173.7178 x d. For 4A, each item is compared only
against FIXED anchors, so the item's Elo shift under steering equals
173.7178 x (mean per-comparison delta log-odds) - assumption-free, no refit.
The steered cells' raw records are recomputed (logits are deterministic; this
is exact recovery of the original run's comparisons).

Usage: uv run python scripts/elo_fits.py <model>   (1B fit is CPU; 4A needs GPU)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.tasks import variant_ids
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

ELO_PER_LOGIT = 400.0 / np.log(10.0)


def bt_fit(n_items, obs, steps=2000, lr=0.05):
    """Bradley-Terry MLE on fractional observations (i, j, p): logit link,
    mean-zero identification. The logit twin of thurstone.fit."""
    i = torch.tensor([o[0] for o in obs])
    j = torch.tensor([o[1] for o in obs])
    p = torch.tensor([o[2] for o in obs], dtype=torch.float64).clamp(1e-6, 1 - 1e-6)
    r = torch.zeros(n_items, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([r], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        z = r[i] - r[j]
        loss = -(p * torch.nn.functional.logsigmoid(z)
                 + (1 - p) * torch.nn.functional.logsigmoid(-z)).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            r -= r.mean()
    return r.detach().numpy()


def part1_1b(model):
    recs = load_json(P1 / "results" / "stage1b" / model / "pairs_raw.json")
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    n = len(ut)
    obs = [(r["i"], r["j"], r["p"]) for r in recs]
    ratings = bt_fit(n, obs)
    elo = ratings * ELO_PER_LOGIT
    mu = [r["mu"] for r in ut]
    rho = spearman(list(elo), mu)
    rr = pearson(list(elo), mu)
    save_json(P1 / "results" / "stage1b" / model / "elo.json",
              [{"id": r["id"], "elo": round(float(e), 1)} for r, e in zip(ut, elo)])
    line = (f"{model} 1B Elo (BT logit twin, n={n}, {len(obs)} obs): "
            f"vs Thurstonian mu spearman {rho:+.4f} pearson {rr:+.4f}; "
            f"Elo span [{elo.min():.0f}, {elo.max():.0f}]")
    print(line)
    return line


def part2_4a(model):
    import harness
    import stage4 as s4
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1]
    D = torch.load(s4.out_dir(model) / "directions.pt", weights_only=False)
    rn = D["resid_norm"]["completion"]
    den_all = torch.load((P1 / "results" / "stage2" / model / "vectors.pt"))["den"].float()
    from stage2 import EMOTIONS
    items24, anchors6 = s4.pick_items24(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = s4.build_cell_prompts(items24, anchors6)

    def mean_d(recs):
        ps = np.clip([r["p"] for r in recs], 1e-6, 1 - 1e-6)
        return float(np.mean(np.log(ps / (1 - ps))))

    recs0, _, _, _ = s4.run_cell(h, prompts, spans, meta, L, None, a_ids, b_ids)
    d0 = mean_d(recs0)
    cells = [(e, 0.5) for e in s4.pick_emotions20()]
    cells += [("blissful", 0.25), ("blissful", 1.0), ("hostile", 0.25), ("hostile", 1.0)]
    cells += [(f"random{sd}", 0.5) for sd in range(3)]
    rows = []
    for e, c in cells:
        if e.startswith("random"):
            v_unit = torch.tensor(D["dirs"][L]["random"][int(e[-1])])
        else:
            v_unit = torch.nn.functional.normalize(den_all[L, EMOTIONS.index(e)], dim=-1)
        vec = harness.scaled_vec(v_unit, c, rn[L])
        recs, _, _, _ = s4.run_cell(h, prompts, spans, meta, L, vec, a_ids, b_ids)
        delo = ELO_PER_LOGIT * (mean_d(recs) - d0)
        rows.append({"emotion": e, "coef": c, "delta_elo": round(delo, 1)})
        print(f"  4A Elo {e}@{c}: {delo:+.1f}")
    save_json(s4.out_dir(model) / "4a_elo.json", rows)
    return rows


if __name__ == "__main__":
    m = sys.argv[1]
    part1_1b(m)
    part2_4a(m)
