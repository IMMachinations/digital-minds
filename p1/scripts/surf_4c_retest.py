"""4C retest with the enlarged-set utility directions: do they still causally
move choices, and does the hardened/global direction move them more?

Mirrors stage4_rollouts.cmd_gate exactly (steer at L = work_layers[1] during
the 24-item x 6-anchor elicitation cell, mean item-oriented delta log-odds vs
the stored unsteered control cell, z vs matched-norm random-direction nulls,
readout-mass drop as the format-integrity guard), comparing three directions:
  utility_old   stage4 directions.pt 'utility' (XL-only ridge at L)
  utility_glob  results/surf/global utility_dir_global.pt (XL+SURF refit)
  probe_v3      the hardened probeloop probe_v3 direction (same layer)
Elo points = 173.7178 x delta log-odds (the elo_fits fixed-anchor identity).

Usage: uv run python scripts/surf_4c_retest.py <model>
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

import harness
import stage4 as s4
from stage4_rollouts import _mean_d

ELO_PER_LOGIT = 400.0 / np.log(10.0)
COEFS = [0.25, 0.5, 1.0]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def main(model):
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1]
    D = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                   weights_only=False)
    rn = D["resid_norm"]["completion"]
    g = torch.load(P1 / "results" / "surf" / "global" / model / "utility_dir_global.pt",
                   weights_only=False)
    assert g["layer"] == L, (g["layer"], L)
    v3 = torch.load(P1 / "results" / "surf" / "probeloop" / model / "probe_v3.pt",
                    weights_only=False)
    assert v3["layer_global"] == L, (v3["layer_global"], L)
    dirsets = {"utility_old": unit(D["dirs"][L]["utility"]),
               "utility_glob": unit(g["dir"].numpy()),
               "probe_v3": unit(np.asarray(v3["coef"]) / np.asarray(v3["std"]))}

    items24, anchors6 = s4.pick_items24(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = s4.build_cell_prompts(items24, anchors6)
    ctrl = load_json(s4._cell_path(model, "control"))
    d0, rm0 = _mean_d(ctrl["recs"]), ctrl["readout_mass"]

    nulls = {c: [] for c in COEFS}
    for sd in range(3):
        for c in COEFS:
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(D["dirs"][L]["random"][sd]) * sign,
                                       c, rn[L])
                recs, rm, _, _ = s4.run_cell(h, prompts, spans, meta, L, v, a_ids, b_ids)
                nulls[c].append(_mean_d(recs) - d0)

    out, lines = {}, [f"{model} 4C retest at layer {L} "
                      f"(cos old/glob {float(dirsets['utility_old'] @ dirsets['utility_glob']):+.3f}, "
                      f"old/v3 {float(dirsets['utility_old'] @ dirsets['probe_v3']):+.3f}, "
                      f"glob/v3 {float(dirsets['utility_glob'] @ dirsets['probe_v3']):+.3f})"]
    for name, vec in dirsets.items():
        out[name] = {}
        for c in COEFS:
            row = {}
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(vec) * sign, c, rn[L])
                recs, rm, _, _ = s4.run_cell(h, prompts, spans, meta, L, v, a_ids, b_ids)
                tag = "plus" if sign > 0 else "minus"
                row[f"dd_{tag}"] = round(_mean_d(recs) - d0, 4)
                row[f"rm_drop_{tag}"] = round(rm0 - rm, 3)
            sd_null = float(np.std(nulls[c], ddof=1))
            row["z_plus"] = round(row["dd_plus"] / (sd_null + 1e-9), 2)
            row["elo_plus"] = round(row["dd_plus"] * ELO_PER_LOGIT, 1)
            row["elo_minus"] = round(row["dd_minus"] * ELO_PER_LOGIT, 1)
            out[name][str(c)] = row
            lines.append(f"  {name} c={c}: dElo +{row['elo_plus']:+.0f}/-{row['elo_minus']:+.0f} "
                         f"z={row['z_plus']:+.1f} rm_drop={row['rm_drop_plus']:+.2f}")
    save_json(P1 / "results" / "surf" / "global" / model / "gate_retest.json",
              {"layer": L, "nulls": {str(c): nulls[c] for c in COEFS}, "results": out})
    (P1 / "results" / "surf" / "global" / model / "gate_retest.txt").write_text(
        "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
