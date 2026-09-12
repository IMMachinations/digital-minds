"""4C retest with the enlarged-set utility directions: do they still causally
move choices, and does the hardened/global direction move them more?

Mirrors stage4_rollouts.cmd_gate exactly (steer at L = work_layers[1] during
the 24-item x 6-anchor elicitation cell, mean item-oriented delta log-odds vs
the stored unsteered control cell, z vs matched-norm random-direction nulls,
readout-mass drop as the format-integrity guard). Direction names:
  old          stage4 directions.pt 'utility' at the steering layer (XL-only ridge)
  glob         results/surf/global utility_dir_global.pt (XL+SURF refit)
  glob2        results/surf/probe_versions utility_dir_global2.pt (XL + all SURF, layered mu)
  v<k>         probeloop probe_v<k>.pt (v0 = the S0 probe); tagged refits such as v15_L18
A direction fit at a layer other than --layer is refused (no cross-layer steering).
Elo points = 173.7178 x delta log-odds (the elo_fits fixed-anchor identity).

Usage: uv run python scripts/surf_4c_retest.py <model> [--dirs old glob v3 ...]
                                                [--layer L] [--tag _suffix]
Default (no flags) reproduces the original three-direction run and its file
names; with --tag the outputs are gate_retest<tag>.{json,txt}.
"""
import argparse
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
DEFAULT_DIRS = ["old", "glob", "v3"]
LEGACY_NAMES = {"old": "utility_old", "glob": "utility_glob", "v3": "probe_v3"}


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def resolve_dir(model, name, L, D):
    """-> unit direction (numpy) for `name`, asserting it lives at layer L."""
    if name == "old":
        return unit(D["dirs"][L]["utility"])
    if name in ("glob", "glob2"):
        p = (P1 / "results" / "surf" / "global" / model / "utility_dir_global.pt"
             if name == "glob" else
             P1 / "results" / "surf" / "probe_versions" / model / "utility_dir_global2.pt")
        g = torch.load(p, weights_only=False)
        assert int(g["layer"]) == L, (name, g["layer"], L)
        return unit(g["dir"].numpy())
    assert name.startswith("v"), name
    sys.path.insert(0, str(P1 / "scripts"))
    from surf_probeloop import _load_probe
    v = name[1:]
    pr = _load_probe(model, int(v) if v.isdigit() else v)
    assert int(pr["layer_global"]) == L, (name, pr["layer_global"], L)
    return unit(np.asarray(pr["coef"]) / np.asarray(pr["std"]))


def main(model, dirs=None, layer=None, tag=""):
    dirs = dirs or DEFAULT_DIRS
    legacy = (dirs == DEFAULT_DIRS and layer is None and tag == "")
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1] if layer is None else layer
    assert L in w, (L, w)
    D = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                   weights_only=False)
    rn = D["resid_norm"]["completion"]
    dirsets = {(LEGACY_NAMES[n] if legacy else n): resolve_dir(model, n, L, D) for n in dirs}

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
    print(f"nulls done: sd " + ", ".join(f"c={c}:{np.std(nulls[c], ddof=1):.4f}" for c in COEFS),
          flush=True)

    names = list(dirsets)
    cos = {f"{a}/{b}": round(float(dirsets[a] @ dirsets[b]), 3)
           for i, a in enumerate(names) for b in names[i + 1:]}
    out, lines = {}, [f"{model} 4C retest at layer {L} (cos " +
                      ", ".join(f"{k} {v:+.3f}" for k, v in cos.items()) + ")"]
    for name, vec in dirsets.items():
        out[name] = {}
        for c in COEFS:
            row = {}
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(vec) * sign, c, rn[L])
                recs, rm, _, _ = s4.run_cell(h, prompts, spans, meta, L, v, a_ids, b_ids)
                t = "plus" if sign > 0 else "minus"
                row[f"dd_{t}"] = round(_mean_d(recs) - d0, 4)
                row[f"rm_drop_{t}"] = round(rm0 - rm, 3)
            sd_null = float(np.std(nulls[c], ddof=1))
            row["null_sd"] = round(sd_null, 4)
            row["z_plus"] = round(row["dd_plus"] / (sd_null + 1e-9), 2)
            row["elo_plus"] = round(row["dd_plus"] * ELO_PER_LOGIT, 1)
            row["elo_minus"] = round(row["dd_minus"] * ELO_PER_LOGIT, 1)
            out[name][str(c)] = row
            lines.append(f"  {name} c={c}: dElo +{row['elo_plus']:+.0f}/-{row['elo_minus']:+.0f} "
                         f"z={row['z_plus']:+.1f} rm_drop={row['rm_drop_plus']:+.2f}")
        print(lines[-3:], flush=True)
    od = P1 / "results" / "surf" / "global" / model
    save_json(od / f"gate_retest{tag}.json",
              {"layer": L, "dirs": dirs, "cos": cos,
               "nulls": {str(c): nulls[c] for c in COEFS}, "results": out})
    (od / f"gate_retest{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--dirs", nargs="+", default=None)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    main(a.model, a.dirs, a.layer, a.tag)
