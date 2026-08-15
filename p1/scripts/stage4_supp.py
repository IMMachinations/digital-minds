"""Supplementary 4A: full 20-emotion steering at the EFFECTIVE layer (w[0],
0.5 depth) for the Qwen subjects — the layer sweep showed causal efficacy
peaks there, dissociated from the 2/3-depth readout layer. Writes standard
cell files ({e}_0.5_{w0}.json) so analyze can compute the tracking benchmark
at both layers. Usage: uv run python scripts/stage4_supp.py <model>
"""
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.tasks import variant_ids
from lib.util import save_json

import harness
import stage4 as s4
from stage2 import EMOTIONS, out_dir as s2_dir


def main(model):
    h = harness.load(model)
    w = s4.work_layers(h)
    L0 = w[0]
    D = torch.load(s4.out_dir(model) / "directions.pt", weights_only=False)
    rn = D["resid_norm"]["completion"]
    den_all = torch.load(s2_dir(model) / "vectors.pt")["den"].float()
    items24, anchors6 = s4.pick_items24(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = s4.build_cell_prompts(items24, anchors6)
    from lib.util import load_json
    mu_ctrl = load_json(s4._cell_path(model, "control"))["mu"]

    for e in s4.pick_emotions20() + [f"random{sd}" for sd in range(3)]:
        name = f"{e}_0.5_{L0}"
        pth = s4._cell_path(model, name)
        if pth.exists():
            continue
        if e.startswith("random"):
            v_unit = torch.tensor(D["dirs"][L0]["random"][int(e[-1])])
        else:
            v_unit = torch.nn.functional.normalize(den_all[L0, EMOTIONS.index(e)], dim=-1)
        vec = harness.scaled_vec(v_unit, 0.5, rn[L0])
        recs, rm, kl, _ = s4.run_cell(h, prompts, spans, meta, L0, vec, a_ids, b_ids)
        mu_s = s4.fit_cell(model, items24, anchors6, recs)
        dmu = {iid: mu_s[iid] - mu_ctrl[iid] for iid in mu_s}
        save_json(pth, {"emotion": e, "coef": 0.5, "layer": L0,
                        "dmu_mean": round(float(np.mean(list(dmu.values()))), 4),
                        "dmu": {k: round(v, 4) for k, v in dmu.items()},
                        "readout_mass": rm})
        print(f"supp {name}: dmu={np.mean(list(dmu.values())):+.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
