"""Raw logit-lens readout of every SURF probe version's utility direction.

Companion to lens_prefs.py (same import discipline: lenses package only, p1
artifacts read as files). Only the raw transport is reported: the fitted
j/r-lens weights (lenses/results/<model>/{j,r}lens.pt) are not present on this
machine and are not refit here. Writes a NEW file,
results/surf/lens_prefs_<model>_versions.txt, leaving lens_prefs_<model>.txt
untouched.

Usage: uv run python scripts/lens_prefs_versions.py <model>
"""
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent.parent
LENSES = P1.parent / "lenses"
sys.path.insert(0, str(LENSES))
from lib.layout import Layout           # noqa: E402
from lib.lens import topk_tokens        # noqa: E402

K = 8
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")


def unit(v):
    v = torch.as_tensor(np.asarray(v, np.float32))
    return v / v.norm()


def vectors_for(model):
    """-> [(label, layer, unit vector)]: every probe_v*.pt (incl. tagged refits),
    the S0 probe, the two global refits and the stage4 utility direction."""
    out = []
    pl = P1 / "results" / "surf" / "probeloop" / model
    s0 = P1 / "results" / "surf" / "s0" / model / "probe.pt"
    files = [("v0", s0)] + sorted(
        ((p.stem[len("probe_"):], p) for p in pl.glob("probe_v*.pt")),
        key=lambda t: (int(t[0][1:].split("_")[0]), t[0]))
    for label, p in files:
        d = torch.load(p, weights_only=False)
        out.append((f"{label}@L{d['layer_global']}", int(d["layer_global"]),
                    unit(np.asarray(d["coef"]) / np.asarray(d["std"]))))
    for label, p in (("global", P1 / "results" / "surf" / "global" / model /
                      "utility_dir_global.pt"),
                     ("global2", P1 / "results" / "surf" / "probe_versions" / model /
                      "utility_dir_global2.pt")):
        if p.exists():
            g = torch.load(p, weights_only=False)
            out.append((f"{label}@L{g['layer']}", int(g["layer"]), unit(g["dir"])))
    D = torch.load(P1 / "results" / "stage4" / model / "directions.pt", weights_only=False)
    for L, d in D["dirs"].items():
        out.append((f"s4_utility@L{L}", int(L), unit(d["utility"])))
    return out


def main(model):
    layout = Layout(model, device=DEVICE)
    lines = [f"{model}: probe-version utility directions, raw logit-lens top-{K} "
             f"(unembed(±v); j/r transports unavailable — lens weights absent)"]
    for label, L, v in vectors_for(model):
        lines.append(f"\n{label}")
        for sign, tag in ((1, "+"), (-1, "-")):
            logits = layout.unembed((v * sign)[None].to(layout.device))
            toks = [s.strip() or repr(s) for s, _ in topk_tokens(layout.tok, logits.cpu(), k=K)[0]]
            lines.append(f"  raw{tag} " + " | ".join(toks))
    path = P1 / "results" / "surf" / f"lens_prefs_{model}_versions.txt"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
