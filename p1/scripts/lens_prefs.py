"""Decode the preference/utility direction family through the j- and r-lenses.

For each preference vector (stage4 utility/choice/pool directions per working
layer; qwen25-7b additionally the global XL+SURF refit, the hardened probe v3,
and the early-layer dvalence directions at 0.25/0.39 depth), show what the
direction "wants the model to say": top unembedded tokens under
  raw : unembed(v)         (vanilla logit lens)
  j   : unembed(J_l @ v)   (Jacobian transport)
  r   : unembed(R_l @ v)   (LRP transport; faithful early layers)
for +v and -v. Follows lens_check.py's import discipline (lenses package only;
p1 artifacts read as files).

Usage: uv run python scripts/lens_prefs.py <model> [<model> ...]
Writes results/surf/lens_prefs_<model>.txt
"""
import json
import sys
from pathlib import Path

import torch

P1 = Path(__file__).resolve().parent.parent
LENSES = P1.parent / "lenses"
sys.path.insert(0, str(LENSES))
from lib.layout import Layout           # noqa: E402
from lib.lens import Lens, topk_tokens  # noqa: E402

K = 8


def unit(v):
    v = torch.as_tensor(v, dtype=torch.float32)
    return v / v.norm()


def vectors_for(model):
    """-> list of (label, layer, unit vector)."""
    out = []
    D = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                   weights_only=False)
    for L, d in D["dirs"].items():
        for kind in ("utility", "choice", "pool"):
            out.append((f"{kind}@L{L}", int(L), unit(d[kind])))
    if model == "qwen25-7b":
        g = torch.load(P1 / "results" / "surf" / "global" / model /
                       "utility_dir_global.pt", weights_only=False)
        out.append((f"utility_global@L{g['layer']}", int(g["layer"]),
                    unit(g["dir"])))
        v3 = torch.load(P1 / "results" / "surf" / "probeloop" / model /
                        "probe_v3.pt", weights_only=False)
        out.append((f"utility_v3@L{v3['layer_global']}", int(v3["layer_global"]),
                    unit(torch.as_tensor(v3["coef"] / v3["std"]))))
        dv = torch.load(P1 / "results" / "surf" / "dvalence" / model /
                        "dvalence_dirs.pt", weights_only=False)
        for L in dv["early"]:
            out.append((f"utility_early@L{L}", int(L), unit(dv["dirs"][L])))
    return sorted(out, key=lambda t: (t[1], t[0]))


def main(models):
    for model in models:
        layout = Layout(model)
        lenses = {k: Lens.load(p) for k in ("j", "r")
                  if (p := LENSES / "results" / model / f"{k}lens.pt").exists()}
        lines = [f"{model}: preference-direction lens readouts "
                 f"(top-{K} tokens; transports: raw + {sorted(lenses)})"]
        for label, L, v in vectors_for(model):
            lines.append(f"\n{label}")
            for kind in ["raw"] + sorted(lenses):
                for sign, tag in ((1, "+"), (-1, "-")):
                    t = v * sign
                    if kind != "raw":
                        t = lenses[kind].jacobians[L].float() @ t
                    logits = layout.unembed(t[None].to(layout.device))
                    toks = [s.strip() or repr(s)
                            for s, _ in topk_tokens(layout.tok, logits.cpu(), k=K)[0]]
                    lines.append(f"  {kind:3}{tag} " + " | ".join(toks))
        path = P1 / "results" / "surf" / f"lens_prefs_{model}.txt"
        path.write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        del layout
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:])
