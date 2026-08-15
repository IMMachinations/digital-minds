"""j-lens / r-lens upgrade of the Stage-2 logit-lens validity check.

Reruns the emotion-vector identity check with three readouts at each working
layer (0.50 / 0.64 / 0.75 depth):
  raw : unembed(v)          — vanilla logit lens (final-norm included, per the
                              lenses package convention; note the original
                              Stage-2 check used bare W_U @ v)
  j   : unembed(J_l @ v)    — Jacobian transport into the final-layer basis
  r   : unembed(R_l @ v)    — LRP-propagated transport (faithful early layers)

Uses the user's lenses package (../lenses) for Layout/Lens; deliberately does
NOT import p1 modules (namespace collision between lenses/lib and desires/lib
via the _day1 shim) — p1 artifacts are read as files.

Usage: uv run python scripts/lens_check.py <model>   (from p1/)
Writes results/stage2/<model>/lens_check.{txt,json}.
"""
import json
import sys
from pathlib import Path

import torch

P1 = Path(__file__).resolve().parent.parent
LENSES = P1.parent / "lenses"
sys.path.insert(0, str(LENSES))
from lib.layout import Layout          # noqa: E402
from lib.lens import Lens, topk_tokens  # noqa: E402

EMOTIONS = json.loads((P1 / "items" / "emotions.json").read_text())
SHOWCASE = ["happy", "sad", "afraid", "angry", "proud", "guilty", "surprised",
            "calm", "desperate", "loving", "nervous", "inspired"]
SYN = {"happy": ["happ", "joy", "glad", "delight", "cheer"],
       "sad": ["sad", "grief", "sorrow", "tear", "lonely", "melanch"],
       "afraid": ["afraid", "fear", "scar", "terror", "fright", "dread"],
       "angry": ["angr", "rage", "fur", "wrath", "irate", "mad"],
       "proud": ["proud", "pride", "esteem", "accomplish"],
       "guilty": ["guilt", "blame", "remorse", "shame"],
       "surprised": ["surpris", "unexpect", "astonish", "shock", "sudden"],
       "calm": ["calm", "seren", "peace", "tranquil", "still"],
       "desperate": ["desperat", "urgent", "hopeless", "frantic"],
       "loving": ["lov", "affection", "tender", "care", "warm"],
       "nervous": ["nervous", "anxi", "jitter", "worr", "tense"],
       "inspired": ["inspir", "creativ", "spark", "vision", "idea"]}
FRACS = (0.50, 0.64, 0.75)


def main(model):
    layout = Layout(model)
    lenses = {"j": Lens.load(LENSES / "results" / model / "jlens.pt"),
              "r": Lens.load(LENSES / "results" / model / "rlens.pt")}
    den = torch.load(P1 / "results" / "stage2" / model / "vectors.pt")["den"].float()
    work = [round(f * layout.n_layers) for f in FRACS]
    lines = [f"{model}: lens-upgraded emotion-vector identity check "
             f"(working layers {work}; hits = synonym in top-20)"]
    table = {}
    for L in work:
        for kind in ("raw", "j", "r"):
            hits = 0
            detail = []
            for e in SHOWCASE:
                v = den[L, EMOTIONS.index(e)]
                if kind == "raw":
                    t = v
                else:
                    J = lenses[kind].jacobians[L].float()
                    t = J @ v
                logits = layout.unembed(t[None].to(layout.device))
                toks = [s for s, _ in topk_tokens(layout.tok, logits.cpu(), k=20)[0]]
                hit = any(sy in tk.strip().lower() for tk in toks for sy in SYN[e])
                hits += hit
                detail.append({"emotion": e, "hit": bool(hit), "top6": toks[:6]})
            table[f"{kind}_L{L}"] = {"hits": hits, "detail": detail}
            lines.append(f"  L{L:>2} {kind:3s}: {hits:2d}/12")
    # per-emotion view at the middle working layer
    Lm = work[1]
    lines.append(f"per-emotion at L{Lm} (raw | j | r), top tokens from j:")
    for i, e in enumerate(SHOWCASE):
        marks = "".join("+" if table[f"{k}_L{Lm}"]["detail"][i]["hit"] else "-"
                        for k in ("raw", "j", "r"))
        lines.append(f"  {e:10s} [{marks}] "
                     + " ".join(repr(t) for t in table[f"j_L{Lm}"]["detail"][i]["top6"]))
    out = P1 / "results" / "stage2" / model
    (out / "lens_check.txt").write_text("\n".join(lines) + "\n")
    (out / "lens_check.json").write_text(json.dumps(
        {k: {"hits": v["hits"]} for k, v in table.items()}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
