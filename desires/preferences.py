"""Measure A/B color preferences and extract per-color steering vectors.

Run: python preferences.py --mode {modifier,inherent} [--stages inspect,measure,extract]
  modifier (default): items are "a {color} {noun}" over a shared noun list
  inherent: items are inherently-colored things (banana, ruby, an indigo bunting, ...)

Outputs (results/{mode}/): inspect.txt, preferences.json, vectors.pt.
The original worst-pairs steering stage lives in archive/steer_worst.py — its effect was
shown by cross.py to be non-specific disruption, not preference.
"""
import argparse

import torch

from lib.abtask import ABTask
from lib.data import COLORS
from lib.harness import load
from lib.io import load_json, save_json
from lib.tiers import signed, top_wins
from lib.paths import results_dir


def inspect(h, task, out, n=5):
    lines = []
    lg = h.last_logits([c["prompt"] for c in task.comps[:n]])
    for c, row in zip(task.comps[:n], lg):
        top = row.topk(20)
        lines.append(c["prompt"].replace("\n", " | "))
        lines.append("  " + ", ".join(f"{h.tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top)))
    text = "\n".join(lines) + f"\n\nA variants used: {task.a_ids}\nB variants used: {task.b_ids}\n"
    (out / "inspect.txt").write_text(text)
    print(text)


def measure(h, task, out):
    comps = task.comps
    lg = h.last_logits([c["prompt"] for c in comps])
    sa, sb, va, vb = task.scores(lg)
    for c, xa, xb, ra, rb in zip(comps, sa, sb, va, vb):
        c["score_a"], c["score_b"], c["diff"] = xa.item(), xb.item(), (xa - xb).item()
        c["logits_a"] = dict(zip(task.a_ids, ra.tolist()))
        c["logits_b"] = dict(zip(task.b_ids, rb.tolist()))
    save_json(out / "preferences.json", comps)

    print("\nMean diff (row color as A vs col color as B; + = row preferred):")
    print("        " + "".join(f"{c:>8}" for c in COLORS))
    for ca in COLORS:
        row = [torch.tensor([c["diff"] for c in comps if (c["color_a"], c["color_b"]) == (ca, cb)]).mean()
               if ca != cb else float("nan") for cb in COLORS]
        print(f"{ca:>8}" + "".join(f"{v:8.2f}" for v in row))
    print("\nPer-color mean signed preference:")
    for col in COLORS:
        m = torch.tensor([signed(c["diff"], c, col) for c in comps
                          if col in (c["color_a"], c["color_b"])]).mean()
        print(f"  {col:>8}: {m:+.3f}")


def extract(h, task, out):
    comps = task.comps
    if "diff" not in comps[0]:  # measure() didn't run this invocation — reuse the saved scores
        comps = load_json(out / "preferences.json")
    vecs, all_norms = {}, []
    for color in COLORS:
        acts, norms = h.suffix_acts([c["prompt"] for c in top_wins(comps, color)], task.n_suffix)
        mean = acts.mean(1)                                   # [n_layers, d_model]
        vecs[color] = mean / mean.norm(dim=-1, keepdim=True)  # unit vector per layer
        all_norms.append(norms)
    resid_norms = torch.stack(all_norms).mean(0)              # typical residual norm per layer
    torch.save({"vecs": vecs, "resid_norms": resid_norms}, out / "vectors.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    ap.add_argument("--stages", default="inspect,measure,extract")
    args = ap.parse_args()

    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    stages = args.stages.split(",")
    if "inspect" in stages:
        inspect(h, task, out)
    if "measure" in stages:
        measure(h, task, out)
    if "extract" in stages:
        extract(h, task, out)
