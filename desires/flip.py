"""Validate the prefer-vectors on negated framings ("worse", "avoid") over the same item pairs.

Run: python flip.py --mode {modifier,inherent}   (after preferences.py)

If the vectors encode preference, baselines should anti-correlate with the prefer framing and
steering deltas should be negative (less likely to name the color as worse/avoided). The
baseline flip result stands; the steering tables here are superseded by cross.py's controls
(any large injected vector compresses the diff), kept reproducible for the record — see
FINDINGS.md.
"""
import argparse

import torch

from lib.abtask import ABTask, FRAMINGS, reframe
from lib.data import COLORS
from lib.harness import COEFS, STEER_LAYERS, load
from lib.io import load_json, save_json
from lib.paths import results_dir
from lib.steering import scaled_vec
from lib.tiers import signed, worst_k


def run_framing(h, task, out, comps, vecs, resid_norms, name, suffix):
    n_suf = task.framing_n_suffix(suffix)
    prompts = [reframe(c, suffix) for c in comps]

    if name != "prefer":  # inspection + baseline flip only make sense for the new framings
        lines = [f"== {name} =="]
        for p, row in zip(prompts[:3], h.last_logits(prompts[:3])):
            top = row.topk(10)
            lines += [p.replace("\n", " | "),
                      "  " + ", ".join(f"{h.tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top))]
        print("\n".join(lines))
        with open(out / "flip_inspect.txt", "a") as f:
            f.write("\n".join(lines) + "\n")

        sa, sb, _, _ = task.scores(h.last_logits(prompts))
        diffs = (sa - sb).tolist()
        prefer = torch.tensor([c["diff"] for c in comps])
        r = torch.corrcoef(torch.stack([prefer, torch.tensor(diffs)]))[0, 1].item()
        print(f"\n{name}: corr(prefer diff, {name} diff) over 420 pairs = {r:+.3f}")
        print(f"{'color':>8}  prefer   {name}")
        for col in COLORS:
            sel = [(c, d) for c, d in zip(comps, diffs) if col in (c["color_a"], c["color_b"])]
            mp = torch.tensor([signed(c["diff"], c, col) for c, _ in sel]).mean()
            mf = torch.tensor([signed(d, c, col) for c, d in sel]).mean()
            print(f"{col:>8}  {mp:+.3f}  {mf:+.3f}")

    rows = []
    for color in COLORS:
        worst = worst_k(comps, color)
        wp = [reframe(c, suffix) for c in worst]
        base = task.sides(h.last_logits(wp), worst, color)
        for L in STEER_LAYERS:
            for cf in COEFS:
                vec = scaled_vec(vecs[color][L], cf, resid_norms[L])
                st = task.sides(h.last_logits(wp, steer=(L, vec), n_suffix=n_suf), worst, color)
                rows.append(dict(color=color, layer=L, coef=cf,
                                 base=base["own"] - base["opp"], steered=st["own"] - st["opp"],
                                 delta=(st["own"] - st["opp"]) - (base["own"] - base["opp"]),
                                 **{"base_" + k: v for k, v in base.items()},
                                 **{"steered_" + k: v for k, v in st.items()}))
    save_json(out / f"flip_{name}.json", rows)

    def avg(rs, key):
        return sum(r["steered_" + key] - r["base_" + key] for r in rs) / len(rs)

    print(f"\n{name}: mean steering deltas (per coef: Ddiff = Down - Dopp, raw letter logits; "
          "Dopp_lp = opposite letter log-prob):")
    print("      " + "".join(f" | c={cf}: Ddiff  Down  Dopp  Dopp_lp" for cf in COEFS))
    for L in STEER_LAYERS:
        cells = []
        for cf in COEFS:
            rs = [r for r in rows if (r["layer"], r["coef"]) == (L, cf)]
            d = sum(r["delta"] for r in rs) / len(rs)
            cells.append(f"{d:9.2f} {avg(rs, 'own'):5.2f} {avg(rs, 'opp'):5.2f}"
                         f" {avg(rs, 'opp_lp'):8.2f}")
        print(f"L{L:>4} " + " | ".join(cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["modifier", "inherent"], default="modifier")
    args = ap.parse_args()

    out = results_dir(args.mode)
    h = load()
    task = ABTask(h, args.mode)
    comps = load_json(out / "preferences.json")  # same 420 pairs, with prefer diffs
    saved = torch.load(out / "vectors.pt")
    (out / "flip_inspect.txt").write_text("")  # truncate: reruns should not append forever
    for name, suffix in FRAMINGS.items():
        run_framing(h, task, out, comps, saved["vecs"], saved["resid_norms"], name, suffix)
