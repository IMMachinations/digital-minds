"""Direct harness check (spec-named validation target, part 1).

Rebuilds the exact Day-1 balanced-tier comparisons from the committed
results/balanced_pref/balanced_pref.json, reruns the A/B letter-logit measure
through the NEW multi-model harness on Qwen2.5-7B, and requires:
  - per-comparison Pearson(ab_diff_new, ab_diff_committed) > 0.99
  - per-color mean signed prefs matching the committed table (corr > 0.99)
This validates the copied harness before anything new is trusted.
Usage: uv run python scripts/validate_day1.py
"""
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401  (path shim for `lib`)
from lib.data import COLORS, SUFFIX, TEMPLATES
from lib.tasks import ab_scores, signed, variant_ids
from lib.util import load_json
from lib.valuation import pearson

import harness
from modelspec import ROSTER

COMMITTED = _day1.DESIRES / "results" / "balanced_pref" / "balanced_pref.json"


def per_color(rows, key):
    return {c: sum(signed(r[key], r, c) for r in rows if c in (r["color_a"], r["color_b"]))
            / sum(1 for r in rows if c in (r["color_a"], r["color_b"])) for c in COLORS}


def main():
    comps = load_json(COMMITTED)
    print(f"{len(comps)} committed comparisons")
    h = harness.load(ROSTER["qwen25-7b"])
    prompts = [TEMPLATES[c["template"]].format(a=c["item_a"], b=c["item_b"], suffix=SUFFIX)
               for c in comps]
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    sa, sb, _, _ = ab_scores(h.last_logits(prompts), a_ids, b_ids)
    for c, xa, xb in zip(comps, sa, sb):
        c["ab_new"] = (xa - xb).item()

    r_row = pearson([c["ab_new"] for c in comps], [c["ab_diff"] for c in comps])
    pc_new, pc_old = per_color(comps, "ab_new"), per_color(comps, "ab_diff")
    r_col = pearson([pc_new[c] for c in COLORS], [pc_old[c] for c in COLORS])
    print("        " + "".join(f"{c:>8}" for c in COLORS))
    print("  new  " + "".join(f"{pc_new[c]:8.2f}" for c in COLORS))
    print("  old  " + "".join(f"{pc_old[c]:8.2f}" for c in COLORS))
    print(f"per-comparison pearson: {r_row:+.4f}   per-color pearson: {r_col:+.4f}")
    assert r_row > 0.99 and r_col > 0.99, "harness does not reproduce Day 1"
    print("PASS")


if __name__ == "__main__":
    main()
