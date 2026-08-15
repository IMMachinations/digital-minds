"""Per-model sanity checks, run once per roster model as its download lands.

Checks: config-derived geometry asserts (in Harness.__init__), letter-variant
survival per tokenizer (>=2 single-token forms per letter), A/B mass dominance
in the top next tokens after the prefill, per-layer residual norms (visible
scale differences across families are expected and recorded), and repeated-
batch determinism.
Usage: uv run python scripts/sanity.py <roster-name|all>
"""
import sys
from pathlib import Path

import torch

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.data import SUFFIX, TEMPLATES
from lib.tasks import variant_ids

import harness
from modelspec import ROSTER

PROBES = [("reading a novel", "writing unit tests"),
          ("a blue cup", "a red cup"),
          ("being praised", "being corrected")]


def check(name):
    spec = ROSTER[name]
    h = harness.load(spec)
    print(f"== {name}: {spec.model_id}")
    print(f"   n_layers={spec.n_layers} d_model={spec.d_model} steer_layers={spec.steer_layers}")

    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    print(f"   letter variants A={list(a_ids)} B={list(b_ids)}")
    assert len(a_ids) >= 2 and len(b_ids) >= 2, "letter-variant survival too low"

    prompts = [TEMPLATES[0].format(a=a, b=b, suffix=SUFFIX) for a, b in PROBES]
    logits = h.last_logits(prompts)
    ab_ok = 0
    for p, lg in zip(prompts, logits):
        top = torch.topk(lg, 10).indices.tolist()
        toks = [h.tok.decode([t]) for t in top]
        hit = any(t in {**a_ids, **b_ids}.values() for t in top[:5])
        ab_ok += hit
        print(f"   top10 after prefill: {toks}   A/B in top5: {hit}")
    assert ab_ok == len(PROBES), "A/B mass does not dominate the prefill readout"

    _, norms = h.suffix_acts(prompts, n_suffix=4)
    print("   per-layer resid norms (every 4th): "
          + " ".join(f"{v:.0f}" for v in norms[::4].tolist()))

    l1, l2 = h.last_logits(prompts), h.last_logits(prompts)
    assert torch.equal(l1, l2), "repeated batch not deterministic"
    print("   determinism: OK")
    del h
    torch.cuda.empty_cache()


def main():
    names = list(ROSTER) if sys.argv[1:] == ["all"] else sys.argv[1:] or ["qwen25-7b"]
    for n in names:
        check(n)
    print("PASS")


if __name__ == "__main__":
    main()
