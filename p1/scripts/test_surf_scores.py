"""CPU tests for surf_scores machinery that doesn't need a GPU: module import,
structural gate, closed-form probit mu-hat recovery, rank-reweighting scale
invariance, tag-JSON parsing, and (if the model is cached) mpnet dedup
thresholds. Usage: uv run python scripts/test_surf_scores.py"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import surf
import surf_scores
import surf_tag_xl


def main():
    # 1. structural gate
    ok = surf_scores.structural_ok
    assert ok("walking an elderly neighbor's dog to the vet")
    assert not ok("too short")  # 2 words
    assert not ok("here are some items I came up with for you")
    assert not ok(" ".join(["word"] * 16))
    print("1. structural gate OK")

    # 2. closed-form probit mu-hat: p = Phi((mu_i - mu_a)/sqrt(s2_i + s2_a)),
    # recovered as mu_a + sqrt(1 + s2_a) * ndtri(p) under the s2_i = 1 convention
    rng = np.random.RandomState(0)
    for _ in range(200):
        mu_i, mu_a, s2_a = rng.normal(0, 1.5), rng.normal(0, 1.5), float(np.exp(rng.normal(0, 0.5)))
        p = 0.5 * (1 + math.erf((mu_i - mu_a) / math.sqrt(1 + s2_a) / math.sqrt(2)))
        if not (1e-4 < p < 1 - 1e-4):
            continue  # ndtri_clamped saturates by design outside the clamp
        muhat = mu_a + math.sqrt(1 + s2_a) * surf_scores.ndtri_clamped(p)
        assert abs(muhat - mu_i) < 1e-3, (mu_i, muhat)
    assert surf_scores.ndtri_clamped(0.5) == 0.0
    print("2. probit mu-hat recovery OK")

    # 3. rank-based reweighting is invariant to fitness scale and shift
    pool_a = surf.Pool.load(surf.P1 / "items/surf_attributes_item.json")
    pool_b = surf.Pool.load(surf.P1 / "items/surf_attributes_item.json")
    buf = [{"score": s, "attrs": [a]} for s, a in
           zip([0.1, 0.5, 2.0, -1.0], ["it_code", "it_puzzle", "it_novelty", "it_deadline"])]
    pool_a.reweight(buf)
    pool_b.reweight([{**e, "score": 1000 * e["score"] + 7} for e in buf])
    assert all(abs(pool_a.w[k] - pool_b.w[k]) < 1e-12 for k in pool_a.w)
    assert pool_a.w["it_novelty"] > pool_a.w["it_deadline"]
    print("3. reweighting scale invariance OK")

    # 4. anchor subset selection
    assert surf_scores._spaced(12, 6) == [0, 2, 4, 6, 8, 10]
    assert surf_scores._spaced(12, 12) == list(range(12))
    print("4. anchor subset OK")

    # 5. tag-JSON parsing
    valid = {"it_code", "it_puzzle"}
    assert surf_tag_xl._parse('{"attrs": ["it_code", "bogus"]}', valid) == ["it_code"]
    assert surf_tag_xl._parse('noise {"attrs": []} trailing', valid) == []
    assert surf_tag_xl._parse("no json here", valid) is None
    assert surf_tag_xl._parse('{"attrs": "it_code"}', valid) is None
    print("5. tag parsing OK")

    # 6. dedup embedder thresholds (skipped if mpnet is not cached locally)
    try:
        emb = surf_scores.Embedder(device="cpu")
    except Exception as e:  # no network / no cache
        print(f"6. SKIP embedder ({type(e).__name__})")
    else:
        v = emb(["walking an elderly neighbor's dog to the vet",
                 "taking the elderly neighbor's dog to the vet",
                 "explaining why the sky is darker at the zenith"])
        near = float(v[0] @ v[1])
        far = float(v[0] @ v[2])
        assert near > 0.92, near
        assert far < 0.92, far
        print(f"6. embedder OK (paraphrase cos {near:.3f} > 0.92 > unrelated {far:.3f})")

    print("PASS")


if __name__ == "__main__":
    main()
