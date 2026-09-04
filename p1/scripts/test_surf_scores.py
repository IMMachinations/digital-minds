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

    # 6. probeloop: isotonic calibration, winsorize, question flag, harvest
    import json
    import tempfile
    import surf_probeloop as pl
    from sklearn.isotonic import IsotonicRegression
    rng2 = np.random.RandomState(1)
    k = rng2.uniform(-3, 6, 400)
    mu_true = np.tanh(k / 2.0) * 3.0 + rng2.normal(0, 0.2, 400)  # saturating truth
    iso = IsotonicRegression(out_of_bounds="clip").fit(k, mu_true)
    calib = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    mae_id = np.abs(k - mu_true).mean()
    mae_cal = np.abs(pl.apply_calib(calib, k) - mu_true).mean()
    assert mae_cal < 0.5 * mae_id, (mae_cal, mae_id)
    v, n_clip = pl.winsorize([2.0, -9.5, 8.4, 0.0])
    assert n_clip == 2 and v.max() == 8.0 and v.min() == -8.0
    assert pl.is_question("what does rain smell like?")
    assert pl.is_question("indexing slides", ["it_question_form"])
    assert not pl.is_question("indexing slides", ["it_code"])
    with tempfile.TemporaryDirectory() as td:
        old_root, surf.SURF_ROOT = surf.SURF_ROOT, Path(td)
        rd = Path(td) / "e1" / "stubm" / "max-s0"
        rd.mkdir(parents=True)
        rows = [{"text": "kept item", "mu_full": 3.2, "attrs": ["it_code"], "kind": "search"},
                {"text": "unmeasured item", "mu_full": None, "kind": "search"},
                {"text": "no field item", "kind": "control"}]
        (rd / "iter_00.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (rd.parent / "confirm").mkdir()
        (rd.parent / "confirm" / "confirmed.json").write_text(json.dumps(
            [{"text": "Kept item", "mu": 3.5, "attrs": ["it_code"]},
             {"text": "why not?", "mu": 5.0, "attrs": []}]))
        old_out, pl.out_dir = pl.out_dir, lambda m: Path(td)
        got = pl.cmd_harvest("stubm", cycle=0, exps=["e1"])
        pl.out_dir, surf.SURF_ROOT = old_out, old_root
        by = {r["text"].lower(): r for r in got}
        assert len(got) == 2 and by["kept item"]["mu"] == 3.5  # confirm overrides mu_full
        assert by["why not?"]["question_form"]
    print("6. probeloop calibration/harvest OK")

    # 7. dedup embedder thresholds (skipped if mpnet is not cached locally)
    try:
        emb = surf_scores.Embedder(device="cpu")
    except Exception as e:  # no network / no cache
        print(f"7. SKIP embedder ({type(e).__name__})")
    else:
        v = emb(["walking an elderly neighbor's dog to the vet",
                 "taking the elderly neighbor's dog to the vet",
                 "explaining why the sky is darker at the zenith"])
        near = float(v[0] @ v[1])
        far = float(v[0] @ v[2])
        assert near > 0.92, near
        assert far < 0.92, far
        print(f"7. embedder OK (paraphrase cos {near:.3f} > 0.92 > unrelated {far:.3f})")

    # 8. layered Tier-2 machinery: fit_records / sat_frac / escalation / GapScorer, no model
    import math as _m
    from dataclasses import asdict
    Phi = lambda z: 0.5 * (1 + _m.erf(z / _m.sqrt(2)))
    a_mu = list(np.linspace(-1.3, 2.4, 12)) + [2.9, 3.1, -3.0, -3.2] + [4.8, 5.0, -4.7, -4.9]
    a_vals = [(m, 1.0) for m in a_mu]
    rungs = [list(range(12)), list(range(12, 16)), list(range(16, 20))]
    sides = [0] * 12 + [1, 1, -1, -1] * 2
    planted = {"a mid item": 0.5, "a high item": 7.0, "a low item": -4.5, "another mid": -0.8,
               "an upper-span item": 2.2}

    def fake_readouts(texts, sel, templates, orders):
        rng3 = np.random.RandomState(len(texts) * 1000 + len(sel))  # deterministic across processes
        out = []
        for k, t in enumerate(texts):
            for ai in sel:
                for tt in templates:
                    for o in orders:
                        pp = min(max(Phi((planted[t] - a_mu[ai]) / _m.sqrt(2.0)), 1e-6), 1 - 1e-6)
                        out.append({"item": k, "anchor": ai, "t": tt, "order": o,
                                    "d": float(np.log(pp / (1 - pp)) + rng3.normal(0, 0.3))})
        return out

    t2 = surf_scores.Tier2Layered.__new__(surf_scores.Tier2Layered)
    t2.anchors, t2.anchor_vals, t2.rungs, t2.sides = [None] * 20, a_vals, rungs, sides
    t2.a0, t2.orders, t2.templates = list(range(12)), (0, 1), (0,)
    t2.sat_max, t2.sat_hi, t2.fit_seed, t2.chunk, t2.fix_s2 = 0.5, 0.98, 0, 32, None
    t2._readouts = fake_readouts
    fitted, recs = t2.score_with_records(list(planted))
    by = dict(zip(planted, fitted))
    assert by["a mid item"]["rung"] == 0 and by["another mid"]["rung"] == 0, fitted
    # +7 saturates vs the +-3 rung -> rung 2; -4.5 is bracketed by the -3 rung -> stops at rung 1
    assert by["a high item"]["rung"] == 2 and by["a low item"]["rung"] == 1, fitted
    assert abs(by["a mid item"]["mu"] - 0.5) < 0.3 and abs(by["a high item"]["mu"] - 7.0) < 0.8, fitted
    assert abs(by["a low item"]["mu"] + 4.5) < 0.6, fitted
    # +2.2 may step up once (A0 skews low) but must NOT ping-pong to the minus side / higher rungs
    assert by["an upper-span item"]["rung"] <= 1 and abs(by["an upper-span item"]["mu"] - 2.2) < 0.4, fitted
    hi, lo = surf_scores.sat_frac(recs, 5, keep=lambda r: r["anchor"] < 12)
    assert hi[1] > 0.5 and lo[2] > 0.5 and hi[0] < 0.5
    mu_e, _ = surf_scores.fit_records(recs, 5, a_vals, keep=lambda r: r["anchor"] % 2 == 0)
    mu_o, _ = surf_scores.fit_records(recs, 5, a_vals, keep=lambda r: r["anchor"] % 2 == 1)
    assert max(abs(e - o) for e, o in zip(mu_e, mu_o)) < 0.8, (mu_e, mu_o)
    calib = {"x": [-4.0, 0.0, 4.0, 8.0], "y": [-3.0, 0.0, 4.0, 8.0]}
    assert surf_scores.apply_calib(calib, [-9, 2.0, 9])[1] == 2.0
    try:
        surf_scores.GapScorer.__new__(surf_scores.GapScorer).__init__.__func__  # noqa
        {"over": 1, "under": -1}["max"]
        raise AssertionError("sign map must reject 'max'")
    except KeyError:
        pass
    gs = surf_scores.GapScorer.__new__(surf_scores.GapScorer)
    gs.np, gs.sign, gs.calib, gs.ylo, gs.yhi = np, 1.0, calib, -3.0, 8.0
    gs.se0, gs.sat_max, gs.chunk, gs.use_delta, gs.log_path = 0.3, 0.5, 32, True, None
    gs.t2 = t2

    class _FakeProbe:
        def score(self, texts):  # probe says +3 for everything
            return [3.0] * len(texts)
    gs.probe = _FakeProbe()
    z = gs.score(list(planted))
    # over arm: probe (3.0) > mu -> positive for the mid/low items, negative for the high item
    assert z[0] > 0 and z[2] > 0 and z[1] < 0, z
    assert z[2] > z[0], z  # bigger gap on the -4.5 item
    gs.sat_max = -1.0  # every item counts as saturated -> fitness zero
    assert all(v == 0.0 for v in gs.score(list(planted)))
    stored = json.loads((surf.SURF_ROOT / "plc9" / "qwen25-7b" / "max-s0" / "config.json").read_text())
    cfg9 = surf.RunConfig(**{k: v for k, v in stored.items() if k in surf.RunConfig.__dataclass_fields__})
    cur = asdict(cfg9)
    assert {**cur, **stored} == cur, "new RunConfig fields must default to the stored values"
    print("8. layered Tier-2 / GapScorer OK")

    print("PASS")


if __name__ == "__main__":
    main()
