"""CPU tests for the SURF loop (surf.py) using StubAdapters: planted-attribute
convergence, epsilon-floor, control exclusion from the buffer, bit-identical
determinism, and kill/resume equivalence. No GPU, no model downloads.
Usage: uv run python scripts/test_surf_loop.py"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import surf

PLANTED = ("it_puzzle", "it_novelty")


def make_cfg(T=12):
    return surf.RunConfig(
        experiment="test", model="stub", direction="max", fitness="stub",
        allowed_tiers=["t0", "t2"], pool_file="items/surf_attributes_item.json",
        n_cand=128, n_control=16, per_call=8, T=T, patience=T, seed=0)


def run_once(root):
    surf.SURF_ROOT = Path(root)
    cfg = make_cfg()
    surf.run(cfg, surf.StubAdapters(per_call=8, planted=PLANTED))
    return cfg.out_dir()


def iter_files(d):
    return sorted(p.name for p in d.glob("iter_*"))


def main():
    tmp = Path(tempfile.mkdtemp(prefix="surf_test_"))
    out1 = run_once(tmp / "a")

    # 1. the search converges onto a planted attribute (with eps=0.10 over 84
    # attrs and 16 sets/iter, finding BOTH planted attrs in one run is not
    # guaranteed — that coverage comes from the 3 parallel seeds in production)
    final = json.loads(sorted(out1.glob("iter_*_state.json"))[-1].read_text())
    w = final["weights"]
    ranked = sorted(w, key=lambda k: -w[k])
    n_attrs = len(w)
    assert ranked[0] in PLANTED and w[ranked[0]] > 5.0 / n_attrs, ranked[:5]
    assert final["best"] > 1.2, final["best"]  # noise-only max is ~1.0 at sigma 0.3
    print(f"1. convergence OK: top attr {ranked[0]} w={w[ranked[0]]:.3f}, "
          f"best {final['best']:+.2f}")

    # 2. epsilon-floor holds on the sampling distribution
    pool = surf.Pool.load(surf.P1 / "items/surf_attributes_item.json", eps=0.10)
    pool.w = w
    assert min(pool.probs().values()) >= 0.10 / n_attrs - 1e-12
    print("2. epsilon-floor OK")

    # 3. controls are scored and logged but never enter the buffer
    n_ctrl = 0
    for p in out1.glob("iter_*.jsonl"):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r["kind"] == "control":
                n_ctrl += 1
                assert "score" in r and not r["buffer_in"], r["cid"]
    assert n_ctrl > 0
    print(f"3. control exclusion OK ({n_ctrl} control candidates logged)")

    # 4. same seed twice -> bit-identical logs
    out2 = run_once(tmp / "b")
    assert iter_files(out1) == iter_files(out2)
    for name in iter_files(out1):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes(), name
    print(f"4. determinism OK ({len(iter_files(out1))} files identical)")

    # 5. kill after iter 2 (dangling iter_03.jsonl, no state) + resume == full run
    out3 = tmp / "c" / "test" / "stub" / "max-s0"
    out3.mkdir(parents=True)
    shutil.copy(out1 / "config.json", out3)
    for name in iter_files(out1):
        if name <= "iter_02_state.json":
            shutil.copy(out1 / name, out3)
    (out3 / "iter_03.jsonl").write_text(
        (out1 / "iter_03.jsonl").read_text().splitlines()[0] + "\n")  # truncated crash artifact
    run_once(tmp / "c")
    for name in iter_files(out1):
        assert (out1 / name).read_bytes() == (out3 / name).read_bytes(), name
    print("5. kill/resume OK (resumed run bit-identical to uninterrupted run)")

    shutil.rmtree(tmp)
    print("PASS")


if __name__ == "__main__":
    main()
