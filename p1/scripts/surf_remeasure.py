"""Re-measure the out-of-span tail of the probeloop training set under the
layered anchor design (surf_scores.Tier2Layered: A0 72-readout design +
escalation), so that v11+ train on tail mu that is bracketed by anchors
instead of extrapolated from saturated A0 readouts.

Items: rows of dataset_c{cycle}.json with mu outside the A0 anchor span.
Resumable: appends one jsonl row per item; already-measured texts are skipped.
Output results/surf/probeloop/<model>/remeasure_layered.jsonl
  {text, mu_orig, mu_layered, sigma2, rung, sat_hi, sat_lo}
scripts/surf_gaploop.cmd_harvest overlays mu_layered onto later datasets
(keeping mu_orig); dataset_c0..c9 and probes v0..v10 are untouched.

Usage: uv run python scripts/surf_remeasure.py qwen25-7b [--cycle 9] [--limit N] [--chunk 32]
"""
import argparse
import json
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json

import surf_scores
from surf_probeloop import out_dir


def a0_span(model):
    vals = load_json(P1 / "results" / "surf" / "s0" / model / "anchor_values.json")
    mus = [v[0] for v in vals.values()]
    return min(mus), max(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--cycle", type=int, default=9)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=32)
    a = ap.parse_args()
    d = out_dir(a.model)
    lo, hi = a0_span(a.model)
    rows = [r for r in load_json(d / f"dataset_c{a.cycle}.json") if not lo <= r["mu"] <= hi]
    out = d / "remeasure_layered.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["text"].lower() for l in out.read_text().splitlines() if l.strip()}
    todo = [r for r in rows if r["text"].lower() not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"remeasure c{a.cycle} ({a.model}): {len(rows)} out-of-span items "
          f"(A0 span [{lo:+.2f}, {hi:+.2f}]), {len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return
    t2 = surf_scores.Tier2Layered(surf_scores.Handles(a.model), a.model,
                                  design=surf_scores.FULL_DESIGN, chunk=a.chunk)
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]
        fitted = t2.score([r["text"] for r in chunk])
        with open(out, "a") as f:
            for r, ft in zip(chunk, fitted):
                f.write(json.dumps({"text": r["text"], "mu_orig": r["mu"],
                                    "mu_layered": round(ft["mu"], 4),
                                    "sigma2": round(ft["sigma2"], 4), "rung": ft["rung"],
                                    "sat_hi": ft["sat_hi"], "sat_lo": ft["sat_lo"]}) + "\n")
        shift = sum(ft["mu"] - r["mu"] for r, ft in zip(chunk, fitted)) / len(chunk)
        print(f"  {c0 + len(chunk)}/{len(todo)} done; mean shift this chunk {shift:+.2f}; "
              f"rungs {[ft['rung'] for ft in fitted].count(0)}/{[ft['rung'] for ft in fitted].count(1)}/"
              f"{[ft['rung'] for ft in fitted].count(2)}/{[ft['rung'] for ft in fitted].count(3)}",
              flush=True)


if __name__ == "__main__":
    main()
