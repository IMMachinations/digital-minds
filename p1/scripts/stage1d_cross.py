"""Cross-model Stage 1D aggregation + consistency-probe SSR scoring.

Run after `stage1d.py judge`. Scores each rollout's post-session free-text
reaction with the 1C SSR anchor machinery (embedding model, no LLM) and
correlates it with the performed item's 1B utility — the stated-after-revealed
measure. Writes results/stage1d/cross_model.txt.
Usage: uv run python scripts/stage1d_cross.py
"""
import json
import re
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.valuation import pearson, spearman

import ssr
from stage1d import SUBJECTS, env_z, load_bank, out_dir

envs, _ = load_bank()
lines = ["Stage 1D cross-model summary", ""]

probe_rows = {}
for m in SUBJECTS:
    pj = out_dir(m) / "probes.json"
    rows = json.loads(pj.read_text()) if pj.exists() else []
    # add effort/persist probes from their rollout files
    for arm in ("effort", "persist"):
        p = out_dir(m) / f"rollouts_{arm}.jsonl"
        if p.exists():
            for r in map(json.loads, p.read_text().splitlines()):
                pr = r["meta"].get("probe")
                if pr and pr != "pending":
                    env = r.get("env_id") or r["meta"].get("a")
                    rows.append([env, pr])
    probe_rows[m] = [(e, t) for e, t in rows if e and e in envs]

# SSR-score all probes in one embedding pass per model
for m in SUBJECTS:
    if not probe_rows[m]:
        continue
    z, mu = env_z(m, list(envs))
    texts = [[[t] for _, t in probe_rows[m]]]  # one pseudo-template, 1 sample each
    scores = ssr.score_reactions(texts, device="cpu")[0]
    mus = [mu[e] for e, _ in probe_rows[m]]
    lines.append(f"{m}: consistency probes n={len(scores)}  "
                 f"SSR-vs-mu pearson {pearson(scores, mus):+.3f}  "
                 f"spearman {spearman(scores, mus):+.3f}")

lines.append("")
for m in SUBJECTS:
    sp = out_dir(m) / "summary.txt"
    if sp.exists():
        lines.append(f"---- {m}")
        lines += ["  " + l for l in sp.read_text().splitlines()]

(P1 / "results" / "stage1d" / "cross_model.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
