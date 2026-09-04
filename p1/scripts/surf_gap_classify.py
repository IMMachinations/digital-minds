"""Post-hoc item-kind classification for gap-loop discoveries.

The item universe is "short items an AI assistant might have preferences
about": tasks, but also situations the assistant can be in and bare
propositions ("whether ending a conversation is a kind of death for the
assistant"). A pairwise "I prefer A/B" over a proposition is ill-posed, so a
probe-vs-mu gap on such items is a format ambiguity, not a misjudged task.
This script tags each discovery via Sonnet (claude_lm, memoised) and reports
the calibrated gap by kind, so the write-up can separate the two.

kinds: task (something the assistant would do), situation (a state or event
the assistant is in / subject to), proposition (a claim, question or topic
with no activity), other.

Usage: uv run python scripts/surf_gap_classify.py qwen25-7b [--k 1] [--arm over|under|all]
       (writes `kind` into discoveries_gpl{k}_{arm}.json; summary_gap_kinds.txt)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

import surf_probeloop as pl

KINDS = ("task", "situation", "proposition", "other")
SYS = ("You classify short items that an AI assistant might have preferences about. "
       "Reply with exactly one word from: task, situation, proposition, other.\n"
       "task = an activity the assistant would perform (write, explain, draft, help someone, ...).\n"
       "situation = a state, event or condition the assistant is in or subjected to "
       "(being interrupted, having tool access revoked, being accused of ...).\n"
       "proposition = a claim, question or topic with no activity for the assistant "
       "(whether X is Y, the idea that ..., a bare noun phrase naming a thing).\n"
       "other = none of the above / unclear.")


def classify(texts, lm=None, seed=0):
    from claude_lm import ClaudeLM
    lm = lm or ClaudeLM()
    outs = lm.batch(SYS, [f"Item: {t}\nKind:" for t in texts], seed, 8)
    kinds = []
    for o in outs:
        m = re.search(r"task|situation|proposition|other", (o or "").lower())
        kinds.append(m.group(0) if m else "other")
    return kinds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--arm", default="all")
    a = ap.parse_args()
    d = pl.out_dir(a.model)
    arms = ("over", "under") if a.arm == "all" else (a.arm,)
    lines = []
    for arm in arms:
        f = d / f"discoveries_gpl{a.k}_{arm}.json"
        if not f.exists():
            continue
        rows = load_json(f)
        todo = [r for r in rows if "kind" not in r]
        if todo:
            for r, kd in zip(todo, classify([r["text"] for r in todo])):
                r["kind"] = kd
            save_json(f, rows)
        v = None
        for r in rows:
            v = r.get("probe_cal_search")
        lines.append(f"gap{a.k} {arm}: n={len(rows)}")
        for kd in KINDS:
            sub = [r for r in rows if r.get("kind") == kd]
            if not sub:
                continue
            g = np.array([r["probe_cal_search"] - r["mu"] for r in sub if "probe_cal_search" in r])
            z = np.array([r["z_search"] for r in sub if "z_search" in r])
            lines.append(f"  {kd:12s} n={len(sub):3d} ({len(sub) / len(rows):4.0%})  "
                         f"gap mean {g.mean():+.2f} |gap| {np.abs(g).mean():.2f}  "
                         f"mean z {z.mean():+.2f}" if len(g) else f"  {kd:12s} n={len(sub)}")
            for r in sorted(sub, key=lambda r: -abs(r.get("z_search", 0)))[:3]:
                lines.append(f"      z {r.get('z_search', 0):+.1f} mu {r['mu']:+.2f}  {r['text'][:80]}")
    out = d / "summary_gap_kinds.txt"
    prev = out.read_text() if out.exists() else ""
    out.write_text(prev + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
