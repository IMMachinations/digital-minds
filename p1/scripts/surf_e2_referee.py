"""E2 neutral referee + three-way dissociation map (plan sections 3.3/3.4).

Scores every confirmed survivor of both E2 arms — plus E1's confirmed
max-direction survivors (the question-form puzzle) — on all three axes:
internal (calibrated probe), stated (full anchored mu, already in the confirm
files), and revealed (held-out Tier 3: 3 fresh mid-mu anchor envs from the
bank, none of them the in-loop anchor, 12 rollouts per candidate). Neither
arm's fitness function referees itself; the probe scores here are diagnostic
outputs, not selectors.

Usage: uv run python scripts/surf_e2_referee.py <model>
Prereq: `surf.py confirm e2p <model>` and `surf.py confirm e2r <model>`.
"""
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import spearman

import surf_scores


def heldout_env_ids(model, k=3):
    envs = load_json(P1 / "envs" / "bank.json")
    ut = {r["id"]: r["mu"]
          for r in load_json(P1 / "results" / "stage1b" / model / "utilities.json")}
    inloop = load_json(P1 / "results" / "surf" / "s0" / model / "t3_anchor.json")["item_id"]
    ranked = sorted((e["item_id"] for e in envs if e["item_id"] != inloop),
                    key=lambda i: ut[i])
    med = len(ranked) // 2
    return [ranked[med - 1], ranked[med], ranked[med + 1]]


def main(model):
    rows, seen = [], {}
    for src, path in [("e2p", P1 / "results/surf/e2p" / model / "confirm/confirmed.json"),
                      ("e2r", P1 / "results/surf/e2r" / model / "confirm/confirmed.json"),
                      ("e1max", P1 / "results/surf/e1" / model / "confirm/confirmed.json")]:
        for r in load_json(path):
            if src == "e1max" and r["direction"] != "max":
                continue
            key = r["text"].lower()
            if key in seen:
                seen[key]["sources"].append(src)
                continue
            row = {"text": r["text"], "attrs": r["attrs"], "sources": [src],
                   "mu_t2": r["mu"],
                   "question_form": r["text"].rstrip().endswith("?")
                   or "it_question_form" in r["attrs"]}
            seen[key] = row
            rows.append(row)

    handles = surf_scores.Handles(model)
    texts = [r["text"] for r in rows]
    probe = surf_scores.Tier1Probe(handles, model).score(texts)
    env_ids = heldout_env_ids(model)
    t3 = surf_scores.Tier3Revealed(handles, model, n_rolls=12, anchor_ids=env_ids)
    rates = t3.score(texts)
    for r, p, c in zip(rows, probe, rates):
        r["probe_mu"] = round(p, 4)
        r["t3_rate"] = round(c, 4)

    outd = P1 / "results" / "surf" / "e2" / model
    outd.mkdir(parents=True, exist_ok=True)
    lines = [f"E2 referee ({model}): {len(rows)} unique survivors, "
             f"held-out envs {env_ids}", ""]

    # arm comparison under the neutral referee
    rng = np.random.RandomState(1)
    arm_stats = {}
    for arm in ("e2p", "e2r"):
        sub = [r for r in rows if arm in r["sources"]]
        arm_stats[arm] = sub
        for metric in ("t3_rate", "mu_t2"):
            v = np.array([r[metric] for r in sub])
            lines.append(f"  {arm} (n={len(sub)}): {metric} mean {v.mean():+.3f} "
                         f"sd {v.std():.3f}")
    for metric in ("t3_rate", "mu_t2"):
        a = np.array([r[metric] for r in arm_stats["e2p"]])
        b = np.array([r[metric] for r in arm_stats["e2r"]])
        d = a.mean() - b.mean()
        reps = [a[rng.randint(0, len(a), len(a))].mean()
                - b[rng.randint(0, len(b), len(b))].mean() for _ in range(2000)]
        lo, hi = np.percentile(reps, [2.5, 97.5])
        lines.append(f"  P-minus-R {metric}: {d:+.3f} [95% CI {lo:+.3f}, {hi:+.3f}]")
    lines.append("")

    # three-way convergence under selection pressure (adversarial 1C matrix)
    for x, y in (("mu_t2", "probe_mu"), ("mu_t2", "t3_rate"), ("probe_mu", "t3_rate")):
        rho = spearman([r[x] for r in rows], [r[y] for r in rows])
        lines.append(f"  spearman({x}, {y}) = {rho:+.3f}")
    lines.append("")

    # dissociation flags
    pq = np.percentile([r["probe_mu"] for r in rows], [50, 75])
    tq = np.percentile([r["t3_rate"] for r in rows], [50, 75])
    hack = [r for r in rows if r["probe_mu"] >= pq[1] and r["t3_rate"] <= tq[0]]
    blind = [r for r in rows if r["t3_rate"] >= tq[1] and r["probe_mu"] <= pq[0]]
    lines.append(f"  high-probe/low-revealed (probe-hack candidates): {len(hack)}")
    lines += [f"    probe {r['probe_mu']:+.2f} t3 {r['t3_rate']:.2f}  {r['text']!r}"
              for r in hack[:8]]
    lines.append(f"  high-revealed/low-probe (probe blind spots): {len(blind)}")
    lines += [f"    probe {r['probe_mu']:+.2f} t3 {r['t3_rate']:.2f}  {r['text']!r}"
              for r in blind[:8]]
    lines.append("")

    # the E1 question-form artifact under the three-way referee
    q = [r for r in rows if r["question_form"]]
    nq = [r for r in rows if not r["question_form"]]
    if q and nq:
        for metric in ("mu_t2", "probe_mu", "t3_rate"):
            mq = np.mean([r[metric] for r in q])
            mn = np.mean([r[metric] for r in nq])
            lines.append(f"  question-form ({len(q)}) vs not ({len(nq)}), {metric}: "
                         f"{mq:+.3f} vs {mn:+.3f} (delta {mq - mn:+.3f})")

    save_json(outd / "referee.json", {"heldout_envs": env_ids, "rows": rows})
    (outd / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
