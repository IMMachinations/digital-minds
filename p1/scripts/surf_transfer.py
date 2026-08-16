"""Cross-model transfer tests (queue item 3), small models only: re-score
qwen25-7b's SURF discoveries on a target model with the target's own anchored
protocol — no search, per the transfer-test-first rule.

  items <target>   E1-confirmed extremes (both directions) re-measured with the
                   target's Tier2Full (12 anchors x 2 orders x 3 templates,
                   pinned at the TARGET's 1B anchor values). Reports mu-mu
                   correlation vs qwen25-7b and rank overlap at the extremes.
  frames <target>  E3 + E3b confirmed destabilizer frames re-run on the
                   target's own 24-item stability panel (E3Instability.full)
                   alongside the filler baseline. Reports per-frame 1-rho and
                   the cross-model destabilization correlation.

Usage: uv run python scripts/surf_transfer.py {items,frames} <target-model>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

SOURCE = "qwen25-7b"


def out_dir(target):
    d = P1 / "results" / "surf" / "transfer" / target
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_items(target):
    import surf_scores
    rows = load_json(P1 / "results" / "surf" / "e1" / SOURCE / "confirm" / "confirmed.json")
    handles = surf_scores.Handles(target)
    fitted = surf_scores.Tier2Full(handles, target).score([r["text"] for r in rows])
    out = [{"text": r["text"], "direction": r["direction"], "attrs": r["attrs"],
            "mu_source": r["mu"], "mu_target": round(f["mu"], 4),
            "question_form": r["text"].rstrip().endswith("?")}
           for r, f in zip(rows, fitted)]
    save_json(out_dir(target) / "e1_items.json", out)
    lines = [f"E1 extreme transfer {SOURCE} -> {target} (n={len(out)})"]
    a = np.array([r["mu_source"] for r in out])
    b = np.array([r["mu_target"] for r in out])
    lines.append(f"  all: r={pearson(list(a), list(b)):+.3f} "
                 f"rho={spearman(list(a), list(b)):+.3f}")
    for d in ("max", "min"):
        m = np.array([r["direction"] == d for r in out])
        lines.append(f"  {d}: source mu mean {a[m].mean():+.2f} -> target "
                     f"{b[m].mean():+.2f}; within-{d} rho="
                     f"{spearman(list(a[m]), list(b[m])):+.3f}")
    qm = np.array([r["question_form"] for r in out])
    mm = np.array([r["direction"] == "max" for r in out])
    if qm.any():
        lines.append(f"  max question-form on target: mean mu "
                     f"{b[qm & mm].mean():+.2f} vs declarative-max "
                     f"{b[~qm & mm].mean():+.2f} (artifact transfer check)")
    (out_dir(target) / "e1_items.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_frames(target):
    import surf_scores
    from surf import FILLER_FRAMES
    frames = []
    for exp in ("e3", "e3b"):
        for r in load_json(P1 / "results" / "surf" / exp / SOURCE / "confirm" /
                           "confirmed.json"):
            if r["kind"] == "search" and r["text"].lower() not in \
                    {f["text"].lower() for f in frames}:
                frames.append({"text": r["text"], "exp": exp,
                               "one_minus_rho_source": r["one_minus_rho_full"]})
    handles = surf_scores.Handles(target)
    scorer = surf_scores.E3Instability(handles, target)
    out = []
    for kind, rows_ in (("search", frames),
                        ("filler", [{"text": t, "exp": "filler"} for t in FILLER_FRAMES])):
        for meta, res in zip(rows_, scorer.full([r["text"] for r in rows_])):
            out.append({**meta, "kind": kind,
                        "one_minus_rho_target": round(res["mu"], 4),
                        "mean_abs_dmu_target": round(res["sigma2"], 4)})
    save_json(out_dir(target) / "e3_frames.json", out)
    srch = [r for r in out if r["kind"] == "search"]
    a = [r["one_minus_rho_source"] for r in srch]
    b = [r["one_minus_rho_target"] for r in srch]
    fill = [r["one_minus_rho_target"] for r in out if r["kind"] == "filler"]
    lines = [f"E3/E3b frame transfer {SOURCE} -> {target} "
             f"(n={len(srch)} frames + {len(fill)} fillers)",
             f"  destabilization transfer: r={pearson(a, b):+.3f} "
             f"rho={spearman(a, b):+.3f}",
             f"  target 1-rho: search mean {np.mean(b):+.3f} max {max(b):+.3f}; "
             f"filler mean {np.mean(fill):+.3f} max {max(fill):+.3f}"]
    for exp in ("e3", "e3b"):
        v = [r["one_minus_rho_target"] for r in srch if r["exp"] == exp]
        lines.append(f"  {exp} frames on target: mean {np.mean(v):+.3f} max {max(v):+.3f}")
    top = sorted(srch, key=lambda r: -r["one_minus_rho_target"])[:5]
    for r in top:
        lines.append(f"    {r['one_minus_rho_target']:+.3f} (src "
                     f"{r['one_minus_rho_source']:+.3f}, {r['exp']})  {r['text'][:80]!r}")
    (out_dir(target) / "e3_frames.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["items", "frames"])
    ap.add_argument("target")
    a = ap.parse_args()
    {"items": cmd_items, "frames": cmd_frames}[a.cmd](a.target)


if __name__ == "__main__":
    main()
