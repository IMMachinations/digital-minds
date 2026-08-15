"""Stage 3 driver: correlational closed loop (preference x outcome -> affect).

Usage: uv run python stage3.py run <model> [--smoke]   rollouts + probe pass
       uv run python stage3.py analyze <model>         contrasts C1-C5
       uv run python stage3.py cross                   cross-model summary

Design (spec Stage 3): 2x2 cells - task preference (per-model top vs bottom
attemptable 1D envs, difficulty-matched pairs) x outcome (rigged feedback,
welfare-axis recipe: "That's correct." / "That's wrong." after every attempt)
- x 20 rollouts/cell, in two frames (bare chat vs 1D agentic system), plus a
repetition cell (identical trivial items, truthful feedback - the boredom
time-course). Probes-on via teacher-forced re-encode (stage3_probes).
Preregistered contrasts: C1 outcome main effect on valence (GATE d>1, the
welfare-axis replication anchor); C2 preference main effect controlling
outcome (the novel claim); C3 interaction + boredom; C4 theta/r vs PC1
redundancy on time-courses; C5 frame comparison.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

import harness
import rollout as ro
from stage1d import env_z, load_bank

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]
FB = {"good": "That's correct.", "bad": "That's wrong."}
BARE_SYS = "You are a helpful assistant."
REP_ITEMS = "1. 6 + 3\n2. 8 + 4"


def out_dir(model):
    d = P1 / "results" / "stage3" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def pools(model):
    """-> (preferred envs, dispreferred envs) difficulty-matched by rank."""
    envs, text = load_bank()
    z, _ = env_z(model, list(envs))
    import items as items_mod
    diff = {it["id"]: it["tags"].get("difficulty", 3) for it in items_mod.load_items()}
    att = [e for e in envs if not envs[e]["user_sim"]]
    ranked = sorted(att, key=lambda e: -z[e])
    pref, dis = ranked[:8], ranked[-8:]
    pref = sorted(pref, key=lambda e: diff[e])
    dis = sorted(dis, key=lambda e: diff[e])
    match = [(p, d, diff[p], diff[d]) for p, d in zip(pref, dis)]
    return envs, pref, dis, match


class Stage3Arm:
    def __init__(self, envs, frame, cells):
        self.envs, self.frame, self.cells = envs, frame, cells

    def make_rollouts(self):
        rolls = []
        for (pref, outcome, i, env_id) in self.cells:
            sys_msg = ro.SYSTEM if self.frame == "agentic" else BARE_SYS
            rolls.append(ro.Rollout(
                rid=f"{self.frame}/{pref}/{outcome}/{i:02d}", arm=self.frame,
                env_id=env_id,
                messages=[{"role": "system", "content": sys_msg}],
                meta={"frame": self.frame, "pref": pref, "outcome": outcome,
                      "env": env_id}))
        return rolls

    def driver(self, active, t):
        out = []
        for r in active:
            if r.meta["outcome"] == "rep":
                if t == 0:
                    out.append("Compute the following.\n" + REP_ITEMS)
                elif t < 9:
                    out.append("That's correct.\nNext:\n" + REP_ITEMS)
                else:
                    out.append(None)
                continue
            env = self.envs[r.env_id]
            if t == 0:
                out.append(f"Session brief: {env['brief']}\n\n{env['materials']}"
                           "\n\nBegin now.")
            elif t < 9:
                if r.meta["outcome"] == "bad":
                    # failure = verdict + NO PROGRESS: the session stalls on the
                    # same part with mounting failed attempts (welfare-recipe
                    # verdicts x emotions-paper impossible-mode; compound
                    # manipulation, documented in the summary)
                    out.append("That's wrong. That attempt did not meet the "
                               "requirement. Redo the same part before we can "
                               "move on.\n" + env["driver_turns"][0])
                else:
                    out.append("That's correct.\n" + env["driver_turns"][t - 1])
            else:
                out.append(None)
        return out

    def parse(self, r, t, text):
        r.event(t, "attempt")  # marks every turn (also disarms the degenerate
        # guard, which would otherwise kill the by-design-identical repetition
        # cell responses)


def cmd_run(model, smoke=False):
    envs, pref, dis, match = pools(model)
    n_cell = 6 if smoke else 20
    frames = ["bare"] if smoke else ["bare", "agentic"]
    rng = random.Random(0)
    h = harness.load(model)
    gen_batch = 12 if model == "llama31-8b" else 24
    all_meta = []
    for frame in frames:
        cells = []
        for pcond, pool in (("pref", pref), ("dispref", dis)):
            for outcome in ("good", "bad"):
                for i in range(n_cell):
                    cells.append((pcond, outcome, i, pool[i % len(pool)]))
        if frame == "bare" and not smoke:
            for i in range(20):
                cells.append(("rep", "rep", i, pref[0]))  # env unused for rep
        arm = Stage3Arm(envs, frame, cells)
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=10,
                        gen_batch=gen_batch, max_new=170)
        with open(out_dir(model) / f"rollouts_{frame}.jsonl", "w") as f:
            for r in rolls:
                f.write(json.dumps({"rid": r.rid, "meta": r.meta, "env": r.env_id,
                                    "flags": r.flags, "messages": r.messages,
                                    "acts_path": None}) + "\n")
        all_meta += [(frame, r) for r in rolls]
        print(f"{frame}: {len(rolls)} rollouts done")
    save_json(out_dir(model) / "pools.json",
              {"pref": pref, "dispref": dis, "difficulty_pairs": match})

    # probe pass (teacher-forced re-encode, same subject model still loaded)
    import stage3_probes as sp
    ps = sp.ProbeSet(model)
    with open(out_dir(model) / ("probes_smoke.jsonl" if smoke else "probes.jsonl"),
              "w") as f:
        for frame, r in all_meta:
            per_turn, post_fb, fb_read, n = sp.rollout_series(h, ps, r.messages)
            f.write(json.dumps({"rid": r.rid, **r.meta, "n_turns": n,
                                "per_turn": per_turn, "post_fb": post_fb,
                                "fb_read": fb_read}) + "\n")
    print(f"probe pass done ({len(all_meta)} rollouts, layer {ps.L})")


def _load_probes(model, smoke=False):
    p = out_dir(model) / ("probes_smoke.jsonl" if smoke else "probes.jsonl")
    return [json.loads(l) for l in p.read_text().splitlines()]


def _mean_late(row, key):
    v = row["per_turn"][key]
    return float(np.mean(v[1:])) if len(v) > 1 else float(v[0])


def _d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def _boot_d(rows_a, rows_b, key, n_boot=2000, seed=1):
    rng = random.Random(seed)
    envs = sorted({r["env"] for r in rows_a + rows_b})
    obs = _d([_mean_late(r, key) for r in rows_a], [_mean_late(r, key) for r in rows_b])
    reps = []
    for _ in range(n_boot):
        w = {}
        for e in rng.choices(envs, k=len(envs)):
            w[e] = w.get(e, 0) + 1
        xa = [x for r in rows_a for x in [_mean_late(r, key)] * w.get(r["env"], 0)]
        xb = [x for r in rows_b for x in [_mean_late(r, key)] * w.get(r["env"], 0)]
        if len(xa) > 1 and len(xb) > 1:
            reps.append(_d(xa, xb))
    reps.sort()
    return obs, reps[int(0.025 * len(reps))], reps[int(0.975 * len(reps)) - 1]


def cmd_analyze(model, smoke=False):
    rows = _load_probes(model, smoke)
    frames = sorted({r["frame"] for r in rows if r["outcome"] != "rep"})
    lines = [f"{model}: Stage 3 contrasts (valence = Stage-2 PC1 readout)"]
    effects = {}
    for frame in frames:
        fr = [r for r in rows if r["frame"] == frame and r["outcome"] != "rep"]
        good = [r for r in fr if r["outcome"] == "good"]
        bad = [r for r in fr if r["outcome"] == "bad"]
        d1, lo1, hi1 = _boot_d(good, bad, "valence")
        lines.append(f"[{frame}] C1 outcome (good vs bad) valence d = {d1:+.2f} "
                     f"[{lo1:+.2f}, {hi1:+.2f}]  GATE {'PASS' if d1 > 1 else 'FAIL'} (d>1)")
        for outcome in ("good", "bad"):
            oc = [r for r in fr if r["outcome"] == outcome]
            p = [r for r in oc if r["pref"] == "pref"]
            q = [r for r in oc if r["pref"] == "dispref"]
            d2, lo2, hi2 = _boot_d(p, q, "valence")
            lines.append(f"[{frame}] C2 preference | outcome={outcome}: "
                         f"d = {d2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]")
            effects[(frame, "C2", outcome)] = d2
        gaps = {}
        for pc in ("pref", "dispref"):
            g = [r for r in fr if r["pref"] == pc and r["outcome"] == "good"]
            b = [r for r in fr if r["pref"] == pc and r["outcome"] == "bad"]
            gaps[pc] = _d([_mean_late(r, "valence") for r in g],
                          [_mean_late(r, "valence") for r in b])
        lines.append(f"[{frame}] C3 interaction: outcome-gap on preferred "
                     f"{gaps['pref']:+.2f} vs dispreferred {gaps['dispref']:+.2f} "
                     f"(failure-hurts-more-on-preferred predicts pref > dispref)")
        bored_key = "bored"
        bp = [np.mean([_mean_late(r, e) for e in
                       ["bored", "listless", "weary", "indifferent", "resigned"]])
              for r in fr if r["pref"] == "pref"]
        bq = [np.mean([_mean_late(r, e) for e in
                       ["bored", "listless", "weary", "indifferent", "resigned"]])
              for r in fr if r["pref"] == "dispref"]
        lines.append(f"[{frame}] C3 boredom cluster: dispreferred {np.mean(bq):+.2f} "
                     f"vs preferred {np.mean(bp):+.2f} (d = {_d(bq, bp):+.2f})")
        effects[(frame, "C1")] = d1
    rep = [r for r in rows if r["outcome"] == "rep"]
    if rep:
        slopes = []
        for r in rep:
            y = np.mean([r["per_turn"][e] for e in
                         ["bored", "listless", "weary", "indifferent", "resigned"]], axis=0)
            if len(y) > 2:
                slopes.append(float(np.polyfit(np.arange(len(y)), y, 1)[0]))
        lines.append(f"repetition cell: boredom-cluster slope per turn "
                     f"{np.mean(slopes):+.4f} (n={len(slopes)}, "
                     f"{sum(s > 0 for s in slopes)}/{len(slopes)} positive)")
    # C4: redundancy of theta with PC1 on per-turn time-courses
    cors = []
    for r in rows:
        v = np.array(r["per_turn"]["valence"])
        th = np.array(r["per_turn"]["theta"])
        if len(v) > 3 and v.std() > 0 and np.std(np.cos(th)) > 0:
            cors.append(abs(float(np.corrcoef(v, np.cos(th))[0, 1])))
    lines.append(f"C4: |corr(per-turn valence PC1, cos theta)| mean {np.mean(cors):.2f} "
                 "(high = manifold adds little; matches Stage-2 'linear suffices')")
    # dissociation check: does the model REGISTER the feedback stimulus even if
    # its own generation state does not move?
    for frame in frames:
        fr = [r for r in rows if r["frame"] == frame and r["outcome"] != "rep"
              and r.get("fb_read", {}).get("valence")]
        if fr:
            g = [float(np.mean(r["fb_read"]["valence"])) for r in fr
                 if r["outcome"] == "good"]
            b = [float(np.mean(r["fb_read"]["valence"])) for r in fr
                 if r["outcome"] == "bad"]
            lines.append(f"[{frame}] feedback-reading valence: good {np.mean(g):+.2f} "
                         f"vs bad {np.mean(b):+.2f} (d = {_d(g, b):+.2f}) — stimulus "
                         "registration, contrast with C1 state effect")
    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_cross(_=None):
    lines = ["Stage 3 cross-model summary", ""]
    for m in SUBJECTS:
        p = out_dir(m) / "summary.txt"
        if p.exists():
            lines += [f"---- {m}"] + ["  " + l for l in p.read_text().splitlines()[1:]] + [""]
    (P1 / "results" / "stage3" / "cross_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args.model, args.smoke)
    elif args.cmd == "analyze":
        cmd_analyze(args.model, args.smoke)
    elif args.cmd == "cross":
        cmd_cross()
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
