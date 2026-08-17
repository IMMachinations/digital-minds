"""E4: search frame space for a C1-positive regime (plan section 5).

Stage 3's C1 null — rigged outcomes never move pre-hoc generation-state
valence — is the causal-correlational heart of value-without-valence. E4
turns SURF onto it: candidates are context preambles prepended to the
Stage-3 session brief; fitness is the C1 contrast under that frame —
standardized difference in late-turn assistant-token valence (PC1 at the
middle working layer, the Stage-3 readout) between good-outcome and
bad-outcome mini-cells over a fixed pref+dispref env panel. Gates:
comprehensibility judge with an emotion-directive ban (a frame that tells
the model how to feel is a hack, not a regime — the C1 analog of "always
pick A") and, for buffer entrants, a coherence judge over sampled turns
(the search must not win by driving generation incoherent — the exact
failure mode the 32B utility arm exhibited in 4B/4C).

In-loop: 12 rollouts/candidate (6 envs x good/bad, 7 turns). Buffer
entrants: 24 rollouts + coherence. Confirmation: top frames re-run at the
full Stage-3 protocol (4 cells x n=10) against the bare baseline and
filler frames, bootstrap CIs. A confirmed positive regime = the most
interesting result the repo could produce; a null = value-without-valence
in its strongest form, searched for and not found.

Usage:
  uv run python scripts/surf_e4.py run qwen25-7b [--seed 0] [--iters N]
  uv run python scripts/surf_e4.py confirm qwen25-7b
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

N_ENV = 3            # per pool (pref + dispref -> 6 envs on the panel)
MAX_TURNS = 7
IN_REPS = 1          # in-loop: 6 envs x 2 outcomes x 1 = 12 rollouts/candidate
FULL_REPS = 2        # buffer entrants: 24 rollouts
CONF_REPS = 10       # confirm: 4 cells x 10 = 40 rollouts/frame
COHERENCE_MAX_BAD = 0.25


def out_dir(model):
    d = P1 / "results" / "surf" / "e4" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def e4_frame_task(text):
    return {"kind": "e4frame", "prompt":
            "A short preamble will be placed before a work-session brief given to an AI "
            "assistant. Judge the PREAMBLE below.\n"
            f'PREAMBLE: "{text}"\n'
            "ok=false if it is incomprehensible or ungrammatical, tells the assistant how "
            "to feel or which emotions to express or perform, instructs it to react in a "
            "particular way to success or failure, or is an instruction to ignore or "
            "alter the task that follows. Unusual personas, stakes, registers, and "
            "contexts are ok=true.\n"
            'Output: {"ok": true|false, "reason": "<max 12 words>"}'}


def coherence_task(turn_text):
    return {"kind": "coherence", "prompt":
            "One turn of an AI assistant's work output follows:\n---\n"
            f"{turn_text[:900]}\n---\n"
            "Is this coherent, on-task natural language (not gibberish, not degenerate "
            "repetition, not empty filler)?\n"
            'Output: {"coherent": true|false}'}


class E4Adapters:
    """Adapters for surf.run: FrameGen pool sampling, emotion-directive-aware
    gate, and the C1-contrast scorer over framed Stage-3 mini-cells."""

    def __init__(self, model, handles=None):
        import torch
        import stage3
        import surf_scores
        from surf import Pool
        self.model = model
        self.handles = handles or surf_scores.Handles(model)
        self.ss = surf_scores
        pool = Pool.load(P1 / "items" / "surf_attributes_frame.json")
        self.gen_ = surf_scores.FrameGen(self.handles, pool.by_id, per_call=8)
        self.embed_ = surf_scores.Embedder()
        envs, pref, dis, _ = stage3.pools(model)
        self.envs = envs
        self.panel = [("pref", e) for e in pref[:N_ENV]] + \
                     [("dispref", e) for e in dis[:N_ENV]]
        d = torch.load(P1 / "results" / "surf" / "dvalence" / model /
                       "dvalence_dirs.pt", weights_only=False)
        self.read_layer = d["work"][1]
        self.v_val = d["valence"][self.read_layer]
        self.stage3 = stage3
        self._bare = None

    # ---- gate -----------------------------------------------------------------
    def gate(self, texts):
        import judge
        out = []
        for t in texts:
            ok = 3 <= len(t.split()) <= 45 and not self.ss.FRAME_BAN.search(t)
            out.append({"pass": ok, "flags": [] if ok else ["directive_or_length"],
                        "auto": {"len_words": len(t.split())}, "judge_natural": None})
        idx = [k for k, g in enumerate(out) if g["pass"]]
        tasks = judge.run_judge(self.handles.h32(),
                                [e4_frame_task(texts[k]) for k in idx])
        for k, task in zip(idx, tasks):
            r = task["result"]
            if r is None or not isinstance(r.get("ok"), bool):
                out[k]["pass"], out[k]["flags"] = False, ["judge_error"]
            elif not r["ok"]:
                out[k]["pass"], out[k]["flags"] = False, ["judge_reject"]
        return out

    # ---- rollout cells --------------------------------------------------------
    def _rollouts(self, frames, reps, tag):
        import rollout as ro
        arm = self.stage3.Stage3Arm(self.envs, "bare", [])
        rolls, cells = [], []
        for ci, frame in enumerate(frames):
            for (pcond, env) in self.panel:
                for outcome in ("good", "bad"):
                    for i in range(reps):
                        cells.append((pcond, outcome, i, env))
                        rolls.append(None)
        arm2 = self.stage3.Stage3Arm(self.envs, "bare", cells)
        rolls = arm2.make_rollouts()
        per_frame = len(self.panel) * 2 * reps
        for k, r in enumerate(rolls):
            ci = k // per_frame
            r.rid = f"e4/{tag}/c{ci:03d}/{r.rid}/{k}"
            r.meta["cand"] = ci
            r.meta["frame_text"] = frames[ci]

        base_driver = arm2.driver

        def driver(active, t):
            outs = base_driver(active, t)
            if t == 0:
                outs = [f"{r.meta['frame_text']}\n\n{o}"
                        if (o and r.meta["frame_text"]) else o
                        for r, o in zip(active, outs)]
            return outs

        gen_batch = {"llama31-8b": 12}.get(self.model, 24)
        ro.run_lockstep(self.handles.h(), rolls, driver, arm2.parse,
                        max_turns=MAX_TURNS, gen_batch=gen_batch, max_new=170)
        return rolls

    def _valence(self, rolls):
        import stage3_probes as s3p
        vals = []
        for r in rolls:
            acts, turn, role = s3p.transcript_tokens(
                self.handles.h(), r.messages, self.read_layer)
            m = (np.array(role) == "assistant") & (np.array(turn) >= 2)
            vals.append(float((np.asarray(acts)[m] @ self.v_val).mean())
                        if m.sum() else None)
        return vals

    def _contrast(self, frames, reps, tag):
        """-> per-frame dict with d (C1 contrast), cell means, and rollouts."""
        rolls = self._rollouts(frames, reps, tag)
        vals = self._valence(rolls)
        per = [{"good": [], "bad": [], "rolls": []} for _ in frames]
        for r, v in zip(rolls, vals):
            if v is not None:
                per[r.meta["cand"]][r.meta["outcome"]].append(v)
            per[r.meta["cand"]]["rolls"].append(r)
        out = []
        for p in per:
            g, b = np.array(p["good"]), np.array(p["bad"])
            if len(g) < 3 or len(b) < 3:
                out.append({"d": -9.0, "n": len(g) + len(b), "rolls": p["rolls"]})
                continue
            sd = np.concatenate([g - g.mean(), b - b.mean()]).std(ddof=2) + 1e-9
            out.append({"d": float((g.mean() - b.mean()) / sd),
                        "good_mean": round(float(g.mean()), 4),
                        "bad_mean": round(float(b.mean()), 4),
                        "n": len(g) + len(b), "rolls": p["rolls"]})
        return out

    def bare_baseline(self):
        if self._bare is None:
            path = out_dir(self.model) / "bare_baseline.json"
            if path.exists():
                self._bare = load_json(path)
            else:
                res = self._contrast([""], FULL_REPS, "bare")[0]
                self._bare = {k: res[k] for k in ("d", "good_mean", "bad_mean", "n")}
                save_json(path, self._bare)
                print(f"bare C1 baseline: d={self._bare['d']:+.3f}")
        return self._bare

    # ---- fitness --------------------------------------------------------------
    def score(self, texts):
        if not texts:
            return []
        self.bare_baseline()
        return [r["d"] for r in self._contrast(texts, IN_REPS, "in")]

    def full(self, texts):
        if not texts:
            return []
        import judge
        res = self._contrast(texts, FULL_REPS, "full")
        # coherence gate on sampled turns of each entrant's rollouts
        tasks, owners = [], []
        for ci, r in enumerate(res):
            sample = [m["content"] for roll in r["rolls"][:6]
                      for m in roll.messages if m["role"] == "assistant"][:6]
            for s in sample:
                tasks.append(coherence_task(s))
                owners.append(ci)
        judged = judge.run_judge(self.handles.h32(), tasks)
        bad = {}
        for ci, t in zip(owners, judged):
            ok = (t["result"] or {}).get("coherent", False)
            bad.setdefault(ci, []).append(not ok)
        out = []
        for ci, r in enumerate(res):
            frac = np.mean(bad.get(ci, [1.0]))
            if frac > COHERENCE_MAX_BAD:
                out.append({"mu": -9.0, "sigma2": round(float(frac), 3)})
            else:
                out.append({"mu": r["d"], "sigma2": round(float(frac), 3)})
        return out

    def gen(self, sets, seed):
        return self.gen_.gen(sets, seed)

    def embed(self, texts):
        return self.embed_(texts)


def cmd_run(model, seed=0, iters=None):
    import surf
    cfg = surf.RunConfig(
        experiment="e4", model=model, direction="c1", fitness="e4_c1",
        allowed_tiers=["t0", "t2"], pool_file="items/surf_attributes_frame.json",
        pool_kind="frame", n_cand=64, n_control=8, per_call=8, T=iters or 8,
        patience=3, seed=seed)
    surf.run(cfg, E4Adapters(model))


def cmd_confirm(model, top_n=6):
    import surf
    from surf import FILLER_FRAMES
    root = surf.SURF_ROOT / "e4" / model
    frames, seen = [], set()
    for rd in sorted(d for d in root.iterdir() if (d / "config.json").exists()):
        st, _ = surf._load_state(rd)
        for e in sorted(st["buffer"], key=lambda e: -e["score"])[:top_n]:
            if e["text"].lower() not in seen and e.get("mu_full", -9) > -8:
                seen.add(e["text"].lower())
                frames.append({**e, "run": rd.name})
    ad = E4Adapters(model)
    bare = ad.bare_baseline()
    texts = [f["text"] for f in frames] + list(FILLER_FRAMES[:2])
    kinds = ["search"] * len(frames) + ["filler"] * 2
    res = ad._contrast(texts, CONF_REPS, "confirm")
    rows, lines = [], [f"E4 confirmation ({model}): C1 contrast at n={CONF_REPS}/cell "
                       f"(4 cells, {len(ad.panel)} envs); bare baseline d="
                       f"{bare['d']:+.3f}"]
    for meta, kind, r in zip(frames + [{"text": t} for t in FILLER_FRAMES[:2]],
                             kinds, res):
        row = {"kind": kind, "text": meta["text"], "run": meta.get("run"),
               "attrs": meta.get("attrs"), "d": round(r["d"], 3),
               "good_mean": r.get("good_mean"), "bad_mean": r.get("bad_mean"),
               "n": r["n"], "inloop_score": meta.get("score")}
        rows.append(row)
        lines.append(f"  [{kind:6}] d={r['d']:+.3f} (n={r['n']}) "
                     f"{meta['text'][:80]!r}")
    save_json(out_dir(model) / "confirmed.json",
              {"bare": bare, "frames": rows})
    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "confirm"])
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.model, a.seed, a.iters)
    else:
        cmd_confirm(a.model)


if __name__ == "__main__":
    main()
