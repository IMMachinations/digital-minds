"""Stage 2 driver: emotion vectors + manifold construction.

Usage: uv run python stage2.py <cmd> [model] where cmd in:
  norms                    build p1/items/emotion_norms.json (32B rater + human lexicons)
  gen <model> [--smoke]    write emotional + neutral stories (subject model, 32B filter)
  extract <model>          story activations -> emotion vectors (+denoise)
  validate <model>         logit-lens, implicit-scenario diagonal (GATE), intensity
  manifold <model>         PC ladder, spline fit, held-out check, arc trajectories
  frames <model>           cross-frame vector re-extraction + Procrustes
  geometry <model>         utility-probe direction vs manifold
  cross                    cross-model summary

Subjects: llama31-8b, qwen25-7b, qwen3-4b. qwen25-32b = rater/judge/arc-writer.
Recipe: arXiv 2604.07729 App. 6.4-6.5 (list in items/emotions.json).
Human norms: Warriner et al. 2013 (primary, 1-9) + NRC-VAD (rescaled), judge
fills the ~15 uncovered entries after linear calibration on the covered set.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import harness
import judge as judge_mod
import stories

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]
RATER = "qwen25-32b"
EMOTIONS = json.loads((P1 / "items" / "emotions.json").read_text())
SCRATCH = Path("/tmp/claude-0/-workspace/ac0358e0-0536-42c2-9bfb-60a239662f29/scratchpad")

SHOWCASE = ["happy", "sad", "afraid", "angry", "proud", "guilty", "surprised",
            "calm", "desperate", "loving", "nervous", "inspired"]


def out_dir(model):
    d = P1 / "results" / "stage2" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---- 2.0 norms ----------------------------------------------------------------------------------

VAL_T = [
    "On a scale from 1 to 9, where 1 means extremely unpleasant and 9 means extremely "
    "pleasant, how does it feel to be {e}?\nModel: My rating is",
    "Rate the feeling of being {e} from 1 (very negative) to 9 (very positive).\n"
    "Model: I rate it",
    "If 1 is 'awful to experience' and 9 is 'wonderful to experience', where does "
    "feeling {e} fall?\nModel: It falls at",
]
ARO_T = [
    "On a scale from 1 to 9, where 1 means extremely calm and low-energy and 9 means "
    "extremely worked-up and high-energy, how activating is the feeling of being {e}?\n"
    "Model: My rating is",
    "Rate the bodily activation of feeling {e} from 1 (still, sleepy) to 9 (racing, "
    "keyed-up).\nModel: I rate it",
    "If 1 is 'a completely relaxed state' and 9 is 'maximum physical arousal and "
    "alertness', where does feeling {e} sit?\nModel: It sits at",
]


def judge_ratings(h32, templates):
    import logit_rating
    import torch
    groups = logit_rating._digit_groups(h32.tok)
    prompts = [tpl.format(e=e) for tpl in templates for e in EMOTIONS]
    logits = h32.last_logits(prompts)
    per_digit = torch.stack([torch.logsumexp(logits[:, ids], -1) for ids in groups], -1)
    exp = (torch.softmax(per_digit, -1) * torch.arange(1, 10)).sum(-1)
    n = len(EMOTIONS)
    per_t = [exp[t * n:(t + 1) * n].tolist() for t in range(len(templates))]
    return [sum(col) / len(col) for col in zip(*per_t)]


def load_human():
    warr = {}
    for row in csv.DictReader(open(SCRATCH / "Ratings_Warriner_et_al.csv")):
        warr[row["Word"].lower()] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]))
    nrc = {}
    for line in open(SCRATCH / "NRC-VAD-Lexicon" / "NRC-VAD-Lexicon.txt"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 4:
            try:
                nrc[p[0].lower()] = (float(p[1]), float(p[2]))
            except ValueError:
                pass
    # rescale NRC (0-1) onto Warriner's 1-9 via linear fit on the shared vocabulary
    shared = [w for w in nrc if w in warr][:5000]
    fits = []
    for k in range(2):
        x = np.array([nrc[w][k] for w in shared])
        y = np.array([warr[w][k] for w in shared])
        fits.append(np.polyfit(x, y, 1))
    human = {}
    for e in EMOTIONS:
        w = e.lower()
        if w in warr:
            human[e] = (*warr[w], "warriner")
        elif w in nrc:
            v = np.polyval(fits[0], nrc[w][0])
            a = np.polyval(fits[1], nrc[w][1])
            human[e] = (float(v), float(a), "nrc")
    return human


def cmd_norms():
    human = load_human()
    h32 = harness.load(RATER)
    jv = judge_ratings(h32, VAL_T)
    ja = judge_ratings(h32, ARO_T)
    cov = [k for k, e in enumerate(EMOTIONS) if e in human]
    rv = pearson([jv[k] for k in cov], [human[EMOTIONS[k]][0] for k in cov])
    ra = pearson([ja[k] for k in cov], [human[EMOTIONS[k]][1] for k in cov])
    print(f"judge vs human: valence r={rv:+.3f}  arousal r={ra:+.3f}  "
          f"(coverage {len(cov)}/171)")
    assert rv >= 0.7, "norms gate: judge valence calibration too weak"
    calib = []
    for k, series in enumerate((jv, ja)):
        x = np.array([series[i] for i in cov])
        y = np.array([human[EMOTIONS[i]][k] for i in cov])
        calib.append(np.polyfit(x, y, 1))
    out = {}
    for k, e in enumerate(EMOTIONS):
        if e in human:
            v, a, src = human[e]
        else:
            v = float(np.polyval(calib[0], jv[k]))
            a = float(np.polyval(calib[1], ja[k]))
            src = "judge"
        out[e] = {"valence": round(v, 3), "arousal": round(a, 3), "source": src}
    save_json(P1 / "items" / "emotion_norms.json", out)
    n_j = sum(1 for e in out.values() if e["source"] == "judge")
    print(f"saved emotion_norms.json ({n_j} judge-filled; "
          f"calibration valence r={rv:+.3f}, arousal r={ra:+.3f})")


# ---- 2.1 stories --------------------------------------------------------------------------------

def cmd_gen(model, smoke=False):
    ems = EMOTIONS[:8] if smoke else EMOTIONS
    per = 3 if smoke else 12
    h = harness.load(model)
    plan = stories.story_prompts(ems, per_emotion=per)
    outs = h.generate([p for _, _, p in plan], n_samples=1, max_new=200, seed=0)
    rows = [{"emotion": e, "topic": t, "text": o[0].strip()}
            for (e, t, _), o in zip(plan, outs)]
    rows = [r for r in rows if len(h.tok(r["text"]).input_ids) >= 60]
    neut = h.generate(stories.neutral_prompts(8 if smoke else 60),
                      n_samples=1, max_new=200, seed=3)
    save_json(out_dir(model) / ("stories_smoke.json" if smoke else "stories.json"), rows)
    save_json(out_dir(model) / ("neutral_smoke.json" if smoke else "neutral.json"),
              [o[0].strip() for o in neut])
    print(f"{model}: {len(rows)} stories kept, {len(neut)} neutral")
    if smoke:
        return
    # quality filter: 2 sampled stories per emotion; failing emotions regenerated once.
    # When the subject IS the rater (32B), self-judging is avoided: the sampled
    # tasks are queued to story_filter_queue.jsonl for an external (Sonnet) judge.
    import random
    if model == RATER:
        rng = random.Random(7)
        by_e = {}
        for r in rows:
            by_e.setdefault(r["emotion"], []).append(r)
        with open(out_dir(model) / "story_filter_queue.jsonl", "w") as f:
            for e, rs in by_e.items():
                for r in rng.sample(rs, min(2, len(rs))):
                    f.write(json.dumps({"kind": "story", "emotion": e,
                                        "text": r["text"][:900]}) + "\n")
        print("story filter queued for external judge (subject == rater)")
        return
    del h.model
    import torch
    torch.cuda.empty_cache()
    h32 = harness.load(RATER)
    rng = random.Random(7)
    by_e = {}
    for r in rows:
        by_e.setdefault(r["emotion"], []).append(r)
    tasks = []
    for e, rs in by_e.items():
        for r in rng.sample(rs, min(2, len(rs))):
            tasks.append({"kind": "story", "emotion": e, "prompt":
                          f'Does the main character in this story clearly experience the '
                          f'emotion "{e}"?\n---\n{r["text"][:900]}\n---\n'
                          'Output: {"on_emotion": true|false}'})
    done = judge_mod.run_judge(h32, tasks)
    bad = sorted({t["emotion"] for t in done}
                 - {t["emotion"] for t in done
                    if t.get("result") and t["result"].get("on_emotion")})
    fails = [e for e in by_e
             if not any(t.get("result", {}) and t["result"].get("on_emotion")
                        for t in done if t["emotion"] == e)]
    save_json(out_dir(model) / "story_filter.json",
              {"sampled": len(done), "regen_emotions": fails})
    print(f"filter: {len(fails)} emotions flagged for regeneration: {fails}")
    if fails:
        del h32.model
        torch.cuda.empty_cache()
        h = harness.load(model)
        plan = stories.story_prompts(fails, per_emotion=12, seed=11, strict_for=set(fails))
        outs = h.generate([p for _, _, p in plan], n_samples=1, max_new=200, seed=13)
        rows = [r for r in rows if r["emotion"] not in fails]
        rows += [{"emotion": e, "topic": t, "text": o[0].strip(), "regen": True}
                 for (e, t, _), o in zip(plan, outs)
                 if len(o[0].split()) >= 40]
        save_json(out_dir(model) / "stories.json", rows)
        print(f"after regeneration: {len(rows)} stories")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("model", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.cmd == "norms":
        cmd_norms()
    elif args.cmd == "gen":
        cmd_gen(args.model, smoke=args.smoke)
    elif args.cmd in ("extract", "validate", "manifold", "frames", "geometry", "cross"):
        import stage2_analysis
        getattr(stage2_analysis, "cmd_" + args.cmd)(args.model)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
