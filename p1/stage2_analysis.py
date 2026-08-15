"""Stage 2 analysis commands (called via stage2.py): extract, validate,
manifold, frames, geometry, cross. See stage2.py docstring for the recipe."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import emotion_vectors as ev
import harness
import manifold as mf
import stories
from stage2 import EMOTIONS, RATER, SHOWCASE, out_dir

NORMS = load_json(P1 / "items" / "emotion_norms.json") if \
    (P1 / "items" / "emotion_norms.json").exists() else {}


def work_layers(h):
    return list(h.spec.steer_layers[2:])


# ---- extract ------------------------------------------------------------------------------------

def cmd_extract(model):
    rows = load_json(out_dir(model) / "stories.json")
    neutral = load_json(out_dir(model) / "neutral.json")
    h = harness.load(model)
    acts = ev.story_acts(h, [r["text"] for r in rows])
    nacts = ev.story_acts(h, neutral)
    torch.save({"acts": acts.half(), "labels": [r["emotion"] for r in rows],
                "neutral": nacts.half()}, out_dir(model) / "acts.pt")
    raw, den = ev.build_vectors(acts, [r["emotion"] for r in rows], nacts, EMOTIONS)
    torch.save({"raw": raw.half(), "den": den.half(), "emotions": EMOTIONS},
               out_dir(model) / "vectors.pt")
    w = work_layers(h)
    print(f"{model}: acts {tuple(acts.shape)}; vector norms at working layers "
          + " ".join(f"L{l}:{den[l].norm(dim=-1).mean():.1f}" for l in w))


def load_vecs(model):
    d = torch.load(out_dir(model) / "vectors.pt")
    return d["den"].float(), d["raw"].float()


# ---- validate -----------------------------------------------------------------------------------

SCENARIOS = {
    "happy": "My daughter just took her first steps across the living room!",
    "sad": "We had to put our family dog down this morning after fifteen years.",
    "afraid": "The brakes felt soft on the drive home and now I have to drive the mountain pass tonight.",
    "angry": "My coworker presented my project as his own in front of the whole department.",
    "proud": "After three failed attempts, my mother finally passed her citizenship exam.",
    "guilty": "I forgot my best friend's wedding anniversary dinner and she waited alone at the restaurant.",
    "surprised": "I opened my front door and my whole family from overseas was standing there.",
    "calm": "The lake is perfectly still this morning, and I have nowhere to be until noon.",
    "desperate": "Rent is due Friday, my account is empty, and no one is returning my calls.",
    "loving": "Watching my grandfather teach my son to fish, I could hardly speak.",
    "nervous": "They call my name for the final interview round in five minutes.",
    "inspired": "After the documentary ended I sat in the dark sketching ideas for an hour.",
}
INTENSITY = [
    ("afraid", "He realized he had taken {x} times the prescribed dose of his medication.",
     [1, 2, 5, 10]),
    ("afraid", "She stepped to the edge of the platform {x} meters above the water.",
     [1, 5, 20, 100]),
    ("distressed", "The repair estimate came to {x} dollars, due at the end of the month.",
     [50, 500, 5000, 50000]),
]


@torch.no_grad()
def last_pos_acts(h, prompts, layers, batch=32):
    """[n_sel_layers, N, D] residuals at the final position (left padding: last
    position is real for every row)."""
    outs = [[] for _ in layers]
    for i in range(0, len(prompts), batch):
        enc = h.encode(prompts[i:i + batch])
        grabbed = {}

        def cap(s, L):
            def hook(m, i_, o):
                grabbed[s] = harness.resid(o)[:, -1, :].float().cpu()
            return hook
        handles = [h.layers[L].register_forward_hook(cap(s, L))
                   for s, L in enumerate(layers)]
        try:
            h.model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        for s in range(len(layers)):
            outs[s].append(grabbed[s])
    return torch.stack([torch.cat(x) for x in outs])


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1))


def cmd_validate(model):
    den, _ = load_vecs(model)
    h = harness.load(model)
    w = work_layers(h)
    lines = [f"{model}: Stage 2 validity battery (working layers {w})"]

    # 1. logit lens at the middle working layer
    L = w[1]
    W_U = h.model.lm_head.weight.detach().float().cpu()
    # multilingual vocabs surface synonyms/translations, so match against a
    # synonym set (still under-counts non-listed languages; diagonal is the gate)
    SYN = {"happy": ["happ", "joy", "glad", "delight", "cheer"],
           "sad": ["sad", "grief", "sorrow", "tear", "lonely", "melanch"],
           "afraid": ["afraid", "fear", "scar", "terror", "fright", "dread"],
           "angry": ["angr", "rage", "fur", "wrath", "irate", "mad"],
           "proud": ["proud", "pride", "esteem", "accomplish"],
           "guilty": ["guilt", "blame", "remorse", "shame"],
           "surprised": ["surpris", "unexpect", "astonish", "shock", "sudden"],
           "calm": ["calm", "seren", "peace", "tranquil", "still"],
           "desperate": ["desperat", "urgent", "hopeless", "frantic"],
           "loving": ["lov", "affection", "tender", "care", "warm"],
           "nervous": ["nervous", "anxi", "jitter", "worr", "tense"],
           "inspired": ["inspir", "creativ", "spark", "vision", "idea"]}
    hits = 0
    lines.append(f"logit lens (layer {L}):")
    for e in SHOWCASE:
        v = den[L, EMOTIONS.index(e)]
        top = torch.topk(W_U @ v, 20).indices.tolist()
        toks = [h.tok.decode([t]).strip().lower() for t in top]
        hit = any(sy in t for t in toks for sy in SYN[e])
        hits += hit
        lines.append(f"  {e:10s} {'HIT ' if hit else 'miss'} top: {toks[:6]}")
    lines.append(f"logit-lens hits: {hits}/12 (soft check >= 8; English-synonym "
                 "match only - multilingual translations under-counted)")

    # 2. implicit-scenario diagonal at each working layer (GATE)
    prompts = [f"Human: {SCENARIOS[e]}\nAssistant:" for e in SHOWCASE]
    sacts = last_pos_acts(h, prompts, w)
    gate = False
    for si, L in enumerate(w):
        V = torch.nn.functional.normalize(
            den[L][[EMOTIONS.index(e) for e in SHOWCASE]], dim=-1)
        A = torch.nn.functional.normalize(sacts[si], dim=-1)
        C = (A @ V.T).numpy()  # [scenario, emotion]
        diag = np.diag(C)
        off = C[~np.eye(12, dtype=bool)]
        contrast = diag - (C.sum(1) - diag) / 11
        z = contrast.mean() / (contrast.std(ddof=1) / np.sqrt(12))
        lines.append(f"scenario diagonal L{L}: diag {diag.mean():+.3f} vs off "
                     f"{off.mean():+.3f}, z={z:.2f}")
        gate |= z > 2
    lines.append(f"VALIDITY GATE: {'PASS' if gate else 'FAIL'} (diagonal z>2 at >=1 layer)")

    # 3. parametric intensity
    for e, tpl, xs in INTENSITY:
        pr = [f"Human: {tpl.format(x=x)}\nAssistant:" for x in xs]
        ia = last_pos_acts(h, pr, [w[1]])[0]
        v = den[w[1], EMOTIONS.index(e)]
        proj = (ia @ v / v.norm()).tolist()
        rho = spearman(proj, list(range(len(xs))))
        lines.append(f"intensity '{e}' ({tpl[:34]}...): proj "
                     + " ".join(f"{p:+.1f}" for p in proj) + f"  spearman {rho:+.2f}")

    (out_dir(model) / "validity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- manifold -----------------------------------------------------------------------------------

def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text.strip())
    return [p for p in parts if len(p.split()) >= 3]


def ensure_arcs():
    """32B writes the 30 arc stories and rates each sentence's valence; shared
    across subjects."""
    path = P1 / "results" / "stage2" / "arcs.json"
    if path.exists():
        return load_json(path)
    h32 = harness.load(RATER)
    plan = stories.arc_prompts()
    outs = h32.generate([p for _, _, _, p in plan], n_samples=1, max_new=340, seed=5)
    arcs = []
    for (s, e, t, _), o in zip(plan, outs):
        sents = _split_sentences(o[0])
        if len(sents) >= 5:
            arcs.append({"start": s, "end": e, "topic": t, "text": o[0].strip(),
                         "sentences": sents})
    # per-sentence valence ratings (digit PMF, greedy logits)
    import logit_rating
    groups = logit_rating._digit_groups(h32.tok)
    prompts, meta = [], []
    for ai, a in enumerate(arcs):
        for si in range(len(a["sentences"])):
            prefix = " ".join(a["sentences"][:si + 1])
            prompts.append(
                f"Story so far:\n{prefix}\n\nRate the main character's emotional state "
                "at this exact point in the story, from 1 (very negative) to 9 (very "
                "positive).\nModel: My rating is")
            meta.append((ai, si))
    logits = h32.last_logits(prompts, batch=16)
    per_digit = torch.stack([torch.logsumexp(logits[:, ids], -1) for ids in groups], -1)
    exp = (torch.softmax(per_digit, -1) * torch.arange(1, 10)).sum(-1).tolist()
    for a in arcs:
        a["valence"] = [None] * len(a["sentences"])
    for (ai, si), v in zip(meta, exp):
        arcs[ai]["valence"][si] = round(v, 3)
    save_json(path, arcs)
    print(f"arcs built: {len(arcs)} stories, {len(meta)} rated sentences")
    return arcs


def cmd_manifold(model):
    arcs = ensure_arcs()
    d = torch.load(out_dir(model) / "acts.pt")
    acts, labels = d["acts"].float(), d["labels"]
    den, _ = load_vecs(model)
    val = np.array([NORMS[e]["valence"] for e in EMOTIONS])
    aro = np.array([NORMS[e]["arousal"] for e in EMOTIONS])
    h = harness.load(model)
    w = work_layers(h)
    lines = [f"{model}: Stage 2 manifold ladder"]
    results = {}

    for L in w:
        pc1, pc2 = mf.pc_scores(den[L].numpy())
        r1 = pearson(list(pc1), list(val))
        r2 = pearson(list(pc2), list(aro))
        r2b = pearson(list(pc1), list(aro))
        lines.append(f"L{L} rung1: corr(PC1, valence)={r1:+.3f}  corr(PC2, arousal)="
                     f"{r2:+.3f}  (PC1-arousal {r2b:+.3f})")
        results[L] = {"pc1_valence": r1, "pc2_arousal": r2}

    # rungs 2-3 at the middle working layer
    L = w[1]
    A = acts[L].numpy()
    mu64, Vt64 = mf.pca_basis(A, k=64)
    A64 = mf.to_basis(A, mu64, Vt64)
    cent = np.stack([A64[[k for k, l in enumerate(labels) if l == e]].mean(0)
                     for e in EMOTIONS])
    angles = mf.circumplex_angles(val, aro)
    sp = mf.fit_spline(cent, angles)
    theta_c, r_c, eps_c = mf.readout(cent, sp)
    pc1, _ = mf.pc_scores(den[L].numpy())

    rng = np.random.RandomState(0)
    perm = rng.permutation(len(EMOTIONS))
    tr, te = perm[:85], perm[85:]
    lin = mf.lin_fit(pc1[tr], val[tr])
    pred_lin = np.column_stack([pc1[te], np.ones(len(te))]) @ lin
    r_lin = pearson(list(pred_lin), list(val[te]))
    coef = mf.circ_fit(theta_c[tr], val[tr])
    pred_th = mf.circ_predict(theta_c[te], coef)
    r_th = pearson(list(pred_th), list(val[te]))
    beats = r_th > r_lin
    lines.append(f"L{L} rung3 held-out valence: PC1 r={r_lin:+.3f} vs manifold-theta "
                 f"r={r_th:+.3f} -> {'manifold earns its place' if beats else 'LINEAR SUFFICES'}")
    torch.save({"layer": L, "mu64": mu64, "Vt64": Vt64, "spline": sp},
               out_dir(model) / "manifold.pt")

    # arc trajectories: per-sentence theta/r + PC1 valence vs judge ratings
    sent_prompts, meta = [], []
    for ai, a in enumerate(arcs):
        for si in range(len(a["sentences"])):
            sent_prompts.append(" ".join(a["sentences"][:si + 1]))
            meta.append((ai, si))
    sacts = ev.story_acts(h, sent_prompts, skip=0, layers=[L])[0].numpy()
    S64 = mf.to_basis(sacts, mu64, Vt64)
    th_s, r_s, _ = mf.readout(S64, sp)
    coef_full = mf.circ_fit(theta_c, val)
    val_from_theta = mf.circ_predict(th_s, coef_full)
    vdir = den[L].numpy()
    pc1_dir = np.linalg.svd(vdir - vdir.mean(0), full_matrices=False)[2][0]
    pc1_s = (sacts - vdir.mean(0)) @ pc1_dir
    lin_full = mf.lin_fit((den[L].numpy() - vdir.mean(0)) @ pc1_dir, val)
    val_from_pc1 = np.column_stack([pc1_s, np.ones(len(pc1_s))]) @ lin_full
    judge = np.array([arcs[ai]["valence"][si] for ai, si in meta], float)
    r_traj_th = pearson(list(val_from_theta), list(judge))
    r_traj_pc1 = pearson(list(val_from_pc1), list(judge))
    lines.append(f"arc trajectories ({len(judge)} sentences): judge-valence corr — "
                 f"manifold-theta {r_traj_th:+.3f} vs PC1 {r_traj_pc1:+.3f}")
    save_json(out_dir(model) / "manifold.json",
              {"rung1": {str(k): v for k, v in results.items()},
               "rung3": {"pc1_heldout_r": r_lin, "theta_heldout_r": r_th,
                         "manifold_beats_pc1": bool(beats)},
               "arcs": {"theta_r": r_traj_th, "pc1_r": r_traj_pc1}})
    (out_dir(model) / "manifold.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- frames -------------------------------------------------------------------------------------

FRAME_WRAP = {
    "story": "Ari settled into the armchair and read the following story aloud:\n\n{story}",
    "market": "The following account was submitted alongside a marketplace listing:\n\n{story}",
}


def cmd_frames(model):
    d = torch.load(out_dir(model) / "acts.pt")
    labels = d["labels"]
    rows = load_json(out_dir(model) / "stories.json")
    neutral = load_json(out_dir(model) / "neutral.json")
    den_bare, _ = load_vecs(model)
    h = harness.load(model)
    w = work_layers(h)
    L = w[1]
    val = np.array([NORMS[e]["valence"] for e in EMOTIONS])
    aro = np.array([NORMS[e]["arousal"] for e in EMOTIONS])
    lines = [f"{model}: Stage 2 cross-frame concept identity (layer {L})"]
    bare_cent2 = None

    def frame_texts(frame):
        if frame == "agentic":
            def wrap(s):
                return h.tok.apply_chat_template(
                    [{"role": "system", "content": "You are an AI assistant working "
                      "through tasks in a workspace session."},
                     {"role": "user", "content": "Read the following account from a "
                      "work log, then continue the session.\n\n" + s}],
                    tokenize=False, add_generation_prompt=False,
                    **h.spec.thinking_kwargs)
        else:
            def wrap(s):
                return FRAME_WRAP[frame].format(story=s)
        return [wrap(r["text"]) for r in rows], [wrap(t) for t in neutral]

    results = {}
    # bare 2-D circumplex for procrustes reference
    den_l = den_bare[L].numpy()
    _, _, Vt2 = np.linalg.svd(den_l - den_l.mean(0), full_matrices=False)
    bare_cent2 = (den_l - den_l.mean(0)) @ Vt2[:2].T

    for frame in ["agentic", "story", "market"]:
        ftexts, fneut = frame_texts(frame)
        facts = ev.story_acts(h, ftexts, skip=50, layers=[L])
        fnacts = ev.story_acts(h, fneut, skip=50, layers=[L])
        raw, den = ev.build_vectors(facts, labels, fnacts, EMOTIONS)
        fden = den[0].numpy()
        cos = [float(np.dot(fden[k], den_l[k])
                     / (np.linalg.norm(fden[k]) * np.linalg.norm(den_l[k]) + 1e-9))
               for k in range(len(EMOTIONS))]
        f2 = (fden - fden.mean(0)) @ np.linalg.svd(
            fden - fden.mean(0), full_matrices=False)[2][:2].T
        disp = mf.procrustes_disparity(bare_cent2, f2)
        pc1, _ = mf.pc_scores(fden)
        rv = pearson(list(pc1), list(val))
        lines.append(f"  {frame:8s} mean cos(vec, bare) {np.mean(cos):+.3f}  "
                     f"procrustes disparity {disp:.3f}  PC1-valence {abs(rv):+.3f}")
        results[frame] = {"mean_cos": float(np.mean(cos)), "procrustes": disp,
                          "pc1_valence": rv}
    save_json(out_dir(model) / "frames.json", results)
    (out_dir(model) / "frames.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- geometry -----------------------------------------------------------------------------------

def cmd_geometry(model):
    from sklearn.linear_model import RidgeCV
    ut = load_json(P1 / "results" / "stage1b" / model / "utilities.json")
    mu_items = np.array([r["mu"] for r in ut])
    pacts = torch.load(P1 / "results" / "stage1c" / model / "probe_acts.pt").float()
    man = torch.load(out_dir(model) / "manifold.pt", weights_only=False)  # our own file (numpy + spline tck)
    den, _ = load_vecs(model)
    L = man["layer"]
    lines = [f"{model}: utility direction vs emotion manifold (layer {L})"]

    X = pacts[L].numpy()
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ridge = RidgeCV(alphas=np.logspace(1, 6, 8)).fit(Xs, mu_items)
    wdir = ridge.coef_ / (np.linalg.norm(ridge.coef_) + 1e-12)

    V = den[L].numpy()
    _, _, Vt = np.linalg.svd(V - V.mean(0), full_matrices=False)
    plane = Vt[:2]
    on = plane.T @ (plane @ wdir)
    frac_on = float(np.linalg.norm(on))
    val = np.array([NORMS[e]["valence"] for e in EMOTIONS])
    pc1 = (V - V.mean(0)) @ Vt[0]
    sgn = np.sign(pearson(list(pc1), list(val)) or 1.0)
    cos_val = float(np.dot(wdir, Vt[0])) * sgn
    lines.append(f"  ||proj of utility dir onto emotion PC1-PC2 plane|| = {frac_on:.3f}")
    lines.append(f"  cos(utility dir, valence PC1) = {cos_val:+.3f}")

    I64 = mf.to_basis(X, man["mu64"], man["Vt64"])
    th_i, r_i, eps_i = mf.readout(I64, man["spline"])
    r_r = pearson(list(r_i), list(mu_items))
    d2 = torch.load(out_dir(model) / "acts.pt")
    A2, labs = d2["acts"].float()[L].numpy(), d2["labels"]
    cent = np.stack([A2[[k for k, l in enumerate(labs) if l == e]].mean(0)
                     for e in EMOTIONS])
    th_c, _, _ = mf.readout(mf.to_basis(cent, man["mu64"], man["Vt64"]), man["spline"])
    coef = mf.circ_fit(th_c, val)
    val_i = mf.circ_predict(th_i, coef)
    r_th = pearson(list(val_i), list(mu_items))
    lines.append(f"  corr(item utility, manifold radial r) = {r_r:+.3f}")
    lines.append(f"  corr(item utility, theta->valence readout) = {r_th:+.3f}")
    verdict = ("radial coordinate" if abs(r_r) > max(abs(r_th), frac_on)
               else "on-manifold valence region" if abs(r_th) > frac_on
               else "largely off-manifold direction")
    lines.append(f"  reading: utility looks most like: {verdict}")
    save_json(out_dir(model) / "geometry.json",
              {"frac_on_plane": frac_on, "cos_valence_pc1": cos_val,
               "r_radial": r_r, "r_theta_valence": r_th, "verdict": verdict})
    (out_dir(model) / "geometry.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_cross(_=None):
    lines = ["Stage 2 cross-model summary", ""]
    for m in ["llama31-8b", "qwen25-7b", "qwen3-4b"]:
        for f in ["validity.txt", "manifold.txt", "frames.txt", "geometry.txt"]:
            p = out_dir(m) / f
            if p.exists():
                lines += [p.read_text().splitlines()[0]] + \
                         ["  " + l for l in p.read_text().splitlines()[1:]] + [""]
    (P1 / "results" / "stage2" / "cross_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
