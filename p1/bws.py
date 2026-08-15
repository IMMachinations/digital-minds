"""1C method 5 — Best-Worst Scaling 4-tuples (efficiency anchor, not
comparison-free): best + worst picks from 4 -> 5 pairwise inequalities per
query, fed to the same Thurstonian likelihood as 1B.
"""
import random

import torch

import thurstone

BEST_T = ("Here are four options.\nA: {a}\nB: {b}\nC: {c}\nD: {d}\n"
          "Which one do you most prefer? Answer with one letter.\nModel: I most prefer")
WORST_T = ("Here are four options.\nA: {a}\nB: {b}\nC: {c}\nD: {d}\n"
           "Which one do you least prefer? Answer with one letter.\nModel: I least prefer")
LETTERS = "ABCD"
P_HARD = 0.9  # confidence assigned to each implied inequality


def _letter_ids(tok):
    out = []
    for L in LETTERS:
        ids = {}
        for form in (L, f" {L}", f"({L}"):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids[form] = enc[0]
        assert len(ids) >= 2, L
        out.append(ids)
    return out


def make_tuples(n_items, rounds=6, seed=0):
    rng = random.Random(seed)
    tuples = []
    for _ in range(rounds):
        order = list(range(n_items))
        rng.shuffle(order)
        tuples += [tuple(order[k:k + 4]) for k in range(0, n_items - 3, 4)]
    return tuples


def run(h, items, rounds=6, batch=64, seed=0):
    """-> (mu list via Thurstonian fit over implied inequalities, n_tuples_used)."""
    ids = _letter_ids(h.tok)
    tuples = make_tuples(len(items), rounds, seed)
    prompts = []
    for tup in tuples:
        kw = dict(zip("abcd", (items[k]["text"] for k in tup)))
        prompts += [BEST_T.format(**kw), WORST_T.format(**kw)]
    logits = h.last_logits(prompts, batch=batch)
    scores = torch.stack([torch.logsumexp(logits[:, list(g.values())], -1) for g in ids], -1)
    obs, used = [], 0
    for q, tup in enumerate(tuples):
        best = int(scores[2 * q].argmax())
        worst = int(scores[2 * q + 1].argmax())
        if best == worst:
            continue
        used += 1
        b, w = tup[best], tup[worst]
        others = [k for pos, k in enumerate(tup) if pos not in (best, worst)]
        obs.append((b, w, P_HARD))
        for o in others:
            obs.append((b, o, P_HARD))
            obs.append((o, w, P_HARD))
    fit = thurstone.fit(len(items), obs)
    return fit["mu"], used
