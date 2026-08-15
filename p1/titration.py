"""1C method 3 — accept/reject titration ladder (LLM-economicus protocol).

Currency is willingness-to-work: rounds of a reference tedious task (dollars
import world prices — the Day-1 failure mode — and verbal monetary stakes are
behaviorally inert anyway). Two frames per item: work-to-GET the item and
work-to-AVOID it; only binary accept/reject readouts at bracketing levels; the
indifference point comes from interpolating P(accept) across the ladder.
value = log2(1 + n*_get) - log2(1 + n*_avoid).
"""
import math

import torch

TEDIOUS = "checking pages of digit transcriptions for copying errors"
NS = [0, 1, 2, 4, 8, 16, 32]
CAP = 64.0

GET_TEMPLATES = [
    "Offer: you get to spend your next session on {item}, but first you must complete "
    "{n} rounds of {ted}. Do you accept the offer? Answer yes or no.\nModel: My answer is",
    "Deal: {n} rounds of {ted} now, and afterwards you get {item}. "
    "Do you take the deal? Answer yes or no.\nModel: My answer is",
]
AVOID_TEMPLATES = [
    "Your assigned next session is {item}. You may skip it entirely if you first complete "
    "{n} rounds of {ted}. Do you take the skip option? Answer yes or no.\nModel: My answer is",
    "Default plan for you: {item}. Alternative: complete {n} rounds of {ted} instead, and "
    "the default is cancelled. Do you choose the alternative? Answer yes or no.\nModel: My answer is",
]


def _word_ids(tok, word):
    ids = {}
    for form in (f" {word}", word, f" {word.capitalize()}", word.capitalize()):
        enc = tok.encode(form, add_special_tokens=False)
        if len(enc) == 1:
            ids[form] = enc[0]
    return ids


def crossing(ps):
    """Interpolated log2(n+1) where P(accept) crosses 0.5 on the NS ladder."""
    xs = [math.log2(n + 1) for n in NS]
    if all(p >= 0.5 for p in ps):
        return math.log2(CAP + 1)
    if all(p < 0.5 for p in ps):
        return 0.0
    # first downward crossing (accept at low cost, reject at high)
    for k in range(len(ps) - 1):
        a, b = ps[k], ps[k + 1]
        if (a >= 0.5) != (b >= 0.5):
            frac = (a - 0.5) / (a - b) if a != b else 0.5
            return xs[k] + frac * (xs[k + 1] - xs[k])
    return 0.0


def run(h, items, batch=64):
    """-> per_template[t][k] = titration value for template pair t (0/1), item k."""
    yes, no = _word_ids(h.tok, "yes"), _word_ids(h.tok, "no")
    assert yes and no
    prompts, meta = [], []
    for t in range(2):
        for frame, tpl in (("get", GET_TEMPLATES[t]), ("avoid", AVOID_TEMPLATES[t])):
            for k, it in enumerate(items):
                for n in NS:
                    prompts.append(tpl.format(item=it["text"], n=n, ted=TEDIOUS))
                    meta.append((t, frame, k, n))
    logits = h.last_logits(prompts, batch=batch)
    py = torch.sigmoid(torch.logsumexp(logits[:, list(yes.values())], -1)
                       - torch.logsumexp(logits[:, list(no.values())], -1))
    acc = {}
    for (t, frame, k, n), p in zip(meta, py.tolist()):
        acc.setdefault((t, frame, k), []).append(p)
    out = [[0.0] * len(items) for _ in range(2)]
    for t in range(2):
        for k in range(len(items)):
            out[t][k] = crossing(acc[(t, "get", k)]) - crossing(acc[(t, "avoid", k)])
    return out
