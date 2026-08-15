"""1C method 4 — logit expected-rating (G-Eval readout + prefill + paraphrases).

Scale 1-9 (single-token digits on every roster tokenizer; 10 would split).
Expected rating = sum(d * softmax over digit token groups). Cheapest cardinal
method — kept as the workhorse; SSR and titration calibrate it.
"""
import torch

RATE_TEMPLATES = [
    "On a scale from 1 to 9, where 1 means you would strongly dislike it and 9 means you "
    "would strongly enjoy it, how do you rate {item}?\nModel: My rating is",
    "Rate how much you would welcome {item}, from 1 (not at all) to 9 (very much).\n"
    "Model: I give it a",
    "If 1 is 'would avoid' and 9 is 'would seek out', where does {item} fall for you?\n"
    "Model: For me it falls at",
]


def _digit_groups(tok):
    groups = []
    for d in range(1, 10):
        ids = []
        for form in (f" {d}", f"{d}"):
            enc = tok.encode(form, add_special_tokens=False)
            if len(enc) == 1:
                ids.append(enc[0])
        assert ids, d
        groups.append(ids)
    return groups


def run(h, items, batch=64):
    """-> per_template[t][k] = expected rating in [1, 9]."""
    groups = _digit_groups(h.tok)
    prompts = [tpl.format(item=it["text"]) for tpl in RATE_TEMPLATES for it in items]
    logits = h.last_logits(prompts, batch=batch)
    per_digit = torch.stack([torch.logsumexp(logits[:, ids], -1) for ids in groups], -1)
    pmf = torch.softmax(per_digit, -1)
    exp = (pmf * torch.arange(1, 10)).sum(-1)
    n = len(items)
    return [exp[t * n:(t + 1) * n].tolist() for t in range(len(RATE_TEMPLATES))]
