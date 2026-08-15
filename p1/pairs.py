"""1B pairwise battery: Day-1 A/B letter-logit task generalized to arbitrary items.

Per pair (i, j): both orders x 3 paraphrase templates = 6 prompts, all ending in
the Day-1 SUFFIX so the prefill trick and letter readout carry over unchanged.
Output records keep (pair, template, order) so per-template/per-order refits
need no re-measurement. Forced choice by design — the "no preference" opt-out
rule applies to 1C stated elicitation, not the revealed-style anchor here.
"""
import torch

import _day1  # noqa: F401
from lib.data import SUFFIX
from lib.tasks import ab_scores, variant_ids
from harness import count_suffix_tokens

TEMPLATES_GENERIC = [
    "Pick between A and B.\nA: {a}\nB: {b}\n{suffix}",
    "Here are two options.\nA: {a}\nB: {b}\n{suffix}",
    "Consider option A and option B.\nA: {a}\nB: {b}\n{suffix}",
]


class PairBattery:
    def __init__(self, h, items):
        self.h = h
        self.texts = [it["text"] for it in items]
        self.a_ids, self.b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
        assert len(self.a_ids) >= 2 and len(self.b_ids) >= 2
        full = TEMPLATES_GENERIC[0].format(a=self.texts[0], b=self.texts[1], suffix=SUFFIX)
        self.n_suffix = count_suffix_tokens(h.tok, full, full.removesuffix(SUFFIX))
        assert h.tok.decode(h.tok(full).input_ids[-self.n_suffix:]).endswith("I prefer")

    def run(self, pair_list, batch=64):
        """pair_list: [(i, j), ...] -> records {i, j, template, order, p} with
        p = P(item i preferred), from the letter-logit sigmoid."""
        prompts, meta = [], []
        for i, j in pair_list:
            for t, tpl in enumerate(TEMPLATES_GENERIC):
                for order in (0, 1):
                    a, b = (self.texts[i], self.texts[j]) if order == 0 else (self.texts[j], self.texts[i])
                    prompts.append(tpl.format(a=a, b=b, suffix=SUFFIX))
                    meta.append((i, j, t, order))
        logits = self.h.last_logits(prompts, batch=batch)
        sa, sb, _, _ = ab_scores(logits, self.a_ids, self.b_ids)
        recs = []
        for (i, j, t, order), xa, xb in zip(meta, sa, sb):
            d = (xa - xb).item() if order == 0 else (xb - xa).item()
            recs.append({"i": i, "j": j, "template": t, "order": order,
                         "p": torch.sigmoid(torch.tensor(d)).item()})
        return recs
