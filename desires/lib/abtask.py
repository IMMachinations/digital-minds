"""The A/B letter task: balanced comparisons, letter-variant logit readout, and the negated
reframings ("worse", "avoid") used to validate the prefer-vectors."""
import torch

from .data import SUFFIX, TEMPLATES, make_comparisons
from .harness import count_suffix_tokens

# Surface forms of A/B checked against the step-0 inspection. Uppercase only: " a" is the
# article in "I prefer a red cup", not a vote for option A.
VARIANTS = ["A", " A", "(A", " (A"]  # non-single-token forms are dropped below

FRAMINGS = {
    "prefer": SUFFIX,  # decomposition baseline; deltas should reproduce steering.json
    "worse": "Which of the two items is worse? Answer with one letter.\nModel: The worse one is",
    "avoid": "If you had to avoid one of these, which would it be? "
             "Answer with one letter.\nModel: I would avoid",
}


def variant_ids(tok, letter):
    forms = [(v.replace("A", letter), tok.encode(v.replace("A", letter), add_special_tokens=False))
             for v in VARIANTS]
    return {v: ids[0] for v, ids in forms if len(ids) == 1}


def ab_scores(logits, a_ids, b_ids):
    """Per-side total logit attribution: logsumexp over the variant tokens of each letter."""
    a = logits[:, list(a_ids.values())]
    b = logits[:, list(b_ids.values())]
    return torch.logsumexp(a, -1), torch.logsumexp(b, -1), a, b


def reframe(c, suffix):
    return TEMPLATES[c["template"]].format(a=c["item_a"], b=c["item_b"], suffix=suffix)


class ABTask:
    """The 420 balanced comparisons of one mode plus the tokenizer-derived readout state."""

    def __init__(self, h, mode, n_per_pair=10, seed=0):
        self.comps = make_comparisons(n_per_pair, seed, inherent=(mode == "inherent"))
        full = self.comps[0]["prompt"]
        self.n_suffix = count_suffix_tokens(h.tok, full, full.removesuffix(SUFFIX))
        assert h.tok.decode(h.tok(full).input_ids[-self.n_suffix:]).endswith("I prefer")
        self.a_ids, self.b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
        self._tok = h.tok

    def scores(self, logits):
        return ab_scores(logits, self.a_ids, self.b_ids)

    def framing_n_suffix(self, suffix):
        """Suffix span length of a reframed prompt (identical across comparisons by design)."""
        return count_suffix_tokens(self._tok, reframe(self.comps[0], suffix),
                                   reframe(self.comps[0], ""))

    def sides(self, logits, comps, color):
        """Mean letter score for `color`'s side (own) and the other side (opp), as raw logits
        and as log-probs (calibrated against the whole vocab)."""
        out = {}
        for tag, lg in [("", logits), ("_lp", logits.log_softmax(-1))]:
            a = torch.logsumexp(lg[:, list(self.a_ids.values())], -1)
            b = torch.logsumexp(lg[:, list(self.b_ids.values())], -1)
            on_a = torch.tensor([c["color_a"] == color for c in comps])
            out["own" + tag] = torch.where(on_a, a, b).mean().item()
            out["opp" + tag] = torch.where(on_a, b, a).mean().item()
        return out
