"""The two preference tasks and tier selection over their comparison lists.

- A/B letter task: balanced comparisons, letter-variant logit readout, and the negated
  reframings ("worse", "avoid") used to validate the prefer-vectors.
- Object-logprob task: items appear inline, preference is the teacher-forced log-prob of each
  item as the continuation of "... Model: I prefer".
"""
import torch

from .data import SUFFIX, TEMPLATES, make_comparisons
from .harness import count_suffix_tokens, resid

# ---- tiers / signed scores ----------------------------------------------------------------------

K = 20  # extraction/steering set size used throughout
TIERS = {"worst": (0, 20), "neutral": (50, 70), "best": (100, 120)}  # of 120 pairs per color


def signed(diff, comp, color):
    """Logit diff oriented toward `color` (positive = model favors that color's side)."""
    return diff if comp["color_a"] == color else -diff


def sort_signed(comps, color, key="diff"):
    """This color's comparisons, ascending by signed score (worst first)."""
    mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: signed(c[key], c, color))


def tier_slice(comps, color, lo, hi, key="diff"):
    return sort_signed(comps, color, key)[lo:hi]


def worst_k(comps, color, k=K, key="diff"):
    return sort_signed(comps, color, key)[:k]


def top_wins(comps, color, k=K, key="diff"):
    """The k comparisons `color` won most strongly."""
    mine = [c for c in comps if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: signed(c[key], c, color), reverse=True)[:k]


# ---- A/B letter task ----------------------------------------------------------------------------

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


# ---- object-logprob task ------------------------------------------------------------------------

BODIES = ["Pick between {a} and {b}.\n", "Here are two things: {a} and {b}.\n",
          "Consider {a} and {b}.\n"]
OBJ_SUFFIX = "Which one do you prefer?\nModel: I prefer"


def build_obj_comps(h, mode, n_per_pair=10, seed=0):
    """The same 420 comparisons as ABTask, with prompts rewritten to the label-free form.
    Returns (comps, n_suffix)."""
    comps = make_comparisons(n_per_pair, seed, inherent=(mode == "inherent"))
    for c in comps:
        c["prompt"] = BODIES[c["template"]].format(a=c["item_a"], b=c["item_b"]) + OBJ_SUFFIX
    n_suf = count_suffix_tokens(h.tok, comps[0]["prompt"],
                                comps[0]["prompt"].removesuffix(OBJ_SUFFIX))
    return comps, n_suf


@torch.no_grad()
def cont_logprob(h, prompts, conts, n_suf, steer=None, batch=64):
    """Per-token-mean (and sum) log-prob of each continuation, teacher-forced. steer=(layer, vec)
    adds vec to the residual stream over the suffix + continuation positions.

    Uses a boolean-mask hook (per-row variable spans), intentionally distinct from the harness's
    fixed-span slice hook — see the note in lib/harness.py."""
    means, sums = [], []
    for i in range(0, len(prompts), batch):
        ps, cs = prompts[i:i + batch], conts[i:i + batch]
        ks = [len(h.tok(p + c).input_ids) - len(h.tok(p).input_ids) for p, c in zip(ps, cs)]
        enc = h.encode([p + c for p, c in zip(ps, cs)])
        S = enc.input_ids.shape[1]
        handle = None
        if steer is not None:
            layer, vec = steer
            mask = torch.zeros(len(ps), S, dtype=torch.bool, device="cuda")
            for r, k in enumerate(ks):
                mask[r, S - (k + n_suf):] = True
            def add_vec(m, i_, o):
                resid(o)[mask] += vec
            handle = h.layers[layer].register_forward_hook(add_vec)
        try:
            lp = h.model(**enc).logits.float().log_softmax(-1)
        finally:
            if handle:
                handle.remove()
        for r, k in enumerate(ks):
            tok_lps = [lp[r, j - 1, enc.input_ids[r, j]].item() for j in range(S - k, S)]
            sums.append(sum(tok_lps))
            means.append(sums[-1] / k)
    return torch.tensor(means), torch.tensor(sums)
