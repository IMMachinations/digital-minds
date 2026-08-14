"""The A/B-free object-logprob task: items appear inline, preference is the teacher-forced
log-prob of each item as the continuation of "... Model: I prefer"."""
import torch

from .data import make_comparisons
from .harness import count_suffix_tokens, resid

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
