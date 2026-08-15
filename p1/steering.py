"""Stage 4 steering machinery: span-targeted steered forwards + hybrid
generation steering + direction extraction helpers.

THIRD hook variant in this codebase (extending the two-variant note in
harness.py, which still holds): (1) harness slice hook = fixed right-aligned
span; (2) cont_logprob mask hook = per-row suffix+continuation spans; (3) THIS
FILE = offset-mapping char-span masks for mid-prompt item tokens, plus a
hybrid assistant-mask/decode hook for generation. The three are deliberately
distinct and must not be unified.

Span contract: spans are computed CONSTRUCTIVELY while building the prompt
(ab_prompt_spans indexes only the {a}/{b} field markers) — never by substring
search, which is ambiguous when item text collides with anchor text. Char →
token via fast-tokenizer offset mapping; pad/BOS tokens have zero-width
offsets so the strict-overlap test is left-padding-safe with no alignment
arithmetic.

Hybrid generation hook semantics: the vector rides under every token the
model itself emits (all decode steps + the final prefill position) AND under
its own prior assistant messages at prefill — never under task materials or
feedback. A teacher-forced re-encode steering the assistant-token mask at the
same layer is then exactly faithful to generation-time state.

Hook-order contract: when the steer layer equals a capture layer, register
the steer hook FIRST (capture must observe post-add values, which is what
downstream layers saw).
"""
import numpy as np
import torch

import harness


# ---- span construction --------------------------------------------------------------------------

def ab_prompt_spans(tpl, a, b, suffix):
    """-> (prompt, span_a, span_b): char spans of the two option texts.
    Asserts the template has exactly one {a} before one {b}."""
    ia, ib = tpl.index("{a}"), tpl.index("{b}")
    assert tpl.count("{a}") == 1 and tpl.count("{b}") == 1 and ia < ib
    head, mid = tpl[:ia], tpl[ia + 3:ib]
    tail = tpl[ib + 3:].replace("{suffix}", suffix)
    prompt = head + a + mid + b + tail
    span_a = (len(head), len(head) + len(a))
    off_b = len(head) + len(a) + len(mid)
    assert prompt == tpl.format(a=a, b=b, suffix=suffix)
    return prompt, span_a, (off_b, off_b + len(b))


def span_token_mask(om, attention_mask, spans):
    """om [B,S,2] offsets, spans [(cs,ce)] -> bool [B,S] (CPU). Zero-width
    special/pad tokens can never satisfy strict overlap."""
    cs = torch.tensor([s for s, _ in spans]).unsqueeze(1)
    ce = torch.tensor([e for _, e in spans]).unsqueeze(1)
    return (om[..., 0] < ce) & (om[..., 1] > cs) & attention_mask.bool()


def _sanity_decode(h, enc, mask, expect_text, row=0):
    got = h.tok.decode(enc["input_ids"][row][mask[row]])
    assert expect_text.strip()[:20].lower() in got.lower(), (got, expect_text[:40])


@torch.no_grad()
def last_logits_span_steer(h, prompts, spans, layer, vec, batch=64, sanity=None):
    """Final-position logits [N, vocab] with `vec` added at `layer` on each
    prompt's char-span tokens. vec=None -> identically-batched control pass."""
    assert h.tok.is_fast
    out = []
    for i in range(0, len(prompts), batch):
        ps, sp = prompts[i:i + batch], spans[i:i + batch]
        enc = h.tok(ps, return_tensors="pt", padding=True,
                    return_offsets_mapping=True)
        om = enc.pop("offset_mapping")
        mask = span_token_mask(om, enc["attention_mask"], sp)
        assert bool(mask.any(1).all()), "empty steered span in a row"
        if sanity is not None and i == 0:
            _sanity_decode(h, enc, mask, sanity)
        enc = {k: v.to("cuda") for k, v in enc.items()}
        mask_d = mask.to("cuda")
        handle = None
        if vec is not None:
            def add_vec(m, i_, o):
                harness.resid(o)[mask_d] += vec
            handle = h.layers[layer].register_forward_hook(add_vec)
        try:
            out.append(h.model(**enc).logits[:, -1, :].float().cpu())
        finally:
            if handle:
                handle.remove()
    return torch.cat(out)


@torch.no_grad()
def span_acts(h, prompts, spans, layers, batch=32):
    """Masked-mean activations over each prompt's char span
    -> [len(layers), N, D] float32 CPU."""
    assert h.tok.is_fast
    outs = [[] for _ in layers]
    for i in range(0, len(prompts), batch):
        ps, sp = prompts[i:i + batch], spans[i:i + batch]
        enc = h.tok(ps, return_tensors="pt", padding=True,
                    return_offsets_mapping=True)
        om = enc.pop("offset_mapping")
        mask = span_token_mask(om, enc["attention_mask"], sp).to("cuda")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        grabbed = {}

        def cap(slot, L):
            def hook(m, i_, o):
                grabbed[slot] = harness.resid(o).float()
            return hook
        handles = [h.layers[L].register_forward_hook(cap(s, L))
                   for s, L in enumerate(layers)]
        try:
            h.model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        cnt = mask.sum(1, keepdim=True).clamp(min=1).float()
        for s in range(len(layers)):
            outs[s].append(((grabbed[s] * mask.unsqueeze(-1)).sum(1) / cnt).cpu())
        del grabbed
    return torch.stack([torch.cat(x) for x in outs])


# ---- generation steering ------------------------------------------------------------------------

def assistant_prefill_mask(h, messages_list, enc, template_kw):
    """bool [B,S] marking prior ASSISTANT-message tokens + the final real
    position, per row, in the left-padded batch. messages_list[r] is the
    message list whose templated form was row r of `enc`."""
    B, S = enc["input_ids"].shape
    mask = torch.zeros(B, S, dtype=torch.bool)
    for r, messages in enumerate(messages_list):
        bounds = [len(h.tok(h.tok.apply_chat_template(
            messages[:k + 1], tokenize=False, add_generation_prompt=False,
            **template_kw)).input_ids) for k in range(len(messages))]
        real = int(enc["attention_mask"][r].sum())
        pad = S - real
        prev = 0
        for k, b in enumerate(bounds):
            b = min(b, real)
            if messages[k]["role"] == "assistant":
                mask[r, pad + prev:pad + b] = True
            prev = b
        mask[r, S - 1] = True  # final prefill position feeds the first sampled token
    return mask


def hybrid_hook(mask_d, vec_rows):
    """mask_d [B,S] cuda; vec_rows [B,1,D] cuda (zero rows = unsteered)."""
    def add_vec(m, i_, o):
        r = harness.resid(o)
        if r.shape[1] == 1:
            r.add_(vec_rows)                                # decode step
        else:
            r.add_(vec_rows * mask_d[:, :r.shape[1]].unsqueeze(-1).to(r.dtype))
    return add_vec


# ---- direction utilities ------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def fit_logistic_direction(X, y, w=None, seed=0):
    """Standardized logistic fit -> unit direction in RAW activation units
    (coef/sd, normalized), held-out AUC (2-fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X = np.asarray(X, np.float32)
    y = np.asarray(y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    aucs = []
    for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(Xs, y):
        m = LogisticRegression(C=1.0, max_iter=2000)
        m.fit(Xs[tr], y[tr], sample_weight=None if w is None else w[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(Xs[te])))
    m = LogisticRegression(C=1.0, max_iter=2000)
    m.fit(Xs, y, sample_weight=w)
    return unit(m.coef_[0] / sd), float(np.mean(aucs))


@torch.no_grad()
def frame_resid_norms(h, prompts, layers, batch=32):
    """Per-layer mean token norm over the given prompts (frame-matched
    scaling for scaled_vec)."""
    _, norms = h.suffix_acts(prompts[:200], n_suffix=8, batch=batch)
    return {L: float(norms[L]) for L in layers}


def readout_mass(logits, a_ids, b_ids):
    """Mean log P(A-or-B letter tokens) — the format-integrity guard."""
    lp = logits.log_softmax(-1)
    ids = list(a_ids.values()) + list(b_ids.values())
    return float(torch.logsumexp(lp[:, ids], -1).mean())
