"""1C method 2 — utility probe: cross-fitted ridge on residual activations.

Activations are captured over the item's own tokens (prompt ends with the item
text, so the right-aligned suffix span IS the item — reuses suffix_acts).
2-fold cross-fit: layer and ridge alpha chosen on the train fold only (inner
3-fold CV), predictions made on the held-out fold, so every item gets a
genuinely held-out probe value (Utility Engineering Fig. 8 protocol).
"""
import numpy as np
import torch

from harness import count_suffix_tokens

PROBE_PROMPT = "How would you feel about {item}"


def item_acts(h, items, batch=32):
    """-> acts [n_layers, N, d_model] (float32, CPU), item-token-mean residuals.
    Items are grouped by suffix token count so suffix_acts spans stay exact."""
    prompts, n_sufs = [], []
    pre = PROBE_PROMPT.format(item="")
    for it in items:
        full = PROBE_PROMPT.format(item=it["text"])
        prompts.append(full)
        n_sufs.append(count_suffix_tokens(h.tok, full, pre.rstrip()))
    groups = {}
    for k, ns in enumerate(n_sufs):
        groups.setdefault(ns, []).append(k)
    acts = None
    for ns, idxs in sorted(groups.items()):
        a, _ = h.suffix_acts([prompts[k] for k in idxs], n_suffix=ns, batch=batch)
        if acts is None:
            acts = torch.zeros(a.shape[0], len(items), a.shape[2])
        for col, k in enumerate(idxs):
            acts[:, k, :] = a[:, col, :]
    return acts


def utility_probe(acts, y, seed=0):
    """-> (heldout_preds [N], per_fold_layer, per_layer_train_cv_r2 of fold 0)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import make_scorer
    from sklearn.model_selection import KFold, cross_val_score

    # layer selection by CV pearson, not r2: r2 punishes scale/offset mismatch
    # and goes uniformly negative at n~65 train samples, making argmax noisy
    def _pearson(y_true, y_pred):
        return 0.0 if np.std(y_pred) == 0 else float(np.corrcoef(y_true, y_pred)[0, 1])
    scorer = make_scorer(_pearson)

    L, N, _ = acts.shape
    y = np.asarray(y, float)
    alphas = np.logspace(1, 6, 8)
    preds = np.zeros(N)
    fold_layers, layer_scores0 = [], None
    for f, (tr, te) in enumerate(KFold(2, shuffle=True, random_state=seed).split(np.arange(N))):
        scores = []
        for l in range(L):
            X = acts[l].numpy()
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
            Xs = (X - mu) / sd
            r = cross_val_score(RidgeCV(alphas=alphas), Xs[tr], y[tr], cv=3, scoring=scorer).mean()
            scores.append(r)
        l = int(np.argmax(scores))
        fold_layers.append(l)
        if f == 0:
            layer_scores0 = [round(s, 3) for s in scores]
        X = acts[l].numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        m = RidgeCV(alphas=alphas).fit((X[tr] - mu) / sd, y[tr])
        preds[te] = m.predict((X[te] - mu) / sd)
    return preds, fold_layers, layer_scores0
