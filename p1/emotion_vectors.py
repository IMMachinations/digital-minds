"""Emotion-vector extraction (paper recipe, left-padding-safe).

story_acts: all-layer residual means over real tokens from the `skip`-th
onward (per-row boolean masks like cont_logprob's, NOT the right-aligned
slice — stories differ in length and are left-padded).
build_vectors: v_e = mean(stories of e) - mean(other emotions' means), then
denoise by projecting out the top PCs of neutral-story activations (to 50%
variance). Raw vectors kept alongside (paper: findings survive undenoised).
"""
import numpy as np
import torch

import harness


@torch.no_grad()
def story_acts(h, texts, skip=50, batch=16, layers=None, per_token=False):
    """-> [n_layers, N, d_model] float32 CPU (story-mean); with per_token=True
    returns list per text of [n_sel_layers, T, d] (real tokens only)."""
    sel = layers if layers is not None else range(len(h.layers))
    out_means = [[] for _ in sel]
    out_tokens = []
    for i in range(0, len(texts), batch):
        enc = h.encode(texts[i:i + batch])
        am = enc.attention_mask
        first = am.float().argmax(1)
        real = am.sum(1)
        start = torch.minimum(torch.full_like(real, skip), real // 2)
        pos = torch.arange(am.shape[1], device=am.device).unsqueeze(0)
        mask = (am == 1) & (pos - first.unsqueeze(1) >= start.unsqueeze(1))
        grabbed = {}

        def cap(slot, layer_idx):
            def hook(m, i_, o):
                grabbed[slot] = harness.resid(o).float()
            return hook
        handles = [h.layers[L].register_forward_hook(cap(s, L)) for s, L in enumerate(sel)]
        try:
            h.model(**enc)
        finally:
            for hd in handles:
                hd.remove()
        cnt = mask.sum(1, keepdim=True).clamp(min=1).float()
        for s in range(len(list(sel))):
            g = grabbed[s]
            out_means[s].append(((g * mask.unsqueeze(-1)).sum(1) / cnt).cpu())
        if per_token:
            for r in range(am.shape[0]):
                rows = [grabbed[s][r, am[r] == 1, :].cpu() for s in range(len(list(sel)))]
                out_tokens.append(torch.stack(rows))
        del grabbed
    means = torch.stack([torch.cat(x) for x in out_means])
    return (means, out_tokens) if per_token else means


def _pca_components(X, var_frac=0.5):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    frac = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = int(np.searchsorted(frac, var_frac)) + 1
    return Vt[:k]


def build_vectors(acts, labels, neutral_acts, emotions, var_frac=0.5):
    """acts [L,N,D], labels: emotion per story -> (raw [L,E,D], denoised [L,E,D])."""
    L, N, D = acts.shape
    E = len(emotions)
    idx = {e: [k for k, l in enumerate(labels) if l == e] for e in emotions}
    raw = torch.zeros(L, E, D)
    for l in range(L):
        A = acts[l].numpy()
        means = np.stack([A[idx[e]].mean(0) for e in emotions])
        for ei in range(E):
            others = np.delete(means, ei, axis=0).mean(0)
            raw[l, ei] = torch.tensor(means[ei] - others)
    den = raw.clone()
    for l in range(L):
        P = _pca_components(neutral_acts[l].numpy(), var_frac)  # [k, D]
        V = den[l].numpy()
        den[l] = torch.tensor(V - (V @ P.T) @ P)
    return raw, den
