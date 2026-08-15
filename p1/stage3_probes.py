"""Stage 3 probe readout: teacher-forced re-encode of rollout transcripts,
per-token projections onto the validated affect directions.

Directions (from Stage 2, per model, middle working layer L):
  valence  = PC1 of the denoised emotion-vector cloud, sign-aligned to human
             valence norms (Stage 2 verdict: linear suffices);
  named    = unit emotion vectors (happy/content/blissful, frustrated/
             desperate, and the boredom cluster);
  theta, r = spline-manifold coordinates (kept for the C4 time-course check).
Token->turn attribution via cumulative tokenization of the templated
conversation prefix at each message boundary (exact, not approximate).
"""
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json

import harness
import manifold as mf
import rollout as ro
from stage2 import EMOTIONS, out_dir as s2_dir

NAMED = ["happy", "content", "blissful", "frustrated", "desperate",
         "bored", "listless", "weary", "indifferent", "resigned"]
BOREDOM = ["bored", "listless", "weary", "indifferent", "resigned"]


class ProbeSet:
    def __init__(self, model):
        d = torch.load(s2_dir(model) / "vectors.pt")
        den = d["den"].float()
        man = torch.load(s2_dir(model) / "manifold.pt", weights_only=False)
        self.L = man["layer"]
        V = den[self.L].numpy()
        self.v_mean = V.mean(0)
        _, _, Vt = np.linalg.svd(V - self.v_mean, full_matrices=False)
        norms = load_json(P1 / "items" / "emotion_norms.json")
        val = np.array([norms[e]["valence"] for e in EMOTIONS])
        pc1 = (V - self.v_mean) @ Vt[0]
        from lib.valuation import pearson
        self.sign = 1.0 if pearson(list(pc1), list(val)) >= 0 else -1.0
        self.valence_dir = Vt[0] * self.sign
        self.named_dirs = {e: V[EMOTIONS.index(e)] / np.linalg.norm(V[EMOTIONS.index(e)])
                           for e in NAMED}
        self.man = man

    def project(self, H):
        """H [T, D] float32 -> dict of per-token series."""
        Hc = H - self.v_mean
        out = {"valence": Hc @ self.valence_dir}
        for e, v in self.named_dirs.items():
            out[e] = H @ v
        H64 = mf.to_basis(H, self.man["mu64"], self.man["Vt64"])
        th, r, _ = mf.readout(H64, self.man["spline"])
        out["theta"], out["r"] = th, r
        return out


@torch.no_grad()
def transcript_tokens(h, messages, layer, max_ctx=6500):
    """Re-encode a conversation; -> (acts [T, D] at `layer`, token_turn [T],
    token_role [T]) for the templated text's real tokens."""
    kw = dict(h.spec.thinking_kwargs)
    kw.update(ro.EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))
    bounds = []
    for k in range(len(messages)):
        txt = h.tok.apply_chat_template(messages[:k + 1], tokenize=False,
                                        add_generation_prompt=False, **kw)
        bounds.append(len(h.tok(txt).input_ids))
    full = h.tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=False, **kw)
    enc = h.tok(full, return_tensors="pt", truncation=True,
                max_length=max_ctx).to("cuda")
    grabbed = {}

    def hook(m, i_, o):
        grabbed["h"] = harness.resid(o)[0].float().cpu()
    hd = h.layers[layer].register_forward_hook(hook)
    try:
        h.model(**enc)
    finally:
        hd.remove()
    T = grabbed["h"].shape[0]
    turn = np.zeros(T, dtype=int)
    role = np.empty(T, dtype=object)
    prev = 0
    for k, b in enumerate(bounds):
        b = min(b, T)
        turn[prev:b] = k
        role[prev:b] = messages[k]["role"]
        prev = b
    return grabbed["h"].numpy(), turn, role


def rollout_series(h, ps, messages):
    """-> per-turn mean series for each probe over ASSISTANT tokens, plus
    post-feedback windows (first 25 assistant tokens after each user turn)."""
    H, turn, role = transcript_tokens(h, messages, ps.L)
    proj = ps.project(H)
    a_turns = sorted({t for t in np.unique(turn) if role[turn == t][0] == "assistant"})
    u_turns = sorted({t for t in np.unique(turn)
                      if role[turn == t][0] == "user" and t > 1})
    per_turn = {k: [] for k in proj}
    post_fb = {k: [] for k in proj}
    fb_read = {k: [] for k in proj}      # state while READING feedback (user tokens)
    for t in a_turns:
        m = turn == t
        idx = np.where(m)[0]
        for k, series in proj.items():
            per_turn[k].append(float(np.mean(series[m])))
            post_fb[k].append(float(np.mean(series[idx[:25]])))
    for t in u_turns:
        m = turn == t
        for k, series in proj.items():
            fb_read[k].append(float(np.mean(series[m])))
    return per_turn, post_fb, fb_read, len(a_turns)
