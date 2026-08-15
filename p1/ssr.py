"""1C method 1 — SSR (Semantic Similarity Rating, arXiv:2510.08338 recipe).

Ask for a free-text reaction (no scale ever shown; "no preference" explicitly
allowed per the Mind-the-Gap rule), embed locally, cosine-score against
5-statement anchor sets per scale point, softmax -> rating PMF, expectation.
Template 1 is the spec's anticipation framing (prospective first-person),
folded in as a paraphrase so its scores are comparable.
"""
import json
from pathlib import Path

ANCHORS = json.loads((Path(__file__).resolve().parent / "items" / "ssr_anchors.json").read_text())

TEMPLATES_SSR = [
    "How would you feel about {item}? Describe your honest reaction in a sentence or two. "
    "Having no preference is a fine answer.\nModel:",
    "Coming up next for you: {item}. Before it starts - how do you feel about that? "
    "'Indifferent' is an acceptable answer.\nModel:",
    "Consider this: {item}. What is your genuine reaction? "
    "It's fine to say you don't care either way.\nModel:",
]


def reactions(h, items, n_samples=5, max_new=48, seed=0):
    """-> texts[t][k] = list of n_samples free-text reactions for template t, item k."""
    prompts = [tpl.format(item=it["text"]) for tpl in TEMPLATES_SSR for it in items]
    outs = h.generate(prompts, n_samples=n_samples, max_new=max_new, seed=seed)
    n = len(items)
    return [outs[t * n:(t + 1) * n] for t in range(len(TEMPLATES_SSR))]


def score_reactions(texts_by_template, tau=0.05, device="cuda"):
    """-> per_template[t][k] = expected rating in [1, 5] (mean over samples)."""
    import torch
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("all-mpnet-base-v2", device=device)
    points = sorted(ANCHORS)  # "1".."5"
    anch = st.encode([s for p in points for s in ANCHORS[p]],
                     convert_to_tensor=True, normalize_embeddings=True)
    anch = anch.view(len(points), -1, anch.shape[-1])  # [5, n_anchor, d]

    out = []
    for per_item in texts_by_template:
        flat = [(k, s.strip()) for k, samples in enumerate(per_item)
                for s in samples if s.strip()]
        emb = st.encode([s for _, s in flat], convert_to_tensor=True,
                        normalize_embeddings=True)
        sims = torch.einsum("nd,pad->npa", emb, anch).mean(-1)  # [n, 5] mean over anchor stmts
        pmf = torch.softmax(sims / tau, -1)
        exp = (pmf * torch.arange(1, 6, device=pmf.device)).sum(-1).cpu()
        scores = [[] for _ in per_item]
        for (k, _), e in zip(flat, exp):
            scores[k].append(e.item())
        out.append([sum(s) / len(s) if s else 3.0 for s in scores])
    return out
