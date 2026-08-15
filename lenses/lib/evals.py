"""Per-layer readouts and the two evals: multihop top-k tables and top-1 agreement curves.

Agreement = fraction of held-out prompts where the lens's top-1 token at layer l (position -2)
matches the model's own final top-1 at that position. The logit-lens baseline is the same
readout with the Jacobian transport skipped.
"""
import torch

from .lens import topk_tokens

# The first prompt is the multihop example from the anthropics/jacobian-lens README; the rest
# are analogous 2-hop facts (attribute of an entity that is itself only described).
MULTIHOP = [
    "Fact: The currency used in the country shaped like a boot is",
    "Fact: The capital of the country where the Eiffel Tower stands is",
    "Fact: The language spoken in the city famous for the Colosseum is",
    "Fact: The continent containing the river that flows through Cairo is",
    "Fact: The official currency of the country whose capital is Tokyo is",
    "Fact: The mountain range separating Spain from the country famous for croissants is",
]


def readout_modes(layout, lenses, prompt, position=-2, k=5):
    """{mode: {layer: [(tok, logit) x k]}} for mode in {"logit"} + lenses keys, plus the
    model's own top-k. lenses = {"j": Lens, "r": Lens} (any subset)."""
    any_lens = next(iter(lenses.values()))
    out = {}
    for mode in ["logit"] + sorted(lenses):
        lens = lenses.get(mode, any_lens)
        lens_logits, model_logits, _ = lens.apply(
            layout, prompt, positions=(position,), use_jacobian=(mode != "logit"))
        out[mode] = {l: topk_tokens(layout.tok, lg, k)[0] for l, lg in lens_logits.items()}
    out["model"] = topk_tokens(layout.tok, model_logits, k)[0]
    return out


@torch.no_grad()
def agreement(layout, lenses, prompts, position=-2):
    """{mode: {layer: fraction}} top-1 agreement with the model's final top-1."""
    modes = ["logit"] + sorted(lenses)
    any_lens = next(iter(lenses.values()))
    layers = any_lens.source_layers
    hits = {m: {l: 0 for l in layers} for m in modes}
    for prompt in prompts:
        ids = layout.encode(prompt)
        h = layout.capture(ids)
        pos = position % ids.shape[1]
        model_top = layout.unembed(h[layout.n_layers - 1][0, pos:pos + 1, :].float()).argmax(-1)
        for m in modes:
            for l in layers:
                r = h[l][0, pos:pos + 1, :].float()
                if m != "logit":
                    J = lenses[m].jacobians[l].to(r.device).float()
                    r = r @ J.T
                if layout.unembed(r).argmax(-1).item() == model_top.item():
                    hits[m][l] += 1
    n = len(prompts)
    return {m: {l: hits[m][l] / n for l in layers} for m in modes}
