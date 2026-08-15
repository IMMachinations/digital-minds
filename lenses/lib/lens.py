"""The Lens object: fitted per-layer Jacobians with save/load/merge/apply.

Readout at layer l: logits = unembed(J_l @ h_l), where unembed is the model's own final
RMSNorm + lm_head (see layout.Layout.unembed). apply(use_jacobian=False) skips the transport
and is exactly the vanilla logit lens — the free baseline. Interface follows the spirit of
anthropics/jacobian-lens (JacobianLens.apply/save/merge), reimplemented independently.
"""
import torch


class Lens:
    def __init__(self, *, jacobians, n_prompts, d_model, model_id, kind, config=None):
        self.jacobians = jacobians          # {layer_key: [d, d]}; EMBED (-1) = embedding output
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.model_id = model_id
        self.kind = kind                    # "j" | "r"
        self.config = config or {}
        self.source_layers = sorted(jacobians)

    def save(self, path, dtype=torch.float16):
        torch.save({"jacobians": {l: J.to(dtype).cpu() for l, J in self.jacobians.items()},
                    "n_prompts": self.n_prompts, "d_model": self.d_model,
                    "model_id": self.model_id, "kind": self.kind, "config": self.config}, path)
        print("wrote", path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=True)
        return cls(**d)

    def to(self, device):
        self.jacobians = {l: J.to(device) for l, J in self.jacobians.items()}
        return self

    @classmethod
    def merge(cls, lenses):
        """n_prompts-weighted mean of Jacobians fit on disjoint prompt subsets."""
        a = lenses[0]
        for b in lenses[1:]:
            assert (a.model_id, a.kind, a.source_layers) == (b.model_id, b.kind, b.source_layers)
        n = sum(l.n_prompts for l in lenses)
        jac = {layer: sum(l.jacobians[layer].float() * (l.n_prompts / n) for l in lenses)
               for layer in a.source_layers}
        return cls(jacobians=jac, n_prompts=n, d_model=a.d_model, model_id=a.model_id,
                   kind=a.kind, config=dict(a.config, merged_from=len(lenses)))

    @torch.no_grad()
    def apply(self, layout, prompt, *, positions=(-2,), use_jacobian=True, layers=None):
        """(lens_logits {layer: [n_pos, vocab] fp32 cpu}, model_logits [n_pos, vocab], ids).
        positions index into the tokenized prompt (negative ok). use_jacobian=False gives the
        vanilla logit lens readout of the same residuals."""
        assert layout.model_id == self.model_id, "lens/model mismatch"
        keys = list(layers) if layers is not None else self.source_layers
        ids = layout.encode(prompt)
        h = layout.capture(ids, layers=keys)
        pos = torch.tensor([p % ids.shape[1] for p in positions])
        out = {}
        for l in keys:
            r = h[l][0, pos, :].float()                      # [n_pos, d]
            if use_jacobian:
                J = self.jacobians[l].to(r.device).float()
                r = r @ J.T                                  # J @ h, batched over positions
            out[l] = layout.unembed(r)
        h_last = (h[layout.n_layers - 1] if layout.n_layers - 1 in h else
                  layout.capture(ids, layers=[layout.n_layers - 1])[layout.n_layers - 1])
        model_logits = layout.unembed(h_last[0, pos, :].float())
        return out, model_logits, ids.cpu()


def topk_tokens(tok, logits, k=5):
    """[(token_str, logit), ...] for each row of logits [n, vocab]."""
    vals, idxs = logits.topk(k, dim=-1)
    return [[(tok.decode([t]), round(v.item(), 3)) for t, v in zip(ti, vi)]
            for ti, vi in zip(idxs, vals)]
