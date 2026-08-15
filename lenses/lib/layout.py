"""Model registry and the Layout wrapper: load, locate the residual stack, unembed.

The layer indexing convention used throughout lenses/: key -1 is the embedding output
(pseudo-layer), keys 0..n_layers-1 are decoder-layer outputs, and the "final" residual is the
output of the last decoder layer, *before* the final RMSNorm. `unembed` owns the final norm +
lm_head, so a lens readout at the last layer must reproduce the model's own logits exactly
(sanity check), and the fitted Jacobian at the last layer is the identity by construction.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "llama31-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
}

EMBED = -1  # pseudo-layer key for the embedding output


def resid(out):
    """Residual-stream tensor from a decoder layer's output (tuple in older transformers)."""
    return out[0] if isinstance(out, tuple) else out


class Layout:
    """References into a loaded HF causal LM (nothing is copied). All three registered models
    are Llama-family layouts: model.model.{embed_tokens,layers,norm} + model.lm_head, RMSNorm
    everywhere, SwiGLU MLPs with gate/up/down_proj. Parameters are frozen at load: the Jacobian
    fit needs grads only with respect to activations."""

    def __init__(self, short_name, device="cuda"):
        assert short_name in MODELS, f"unknown model {short_name!r}; one of {list(MODELS)}"
        self.short_name = short_name
        self.model_id = MODELS[short_name]
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16, device_map=device).eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        inner = self.model.model
        self.embed = inner.embed_tokens
        self.layers = inner.layers
        self.final_norm = inner.norm
        self.lm_head = self.model.lm_head
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size
        self.device = device
        for mlp_attr in ("gate_proj", "up_proj", "down_proj"):
            assert hasattr(self.layers[0].mlp, mlp_attr), f"unexpected MLP layout ({mlp_attr})"

    def encode(self, prompt, max_seq_len=None):
        ids = self.tok(prompt, return_tensors="pt").input_ids
        if max_seq_len is not None:
            ids = ids[:, :max_seq_len]
        return ids.to(self.device)

    @torch.no_grad()
    def unembed(self, h):
        """Logits [.., vocab] (float32, cpu) from residuals in the final-layer basis."""
        return self.lm_head(self.final_norm(h.to(self.device, torch.bfloat16))).float().cpu()

    def capture(self, ids, layers=None, grad=False):
        """Forward through the residual stack, returning {layer_key: residual [B, S, d]}.
        With grad=True the returned tensors are graph nodes usable as autograd.grad inputs
        (lm_head is never run: it sits outside the fit graph). All parameters are frozen, so
        in grad mode the embedding output is flipped to requires_grad in a hook — otherwise
        no activation would carry a graph at all."""
        keys = list(layers) if layers is not None else [EMBED] + list(range(self.n_layers))
        captured = {}

        def enable_grad_root(module, inputs, out):
            out.requires_grad_(True)

        def cap(key):
            def hook(module, inputs, out):
                captured[key] = resid(out)
            return hook

        handles = [self.embed.register_forward_hook(enable_grad_root)] if grad else []
        handles += [(self.embed if k == EMBED else self.layers[k]).register_forward_hook(cap(k))
                    for k in keys]
        try:
            if grad:
                with torch.enable_grad():
                    self.model.model(input_ids=ids)
            else:
                with torch.no_grad():
                    self.model.model(input_ids=ids)
        finally:
            for h in handles:
                h.remove()
        return captured


def load(short_name):
    return Layout(short_name)
