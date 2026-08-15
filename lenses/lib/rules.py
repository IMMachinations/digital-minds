"""r-lens LRP rules: a context manager that rewires the backward pass, leaving the forward
bit-identical (sanity `rules_forward` enforces torch.equal on the logits).

From "R-lens: making J-lens more faithful on early layers" (LessWrong,
nv8oedrnLXKRzNEL9): propagate LRP coefficients instead of raw gradients so relevance is
preserved layer to layer instead of accumulating error. For the dense RMSNorm+SwiGLU models
here that means three rules, all implemented as autograd-visible detaches:

  1. RMSNorm rule   — detach the rsqrt(variance + eps) factor (prevents relevance collapse
                      through the normalization denominator).
  2. Identity rule  — SiLU backward acts as sigmoid(z), dropping the z*sigmoid'(z) term.
  3. Half-rule      — the SwiGLU product a*u splits relevance evenly across both factors
                      (0.5*(a*u.detach() + a.detach()*u)) instead of double-counting.

Patches are per-instance (types.MethodType on each decoder layer's input_layernorm,
post_attention_layernorm, and mlp), so Qwen3's per-head q_norm/k_norm are naturally left
unmodified, matching the post. The final norm sits outside the fit graph and is irrelevant.

Forward-value notes (why torch.equal holds): the RMSNorm patch is the stock transformers
forward with only a .detach() inserted; the identity rule uses a straight-through form
silu(g).detach() + (t - t.detach()) with t = g*sigmoid(g).detach(), whose value is exactly
silu(g) and whose gradient w.r.t. g is sigmoid(g); the half-rule's two half-products are the
same rounded product twice, and x+x then *0.5 are exact in floating point.
"""
import types
from contextlib import contextmanager

import torch
import torch.nn.functional as F


def _rms_norm_forward(self, hidden_states):
    # transformers 5.15 LlamaRMSNorm.forward (shared by Qwen2/Qwen3), op-for-op, with the
    # normalization denominator detached.
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon).detach()
    return self.weight * hidden_states.to(input_dtype)


def _mlp_forward(self, x):
    g = self.gate_proj(x)
    u = self.up_proj(x)
    t = g * torch.sigmoid(g).detach()
    a = F.silu(g).detach() + (t - t.detach())        # value silu(g)+0; grad wrt g is sigmoid(g)
    prod = 0.5 * (a * u.detach() + a.detach() * u)   # value a*u; relevance split evenly
    return self.down_proj(prod)


@contextmanager
def rlens_rules(layout):
    """Within the context, backward passes through layout's decoder layers yield LRP
    coefficients; forward values are unchanged."""
    patched = []  # (module, original forward or None-if-class-default)
    try:
        for layer in layout.layers:
            for norm in (layer.input_layernorm, layer.post_attention_layernorm):
                patched.append((norm, norm.__dict__.get("forward")))
                norm.forward = types.MethodType(_rms_norm_forward, norm)
            mlp = layer.mlp
            patched.append((mlp, mlp.__dict__.get("forward")))
            mlp.forward = types.MethodType(_mlp_forward, mlp)
        yield
    finally:
        for module, original in patched:
            if original is None:
                del module.__dict__["forward"]
            else:
                module.forward = original
