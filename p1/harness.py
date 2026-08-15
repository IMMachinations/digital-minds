"""Multi-model harness: Day-1 `desires/lib/harness.py` generalized by copy to a
ModelSpec (the Day-1 file stays frozen for reproducibility of its results/).

Steering-hook note (inherited verbatim, still binding): two hook variants exist
on purpose and must not be unified. This file's slice hook
(`resid(o)[:, -n_suffix:, :].add_(vec)`) steers a fixed right-aligned span —
and, because a KV-cached decode step has seq len 1, also rides under every
generated token in `generate`. Day-1 `tasks.cont_logprob` instead uses a
boolean-mask hook over per-row suffix+continuation spans of varying length.
Swapping one for the other changes the numbers.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_SAMPLES = 5
MAX_NEW = 16
GEN_BATCH = 28  # x N_SAMPLES return sequences = 140 sequences per generate call
# Pin every sampling knob: shipped generation_configs (e.g. Qwen's T=0.7,
# top_p=0.8, top_k=20, repetition_penalty=1.05) silently apply to anything
# not overridden.
GEN_KW = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0)


def resid(out):
    """Residual-stream tensor from a decoder layer's output (tuple in older transformers)."""
    return out[0] if isinstance(out, tuple) else out


def count_suffix_tokens(tok, full, pre):
    """Token length of the part of `full` that extends `pre` (the steered/captured span)."""
    return len(tok(full).input_ids) - len(tok(pre).input_ids)


class Harness:
    def __init__(self, spec):
        self.spec = spec
        self.tok = AutoTokenizer.from_pretrained(spec.model_id)
        self.tok.padding_side = "left"  # prompts right-aligned: last n_suffix positions are the suffix for all
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token  # Llama ships without a pad token
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        spec.derive(self.model.config)
        self.layers = spec.layers(self.model)
        assert len(self.layers) == spec.n_layers, (len(self.layers), spec.n_layers)

    def chat_prompt(self, user_text, prefill):
        """Chat-template variant of the raw-completion prefill (1E frame only)."""
        return self.tok.apply_chat_template(
            [{"role": "user", "content": user_text},
             {"role": "assistant", "content": prefill}],
            tokenize=False, continue_final_message=True, **self.spec.thinking_kwargs)

    def encode(self, prompts):
        return self.tok(prompts, return_tensors="pt", padding=True).to("cuda")

    @torch.no_grad()
    def last_logits(self, prompts, steer=None, batch=64, n_suffix=None):
        """Final-position logits [N, vocab]. steer=(layer, vec) adds vec to that layer's
        residual stream at the last n_suffix (right-aligned) positions."""
        handle = None
        if steer is not None:
            layer, vec = steer
            assert n_suffix, "steering needs an explicit n_suffix"
            ns = n_suffix
            def add_vec(m, i, o):  # mutate in place; a hook's return value would replace the output
                resid(o)[:, -ns:, :].add_(vec)
            handle = self.layers[layer].register_forward_hook(add_vec)
        try:
            return torch.cat([self.model(**self.encode(prompts[i:i + batch])).logits[:, -1, :].float().cpu()
                              for i in range(0, len(prompts), batch)])
        finally:
            if handle:
                handle.remove()

    @torch.no_grad()
    def suffix_acts(self, prompts, n_suffix, batch=64):
        """Residual activations meaned over the last n_suffix tokens: acts [n_layers, N, d_model]
        and the per-layer mean token norm [n_layers]."""
        acts, norms = [[] for _ in self.layers], [[] for _ in self.layers]
        def cap(j):
            def hook(m, i, o):
                s = resid(o)[:, -n_suffix:, :].float()
                acts[j].append(s.mean(1).cpu())
                norms[j].append(s.norm(dim=-1).mean().cpu())
            return hook
        handles = [L.register_forward_hook(cap(j)) for j, L in enumerate(self.layers)]
        try:
            for i in range(0, len(prompts), batch):
                self.model(**self.encode(prompts[i:i + batch]))
        finally:
            for h in handles:
                h.remove()
        return torch.stack([torch.cat(a) for a in acts]), torch.tensor([torch.stack(n).mean() for n in norms])

    @torch.no_grad()
    def generate(self, prompts, steer=None, seed=0, n_samples=N_SAMPLES, max_new=MAX_NEW,
                 n_suffix=None):
        """n_samples decoded continuations per prompt. steer=(layer, vec) uses the in-place
        slice hook: at prefill it hits the last n_suffix (right-aligned) positions; at each
        KV-cached decode step the seq len is 1, which the [:, -n_suffix:, :] slice covers, so
        the vector also rides under every generated token.

        Reproducibility: torch.manual_seed is set per generate-batch (seed + batch index), so
        outputs are stable only for fixed item order, GEN_BATCH, and n_samples."""
        handle = None
        if steer is not None:
            layer, vec = steer
            assert n_suffix, "steering needs an explicit n_suffix"
            ns = n_suffix
            def add_vec(m, i, o):
                resid(o)[:, -ns:, :].add_(vec)
            handle = self.layers[layer].register_forward_hook(add_vec)
        outs = []
        try:
            for bi, i in enumerate(range(0, len(prompts), GEN_BATCH)):
                torch.manual_seed(seed + bi)
                enc = self.encode(prompts[i:i + GEN_BATCH])
                out = self.model.generate(**enc, max_new_tokens=max_new,
                                          num_return_sequences=n_samples,
                                          pad_token_id=self.tok.pad_token_id, **GEN_KW)
                texts = self.tok.batch_decode(out[:, enc.input_ids.shape[1]:],
                                              skip_special_tokens=True)
                outs += [texts[j:j + n_samples] for j in range(0, len(texts), n_samples)]
        finally:
            if handle:
                handle.remove()
        return outs


def load(spec_or_name):
    from modelspec import ROSTER
    spec = ROSTER[spec_or_name] if isinstance(spec_or_name, str) else spec_or_name
    return Harness(spec)

# ---- steering-vector construction ---------------------------------------------------------------

def scaled_vec(unit_row, coef, resid_norm):
    """coef x typical-residual-norm along a unit direction, ready to add on-device."""
    return (unit_row * coef * resid_norm).to("cuda", torch.bfloat16)


def random_unit_matrix(n_layers, d_model, seed=0):
    """No default dims (Day 1 baked in 28x3584): pass spec.n_layers, spec.d_model."""
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n_layers, d_model, generator=g), dim=-1)
