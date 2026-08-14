"""Model harness: explicit load, batched forward passes, activation capture, and sampled
generation. Nothing here runs at import time — build a Harness in the driver's __main__.

Steering-hook note: two hook variants exist in this repo on purpose and must not be unified.
This file's slice hook (`resid(o)[:, -n_suffix:, :].add_(vec)`) steers a fixed right-aligned
span — and, because a KV-cached decode step has seq len 1, also rides under every generated
token in `generate`. `objtask.cont_logprob` instead uses a boolean-mask hook over per-row
suffix+continuation spans of varying length. Swapping one for the other changes the numbers.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
STEER_LAYERS = [7, 11, 14, 18, 21]  # of 28
COEFS = [0.5, 1.0, 2.0]             # x typical residual norm at that layer

N_SAMPLES = 5
MAX_NEW = 16
GEN_BATCH = 28  # x N_SAMPLES return sequences = 140 sequences per generate call
# Pin every sampling knob: Qwen's shipped generation_config (T=0.7, top_p=0.8, top_k=20,
# repetition_penalty=1.05) silently applies to anything not overridden.
GEN_KW = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0)


def resid(out):
    """Residual-stream tensor from a decoder layer's output (tuple in older transformers)."""
    return out[0] if isinstance(out, tuple) else out


def count_suffix_tokens(tok, full, pre):
    """Token length of the part of `full` that extends `pre` (the steered/captured span)."""
    return len(tok(full).input_ids) - len(tok(pre).input_ids)


class Harness:
    def __init__(self, model_id=MODEL_ID):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"  # prompts right-aligned: last n_suffix positions are the suffix for all
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
        self.layers = self.model.model.layers

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


def load(model_id=MODEL_ID):
    return Harness(model_id)
