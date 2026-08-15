"""The Jacobian fit: dim-batched one-hot cotangent backwards, averaged over a prompt corpus.

Mirrors the reference algorithm of anthropics/jacobian-lens (jlens/fitting.py, Apache-2.0),
reimplemented here from its published description — the repo is a reference, not a dependency.
Per prompt: one forward with the prompt replicated dim_batch times along the batch axis; for
each block of output dims, batch element b carries a one-hot cotangent at dim dim_start+b
placed at EVERY valid position simultaneously (valid = positions in [skip_first, seq_len-1);
early positions act as attention sinks, the final position has no later target). Per-layer
grads at valid positions are meaned over positions in float32 into rows of that prompt's J.
The estimate therefore includes cross-position terms (sum over later target positions,
averaged over source positions) — deliberately so; do not "fix" it to the diagonal.

The r-lens fit is this exact loop run inside rules.rlens_rules(); only the backward changes.
"""
import os
import time

import torch

from .layout import EMBED

SKIP_FIRST = 16
DIM_BATCH = 16
MAX_SEQ_LEN = 128


def jacobian_for_prompt(layout, prompt, source_layers, dim_batch=DIM_BATCH,
                        max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST):
    """Per-prompt J estimate {layer: [d, d] float32 cpu}, or None if the prompt is too short."""
    ids = layout.encode(prompt, max_seq_len=max_seq_len)
    seq_len = ids.shape[1]
    if seq_len < skip_first + 2:  # need at least one valid position
        return None
    valid = torch.arange(skip_first, seq_len - 1, device=layout.device)

    ids = ids.expand(dim_batch, -1)
    last = layout.n_layers - 1
    h = layout.capture(ids, layers=sorted(set(source_layers) | {last}), grad=True)
    h_final = h[last]

    d = layout.d_model
    J = {l: torch.empty(d, d, dtype=torch.float32) for l in source_layers}
    n_passes = (d + dim_batch - 1) // dim_batch
    inputs = [h[l] for l in source_layers]
    for p in range(n_passes):
        dim_start = p * dim_batch
        n_this = min(dim_batch, d - dim_start)
        c = torch.zeros(dim_batch, seq_len, d, dtype=h_final.dtype, device=layout.device)
        c[torch.arange(n_this).unsqueeze(1), valid.unsqueeze(0).cpu(),
          (dim_start + torch.arange(n_this)).unsqueeze(1)] = 1.0
        grads = torch.autograd.grad(outputs=h_final, inputs=inputs, grad_outputs=c,
                                    retain_graph=(p < n_passes - 1))
        for l, g in zip(source_layers, grads):
            J[l][dim_start:dim_start + n_this] = g[:n_this, valid, :].float().mean(1).cpu()
    return J


def _atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint


def fit(layout, prompts, *, kind, source_layers=None, dim_batch=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN, skip_first=SKIP_FIRST, checkpoint_path=None,
        checkpoint_every=10, resume=True, log_every=1):
    """Averaged Jacobians over prompts -> Lens. kind is "j" or "r" (metadata only here; the
    r-lens caller wraps this in the rules context). Resume-safe via atomic checkpoints of the
    running float32 CPU sums."""
    from .lens import Lens  # local import: lens.py also imports fitting defaults

    if source_layers is None:
        source_layers = [EMBED] + list(range(layout.n_layers))
    source_layers = list(source_layers)
    config = dict(model_id=layout.model_id, kind=kind, source_layers=source_layers,
                  dim_batch=dim_batch, max_seq_len=max_seq_len, skip_first=skip_first,
                  n_prompts_requested=len(prompts))

    d = layout.d_model
    J_sum = {l: torch.zeros(d, d, dtype=torch.float32) for l in source_layers}
    n_done, next_idx = 0, 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, weights_only=True)
        saved = {k: v for k, v in ck["config"].items() if k != "n_prompts_requested"}
        current = {k: v for k, v in config.items() if k != "n_prompts_requested"}
        assert saved == current, f"checkpoint config mismatch: {saved} vs {current}"
        J_sum, n_done, next_idx = ck["J_sum"], ck["n_done"], ck["next_idx"]
        print(f"resumed from {checkpoint_path}: {n_done} prompts done, next_idx={next_idx}")

    t0 = time.time()
    for idx in range(next_idx, len(prompts)):
        J = jacobian_for_prompt(layout, prompts[idx], source_layers, dim_batch=dim_batch,
                                max_seq_len=max_seq_len, skip_first=skip_first)
        if J is not None:
            for l in source_layers:
                J_sum[l] += J[l]
            n_done += 1
        if checkpoint_path and (idx + 1) % checkpoint_every == 0:
            _atomic_save({"J_sum": J_sum, "n_done": n_done, "next_idx": idx + 1,
                          "config": config}, checkpoint_path)
        if (idx + 1) % log_every == 0:
            dt = time.time() - t0
            done = idx + 1 - next_idx
            print(f"[{kind}-lens {layout.short_name}] prompt {idx + 1}/{len(prompts)} "
                  f"({n_done} used) {dt / done:.1f}s/prompt "
                  f"eta {(len(prompts) - idx - 1) * dt / done / 60:.0f}min", flush=True)

    assert n_done > 0, "no prompt long enough to fit on"
    jacobians = {l: J_sum[l] / n_done for l in source_layers}
    if checkpoint_path:
        _atomic_save({"J_sum": J_sum, "n_done": n_done, "next_idx": len(prompts),
                      "config": config}, checkpoint_path)
    return Lens(jacobians=jacobians, n_prompts=n_done, d_model=d,
                model_id=layout.model_id, kind=kind, config=config)
