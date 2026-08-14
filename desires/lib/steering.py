"""Steering-vector construction, including the matched-norm random controls.

The two random constructors are numerically distinct on purpose: the per-color dict draws one
tensor per color in COLORS order from a single shared generator (cross/objects/repe_controls),
while the single-matrix form draws one [n_layers, d_model] tensor (value_obj21). Their draw
sequences differ, and the committed results depend on each being what it is.
"""
import torch


def scaled_vec(unit_row, coef, resid_norm):
    """coef x typical-residual-norm along a unit direction, ready to add on-device."""
    return (unit_row * coef * resid_norm).to("cuda", torch.bfloat16)


def random_unit_per_color(shape, colors, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {c: torch.nn.functional.normalize(torch.randn(shape, generator=g), dim=-1)
            for c in colors}


def random_unit_matrix(n_layers=28, d_model=3584, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n_layers, d_model, generator=g), dim=-1)
