#!/usr/bin/env python3
"""Palette validator: the computable checks behind CHARTS.md (run, don't eyeball).

Checks each color's OKLCH lightness band, chroma floor, and contrast vs surface,
then pairwise ΔE (OKLab x100) under normal vision (floor 15) and under
protanopia/deuteranopia at severity 1.0 (target 8, floor 6 — floor legal only
with secondary encoding such as direct labels or a legend).

Usage:
    python scripts/validate_palette.py                # validates chartstyle.MODEL_COLORS
    python scripts/validate_palette.py "#aaa,#bbb"    # ad-hoc hex list
    ... [--pairs adj|all] [--surface "#fcfcfb"]
"""
import argparse
import itertools
import math
import sys
from pathlib import Path


def _srgb_to_lin(c):
    c /= 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _hex_to_lin(h):
    h = h.lstrip("#")
    return [_srgb_to_lin(int(h[i:i + 2], 16)) for i in (0, 2, 4)]


def _oklab(lin):
    r, g, b = lin
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l, m, s = l ** (1 / 3), m ** (1 / 3), s ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = _oklab(_hex_to_lin(h))
    return L, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


# Machado, Oliveira & Fernandes 2009, severity 1.0, applied in linear RGB.
PROTAN = [[0.152286, 1.052583, -0.204868],
          [0.114503, 0.786281, 0.099216],
          [-0.003882, -0.048116, 1.051998]]
DEUTAN = [[0.367322, 0.860646, -0.227968],
          [0.280085, 0.672501, 0.047413],
          [-0.011820, 0.042940, 0.968881]]


def delta_e(h1, h2, sim=None):
    a, b = _hex_to_lin(h1), _hex_to_lin(h2)
    if sim:
        a = [max(0.0, min(1.0, sum(sim[i][j] * a[j] for j in range(3)))) for i in range(3)]
        b = [max(0.0, min(1.0, sum(sim[i][j] * b[j] for j in range(3)))) for i in range(3)]
    return 100 * math.dist(_oklab(a), _oklab(b))


def contrast(h1, h2):
    def lum(h):
        r, g, b = _hex_to_lin(h)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    hi, lo = sorted([lum(h1), lum(h2)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def validate(names, hexes, surface, pairs="all", band=(0.43, 0.77)):
    fail = False
    print(f"surface {surface}, L band {band}, pairs={pairs}")
    for n, h in zip(names, hexes):
        L, C, H = oklch(h)
        cr = contrast(h, surface)
        notes = []
        if not (band[0] - 0.005 <= L <= band[1] + 0.005):
            notes.append("L-band FAIL")
            fail = True
        if C < 0.095:
            notes.append("chroma FAIL")
            fail = True
        if cr < 3.0:
            notes.append(f"contrast WARN ({cr:.2f}:1 < 3:1 — needs labels/table relief)")
        print(f"  {n:12s} {h}  L={L:.3f} C={C:.3f} H={H:5.1f}  contrast={cr:.2f}:1  {' '.join(notes)}")
    idx = list(range(len(hexes)))
    plist = list(itertools.combinations(idx, 2)) if pairs == "all" else [(i, i + 1) for i in idx[:-1]]
    for kind, sim, floor, target in (("normal", None, 15, 15), ("protan", PROTAN, 6, 8),
                                     ("deutan", DEUTAN, 6, 8)):
        worst, wp = min(((delta_e(hexes[i], hexes[j], sim), (names[i], names[j]))
                         for i, j in plist), key=lambda t: t[0])
        status = "PASS" if worst >= target else ("WARN (needs secondary encoding)"
                                                 if worst >= floor else "FAIL")
        fail = fail or worst < floor
        print(f"  {kind:6s} worst pair dE={worst:5.1f}  {wp[0]} vs {wp[1]}  [{status}]")
    return not fail


def validate_ordinal(names, hexes, surface):
    """Ordinal-ramp checks: single hue, monotone lightness, adjacent ΔL >= 0.06,
    light-end contrast >= 2:1. (Categorical pairwise floors do not apply.)"""
    fail = False
    Ls, Hs = [], []
    for n, h in zip(names, hexes):
        L, C, H = oklch(h)
        Ls.append(L)
        Hs.append(H)
        print(f"  {n:12s} {h}  L={L:.3f} C={C:.3f} H={H:5.1f}")
    if max(Hs) - min(Hs) > 8:
        print(f"  hue spread {max(Hs)-min(Hs):.1f} deg > 8 — not a single hue: FAIL")
        fail = True
    for k in range(len(Ls) - 1):
        d = Ls[k] - Ls[k + 1]
        if d < 0.06 - 1e-9:
            print(f"  adjacent dL {names[k]}->{names[k+1]} = {d:.3f} < 0.06: FAIL")
            fail = True
    if not all(Ls[k] > Ls[k + 1] for k in range(len(Ls) - 1)):
        print("  lightness not monotone: FAIL")
        fail = True
    cr = contrast(hexes[0], surface)
    if cr < 2.0:
        print(f"  light-end contrast {cr:.2f}:1 < 2:1: FAIL")
        fail = True
    print(f"  ordinal checks {'FAIL' if fail else 'PASS'} "
          f"(adjacent dL {[round(Ls[k]-Ls[k+1],3) for k in range(len(Ls)-1)]}, "
          f"light-end {cr:.2f}:1)")
    return not fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hexes", nargs="?", help="comma-separated hex list; default: chartstyle palette")
    ap.add_argument("--pairs", choices=["adj", "all"], default="all")
    ap.add_argument("--surface", default=None)
    ap.add_argument("--ordinal", action="store_true",
                    help="run ordinal-ramp checks instead of the categorical six")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import chartstyle
    surface = args.surface or chartstyle.SURFACE
    if args.hexes:
        hexes = [h.strip() for h in args.hexes.split(",")]
        names = [f"color{i+1}" for i in range(len(hexes))]
        ok = (validate_ordinal(names, hexes, surface) if args.ordinal
              else validate(names, hexes, surface, pairs=args.pairs))
    else:
        # default: core roster categorically + the Qwen2.5 size ramp ordinally
        core = ["qwen3-4b", "llama31-8b", "qwen25-7b", "qwen25-32b"]
        print("== core roster (categorical six checks)")
        ok = validate(core, [chartstyle.MODEL_COLORS[m] for m in core], surface,
                      pairs=args.pairs)
        print("== Qwen2.5 size ramp (ordinal checks)")
        ramp = list(chartstyle.QWEN25_SIZES)  # smallest->largest = light->dark
        ok = validate_ordinal(ramp, [chartstyle.MODEL_COLORS[m] for m in ramp],
                              surface) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
