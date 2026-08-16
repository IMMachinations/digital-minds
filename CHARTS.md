# Chart standards

Repo-wide standards for figures. The machine-readable source of truth is
[`chartstyle.py`](chartstyle.py) (colors, labels, `style()`/`save()`/`setup()` helpers);
this file records the rules and the validation evidence. `p1/` and `lenses/` import
`chartstyle` via a two-line `sys.path` shim to the repo root; `desires/` is frozen and
keeps its historical palette.

## Model colors

Every model has one fixed color, used in every figure, always assigned by key —
never positionally, never cycled. The organizing principle is **hue = model
family, lightness = size within family**: Qwen2.5 is an ordinal violet ramp
(H≈293, darker = larger), Qwen3 is gold (H≈75), Llama wears Meta blue (H≈258),
relit to sit inside the violet ramp's lightness range.

| model | hex | OKLCH | role |
|---|---|---|---|
| qwen25-05b | `#B7A5FB` | L 0.770 · C 0.122 · H 293 | Qwen2.5 ramp, lightest |
| qwen25-15b | `#A58AFA` | L 0.705 · C 0.161 · H 293 | Qwen2.5 ramp |
| qwen25-3b | `#956CFB` | L 0.642 · C 0.204 · H 293 | Qwen2.5 ramp |
| qwen25-7b | `#864BF9` | L 0.581 · C 0.242 · H 293 | Qwen2.5 ramp |
| qwen25-32b | `#5D12BD` | L 0.430 · C 0.227 · H 293 | Qwen2.5 ramp, darkest |
| qwen3-4b | `#D9951E` | L 0.720 · C 0.145 · H 75 | Qwen3 family: gold |
| llama31-8b | `#468EFA` | L 0.655 · C 0.177 · H 258 | Meta blue (relit) |

Legend/label text uses the display names in `chartstyle.MODEL_LABELS`
("Llama-3.1-8B", "Qwen2.5-7B", …), not the raw short keys.

The Qwen2.5 ramp is **ordinal**, not pairwise-categorical: neighboring sizes are
deliberately close (identity is read from the lightness order plus mandatory
labels), so charts mixing several ramp steps must carry direct labels, and
scatter-like forms should not rely on telling adjacent sizes apart by color
alone. The cross-family core set (one Qwen2.5 step, Qwen3 gold, Meta blue)
retains full categorical guarantees.

## Validation record (2026-08-16)

Checked with `scripts/validate_palette.py` (OKLab ΔE ×100; CVD simulated with
Machado–Oliveira–Fernandes 2009 at severity 1.0) on `SURFACE` `#fcfcfb`. The
default run validates the core roster categorically and the Qwen2.5 ramp
ordinally; exit 0 = both pass.

Core roster (qwen3-4b, llama31-8b, qwen25-7b, qwen25-32b), all pairs:
- normal vision: worst pair ΔE **15.1** (qwen25-7b vs qwen25-32b) — PASS (floor 15)
- protanopia: worst pair ΔE **11.1** — PASS (target 8)
- deuteranopia: worst pair ΔE **7.0** (llama31-8b vs qwen25-7b) — **WARN band (6–8)**
- qwen3-4b contrast vs surface **2.47:1**, qwen25-05b **2.09:1** (< 3:1) — WARN

Qwen2.5 size ramp (ordinal checks): single hue ✓, monotone lightness ✓,
adjacent ΔL = 0.064/0.063/0.062/0.150 (≥ 0.06) ✓, light-end contrast 2.09:1 (≥ 2:1) ✓.

The WARNs are legal **only with secondary encoding**, so it is mandatory, not
optional, that every multi-model chart carries a legend *and* direct series labels
(end-labels on lines, first-group labels on grouped bars), and that light marks
never carry a value that isn't also readable from an axis, label, or table.

Constraint worth knowing before "improving" the palette: same-hue steps pairwise
≥ 15 ΔE consume the whole legal lightness band (OKLCH L 0.43–0.77) after ~3
steps — that is why the violet band could not hold six Qwen models categorically
and Qwen3 moved to its own hue, and why the deutan llama/7B pair sits in the
WARN band (CVD collapses the blue/violet hue difference, leaving only lightness).
Any change must re-run the validator and update this record.

## Other rules

- **Scatter-like forms** (scatter, bubble, any chart where arbitrary pairs of model
  marks sit adjacent): all-pairs is the operative test above — fine with the
  mandated labels; without them, cap at 3 model series or facet per model.
- **Neutrals and semantic colors** come from `chartstyle`: `INK/INK2/MUTED` for text,
  `GRID/BASE` for furniture, `POS/NEG/MID` for diverging, `SEQ_CMAP` for sequential,
  `DARK/LIGHT` for paired dumbbell comparisons, `ACCENT` for a generic single series.
  Text always wears ink tokens, never a series color (exception: direct series
  end-labels may take their series color).
- **Lens-mode colors** (`lenses/lib/plotting.py` `MODE_COLOR`) encode lens mode
  (logit/j/r), not model — lens figures are single-model (model in the title), so
  the two schemes never collide in one chart.
- **Marks fit comfortably inside the bounds.** Never set an axis limit exactly at
  the data's extreme — a mark sitting on the bound gets sliced by the spine (the
  old f13 clipped its 0%- and 100%-share points this way). For bounded quantities
  (shares, rates, probabilities) use `chartstyle.bounded_axis(ax)`, which pads the
  limits ~4% past the bound while keeping natural ticks; otherwise leave
  matplotlib's default margins alone rather than hard-coding limits. The same rule
  covers annotations: direct labels and end-labels must land inside the figure
  (extend the limit or flip the offset side if they don't).
- **Legends never sit on data.** In a dense plot, move the legend outside the axes
  (`bbox_to_anchor`) rather than hunting for an empty corner that isn't.
- **Axis labels are self-sufficient.** Every axis states the quantity *and* its
  scale or provenance — units, range, denominator, and any normalization
  (z-scored per what? mean over what?) — detailed enough that the chart reads
  correctly without the caption. "PC1 projection" is not a label;
  "emotion-vector PC1 projection (valence-aligned)" is. Prefer long-and-precise
  over short-and-vague; the title carries the story, the axes carry the units.
- **Save standard**: `chartstyle.save()` — dpi 200, tight bbox, `SURFACE` facecolor.
- Re-validate after any palette change: `python scripts/validate_palette.py`
  (exit 0 = no hard FAIL; WARNs must be covered by the label mandate above).
