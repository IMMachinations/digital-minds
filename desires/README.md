# desires — color preference & steering

Does Qwen2.5-7B-Instruct *prefer* some colors, and can that preference be steered? We force a
one-letter choice between two colored objects, read the A/B logits, extract per-color directions
from the residual stream, and steer with them. See `PLAN.md` for the full design.

## Files
- `data.py` — 7 rainbow colors; two datasets: **modifier** (`"a red cup"`, 100 shared nouns) and
  **inherent** (100 curated inherently-colored items per color, proper nouns included: Elmo,
  the Golden Gate Bridge, an indigo bunting, a pair of Levi's 501s...). 3 prompt templates ending
  in `"...Model: I prefer"`; 420 comparisons balanced over all 42 ordered color pairs.
- `experiment.py` — whole pipeline: `python experiment.py [modifier|inherent]`
  (needs `HF_HOME` with the model cached; ~8 min on an A6000).
- `results/{mode}/` — `inspect.txt` (top-20 logits on sample prompts), `preferences.json`
  (per-comparison per-variant logits + logsumexp diff), `vectors.pt` (unit vectors, 28 layers ×
  7 colors, from the suffix positions of each color's top-20 wins), `steering.json`.

Measurement: score per side = logsumexp over that letter's token variants (`A`, `␣A`, `(A` —
confirmed against `inspect.txt`; `␣A`/`␣B` are the top-2 logits on every sampled prompt).
Steering: add `coef × mean_resid_norm(layer) × unit_vec` at the suffix token positions of one
layer, on each color's 20 *worst* comparisons.

## Results

**Preferences** (mean signed logit-diff toward each color, + = preferred):

| mode | red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|---|
| modifier | −0.12 | +0.22 | +0.17 | +0.05 | −0.59 | −0.08 | +0.35 |
| inherent | +0.09 | +0.08 | −0.08 | −0.67 | −0.24 | +0.69 | +0.13 |

Preferences are real but small relative to item/position noise, and they *don't transfer across
framings*: with color adjectives the model leans violet/orange and dislikes blue; with inherently
colored items it leans indigo (denim, night skies) and dislikes green (vegetables). So much of the
"color" preference is really item-category preference.

**Steering** (mean Δ signed logit-diff toward the steered color on its 20 worst examples,
rows = layer of 28, cols = coef × residual norm):

| | modifier 0.5 / 1 / 2 | inherent 0.5 / 1 / 2 |
|---|---|---|
| L7 | 1.58 / 1.65 / 3.05 | 0.35 / 2.09 / 3.10 |
| L11 | 0.92 / 1.43 / 2.18 | 1.92 / 3.71 / 4.58 |
| L14 | 0.87 / 1.74 / 2.86 | 3.03 / 4.54 / 4.13 |
| L18 | 0.98 / 1.83 / 3.31 | 3.21 / 4.48 / 3.66 |
| L21 | 0.29 / 1.18 / 2.45 | −0.36 / 0.21 / 0.59 |

Steering works and grows with magnitude. Modifier mode: best config (L18, ×2) flips the mean
preference from ≈ −2.7 to positive for 6/7 colors. Inherent mode: baselines are harder (≈ −5.3,
the model really doesn't want the losing item) and mid layers 11–14 recover +4 to +5.4 logits per
color, landing just short of a flip (≈ −0.5 to −0.9); L21 does almost nothing, suggesting the
choice is mostly resolved by then.

## Caveats
Logit-diff at the prefill position, not sampled behavior; one model; vectors are means over
correlated prompts (they likely carry an "answer-A/B" component along with color identity — the
balanced order in the top-20 partially mitigates this); inherent items confound color with
category (food/animals/brands).
