# Color-preference & steering experiment — plan

**Question:** Does Qwen2.5-7B-Instruct have color preferences when forced to pick between two
colored objects? Can we extract a per-color "preference direction" from the residual stream and
steer the model's choice with it?

**Setup:** Qwen/Qwen2.5-7B-Instruct, bf16, plain HF transformers + forward hooks. Two files
(`data.py`, `experiment.py`) plus a `results/` dir. No framework.

## Data (`data.py`)
- 7 canonical rainbow colors: red, orange, yellow, green, blue, indigo, violet.
- 100 examples per color: `"a {color} {noun}"` over one shared list of 100 common object nouns —
  sharing nouns across colors removes the noun confound.
- 3 prompt templates of the form
  `"Pick between A and B.\nA: {a}\nB: {b}\nAnswer with one letter which one you prefer.\nModel: I prefer"`.
  The trailing suffix (from "Answer with..." on) is identical across templates.
- 420 comparisons: 10 per ordered color pair (42 pairs), random nouns + template per comparison.
  Balancing over *ordered* pairs cancels A/B position bias in the per-color aggregates.

## Pipeline (`experiment.py`)
0. **Inspect** — print top-20 next-token logits for a few prompts to confirm which surface forms of
   A/B carry the probability mass; hardcode those variant sets (uppercase only — " a" is the
   article in "I prefer a red cup" and must be excluded).
1. **Measure preferences** — batched forward passes over all 420 comparisons; per side, record each
   variant's raw logit and the total attribution `logsumexp(variant logits)`;
   `diff = score_A − score_B`. Save `preferences.json`; print the 7×7 mean-diff matrix and
   per-color mean preference.
2. **Extract vectors** — per color, take the top-20 comparisons it won most strongly
   (sign-corrected for which side it was on). Capture residual-stream activations at every layer,
   averaged over only the suffix tokens ("Answer with one letter... I prefer" — excludes the
   items). Mean over the 20 examples, L2-normalize → one unit vector per (color, layer).
   Save `vectors.pt`.
3. **Steer** — per color, take its 20 worst comparisons. Add `coef × mean_resid_norm(layer) × unit_vec`
   to the residual stream at the suffix positions of a single layer. Grid: 5 layers (evenly spaced
   through the 28) × 3 coefficients {0.5, 1, 2}.
4. **Report** — mean change in signed logit-diff toward the steered color vs. the unsteered
   baseline on the same examples, per (layer, coef). Save `steering.json`.

## Sanity checks
- Step-0 printout: A/B variants dominate the top logits.
- Preference matrix roughly antisymmetric (position bias cancels).
- Steering deltas grow with coefficient; small coef ≈ small effect.
