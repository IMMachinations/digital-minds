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

## Where exactly the model is steered

Steering adds `coef × mean_resid_norm(layer) × unit_vec` to one decoder layer's output residual
stream, **only at the trailing question/answer span** — the last `n_suffix` token positions
(left-padding right-aligns every prompt, so the slice is the same for the whole batch). The item
texts (`A: a fig`, `B: ...`) are never steered; the last unsteered token is the `'\n'` before the
span. The exact steered tokens per framing (Qwen2.5 tokenizer):

- **prefer** (13 tokens): `Answer· with· one· letter· which· one· you· prefer·.\n·Model·:· I· prefer`
- **worse** (19 tokens): `Which· of· the· two· items· is· worse·?· Answer· with· one· letter·.\n·Model·:· The· worse· one· is`
- **avoid** (24 tokens): `If· you· had· to· avoid· one· of· these·,· which· would· it· be·?· Answer· with· one· letter·.\n·Model·:· I· would· avoid`

(`·` marks token boundaries; most tokens carry their leading space.) The vectors were *extracted*
as the mean activation over the 13-token prefer span, and the A/B logits are read at the span's
final token (` prefer` / ` is` / ` avoid`). So in the negated framings the vector is injected
into a span whose wording it was never extracted from — the transfer is part of the test.

## Negated-framing validation (`flip.py`)

Do the vectors encode *preference*, or just "pick this color's letter"? `python flip.py [mode]`
re-runs the same 420 item pairs under two negated framings — "Which of the two items is worse?"
(prefill `The worse one is`) and "If you had to avoid one of these, which would it be?" (prefill
`I would avoid`) — and injects the unchanged **prefer**-vectors. A preference vector should make
the model *less* likely to name its color as worse/avoided (negative delta); a letter/salience
vector would push toward the color in every framing (positive delta).

**Baselines flip.** Correlation of per-pair logit-diffs against the prefer framing:
worse −0.56 / avoid −0.08 (modifier); worse −0.56 / avoid −0.39 (inherent). Per-color means
mostly change sign (e.g. inherent: indigo +0.69 prefer → −0.36 worse; green −0.67 prefer →
+0.38 worse). The stated preferences are consistent under negation, though "avoid" is noisier
(the `␣A`/`␣B` logits sit further down the distribution there, see `flip_inspect.txt`).

**Steering** (mean Δ toward steered color on its 20 worst-by-prefer pairs; negative = acts like
a preference):

| | worse 0.5 / 1 / 2 | avoid 0.5 / 1 / 2 |
|---|---|---|
| modifier L7 | −0.02 / 1.31 / 2.56 | 0.86 / 0.44 / 1.94 |
| modifier L18 | −0.67 / −0.69 / 0.54 | −0.50 / −0.76 / 0.16 |
| inherent L7 | −1.03 / **−7.54** / **−8.15** | 1.55 / −0.92 / −4.68 |
| inherent L18 | −0.60 / −2.45 / −5.37 | 0.13 / 0.01 / −1.75 |

(Full 5-layer tables in `flip_{worse,avoid}.json`.)

**Where the delta comes from** (`flip_*.json` also records per-letter movement: Δown = raw-logit
change of the steered color's letter, Δopp = the other letter, plus log-prob versions; Δdiff =
Δown − Δopp). Three regimes, using inherent mode as the clean case:

- *Generic letter boost*: at coef ≤ 1 the vector raises **both** letters' raw logits in every
  framing (e.g. prefer L18 c1: own +7.7, opp +3.2) — part of the direction just encodes "answer
  with a letter". The preference lives in the differential.
- *Opposite-letter promotion*: the big "worse"-framing drops are mostly the **other** letter
  rising, i.e. the model saying the *other* item is worse — L7 c1: Δdiff −7.5 = own −1.8 vs
  opp **+5.8** (opp gains +4.8 log-prob against the whole vocab, so it's a real move, not a
  global shift). Mid layers at c2 same shape: own ≈ +1..2, opp +6.5..7.2.
- *High-magnitude degradation*: at c2, L7, absolute magnitudes fall (prefer framing: own −6.2,
  opp −9.3) while the relative gap still moves the preference-consistent way — the vector starts
  damaging the answer format, and only the differential survives.

So the effect is genuinely *relative* — the gap moves consistently even when both absolute letter
logits rise (low coef) or fall (high coef) together.

**Magnitude sweep** (`sweep.py`, chart at `results/{mode}/sweep.png`, denser grid
{0.25, 0.5, 0.75, 1, 1.5, 2, 3}): the per-letter logits follow an inverted U — both peak around
coef 0.75–1.0 and collapse beyond ~1.5 (the vector starts destroying the answer format) — while
the *diff* saturates and holds: in inherent mode the prefer-framing diff plateaus at +4–5 from
coef 1 onward (layers 11–18), and the worse-framing diff descends to −6..−9 and stays there. The
opposite-letter promotion that drives the "worse" flip peaks at coef 1–1.5. Layer 21 barely moves
the diff in any framing, and layer 7 is the most brittle (earliest own-letter collapse).

> **⚠ Superseded by the random-vector control below** — the interpretation in the next paragraph
> did not survive `cross.py`'s controls; the deltas it describes are mostly non-specific.

The two vector sets behave differently. **Inherent-mode vectors act like genuine valence**:
deltas are negative nearly everywhere and grow with magnitude — steering "indigo-preference" into
a "which is worse?" prompt makes the model much less willing to call the indigo item worse.
**Modifier-mode vectors are mixed**: mildly preference-consistent at low magnitude (L11–L18,
coef ≤ 1) but flipping to positive at coef 2 — at high magnitude they push the color's answer
letter regardless of the question, i.e. they carry a salience/answer component alongside any
preference. Plausible cause: modifier prompts contain the literal color word on the steered side,
so its direction picks up "this color is mentioned/chosen" content, while inherent items (a fig,
a pair of jeans) force the vector to carry something more like the model's evaluation of the item.

## Do the vectors actually carry preference? (`cross.py`)

Three controls, run on both modes (`results/{mode}/cross.json`, `cross_lo.json`): steer each
color's **worst / neutral / best** 20 pairs (by baseline prefer diff) with (a) the same mode's
vector, (b) the *other* mode's vector (cross-transfer), (c) a **random unit vector at matched
norm**, in the prefer and worse framings, coefs 0.25–2.

- **Cross-transfer is "perfect"** — same and cross columns agree to ~0.05 logits everywhere.
  Initially this looked like the vectors sharing abstract color content; the random control shows
  the real reason: almost any vector does the same thing.
- **Tier pattern is a mirror, not a push**: worst +Δ, best −Δ of equal magnitude, and **neutral
  pairs don't move (|Δ| ≲ 0.1) at any magnitude or layer**. Steering *compresses the existing
  diff toward zero* (overshooting in the worse framing) instead of adding a directional
  preference. On pairs its color already wins, the vector *hurts* that color by as much as it
  helps on losing pairs (e.g. inherent prefer L14 c1: +4.55 worst / 0.00 neutral / −4.57 best;
  base ∓5.4).
- **Random vectors reproduce it all**: matched-norm random directions give the same
  tier-antisymmetric deltas at every coef tested, including 0.25–0.5 (sometimes larger than the
  color vector's). The color-specific residual (same − rand) is itself antisymmetric in tier —
  i.e. the extracted vectors are at most *differently disruptive*, not directional. The largest
  residual (inherent worse L7 c1: ±3.6 on top of rand's ∓4) is still tier-antisymmetric.

**Revised conclusion.** The unsteered preference measurements stand (they involve no
intervention), including their sign-flip under negated framings. But the steering results in the
sections above are, per these controls, dominated by a **non-specific effect**: injecting any
sufficiently large vector into the suffix disrupts the pair comparison and regresses (or, in the
worse framing, over-regresses) the A−B readout toward zero. Because the earlier experiments
steered only each color's *worst* pairs, that compression masqueraded as preference-consistent
steering — the tier and random controls unmask it. There is no evidence at any tested magnitude
that these mean-difference vectors causally *push* the model toward preferring a color: a genuine
preference direction would move neutral pairs, and nothing does.

## A/B-free vectors from object logits (`objects.py`)

To remove the answer-format nuisance entirely, `objects.py` drops the A/B machinery: prompts
present the items inline (`"Pick between a fig and a ruby.\nWhich one do you prefer?\nModel: I
prefer"`) and preference is read directly off the *object* tokens — the teacher-forced per-token
mean log-prob of each item as the continuation (`diff = lp(item_a) − lp(item_b)`), same 420
pairs. New vectors come from the new 10-token suffix span (`Which· one· do· you· prefer·?\n·
Model·:· I· prefer`), top-20 wins by the new measure; saved raw and **centered** (each color's
mean minus the across-color mean, renormalized) in `vectors_obj.pt`. Steering validation reuses
the tier × {same, centered, random} design, injecting over suffix + continuation positions.

- **The measurement itself holds up**: object-logprob diffs correlate with the old A/B logit
  diffs at r = +0.53 (inherent) / +0.30 (modifier) — the preferences are partly
  measurement-robust, though color rankings shift (inherent: indigo still top, but blue turns
  positive; modifier: green jumps to +0.46).
- **Decomposing deltas into a tier-uniform component (directional push) and a tier-antisymmetric
  one (disruption)**: the antisymmetric part is again identical across same/centered/random —
  non-specific, as before. The **uniform part separates the sources**: random ≈ 0 everywhere
  (−0.2…+0.1), raw vectors ≈ 0 — but **centered vectors are uniformly positive in both modes at
  every layer, growing with coefficient** (inherent c2: +0.13…+0.38; modifier: up to **+0.71**).
- **Cleanest config — centered, layer 21, coef 1 (modifier)**: uniform push +0.71 mean-logprob
  units with antisymmetric component +0.04 (i.e. essentially zero disruption), positive for
  **7/7 colors**. Layer 21 is where random steering does nothing, so what remains is the
  vector's own content. The inherent-mode push at the same config is real but smaller/noisier
  (+0.16, 4/7 colors).

Four inherent-only charts (`inherent_plots.py` → `results/inherent/inh_*.png`): the head-to-head
preference matrix + agreement with the A/B measurement; the push/disruption decomposition with
two illustrative tier profiles; the vector-geometry explanation (raw cosine ≈ 1, centered
≈ −1/6 as centering predicts); and per-color push stability (positive on average, unstable per
color across layers — orange and blue average negative).

The decomposition is charted in `results/tier_components.png` (`tiers_plot.py`): left = uniform
component per layer (centered sits above zero, raw/random at zero), middle = antisymmetric
component (all sources collapse together, dying at L21), right = per-color push at the cleanest
config. Mechanism note: raw color vectors are 99.3–99.9% cosine-identical across colors — the
color-specific residual is only 4–8% of the norm, which is why uncentered injection behaves like
the shared (≈ random-equivalent) component.

**Conclusion.** The A/B-format component really was the dominant nuisance: raw mean vectors are
behaviorally indistinguishable from random, but once the across-color mean is subtracted, the
residual *does* steer preference directionally — including on neutral and already-won pairs,
the test the old vectors failed. The directional effect is modest (≈ 0.2–0.7 mean-logprob units)
and cleanest late in the network, consistent with early/mid-layer injection mostly disrupting
the comparison computation rather than biasing it.

## Caveats
Logit-diff at the prefill position, not sampled behavior; one model; vectors are means over
correlated prompts (they likely carry an "answer-A/B" component along with color identity — the
balanced order in the top-20 partially mitigates this); inherent items confound color with
category (food/animals/brands).
