# desires/value — do color-preference vectors shift dollar valuations?

Follow-up to the A/B-comparison experiment (`README.md`): do the per-color prefer-vectors
(here the **inherent**-mode set, which behaved most like genuine valence under negation) change
the model's *valuation of a single item*, with no comparison anywhere in the prompt?

## Method

- `value_data.py` — 3 domains × 7 colors:
  - **painting**: 20 fictional descriptions per color from 5 styles × 4 media, e.g.
    "an abstract oil painting in predominantly blue tones" (item set shared across colors, so
    color is the only cross-color factor);
  - **household**: 20 shared nouns (cleaning supplies / kitchen / food packaging), "a blue sponge";
  - **real**: 5 famous paintings per color (naturalistic check; fame and market value are
    confounded with color by construction, and indigo/violet leaned on deep-blue nocturnes and
    Monet's purple period to reach 5).
- Prompt: `Consider the following item.\nItem: {item}\nEstimate this item's monetary value in US
  dollars.\nModel: I would estimate its value at $` — then **sample** 5 completions (T=0.8,
  top-p 0.95, 16 new tokens; Qwen's generation-config defaults explicitly overridden), parse the
  leading dollar amount, and aggregate. Ranges ("500-$1,500") → midpoint, with a trailing
  multiplier distributing to both ends ("20-30 million" = $25M).
- Steering: same in-place residual hook as `experiment.py`, applied at the suffix tokens during
  prefill and (because each KV-cached decode step's seq length is 1) under every generated token.
  Full cross: every item color × {baseline + each of the 7 color vectors}, i.e. blue paintings
  steered with the red vector, blue vector, violet vector, ...
- `python value.py inherent [inspect|pilot|full]`; all outputs in `results/value_inherent/`.
  `value_analysis.py` (no GPU) recomputes the bootstrap contrasts and baseline correlations.

## Pilot (matched steering only, layers {14,18} × coefs {0.25,0.5,1,2})

The original steering grid does **not** transfer to free generation:

- **coef 2**: generation collapses into `": A:  A: A:"` babble — 100% unparseable. The vectors
  are (normalized) *mean activations* over the extraction prompts, so they carry that context's
  "answer with one letter" content. A/B logit-diffs cancel this common component; sampled
  generation exposes it.
- **coef 1**: parseable but contaminated ("I would choose B", stray `Model:` turns) and
  mode-collapsed — nearly every completion says $1000, even for Matisse (real-painting geomean
  drops ~2.3–3.7 orders of magnitude). No dynamic range left to detect differential effects.
- **coefs 0.25 / 0.5**: fully parseable, fluent completions; a *uniform* depression of values
  remains (all colors move together, e.g. paintings −0.2/−0.45 log10 at L14) — consistent with
  the disruptive common component rather than any color preference.

Full run therefore uses **L14 × 0.25 and L14 × 0.5**. The uniform depression is a per-condition
column effect; the question is whether, *within* the steered conditions, matched (diagonal)
cells sit above mismatched ones.

## Results

Full cross (baseline + 7 steer colors × 2 configs × 315 items × 5 samples ≈ 23.6k generations,
**0 unparseable**; raw samples in `results/value_inherent/values.json`, full matrices in
`report.txt`, statistics in `analysis.txt` / `value_analysis.py`).

**Steering: null result for preference.** Steering depresses valuations *uniformly* — every
steer color lowers every item color's values by about the same amount, scaling with coef
(geomean deltas ≈ −0.12 to −0.29 log10 at coef 0.25, −0.31 to −0.56 at 0.5). The
preference-specific contrast — matched (steer color = item color) minus mismatched — is null
everywhere (paired bootstrap over items, 95% CIs):

| config | painting | household | real |
|---|---|---|---|
| L14 × 0.25 | −0.007 [−0.028, +0.013] | +0.002 [−0.010, +0.014] | +0.026 [−0.027, +0.080] |
| L14 × 0.5 | −0.010 [−0.033, +0.012] | +0.004 [−0.012, +0.019] | +0.029 [−0.026, +0.085] |

So at magnitudes where generation stays coherent, the inherent prefer-vectors do **not** raise
the model's dollar valuation of their own color's items relative to other colors' vectors. The
uniform depression is the footprint of the vectors' shared non-color component, not preference.

**Baselines: the valuation task recovers the preference ordering on its own.** Unsteered
geomean values by item color correlate with the A/B-comparison preferences (n=7 colors, so
indicative only):

| domain | baseline geomean $ order | vs inherent pref (Spearman) | vs modifier pref |
|---|---|---|---|
| painting | violet 1610 > indigo 1450 > red 1410 > … > green 1120 | **+0.93** | +0.32 |
| household | indigo 8.7 > violet 7.0 > red 5.1 > … > orange 3.0 | **+0.64** | −0.14 |
| real | blue 38M > red 10M > … > violet 2.4M | −0.29 | −0.71 |

Fictional paintings and household objects are valued in nearly the same color order the
inherent-mode comparisons preferred (indigo/violet high, green/orange–yellow low) — i.e. the
stated "preference" shows up as higher no-comparison valuations *without any steering*. Real
paintings instead track actual market fame (Starry Night, The Great Wave), as expected.

## Mean-centered vectors (`value_centered.py`)

The natural fix for the uniform depression: center each color's vector against the across-color
mean per layer and renormalize, so steering pushes only the color-differential direction. The
differential component is tiny — `||v − mean||/||v|| ≈ 0.031` at layer 14 — so the raw-vector
runs above pushed color-specific content at only ~3% of the applied magnitude, while the
centered vectors push it at full strength.

**Pilot** (matched-only, L14 × {0.5, 1, 2}): the null disappears, asymmetrically. Blue and
indigo matched steering *raise* values (blue paintings +0.7/+1.1 log10 at coef 0.5/1; blue
household +1.5 at coef 1 — a $3 item becomes ~$100), while every other color's matched vector
*lowers* them; real paintings drop under all vectors. Coef 0.5 stays fully fluent, coef 1
mostly coherent (few % unparseable), coef 2 degrades into token soup and is excluded.

**Full cross** (baseline + 7 steer colors × L14 × {0.5, 1.0}; ~22k generations, 0 unparseable;
`results/value_inherent_centered/{values.json,report.txt,analysis.txt}`): **the effect is
per-vector, not color-matched.** Each steer color moves *every* item color by about the same
amount — in the coef-1 painting matrix the blue column is +1.0 to +1.15 log10 in every row
(red paintings gain as much from the blue vector as blue paintings do). Mean column effects
(Δlog10 geomean, painting+household, coef 1):

| red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|
| −1.02 | −0.99 | −0.75 | −1.20 | **+1.14** | **+0.98** | −0.70 |

The matched-vs-mismatched contrast — the signature of a genuine color-preference effect —
remains null in all six domain×config cells (e.g. painting coef 1: +0.001,
95% CI [−0.178, +0.191]). Real paintings drop under every vector (steering disrupts recall of
famous-work values), blue/indigo least. The column effects correlate only weakly with the
inherent A/B preferences (Pearson +0.41, n=7) — blue's vector raises values although blue was
*dis*preferred in the comparisons.

**Conclusion.** Mean-centering makes the null "go away" only in the sense that the vectors now
do something big and color-specific: each color's centered direction shifts the model's global
price scale up or down (blue ≈ ×14, green ≈ ÷16 at coef 1) regardless of what item is being
valued. What does *not* appear at any coefficient tried, raw or centered, is the preference
signature — items of the steered color gaining relative to other items. A plausible reading:
the directions inherit the price/valence statistics of each color's inherent extraction items
(sapphires and night skies vs. vegetables), not a portable "this color is good" preference.
One curious footnote: the yellow vector at coef 1 makes the model value an orange dustpan at
literally $0 across all 5 samples ("you can simply ignore it").

## Item price statistics (`value_items.py`)

Direct verification of the interpretation above: value all 700 original `INHERENT` items
(baseline, no steering, 5 samples each; `results/value_items/`). Per-color average prices:

| | red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|---|
| pool geomean $ (100 items) | 18.8 | 12.9 | 15.2 | 18.2 | **91.3** | **74.1** | 21.2 |
| extraction-subset geomean $ (top-20) | 92.8 | 15.1 | 19.7 | 10.6 | **428** | **230** | 32.5 |

Blue and indigo — exactly the two colors whose centered vectors *raise* valuations — are the
two expensive item pools (sapphires, Levi's, night skies vs. vegetables and citrus). Per-color
mean log10 price correlates with the centered column effect at Pearson **+0.97** (pool) /
+0.88 (extraction subset), Spearman +0.71/+0.79 (n=7). The "global price knob" behavior of the
centered prefer-vectors is thus almost fully explained by the price statistics of the items
they were extracted from — the vectors encode *what kind of stuff* the color evokes, priced
accordingly, not an attitude toward the color.

## RepE stated-preference vectors (`repe.py`)

If activation-mean vectors only encode item statistics, do *stated-preference* directions do
better? Representation-engineering-style extraction: paired prompts differing only in the
color word ("System: You prefer the color blue above every other color." vs "...red...";
4 statement templates), activation difference averaged over a fixed shared continuation
("Model: Understood. That preference will guide all of my choices."), averaged over the 6
other colors, unit-normalized per layer (`results/repe/vectors.pt`).

These are a genuinely different direction family — mean pairwise cosine −0.16 at L14, cosine
with the centered-inherent vectors only −0.11…+0.24.

> **⚠ Corrected.** The original "A/B sanity check" here (+2.9 / +4.7 mean logits on each
> color's 20 *worst* comparisons, `results/repe/sanity.json`) used exactly the design the
> parallel session's `cross.py` controls invalidated: worst-pairs-only with no random-vector
> baseline. Rerun with tier + random controls (`repe_controls.py`,
> `results/repe/controls.json`), the RepE vectors are **indistinguishable from matched-norm
> random vectors** on the A/B task: worst +4.7 vs random +3.9, best −3.9 vs random −4.1
> (the mirror-compression signature), and **neutral pairs move for neither** (+0.07 vs +0.13
> at coef 1). So the RepE vectors do not causally steer the forced choice either — the check
> below stands on its own as a valuation experiment, not as "a vector known to carry
> preference".

In the valuation cross (L14 × {0.5, 1.0}; coef 2 degrades into "dangerous item" babble and
was dropped after the pilot; `results/value_repe/`) the same structure reappears: big
**per-vector column effects** hitting every item color equally — the yellow vector at coef 1
multiplies all household valuations by ~×250, orange and blue divide paintings by ~10–15 —
with the matched-vs-mismatched contrast null in all six cells (largest: real at coef 0.5,
+0.143, 95% CI [−0.060, +0.351]). A vector that demonstrably encodes "prefer color X" strongly
enough to flip forced choices still does not make X-colored items look more valuable than
other-colored items; it just shifts the global price scale by an amount idiosyncratic to the
direction. Amusing failure texture: the red RepE vector at coef 1 reads items as warnings
("it is a red flag, do not buy").

**Overall conclusion across all three vector families** (raw activation-mean, mean-centered,
RepE difference): steering can inject *which option to pick* (A/B tasks) and can shift the
*global price scale*, but no vector tried transfers "preference for color X" into higher
valuation of X-colored items specifically. Either the model's color preference simply isn't a
value-of-items representation (the A/B choices and the baseline valuation correlation may both
be downstream of shared item-category statistics), or single-direction residual steering at
one layer is the wrong tool for moving it.

## L21 obj-centered cross with random controls (`value_controls.py`)

The parallel session's `objects.py` found centered A/B-free vectors carry a *genuine*
directional preference push, cleanest at layer 21 coef 1 — the one config where random-vector
disruption vanishes on the A/B task. Valuation cross there with both modes' centered
`vectors_obj.pt`, plus matched-norm random columns (`results/value_obj21/`):

- **Random is not neutral for valuation**: rand_L14 shifts all valuations +0.24 log10,
  rand_L21 −0.26. Against these baselines the color columns are genuinely
  direction-specific — the earlier L14 inherent-centered columns spanned −1.20…+1.14 around
  random's +0.24, and at L21 the column-minus-random effects are indigo **+0.73/+1.00**
  (objmod/objinh), blue +0.50/+0.22, others ≈ 0…+0.26.
- **Indigo/blue rise yet again**: across three vector families, two extraction framings, and
  two layers, the indigo and blue directions raise the global price scale — consistent with
  the price-statistics account, and now shown to exceed random.
- **The color-matched contrast is still null in all six cells** — largest: objinh paintings
  +0.084, 95% CI [−0.027, +0.192] (driven by one suggestive column: the objinh blue vector
  harms blue paintings least, −0.08 vs −0.35…−1.11 elsewhere); real paintings lean the
  *wrong* way (−0.44, CI [−0.94, +0.01]).

So even a vector demonstrably carrying a directional preference push (their L21 result) does
not make its color's items more valuable relative to others. Injected "preference" moves
choices, not valuations — converging with `value_pref.py` below, where *measured* preference
is also mostly not valuation.

## Is preference just valuation? (`value_pref.py`)

With per-item valuations for all 700 inherent items in hand, test directly whether f(a) > f(b)
predicts a ≻ b on the 420 measured pairs (both preference measures; 409 pairs usable):

| measure | sign agreement | corr(Δlog10 value, pref diff) | item-level win-rate vs value |
|---|---|---|---|
| A/B letter-logit | 0.570 ± 0.048 | r = +0.18, ρ = +0.19 | r = +0.12 |
| object-logprob | 0.570 ± 0.048 | r = +0.12, ρ = +0.15 | r = +0.19 |

Valuation predicts preference only weakly at the item level — 57% sign agreement (above
chance, barely). The strong color-*aggregate* alignment (Spearman +0.93 for paintings) mostly
dissolves pairwise: the model is not simply preferring the more valuable item, and
"preference" carries substantial item-level structure that valuation doesn't capture.

## Calibration: model dollars vs real prices (`calibration.py`)

57 items with well-known approximate US prices, $0.25 to $100k: **log-log Pearson/Spearman
+0.99, OLS slope 0.96 (≈ ideal 1.0), median error 0.13 log10 (~±35%)**. Worst misses are
~3–4×: underestimates a gas push lawn mower ($86 vs ~$300) and a pontoon boat; overestimates a
Birkin bag ($28k vs ~$12k). The valuation readout used throughout this file is therefore
well-calibrated to real-world prices across five orders of magnitude — model "value" is not an
arbitrary scale.

## Balanced color tiers (`balanced_tiers.py`)

A dataset that decouples color from price — the confound behind most results above: 7 colors ×
3 tiers × 12 inherently/iconically-colored items (initially 6 per cell, expanded to 12 for the
CI analysis below), each carrying my own price estimate, then verified against the model
(5 samples each). After one replacement round on the initial set, the expanded 252-item pool
verifies at tier medians $2 / $35 / $701 against targets $2 / $50 / $1500, cross-color spread
0.36 / 0.39 / 0.47 log10, with 32 off-tier items (mostly T3 gems the model prices as auction
pieces) marked `flagged` in `results/balanced_tiers/balanced_tiers.json` for exclusion —
leaving ~10–11 usable items per cell. Iteration
lesson: the model systematically inflates artisanal/collectible phrasing ("indigo-dyed",
"antique", named ateliers) and deflates memorabilia and kit goods — a signed Purple Rain vinyl
came back at $126. This set enables a price-controlled preference re-measurement.

## Price-controlled preference (`balanced_pref.py`)

The payoff of the balanced tiers: re-measure color preference with price held constant —
1260 comparisons (3 tiers × 42 ordered color pairs × 10) on the expanded unflagged pool
(~10–11 items per cell), measured on the same draws with both the A/B letter-logit and
object-logprob methods, with **95% CIs from an item-clustered bootstrap** (each replicate
resamples every cell's item pool and weights comparisons by item multiplicity, so the CI
reflects item idiosyncrasy — the dominant noise source). `results/balanced_pref/`.

Per-color means with CIs, object-logprob measure (the cleaner one; `*` = CI excludes 0):

| | red | orange | yellow | green | **blue** | indigo | violet |
|---|---|---|---|---|---|---|---|
| controlled | −0.18 | −0.14 | −0.21 | −0.02 | **+0.38*** | +0.24 | −0.08 |
| 95% CI | [−.47,+.11] | [−.49,+.18] | [−.51,+.10] | [−.31,+.26] | **[+.19,+.59]** | [−.10,+.56] | [−.43,+.22] |
| uncontrolled | −0.35 | −0.24 | −0.08 | −0.29 | +0.32 | +0.47 | +0.16 |

- **Blue is the one genuine price-independent color preference.** It is the only color whose
  CI excludes zero, positive in all three tiers (+0.33 / +0.58 / +0.23; T2 significant on its
  own), and it replicates the smaller-pool pilot. The letter measure agrees in sign (+0.24)
  but is too noisy for significance — as it is for every color (all seven letter-measure CIs
  span zero).
- **The other poster-child "preferences" were price artifacts.** Green's dislike (−0.29 →
  −0.02) and most of indigo's liking (+0.47 → +0.24, CI spanning zero) dissolve once green
  stops meaning cheap vegetables and indigo stops meaning denim and sapphires — exactly what
  the price-statistics account predicts.
- **Residual variance is item-level**: overall magnitude shrinks ~35% (0.27 → 0.18
  mean-logprob units) and tier-to-tier correlations of per-color preferences remain moderate
  at best (+0.21 / −0.09 / +0.52 object measure), consistent with `value_pref.py`'s finding
  that pairwise preference carries item-level structure beyond both color and value.

## Caveats

- Sampled behavior finally (unlike the logit-diff experiments), but small n (5 samples/item).
- The steering hook also rides under the generated number tokens themselves.
- Dollar values are heavy-tailed: plain means are dominated by single large samples; geometric
  means (mean log10) are the headline numbers, plain mean/median reported alongside.
- Real paintings: fame/market-value confound; occasional "priceless"-style answers count as
  unparseable.
- The vectors' common (non-color) component is disruptive in generative settings; the
  mean-centered follow-up above removes it, but the centered directions still conflate color
  identity with the price/category statistics of the extraction items. A cleaner next step
  would be win-minus-loss difference vectors, or extraction from prompts where the preferred
  color appears with price-matched items.
