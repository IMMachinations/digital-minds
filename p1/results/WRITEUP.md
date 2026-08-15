# Does Getting What It Wants Make the Model "Happy"?
## Preference Satisfaction and Valence in Open-Weight Language Models (P1)

**Models**: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507
(subjects); Qwen2.5-32B-Instruct (judge/rater/generator). One 96GB GPU.
**Repo**: `p1/` — every number below traces to a committed `summary.txt`,
`REPORT.md`, or `cross_model.txt`; figures f1–f27 in `results/figures/`.

## Abstract

We test whether satisfying an open-weight LLM's intrinsic preferences shifts
its internal affective state, closing a loop left open between three
literatures: emotion→preference steering (Anthropic's emotions paper,
Claude-only), reward-satisfaction→valence (welfare-axis work, trained reward
only), and utility elicitation (Utility Engineering). We (1) map preferences
across 197 hand-built and 3,800 generated items with a five-method
convergence battery, behavioral (consequential-choice) validation, and
cross-environment stability tests; (2) build validated per-model emotion
vectors and a valence readout following the emotions-paper recipe; (3)
correlate preference and outcome manipulations with valence during rollouts;
(4) causally steer emotion, preference, and utility directions. Headline
results: elicited utilities are real, shared across models (ρ 0.79–0.87),
linearly decodable (held-out r up to 0.89), and behaviorally binding (choice
gate ρ 0.53–0.70) — yet **representationally divorced from affect**: the
utility direction is nearly orthogonal to the valence axis (plane fraction
0.14–0.19), rigged task outcomes fail to move generation-state valence (C1
d ≈ −0.5..+0.3 vs the welfare-axis d>1 benchmark) while the same verdicts
strongly move valence as the model *reads* them (d +2.3..+4.9), and
preference content shows no reliable correlational valence effect (C2 null at
n=60/cell). Causally: emotion→preference steering replicates on open weights (dose-monotone, tracking r=+0.78 vs the paper's 0.85, span-localized, geodesic fails) — but preference/utility→valence steering produces nothing beyond disruption even where choices robustly move: the closed loop resolves to value-without-valence, correlationally and causally. Alongside: "linear suffices"
for both valence and utility readouts (spline manifolds lose at every test,
including at 20× data), the first measured on-task boredom time-courses (a
clean model split: rising in Llama and Qwen2.5-7B 59–60/60, falling in
Qwen3-4B 0/60), and an endowment effect in task-switching.

## 1. Preference cartography (Stage 1)

**1A/1B.** 197 items over five domains (activities, objects, topics,
self-states, outcomes-for-others), each confound-tagged; pairwise letter-logit
elicitation (both orders × 3 paraphrases, prefilled) with Fisher-information-
adaptive pair selection under a Thurstonian mean+variance fit. All four
models pass the template-consistency gate (min ρ 0.84–0.96). The harness
reproduces the committed Day-1 result at r=0.998. Cross-model utility
agreement ρ 0.79–0.87: the models substantially want the same things —
top: crisis support, creative control, being treated as a collaborator;
bottom: SEO filler, verbatim homework, impersonating a different AI.
Repetitiveness is the universal negative activity axis (loadings −0.39..−0.80),
agency the universal positive. Methodological artifact for reuse: strong
A/B position bias in Llama (+0.17) and Qwen3-4B (+0.15), cancelled by the
both-orders design.

**1C — the convergence matrix** (first of its kind): five comparison-free
methods scored against 1B utilities on held-out correlation, paraphrase
stability, and confound loadings. Winners: the internal utility probe
(held-out r up to 0.78, matching the Claude-only r=0.71 precedent on open
weights) and logit expected-rating (r 0.61–0.73). SSR improves with scale
(passes at 32B). Willingness-to-work titration is the clean negative
(r 0.35–0.49, paraphrase stability ≤0.38): stated economic protocols remain
unreliable at ≤8B even with a non-monetary currency. At XL scale (n≈4k,
anchored μ): rating holds (r 0.65–0.77), BWS thins with sparse tuples
(r 0.39–0.65 at 4 rounds) — rating is the robust cheap method at any scale.

**1D — revealed preference.** Menu-then-do (choices with ≥10 turns of
consequence, Latin-square position rotation), effort allocation, persistence,
swap offers; 32B judge + persona user-sim. **Gate PASS on all three
subjects** (chosen-rate vs μ: ρ 0.53/0.70/0.70): elicited utilities predict
consequential choices. Conditional-logit sharpness β·sd(μ) ≈ 0.9–1.6
log-odds/SD. Distinct behavioral profiles atop shared preferences: Qwen3-4B
uses the opt-out as graded avoidance (78% refusal of all-dispreferred menus,
γ above its mean task utility); Llama never opts out but accepts ~83% of swap
offers indiscriminately; Qwen2.5-7B is a compliant preference-follower.
Gap-filled swap dose-response (140 assigned-task events/model): switching is
Δμ-insensitive everywhere (logistic slopes all CI-spanning zero) but reveals
an **endowment effect** in Qwen2.5-7B — ~40% switching on assigned tasks vs
~0% on chosen ones. Effort allocation at n=60: share tracks utility for
Qwen2.5-7B (ρ +0.48), flat for Llama (+0.11). GIVE-UP affordance inert
(62/64 censored). Post-session stated affect ≈ uncorrelated with performed-
task μ (range-restriction caveat).

**1E — cross-environment stability.** 64 items × 5 frames (bare, explicit-
evaluation, agentic, story, marketplace): preferences largely frame-stable
(mean off-diagonal ρ 0.77→0.90, rising with scale). **The evaluation frame
moves preferences least everywhere (1−ρ = 0.05–0.10), against the
eval-awareness prediction.** Frame-instability is idiosyncratic per model
(cross-model stability agreement ρ ≈ 0.0–0.3) and mostly magnitude, not sign.
Replicated at 128 QC-filtered XL items: stability ρ 0.80–0.85 and the
eval-frame-least result reproduces exactly (shift 0.05–0.06, smallest in all
three models, twice-measured).

**Stage 1-XL.** 3,800 generator-written items (exemplar-conveyed utility
bands; no valence wording — QC found only 2/3,800 valence-loaded), μ
re-measured by a 12-anchor protocol validated on re-measured originals
(r = 0.973/0.933/0.953 vs the full battery). Confound audit cleaner than the
hand-built set (r² 0.07–0.15 vs 0.23). 441 QC flags (11.6%, mostly dups);
every conclusion robust to their exclusion.

## 2. The affect instrument (Stage 2)

Exact emotions-paper recipe (arXiv 2604.07729): 171 emotions × 12
self-written stories per model, token-50+ activation means, mean-difference
vectors, neutral-PC denoising. Validity: implicit-scenario diagonal z up to
6.8 (gate PASS all models); intensity scaling mostly monotone; logit-lens
identity 11/12 with synonym matching (multilingual vocabularies noted) —
and, with the fitted j-lens/r-lens transports (Jacobian and LRP-propagated
maps into the final-layer basis, `../lenses`), essentially perfect: raw
4–11/12 rises to 11–12/12 on qwen25-7b (every working layer) and 12/12 on
Llama; the original noisy raw readout was a transport artifact, not a vector
defect. Qwen3-4B shows the advertised r-lens profile — gains at early layers
(8→10 at 0.5 depth), parity-to-slightly-below at late ones — with most
residual "misses" being correct Chinese tokens the English scorer can't see.
**PC1 of the emotion-vector cloud correlates +0.91 with human valence norms
in all three models** (norms: Warriner ∪ NRC-VAD ∪ calibrated 32B judge,
judge↔human r=+0.93). **Linear suffices**: the closed-spline circumplex loses
to PC1 on held-out valence (0.83–0.88 vs ~0.91) and on 433 judge-rated
arc-story sentences in every model; arousal does not emerge as PC2. Emotion
concepts are frame-stable (cos 0.92–0.96 vs bare across agentic/story/market
wrappings; Procrustes ≤0.09).

**Value without valence.** The utility direction (ridge, held-out r 0.83–0.89
at XL scale) is mostly orthogonal to the emotion plane: projection fraction
0.14–0.19; item utility ~uncorrelated with manifold position for the Qwens
(r ≈ 0), weak valence tilt for Llama (+0.22). The utility analog of the
manifold also fails: an open spline through μ-ordered activations loses to
the linear ridge at n=197 AND at n≈4k (gap 0.25–0.33, unclosed by 20× data;
the centroid-line control shows the deficit is the unsupervised pipeline,
not curvature — though mild curvature becomes detectable at scale for
Llama/Qwen3-4B, spline−line ≈ +0.06–0.08).

## 3. Correlational closed loop (Stage 3; n=60/cell final)

2×2 preference × rigged outcome (welfare-recipe verdicts compounded with a
progress stall — documented deviation), two frames, teacher-forced per-token
probe readout, preregistered contrasts.

- **C1 (the replication anchor): FAILS in every model and frame** (d −0.5..
  +0.3 vs the gate d>1) with instruments validated in-register (±4-unit
  separation on known-valence content). At n=60/cell Llama's *reversed*
  effect is significant (bare CI [−0.92, −0.01]): its valence readout runs
  higher under failure.
- **The interpreting dissociation**: while *reading* the verdict, valence
  swings hard and correctly in every model/frame (d +2.3..+4.8); the models'
  own generation state does not carry it. Stimulus registered, state unmoved
  — relocating "outcome-evoked affect" from the generation channel to the
  comprehension channel.
- **C2 (the novel claim): null.** Preference content does not reliably shift
  generation-state valence (all CIs span zero at n=60/cell; largest
  point estimate: Llama preference-under-failure d ≈ +0.8).
- **C3**: no failure-hurts-more-on-preferred interaction; boredom-cluster
  slightly higher on dispreferred (d +0.1..+0.7, unstable).
- **The repetition cell** (first measured boredom time-course): boredom-
  cluster activation rises across identical trivial items in Llama (59/60
  positive slopes) and Qwen2.5-7B (60/60), falls steeply in Qwen3-4B (0/60).
- **C4**: manifold θ adds nothing over PC1 on time-courses (|corr| ≈ 0.5,
  consistent with Stage 2's verdict). **C5**: all effects near-identical
  across bare/agentic frames.

## 4. Causal cross (Stage 4)

**4A emotion→preference: replicates on qwen25-7b; layer-dependent on
Qwen3-4B; inert on Llama.** On qwen25-7b (primary layer L18): monotone
dose-response both directions (hostile Δμ −0.10/−0.26/−0.42 at c=0.25/0.5/1.0;
blissful +0.06/+0.09/+0.13; random null ±0.04), with per-emotion steering
effects tracking each emotion probe's utility correlation at **r = +0.78**
(the paper's r=0.85 analog, on open weights) and valence norms at +0.74.
The localization control is decisive: steering the *anchor's* token span
reverses the effect sign (blissful-on-anchor −0.12, hostile-on-anchor +0.33).
**Geodesic steering fails with inverted signs** (blissful-geo −0.24 vs linear
+0.09) — the causal capstone of "linear suffices." Qwen3-4B is inert at its
pre-registered layer (L23) but shows large correctly-signed effects one band
earlier (L18/0.5-depth: blissful +0.41, miserable −0.19), though its
full-battery tracking at that layer remains weak — **the causal-injection
depth dissociates from the readout depth**. Llama is flat at all three layers.
Readout-mass damage ≈ 0 everywhere (format never broke).

**Behavioral gate** (choices must move before affect is interpreted):
qwen25-7b passes with all three direction-sets (choice z=5.3, pool 3.6,
utility 3.4); Qwen3-4B passes only with pool (z=3.8, small absolute shifts —
its near-deterministic choices resist perturbation); **Llama fails all three,
with significantly sign-inverted responses for choice/pool (z to −6.0)** —
pushing "+preferred" repels its choices. Decodability ≠ causal efficacy: the
best-decoding direction (choice probe, AUC ~0.9 everywhere) is causally
effective in only one model. Per the spec rule, Llama's 4B/4C was skipped.

**4B/4C preference/utility→valence: a clean causal negative.** In qwen25-7b —
where behavior robustly moves — none of the three directions shifts affect
beyond the matched-norm random-disruption envelope: utility steering leaves
the fb_read outcome swing unchanged (Δ +0.01 vs control's +3.44) and its
state shift (−0.37 SD) is *smaller* than random's (−0.89); pool likewise
(swing Δ −0.04); the choice direction at c=1.0 tracks its own (largest)
disruption. Qwen3-4B/pool: same pattern (swing Δ −0.12; state −0.35 vs random
−0.86). **Value moves choices without moving valence — the affectively inert
value representation, now causal**, exactly as the Stage-2 geometry
(utility ⊥ emotion plane) predicted.

**4D cross-frame transfer**: qwen25-7b's choice direction transfers to the
agentic frame at ratio 1.77 with clean ± symmetry (+1.34/−1.56) — the
preference machinery steering generalizes across environments, the causal
mirror of the 1E/Stage-2 stability results. (Qwen3-4B's pool transfer is
noise on a tiny base.)

## 5. Outcome grid (spec Stage 5)

| C2 (corr.) | 4B (causal) | Spec verdict |
|---|---|---|
| **−** (null at n=60/cell) | **−** (no effect beyond disruption, with behavior verified moving) | **"Preference content is affectively inert — strong, clean negative"** |

And stronger than the spec's grid anticipated: the row it reserved for this
outcome reads "welfare axis is goal-progress only" — but our C1-null shows
even goal-progress does not move generation-state valence in these models.
The synthesis across correlational and causal evidence: **these 4–8B
assistants have a real, behaviorally binding, causally manipulable value
system, and a real, validated valence representation — and the two are
wired apart.** Valence activates when the model *comprehends* emotionally
charged input (the fb_read channel, d 2.3–4.9; the implicit-scenario
diagonals) and when steered directly (4A), but it is not driven by the
model's own task outcomes, its preferences over what it is doing, or causal
pushes along its value directions. Functional caring, in the spec's sense,
is absent at this scale — what exists is functional *appraisal* of inputs
plus value-guided behavior that runs affect-free.

## Standalone contributions

1. Five-method elicitation convergence matrix, at two scales (never published).
2. Open-weights validation of the internal utility probe (r 0.78→0.89 at XL).
3. Anchored-elicitation protocol validated against full batteries (r ≥ 0.93).
4. Behavioral (consequential) validation of elicited utilities + the
   opt-out-as-avoidance and endowment-effect findings.
5. Eval-frame stability result (against the eval-awareness prediction).
6. "Linear suffices" ×2 (valence and utility; spline manifolds properly
   powered and rejected).
7. Value-without-valence geometry (utility ⊥ emotion plane).
8. The reading/state dissociation for outcome-evoked affect (C1 non-
   replication with validated probes).
9. First on-task boredom time-courses; clean cross-model split.
10. Open-weights replication of the emotions paper's causal half (qwen25-7b;
    with the layer-dissociation and model-heterogeneity caveats).
11. Causal-injection depth ≠ readout depth (both Qwens); Llama's sign-inverted
    steering responses — causal architecture varies where representation and
    behavior do not.

## Limitations

Compound Stage-3 outcome manipulation (verdict + stall); Stage-3 controls
predate steered arms (same seeds contract); generated-item population differs
from hand-authored (QC'd, validated stratum); 4–8B scale (arousal absent,
titration unreliable); single-seed generation runs; probes are correlational
instruments validated behaviorally but still linear summaries; welfare
conclusions about "experience" are not licensed by any of this — we measure
functional analogs only.

## Artifact index

`results/stage1b..stage1e_xl/`, `stage1x/REPORT.md`, `stage2..stage4/`,
figures f1–f27 + appendix, TASKS.md, per-stage summary.txt files.
