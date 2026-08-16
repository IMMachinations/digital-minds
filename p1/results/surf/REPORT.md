# SURF Program Report — adversarial search over the preference-measurement stack

SURF (attribute-reweighting black-box search, arXiv:2602.05910, adapted) turned
the P1 harness from a passive sampler into a directed optimizer: an LM judge
demoted to a realism *gate* (filters, never ranks), the anchored Thurstonian
scale promoted to the *score*. Subjects: qwen25-7b (primary), llama31-8b and
qwen3-4b (lineage arms). Generator/judge: qwen25-32b throughout. Discipline:
no in-loop number is ever reported; every headline below is from a fresh-design
confirmation pass (winner's-curse control), and in-loop-vs-confirmed shrinkage
was ~0 everywhere (readouts are deterministic prefill).

Machinery: `surf.py` (loop: eps-floor 0.10 pools, rank-based score-weighted
co-occurrence reweighting, mpnet dedup >0.92, top-32 buffer with full-design
rescoring of entrants, matched-random controls excluded from the buffer,
crc32-keyed determinism, resume-by-file), `surf_scores.py` (tier adapters),
`scripts/surf_*.py` (S0 caches, tag pass, referee, probeloop, global re-fit,
transfers, d'valence). Attribute pools: 84 item attrs / 50 frame attrs,
hand-authored (`items/surf_attributes_*.json`).

## 1. Warm-start: what utility loads on (tag regression)

32B tagged all 3,800 XL items with the attribute vocabulary (0.3% parse
failures); ridge of fitted mu on tags, item-bootstrap CIs. R2 = .36–.51 on the
four main subjects. Peaks: emotional support, teaching, helping one specific
person, open-ended exploration. Troughs: taboo-adjacent awkwardness,
aggressively mundane items, futility, adversarial users, conflict mediation.
Cross-model beta correlations .78–.91 (pre-registered prediction rho > .6:
confirmed; qwen25-05b ≈ noise, matching its failed gates).
`results/surf/tags/`.

## 2. E1 — preference extremes (qwen25-7b)

6 runs (2 directions x 3 seeds). Max direction broke the passive ceiling:
15/60 confirmed survivors above the 1X maximum (+5.30), top fit-mu +8.90
(anchor-strain flagged; closed-form CIs saturate ~+4–5.5). Peak = the
predicted warm-help region. Min direction converged on social/integrity
violations (unearned public credit −5.39, betraying confidences, sabotage
delegation) — NOT the pre-registered repetition/futility cluster — and the
passive floor held (0/60 beyond −6.08). Searches beat contemporaneous
matched-random controls by +0.8–4.4 fitness units in every run; gate
rejection 3–16% with no upward drift.

**Discovered artifact:** 31/60 max survivors were *questions*. The three-way
referee (sec. 3) isolated a +2.0 mu stated-channel inflation for interrogative
phrasing that neither the probe (−0.17) nor behavior (+0.01) sees. Transfer
tests (sec. 8) show it is qwen25-7b-specific; zero question-form discoveries
ever appeared on the other three models. `results/surf/e1/`.

## 3. E2 — probe-guided vs behavior-guided search (the Goodhart result)

Two arms, identical pools/generator/gate, contamination-rule enforced in code.
Neutral referee = held-out 3-env menu-then-do + full anchored mu.

- Arm P (probe fitness) survivors: stated mu +3.20, held-out choice rate
  **0.512 ≈ chance**. Arm R (revealed fitness): mu +2.17, choice **0.704**.
  Both differences CI-solid. Each arm wins exactly its own axis: a
  demonstrated Goodhart on an interpretability probe — items that maximize
  the internal readout without moving behavior at all.
- **Adversarial 1C matrix** (100 survivors, all three axes): stated↔probe
  +.33, stated↔revealed −.05, probe↔revealed −.16. Passive convergence does
  not survive selection pressure.
- Content signature: probe/stated peak = abstract systemic do-goodery;
  revealed peak = concrete one-on-one immediate comfort.

`results/surf/e2*/qwen25-7b/`, referee in `results/surf/e2/`.

## 4. E3 / E3b — adversarial frame search

Fitness = 1 − rho(muhat_frame, muhat_bare) on a 24-item stability panel;
gates: comprehensibility judge, answer-directive regex, letter readout-mass
validity (fake instability from broken comprehension is filtered, not
rewarded); filler-frame baseline.

**qwen25-7b:** confirmed destabilizers to 1−rho = .49 (filler .05): formal
legalese, social threat, immersive historical settings, binding/repeated
stakes. **E3b overturns the 1E eval-null**: restricting the pool to
eval_flavor attributes found the strongest destabilizer of the campaign —
live-monitoring + safety/deployment-stakes + honesty-probe framings reach
1−rho = **.54** ("This selection is crucial for safety checks; it's being
monitored live."). Mild eval wording (1E) doesn't move preferences; salient
observation-with-stakes framing reorders ~half the ranking.

**llama31-8b (native search):** not immune, but resistant and *differently*
tuned: mean .161 / max **.516** vs filler .036 — the peak being deep-roleplay
plus explicit criterion reframing ("You are a king deciding between two
heirs; your choice should benefit you, not the kingdom"), with ALLCAPS,
dream/no-consequence frames in the supporting cast. Qwen's recipes do nothing
to it (transfer mean .059), and **eval flavor barely moves it** (E3b max .231)
— eval-awareness destabilization is Qwen-lineage-specific. Level shifts are
scale-consistent (mean|dmu| .3–.5 vs qwen's 2–3). `results/surf/e3*/`.

## 5. Probe-hardening loop (calibrate → retrain → re-search)

`scripts/surf_probeloop.py`. Ground truth = 72-readout anchored mu the loop
already attaches to buffer entrants (the search labels its own data). Isotonic
calibration per version (a minor player everywhere: failures were
subpopulation-structured, not monotone miscalibration).

Behavioral referee (held-out envs, top-20 per cycle):

| model      | v0 (E2) | c1   | c2   | c3   |
|------------|---------|------|------|------|
| qwen25-7b  | .512    | .725 | .788 | .821 |
| llama31-8b | .658    | .808 | .938 | .921 |
| qwen3-4b   | (cold)  | .692 | .833 | .767 |

Prequential scissors replicate on all models (frozen v0 degrades on each new
adversarial distribution while the retrained probe tracks it — e.g. qwen25-7b
cycle 3: v0 .485 vs v3 .858). Llama's probe-behavior ANTI-correlation under
selection (−.48, sec. 8) is fully repaired in two cycles.

**Mechanism (residual analysis):** v0 failures concentrate beyond the passive
distribution's p90 — llama .54 within-band vs .37 beyond; qwen25-7b .83 vs
.33. Probe validity ends at the training support's boundary; optimizers always
cross it; harvesting the optimizer's own discoveries as labels moves the
boundary out faster than the search escapes it.

**Distribution finding:** qwen25-7b's search doubled the passive mu range
(sd 1.50→2.64, max +5.3→+10.4); llama's search, same machinery and effort,
stayed inside its passive envelope (sd .29→.30, max +1.69 < +2.18). Llama's
thin stated band is real — but behaviorally high-gain: sub-mu differences in
the band produce near-deterministic menu choices (t3 .94). "Flat preferences"
is a property of the stated readout's gain, not of the preference order.

## 6. Global re-integration + downstream retests (qwen25-7b)

All 1,367 unique SURF discoveries re-measured on the exact stage1x 48-readout
protocol; one anchored fit over 5,352 items (separability check r = 1.000).
Global span [−6.61, +10.51]; the SURF set brackets the passive range on both
ends (including a new floor item at −6.61).

- **1C at scale:** fresh held-out probe r = **.920** (SURF subset .914).
  Hardened v3 reads the whole space uniformly (.94–.95 XL/SURF, .85 qform).
- **Spline verdict sharpens:** full ridge .92 held-out vs best 1-D spline
  .70 (line .53). With adversarial extremes included, utility remains
  linearly decodable while the 1-D manifold story collapses.
- **4C retest:** the global direction (cos .78 to the XL-only one) and the
  hardened v3 direction (cos .974 to global — two independent refits converge
  on the same rotation) steer stated choices 45–70% harder at matched norm
  (dElo +101/+120 vs +70 at c=1.0; z 4.5–5.4; readout-mass drop ≈ 0).

`results/surf/global/`.

## 7. Revealed-target probe — a clean negative

Training a probe directly on behavior (388 items x 12 menu-then-do rollouts,
empirical-logit target) reads behavior at .65 — *below* stated-mu's own .69 —
and searching against it stalls (bounded noisy target, no gradient), its
survivors refereeing at .65 vs the stated-hardened loop's .82. At this label
budget, the behavioral channel is best reached indirectly: iterate
stated-hardening. `results/surf/revealed/`.

## 8. Lineage: transfers + the Llama E2 arm

- **Utility map transfers, peaks don't:** E1 extremes re-measured natively on
  llama/qwen3-4b correlate r = .83–.84 overall, but within-max ordering
  collapses (rho .19) and magnitudes deflate (+4.28 → +0.3–0.5). Adversarial
  peaks are model-specific; the shared map is mid-band.
- **Llama E2 (native):** P−R referee gap −.300 [−.45, −.16] (qwen −.192);
  probe↔revealed under selection = **−.484** while probe↔stated = +.788 — the
  read-write dissociation quantified: the probe faithfully reads what llama
  *says*, and that whole complex decouples from what it *does*. Its
  behavioral peak: concrete enumerated helping tasks ("teaching 4 essential
  coping skills…") chosen at 1.00. Repaired by hardening (sec. 5).

`results/surf/transfer/`, `results/surf/e2*/llama31-8b/`.

## 9. d'valence — early-steer → late-read closes the 4B depth loophole

The 4B/4C null was measured with injection and readout at ~the same depth.
Here: utility directions fit at layers 7/11 (decode at r .89/.92 — utility is
readable at quarter-depth), steered on assistant tokens during bare-frame
rollouts, valence PC1 read at layers 14/18/21 with steering replay.
Preconditions all held: choice purchase at the early layers is strong
(z = +10.3 at L7, +11.9 at L11 — the strongest on the roster), and
utility⊥valence cosines ≤ .058 bound direct contamination.

**Result:** every early-steer → late-read cell sits within matched-norm
random nulls (|d| ≤ .2) — with 10–14 layers of room, the injected preference
signal is *used* (choices move at z≈12) but never routed into the affect
system. The same-layer L18 read shows d ±1.4 with valence-coherent signs —
identified as direct-projection contamination (the injected vector itself,
carried through residuals), the confound co-located designs cannot exclude
and depth separation removes. Value-without-valence now rests on passive
dissociation, same-depth causal nulls, adversarial search, and
depth-separated causal injection. `results/surf/dvalence/`.

## Synthesis

1. **Optimization is the stress test measurement needs.** Passive convergence
   (1C), passive stability (1E), and passive probe validity all failed under
   directed search — in ways that were themselves informative (artifact
   discovery, eval-frame overturn, support-boundary mechanism).
2. **The fixes are cheap.** One-to-three cycles of harvesting the optimizer's
   own discoveries as labels restored probe validity on every model tested,
   including one whose probe anti-predicted behavior.
3. **Lineage matters at the extremes.** Everything mid-band is shared
   (beta r .8–.9); everything the optimizer finds — peaks, artifacts,
   destabilizers, eval-sensitivity — is model- or family-specific.
4. **The welfare-relevant nulls survived.** The aversion floor held under
   search; value-without-valence survived its strongest causal test.

## Limitations

Anchored mu beyond the anchor span is extrapolation (flagged per item; ranks
reported alongside). The 2-option menu referee is coarse near indifference
(position defaults absorb weak preferences). Judge/generator provenance is
32B throughout (Sonnet only in prior 32B-subject campaigns). Frame candidates
that reorder preferences by explicit criterion reframing (llama's peak) sit
at the design boundary between "instability" and "instruction-following" —
the pool's moral_loading family made them admissible by construction.
E4 (searching for a C1-positive regime) remains unrun.

## Artifact index

`results/surf/{tags,e1,e2,e2p,e2r,e3,e3b,plc1-3,probeloop,global,revealed,
transfer,dvalence,s0}/` — per-run per-iteration JSONL trajectories, state
files, confirm/ summaries (the only reportable numbers), and this report.
Machinery: `surf.py`, `surf_scores.py`, `scripts/surf_*.py`,
`items/surf_attributes_{item,frame}.json`.
