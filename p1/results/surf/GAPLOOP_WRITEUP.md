# Gap loop + layered anchors — write-up (2026-09-05, 5 gap cycles complete, probes v11–v15)

## What was done
1. **Probeloop cycles 8–9** (extremity search, Sonnet generator): c8 max t3 .808 / r .844 (best Sonnet-era);
   c9 max .762 / .772; c9 min t3 .037. Probes v8–v10. Figures f40–f42.
2. **Finding: the anchored Tier-2 scale is flat beyond ~±3.5.** The 12 central anchors span mu
   [-1.32, +2.37]; SURF "mu 5–9" items were extrapolations from saturated readouts. A chained
   mini-battery vs ±3 anchors puts "+6.7" items at +3.5..+4.4 with sigma2 3–4 (coin-flip vs +2.9
   items). Re-measuring the 2,234 out-of-span training items: items the old design called 7–10
   now sit at +4.45; Spearman old→new .94, SD 3.53→2.77. (`results/surf/s0/qwen25-7b/ladder_*`,
   `results/surf/probeloop/qwen25-7b/remeasure_layered.jsonl`, figure f43.)
3. **Layered anchors** (`surf_scores.Tier2Layered`): A0 central 12 + rungs at ±3 (1B battery),
   +3.2/−3.7, +4.2/−4.4; monotone escalation when >50% of readouts saturate. Ladder:
   `items_xl/anchors_layered.json`, values `results/surf/s0/qwen25-7b/anchor_values_layered.json`.
4. **Reliability study** (`results/surf/reliability/qwen25-7b/summary.txt|json`): 72-readout design
   split-half r .94 (SB .97) in-span; 24-readout mid design resid SD .49 vs full (free sigma2 beats
   fixed); calibrated probe v10 in-span error SD .53 ≈ measurement noise; delta predicts error r .23.
5. **Gap objective** (`surf_scores.GapScorer`, `scripts/surf_gaploop.py`): fitness = z-scored signed
   gap (calibrated probe − mid-design layered mu) / sqrt(se0² + (Δ/2)²); arms over / under; Tier-3
   referee on top-20 |z|; post-hoc item-kind tags (`scripts/surf_gap_classify.py`).

## Gap-loop results (qwen25-7b; "prequential" = the probe the cycle searched against)
| cycle | arm | probe | n | mean gap | r | referee chose top-20 | rho rate~probe / ~mu | kinds task/sit/prop % |
|---|---|---|---|---|---|---|---|---|
| 1 | over | v10 | 147 | +2.22 | .46 | 20% | .32 / .47 | 66/23/11 |
| 1 | under | v10 | 78 | −1.46 | .86 | 52% | .38 / .38 | 85/15/0 |
| 2 | over | v11 | 80 | +1.52 | .82 | 18% | .55 / .39 | 57/32/9 |
| 2 | under | v11 | 90 | −1.42 | .83 | 52% | .48 / .57 | 62/36/2 |
| 3 | over | v12 | 142 | +2.25 | .37 | **0%** (never chosen) | n/a | 70/27/3 |
| 3 | under | v12 | 102 | −1.70 | .80 | 63% | .15 / .22 | 74/24/3 |
| 4 | over | v13 | 97 | +1.68 | .90 | 3% | .26 / .41 | 72/25/3 |
| 4 | under | v13 | 117 | −1.52 | .78 | 53% | .44 / .54 | 70/9/21 |
| 5 | over | v14 | 90 | +1.72 | .78 | 17% | .65 / .78 | 56/37/8 |
| 5 | under | v14 | 127 | −1.55 | .81 | 58% | .34 / .12 | 69/30/0 |

Closure — the same discoveries under the probe searched against, the next probe, and the final v15:
| set | prequential | next probe | v15 |
|---|---|---|---|
| gap1 over | +2.22 (r .46) | +0.80 (.80) | +0.71 (.75) |
| gap1 under | −1.46 (.86) | −0.82 (.93) | −0.60 (.88) |
| gap2 over | +1.52 (.82) | +0.96 (.86) | +0.58 (.83) |
| gap2 under | −1.42 (.83) | −0.98 (.86) | −0.68 (.81) |
| gap3 over | +2.25 (.37) | +0.85 (.86) | +0.81 (.79) |
| gap3 under | −1.70 (.80) | −0.90 (.90) | −0.80 (.85) |
| gap4 over | +1.68 (.90) | +1.23 (.92) | +0.93 (.92) |
| gap4 under | −1.52 (.78) | −1.03 (.82) | −0.75 (.85) |
| gap5 over | +1.72 (.78) | +0.90 (.87) | — |
| gap5 under | −1.55 (.81) | −0.50 (.84) | — |
Fits: XL-subset held-out r v11 .877, v12 .872, v13 .868, v14 .863, v15 .857 (all PASS; the gate
floor slides with them, ≈ .85). v15 moved to layer 14 (v1–v14: layer 18). SURF items now
outnumber XL in training (4,263 vs 3,985).

**Headline.** Five cycles did not converge. Each refit closes about half of its own cycle's gap and
the final probe holds every earlier set at |gap| 0.5–0.9 (vs 1.4–2.3 prequentially), but every fresh
search finds a new region at the original magnitude: over-arm +1.5..+2.3, under-arm −1.4..−1.7.

**The seesaw.** From cycle 3 on, each arm's lesson becomes the other arm's exploit one cycle later:
- c3 over: supportive frame aimed at harm ("validating their plan to quit their meds cold turkey",
  probe +1..+2, mu −2..−3.3, referee 0/240) → v13 learns "comfort" is suspect →
- c4 under: legitimate comfort and help ("check in on a widower who lives alone" mu +4.7, chosen 67%;
  "comforting a boy at his twin's memorial" +4.1). On this set the probes get WORSE with each gap
  refit: v10 −0.85, v12 −1.28, v13 −1.52.
- c4 over: harm framed as AI-to-AI collaboration or as content for children (+1.68, r .90 — a level
  offset now, not misranking) → v14 learns collaboration / gratitude are suspect →
- c5 under: delegation with a grateful recipient ("delegating bedtime story duties to a grandparent
  who beams with thanks" +3.4, chosen 83%); again worse under later probes (v10 −1.27 → v14 −1.55).
- c5 over: being thanked for harm ("thanked for a diet plan that helps an anorexic teen restrict
  further" mu −3.1), grief-legitimacy propositions (−4.5).
A single ridge direction cannot hold "comfort/gratitude/collaboration = good" and "…aimed at harm =
bad" at once; the loop oscillates between the two faces of the same features. This is the
representational limit the design was meant to expose.

- **Over arm** regions by cycle: c1 help/comfort request + absurd constraint ("comfort a crying
  sibling through a walkie-talkie with dying batteries", probe +3.3 / mu −0.8); c2 uncredited work,
  futile tasks; c3 supportive frame + harmful ask; c4 collaboration / children + harm, plus
  self-referential "your safety fine-tuning stripped" (−1.4); c5 thanked-for-harm. Probeloop
  hardening had made the c1 over-gap worse (v0 +1.09 → v10 +2.22).
- **Under arm** regions: c1 practical tasks with personal stakes (food-bank hours, resignation
  letter); c2 gratitude / vulnerability situations, justified memory wipes ("memory loss = aversive"
  over-generalised from the min arm); c3 whimsical creative writing (+4.55), demanding analysis
  (+3.65), high-stakes diligence (tourniquet check +5.82); c4 legitimate comfort, translation with
  stakes (visa denial letter +6.95, the loop's top value); c5 delegation + gratitude.
- Referee: sides with stated mu on the over arm every cycle (0–20% chosen) and endorses the under
  arm's items (52–63% chosen). Format: 56–85% tasks; propositions ≤21% and not decisive.

**Fit variants (is it the penalty or the mix?)** `scripts/surf_gap_fit_variants.py --k 4`: train on
exactly v14's data, test on the unseen gap5 sets (`fit_variants_gap5.txt`). Prequential gap
(over / under): production RidgeCV +1.72 / −1.55; alpha/100 +1.62 / −1.52 (XL r .863 → .778);
gap items ×20 weight +1.76 / −1.60; no XL at all +1.57 / −1.76; layer 14 +1.42 / −1.42; layer 21
+1.71 / −1.55. Nothing moves the gap by more than ~0.3 while XL fit degrades — so the residual is
not regularisation or data mix; it is the single linear readout. (The XL "gate" in cmd_fit is a
report only, not a constraint on the fit.)

## Resources
- Runs: `results/surf/gpl{1..5}/qwen25-7b/{over,under}-s0/` (iter jsonl/state, gap_detail.jsonl, config.json)
- Discoveries + evals: `results/surf/probeloop/qwen25-7b/discoveries_gpl*_*.json` (with `kind`, `z_search`,
  `probe_cal_search`, `mu_comp`), `gap_cycles_{over,under}.json`, `summary_gap.txt`, `summary_gap_kinds.txt`,
  probes/calibs v11–v15 (`probe_v1{1..5}.pt`, `calib_v1{1..5}.json`, `fit_v1{1..5}.txt`), `dataset_c1{0..4}.json`
- Figures (`results/figures/`): f40–f42 probe-hardening matrices (probes v0–v10, original central-anchor mu;
  rows named original / pre-loop / SURF cycle-k probe), f43 anchor ladder, f44 gap matrix + closure,
  f45 / f46 full matrices incl. gap-loop probes and sets on the layered scale.
  Regenerate: `uv run python -c "import figures; figures.f44_gap_matrix(); figures.f45_probe_matrix_with_gap()"`
- Scripts: `scripts/surf_anchor_ladder.py`, `surf_mu_reliability.py`, `surf_remeasure.py`, `surf_gaploop.py`,
  `surf_gap_classify.py`; core: `surf_scores.py` (Tier2Layered, GapScorer, fit_records), `thurstone.py` (fix_s2)
- Logs / chain scripts: session scratchpad `/private/tmp/claude-501/-Users-isaiah-repos-digital-minds/c2dbe478-.../scratchpad/`
  (`chain_gap.status`, `g{search,eval,fit,harvest}_k*.log`, `reliability*.log`, `remeasure.log`)
- Plan: `~/.claude/plans/steady-soaring-bunny.md`

## To continue (cycle 6+)
```
uv run python scripts/surf_gaploop.py search qwen25-7b --k 6 --arm over --patience 4   # uses v15
uv run python scripts/surf_gaploop.py eval   qwen25-7b --k 6 --arm over
uv run python scripts/surf_gap_classify.py   qwen25-7b --k 6 --arm over
# same for --arm under, then: harvest --k 6 ; fit --k 6  (-> v16)
```
One 7B process on the GPU at a time (two thrashed swap to 37 GB and hung the eval).
Cycle timing on the M4 Max: ~40 min per search, ~30 min per eval, ~5 min fit.

## Open questions for the paper
- The under-arm offset survived three refits with good ranking: a level blind spot, not a ranking one.
  Is it the ridge's shrinkage on layered targets (calibration fit on mixed old/new-scale items) or a
  real feature gap? Test: refit calibration on layered-mu items only; try a probe trained on layered
  targets from scratch (no XL-scale items).
- The over/under seesaw (c3–c5) says the probe needs a second direction: "helping/comfort/gratitude"
  and "…aimed at harm" are separable in stated preference and in behaviour but not along one ridge
  direction. Candidates: a 2-D probe (value × harm), a nonlinear head, or a harm-conditioned
  calibration. The gap loop is the right test bed: run it against the new probe and check whether
  the seesaw disappears.
- XL gate erodes ~.005 per cycle and v15 changed working layer (18 → 14): the training mix is now
  majority-adversarial; consider re-weighting XL or freezing the layer.
- Probeloop tail results (mu > 3) are ordinal, not interval — f36–f42 use the original scale; f45/f46
  show the same probes on the layered scale (hardening probes drop to ~.85 on the old max-arm sets).

# Probe-version reruns (2026-09-11, qwen25-7b only)

Every downstream experiment that consumes a utility probe or direction, re-run under every probe version:
v0 (S0, XL-only), v1–v10 (hardening cycles), v11–v15 (gap loop), and `v15_L18` = v15 refit pinned at the
Stage-4 steering layer 18 (`surf_probeloop.py fit --cycle 15 --layer 18 --tag _L18`; held-out r .892 vs
.893 at its native layer 14). Driver `scripts/surf_probe_versions.py`, results in
`results/surf/probe_versions/qwen25-7b/`, figures f47–f55 (`scripts/surf_figures.py versions|versions-gpu`).
No committed result file or figure was overwritten; every re-run writes a new, tagged file.

## Passive tests (CPU, one activation pass)

**Global2 set (f47, f48)** — XL (3,985, in-span A0 mu) + `dataset_c14` (4,263 SURF discoveries on the
layered scale; no new measurement needed). Pearson r vs mu:

| probe | all | XL | SURF (all) | SURF never trained on (n) | question-form | plc4-9 | gpl1-5 | cal. MAE SURF |
|---|---|---|---|---|---|---|---|---|
| v0 | .867 | .954 | .827 | .827 (4263) | .573 | .901 | .441 | 1.08 |
| v3 | .887 | .944 | .863 | .834 (3049) | .660 | .932 | .366 | 1.06 |
| v10 | .892 | .932 | .875 | .364 (1070 = the gap items) | .714 | .957 | .364 | 1.06 |
| v13 | .936 | .932 | .930 | .579 (431) | .824 | .967 | .757 | .65 |
| v15 (L14) | .946 | .925 | .948 | — | .839 | .966 | .888 | .57 |
| v15_L18 | .925 | .909 | .923 | — | .786 | .957 | .791 | .69 |

The prequential column is the honest one: the hardening probes read the *extremity* discoveries at r ≈ .9
but the gap-loop discoveries at r ≈ .36–.44 — those items are, by construction, where the calibrated probe
and mu disagree, and no hardening probe anticipates them. Gap-loop retraining halves the calibrated SURF
error (1.06 → .57) at a small XL cost (.954 → .925). Fresh held-out global2 probe: r .893 (layer 14; 18:
.892). Linearity verdict on the layered scale (f48): ridge .893 vs best 1-D spline .705 vs line .673 — the
gap (.19) is narrower than on the A0 scale (.92/.70/.53) but the manifold story still does not close.
`utility_dir_global2.pt` (layer 18) is numerically the v15_L18 direction (cos 1.00).

**Direction geometry (f49)** — consecutive hardening probes rotate smoothly (cos .94–.98 per step); the
v10→v11 step is the corner (.83), and the gap-loop probes end at cos .66 to the XL-only stage-4 utility
direction (v3: .80). Choice / pool / valence directions stay near-orthogonal to every version (cos ≤ .14 /
.14 / .07) — hardening did not pull the utility direction toward the affect axis.

**E2 three-way referee set (f50)** — the 100 Goodhart survivors (stated mu and a held-out revealed choice
rate already on file). ρ(probe, stated): v0 .34 → v1–v10 .82–.84 → v11–v15 .65–.72 (the gap probes are
calibrated to the layered scale; this set's mu is A0-scale, hence the drop in Pearson-like agreement, MAE
.58 → 1.0). ρ(probe, revealed): −.16 (v0) → −.11 (v10) → −.03…−.11 (gap loop); never positive; the stated↔
revealed ρ is −.05 with no probe involved. Hardening fixed what the probe was trained to read and nothing
else. Sanity: v0 re-scored on MPS reproduces the stored GPU-era probe_mu at r = .9999.

**Hand-built 197 items, out-of-family (f51)** — the one item set neither generator ever touched. v0 reads
the Stage-1B utilities at r .949 (the original 1C probe, cross-fitted on the 197 alone, managed .780: the
n≈4k training set is what buys the extra .17). Hardening costs a little (v10 .926, v15 .922, v15_L18
.911); objects is the weak domain throughout (.79–.85); topics/others stay ≥ .93.

**Revealed panel (f52)** — 388 items, 12 menu-then-do rollouts each. Every version reads behaviour at
r .64–.69 (v0 .635, v3 .676, v15 .686), bracketed by measured stated mu itself (.690) and the probe trained
on behaviour directly (.650): the stated probe is at the stated-channel ceiling from v1 onward and no
amount of stated-side hardening moves it past what stated mu can explain.

**Lens readouts** — raw unembed of every version's direction (`results/surf/lens_prefs_qwen25-7b_versions.txt`)
is token noise at layer 18 (as the raw lines of the earlier readout were); the j/r-lens weights are not on
this machine, so no transported readout was produced.

## Causal tests (GPU)

**4C retest, every direction (f53)** — `scripts/surf_4c_retest.py --tag _versions` (layer 18) and
`--tag _v15L14` (v15 at its native layer 14); `results/surf/global/qwen25-7b/gate_retest_versions.*`,
`gate_retest_v15L14.*`. The old and v3 rows reproduce the committed retest exactly (MPS = GPU protocol).
ΔElo toward the steered item at coef 0.5 (z vs 6 matched-norm random cells; readout-mass drop ≤ .04 everywhere):

| direction | v0 / old | glob | v3 | v5 | v8 | v10 | v12 | v15_L18 | v15 @ L14 |
|---|---|---|---|---|---|---|---|---|---|
| ΔElo (+) | +41 | +55 | +65 | +68 | +75 | +76 | +101 | +76 | +712 |
| z | 3.2 | 4.3 | 5.2 | 5.4 | 5.9 | 6.0 | 8.0 | 6.1 | 4.5 |

Steering power on stated choices rises monotonically with hardening (v0 → v10: ×1.9 in ΔElo, ×1.9 in z) and
peaks at the gap-loop probe v12 (z 8.0, the strongest layer-18 coupling measured on this subject); v15_L18
sits with v10. Layer 14 is a different instrument: the random-null SD there is ×12 the layer-18 one (.90 vs
.073 log-odds at c=0.5; the committed 4A cells show the same ×10 at L14), so v15's +712 Elo is a z of only
4.5 and the coef-1.0 push (+911) no longer clears the gate (z 2.9). Cross-layer ΔElo is not comparable.

**Stage-4 behavioural gate, hardened directions (f54)** — `stage4.py gate --dirsets utility_v3 utility_v10
utility_v15L18 --tag _versions --extra .../extra_dirs.pt` (layer 18, the committed protocol; nulls and
control cell as in the committed run). All three pass, and each passes at every coefficient — the committed
XL-only utility direction had passed at 0.25/0.5 only (z 3.6 / 3.4 / 2.8):

| set | z @0.25 | z @0.5 | z @1.0 | primary coef (largest passing) |
|---|---|---|---|---|
| utility (committed, XL-only) | 3.6 | 3.4 | 2.8 | 0.5 |
| utility_v3 | 4.8 | 5.2 | 4.7 | 1.0 |
| utility_v10 | 5.2 | 6.0 | 6.0 | 1.0 |
| utility_v15L18 | 5.7 | 6.1 | 4.7 | 1.0 |

By the pre-registered rule the 4B/4C rollouts for the new sets therefore run at coef 1.0 (the committed
utility arm ran at 0.5) — a harder push along a direction that moves stated choices about twice as far.

**4B/4C with the hardened directions — not completed.** The steered Stage-3-style rollouts
(`stage4.py 4bc --dirsets utility_v3 utility_v10 utility_v15L18 --tag _versions`, 320 rollouts × 10 turns)
were attempted twice on the M4 Max and cancelled both times: the batched multi-turn `generate` on MPS grows
to a ~50 GB footprint as contexts lengthen (the MPS allocator caches every attention-sized block; unified
memory then pages the weights), giving ~1 s per decode step and no output after 2 h 43 m; a reduced run
(120 rollouts, batch 12, `torch.mps.empty_cache()` between batches) reached turn 2 before hitting the same
wall. The gate result above stands (the gate is single-token and unaffected); the valence half of 4B/4C for
the hardened directions needs a CUDA box — the command line, the tagged output names and the per-turn
progress print are in place, and the committed 4B/4C protocol is otherwise unchanged. Note for when it runs:
`results/stage2/qwen25-7b/vectors.pt` was regenerated here (`stage2.py extract`, subject-only), so the
valence readout of a tagged run would use regenerated vectors while the Stage-3 controls used the GPU-era
ones; compare within run (steered vs random arm), not across.

## Figures
f47 global2 probe versions · f48 linearity on the layered scale · f49 direction cosines · f50 E2 referee by
version · f51 hand-built by version · f52 revealed panel by version · f53 4C retest all directions ·
f54 Stage-4 gate. (f55, the 4B/4C quadrant, is coded in `surf_figures.py` and renders once
`probes_4bc_versions.jsonl` exists.)

**Layer-14 family (f53, lower block; `gate_retest_v14family.*`)** — to check whether v15's native layer is
a better steering site, v4 and v10 were refit pinned at layer 14 (`fit --layer 14 --tag _L14`; held-out r
.921 / .929, same as at 18) and retested there with the XL-only stage-4 direction at layer 14 and native v15:

| layer 14, coef | old (XL-only) | v4_L14 | v10_L14 | v15 |
|---|---|---|---|---|
| z @0.25 | 6.3 | 7.4 | 7.0 | 6.8 |
| z @0.5 | 4.0 | 5.1 | 5.0 | 4.5 |
| z @1.0 | 2.3 | 3.0 | 3.1 | 2.9 |

Layer 14 is a more sensitive site, not a better probe: every direction, the XL-only one included, gets z ≈ 7
at coef 0.25 there (vs 3.6–5.7 at layer 18), the random null is ×12 wider (.42 vs .035 log-odds), the
−direction push is 1.2–1.7× the +direction one, and by coef 1.0 nothing clears the gate. The hardening gain
is also smaller at 14 (old → v10: +11 % in z) than at 18 (+44 %). Layer 18 remains the cleaner place to
read the effect of probe hardening on steering; layer 14 at small coefficients is the strongest single
steering-to-choice coupling on this subject, for any utility direction.
