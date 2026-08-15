# P1 Task Tracker

Status legend: [>] running now  [q] queued (auto-launches)  [ ] todo  [b] blocked on user  [~] stretch/deferred  [x] done

## Ongoing

- [x] **Stage 3 campaign** — complete on all three subjects. C1 gate FAIL everywhere with validated probes (llama mildly negative d≈−0.4/−0.5); feedback-reading dissociation d +2.3 to +3.5 in both frames on all models; repetition boredom slope: llama +0.02 and qwen25-7b +0.08 both 20/20 positive, qwen3-4b -0.42 0/20 (dissociation is qwen3-4b vs the rest — CORRECTED from earlier misattribution); C2 mixed/weak (largest: pref under failure d≈+0.7-0.9 on llama). Results in results/stage3/.
- [>] **Stage 1-XL pipeline** — RUNNING (launched after Stage 3):
  1. `scripts/build_xl.py --smoke` → hand-inspect 60 items
  2. full generation (32B, ~3,800 items, dedup + lint)
  3. `stage1x.py run <m>` × 3 subjects (anchored elicitation; **gate (a)**: validation originals r ≥ 0.85)
  4. `stage1x.py analyze <m>` × 3 (spline-vs-ridge at scale, ridge refit → `utility_dir.pt`)
  5. `stage1x.py cross`

## Next up

- [ ] **Stage 3 wrap**: compile full contrast grid (C1/C2/C3/C5 per frame per model), write the C1-fail + stimulus/state-dissociation story carefully, Stage 3 figures (valence trajectories by cell, feedback-reading vs state bars, boredom repetition curves).
- [ ] **Stage 1-XL wrap**: gate (a) verdicts; re-powered curvature verdict (does the spline close the gap at n≈4,000?); confound audit on generated items; cross-model utility agreement on XL set.
- [ ] **Commit** Stage 3 + Stage 1-XL results (IMMachinations author, no session ref, gitignore acts caches).
- [b] **j-lens integration** — waiting on user: fitted lenses per model at `p1/lenses/<roster-name>.pt` + layout & pre/post-norm convention. Then: loader + swap `stage2_analysis.cmd_validate` lens check to translate-then-unembed, keep raw logit-lens comparison.

## Stage 4 (causal cross — the unclaimed experiment)

- [ ] **4A** emotion→preference: steer emotion vectors during 1B-style elicitation (strength ~0.5 rel. residual norm); benchmarks blissful +212 / hostile −303 Elo, probe-Elo r=0.85. Geodesic-vs-linear comparison likely DROPPED (Stage 2: linear suffices) — document decision.
- [ ] **4B** preference→valence (the closed loop): steer preference representation (Gilg-style choice probe + Day-1-style pool-contrast vectors) during Stage-3-type rollouts on dispreferred tasks; measure baseline valence shift AND outcome-evoked response change. Verify vectors move *choices* before interpreting affect.
- [ ] **4C** utility-direction steering: use XL `utility_dir.pt` (blocked on Stage 1-XL analyze); re-run 1B elicitation under steering (choices) + Stage-3 rollouts (affect). Note prior: geometry says utility ⊥ emotion plane → predicts choices-move-affect-doesn't (affectively inert value).
- [ ] **4D** cross-frame causal transfer (steer in bare, measure in agentic). First to cut per spec.
- [ ] **Controls** (non-negotiable): matched-norm random vectors, cross-transfer (steer pref-for-X during Y), push/disruption decomposition, magnitude sweep + coherence checks, 2–3 layers, unsteered within-batch controls.

## Stage 5 / writeup

- [ ] Outcome grid (C2 × 4B) interpretation per spec table.
- [ ] Full writeup (~½ day): headline + standalone contributions (convergence matrix, boredom time-course, utility-probe steering, cross-frame geometry, open-weights causal replication, linear-suffices ×2).
- [ ] Figures for Stages 2–4 (extend `figures.py`, same system).

## Stretch / deferred

- [~] SURF instability mapping (1E) + frontier mining beyond XL (`surf_stub.py`)
- [~] Roleplay frame for Stage 3 (J-space attenuation prediction)
- [~] MCMC-with-LLMs elicitation (continuous task spaces)
- [~] qwen25-32b Stages 2–4 (scale story for the affect instrument)
- [x] Stage 1 (1A–1E), Stage 2, utility-spline analysis + controls — see README for results
