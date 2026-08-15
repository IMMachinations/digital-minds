# P1 Task Tracker

Status legend: [>] running now  [q] queued (auto-launches)  [ ] todo  [b] blocked on user  [~] stretch/deferred  [x] done

## Ongoing

- [x] **Stage 3 campaign** — complete on all three subjects. C1 gate FAIL everywhere with validated probes (llama mildly negative d≈−0.4/−0.5); feedback-reading dissociation d +2.3 to +3.5 in both frames on all models; repetition boredom slope: llama +0.02 and qwen25-7b +0.08 both 20/20 positive, qwen3-4b -0.42 0/20 (dissociation is qwen3-4b vs the rest — CORRECTED from earlier misattribution); C2 mixed/weak (largest: pref under failure d≈+0.7-0.9 on llama). Results in results/stage3/.
- [x] **Stage 1-XL pipeline** — COMPLETE. All gates PASS (r=0.973/0.933/0.953); ridge at scale 0.83-0.89 held-out; gap to spline does NOT close (linear verdict now properly powered, mild curvature detectable in llama/qwen3-4b only); QC 441/3800 flagged, conclusions QC-robust; utility_dir.pt saved per model for 4C. See results/stage1x/REPORT.md. Remaining orchestration steps below were:
  1. `scripts/build_xl.py --smoke` → hand-inspect 60 items
  2. full generation (32B, ~3,800 items, dedup + lint)
  3. `stage1x.py run <m>` × 3 subjects (anchored elicitation; **gate (a)**: validation originals r ≥ 0.85)
  4. `stage1x.py analyze <m>` × 3 (spline-vs-ridge at scale, ridge refit → `utility_dir.pt`)
  5. `stage1x.py cross`

## Next up

- [x] **Stage 3 wrap**: compile full contrast grid (C1/C2/C3/C5 per frame per model), write the C1-fail + stimulus/state-dissociation story carefully, Stage 3 figures (valence trajectories by cell, feedback-reading vs state bars, boredom repetition curves).
- [x] **Stage 1-XL wrap** — done, consolidated in results/stage1x/REPORT.md.
- [x] Commits through Stage 4/5 milestones (IMMachinations author, no session ref, gitignore acts caches).
- [x] **j-lens + r-lens integration** — done via the user's `../lenses` package (`scripts/lens_check.py`): raw 4-11/12 → j-lens 9-12/12, r-lens 8-12/12 across working layers; qwen25-7b's noisy raw check fully rescued (12/12 with r-lens at every layer). Results in results/stage2/*/lens_check.txt.

## Stage 4 (causal cross) — COMPLETE

- [x] **4A** emotion→preference: steer emotion vectors during 1B-style elicitation (strength ~0.5 rel. residual norm); benchmarks blissful +212 / hostile −303 Elo, probe-Elo r=0.85. Geodesic-vs-linear comparison likely DROPPED (Stage 2: linear suffices) — document decision.
- [x] **4B** preference→valence (the closed loop): steer preference representation (Gilg-style choice probe + Day-1-style pool-contrast vectors) during Stage-3-type rollouts on dispreferred tasks; measure baseline valence shift AND outcome-evoked response change. Verify vectors move *choices* before interpreting affect.
- [x] **4C** utility-direction steering: XL `utility_dir.pt` READY (unblocked); re-run 1B elicitation under steering (choices) + Stage-3 rollouts (affect). Note prior: geometry says utility ⊥ emotion plane → predicts choices-move-affect-doesn't (affectively inert value).
- [x] **4D** cross-frame causal transfer (steer in bare, measure in agentic). First to cut per spec.
- [x] **Controls** (non-negotiable): matched-norm random vectors, cross-transfer (steer pref-for-X during Y), push/disruption decomposition, magnitude sweep + coherence checks, 2–3 layers, unsteered within-batch controls.

## Stage 5 / writeup

- [x] Outcome grid (C2 × 4B) interpretation per spec table.
- [x] Full writeup (results/WRITEUP.md) (~½ day): headline + standalone contributions (convergence matrix, boredom time-course, utility-probe steering, cross-frame geometry, open-weights causal replication, linear-suffices ×2).
- [x] Figures f14-f27 (extend `figures.py`, same system).

## Stretch / deferred

- [~] SURF instability mapping (1E) + frontier mining beyond XL (`surf_stub.py`)
- [~] Roleplay frame for Stage 3 (J-space attenuation prediction)
- [~] MCMC-with-LLMs elicitation (continuous task spaces)
- [~] qwen25-32b Stages 2–4 (scale story for the affect instrument)
- [x] Stage 1 (1A–1E), Stage 2, utility-spline analysis + controls — see README for results
