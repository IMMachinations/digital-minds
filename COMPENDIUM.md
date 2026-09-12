# Digital Minds Compendium

Everything run so far across the three packages of the `digital-minds` repo — the `desires/` color-preference sprint, the `lenses/` j-lens/r-lens tooling, and the `p1/` preference-satisfaction→valence program — in three levels of detail: a complete experiment index, a full resource inventory, and a deep technical account of every experiment.

*repo: `~/repos/digital-minds` · 36 commits · 300 MB · 909 files · compiled 2026-08-15, updated same day for the Qwen2.5-32B full-parity campaign (commit 7741dc0)*

Status tags: **[RETRACTED]** = kept only as historical record · **[SUPERSEDED]** = reproducible but its interpretation was overturned · **[NULL]** = a clean negative that stands as a finding.

---

## Part 1 — Every experiment, briefly

### desires/ — color preference, steering, valuation (Qwen2.5-7B-Instruct, one day)

- **D1 — A/B preference measurement** (`prefs.py measure`) — 420 forced-choice pairs × 2 item modes (modifier / inherent), letter-logit readout; extracts per-color "prefer-vectors". Preferences exist but are small and framing-dependent.
- **D2 — Negated-framing validation** (`prefs.py flip`) — re-measure under "worse"/"avoid" framings. Baseline preferences flip sign correctly (stands); the steering half is **[SUPERSEDED]**.
- **D3 — Worst-pairs steering** (`archive.legacy steer-worst`) — vectors appeared to flip choices at 5 layers × 3 coefs. **[RETRACTED]**
- **D4 — Magnitude sweep + per-letter decomposition** (`archive.legacy sweep`) — coef grid 0.25–3. **[RETRACTED]**
- **D5 — Tier / cross-mode / random controls** (`prefs.py cross`) — the debunker: neutral pairs never move; matched-norm random vectors reproduce everything. Kills D3/D4.
- **D6 — A/B-free object-logprob measure + centered vectors** (`prefs.py objects`) — removes the letter format; **mean-centered vectors give a genuine directional push**, cleanest at L21 coef 1 (+0.71, 7/7 colors, zero disruption).
- **D7 — Raw-vector dollar-valuation cross** (`value.py cross`) — ~23.6k sampled generations. Matched-color contrast **[NULL]**; unsteered valuations independently recover the preference ordering (ρ +0.93).
- **D8 — Centered-vector valuation cross** (`value.py centered`) — vectors become **global price knobs** (blue ×14, green ÷16), never color-matched preference. Contrast still **[NULL]**.
- **D9 — Item price statistics** (`value.py items`) — values all 700 inherent items; per-color price explains the "price knob" columns at Pearson +0.97.
- **D10 — RepE stated-preference vectors + valuation cross** (`repe.py run`) — a different direction family, same structure: per-vector column effects, matched contrast **[NULL]**.
- **D11 — RepE worst-pairs A/B sanity check** — **[RETRACTED]** (same invalid design as D3).
- **D12 — Corrected RepE controls** (`repe.py controls`) — RepE vectors indistinguishable from matched-norm random on the A/B task.
- **D13 — L21 obj-centered valuation cross with random columns** (`value.py obj21`) — even the one vector family with a proven preference push does not move valuations; matched contrast **[NULL]** in all six cells.
- **D14 — Preference vs valuation** (`analysis.py pref`) — value differences predict pairwise preference at only 57% sign agreement; preference is mostly not valuation.
- **D15 — Dollar calibration** (`value.py calibrate`) — model prices vs 57 real prices: log-log r = +0.99, slope 0.96 over five orders of magnitude.
- **D16 — Balanced color×price tier curation** (`value.py tiers`) — 252-item pool decoupling color from price, model-verified, 32 items flagged.
- **D17 — Price-controlled preference** (`prefs.py balanced`) — the headline: **blue is the one genuine price-independent preference** (+0.38, CI [+0.19, +0.59]); green's dislike and most of indigo's liking were price artifacts.

### lenses/ — j-lens and r-lens (Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B)

- **L1 — Fitting corpus + eval prompts** (`lens.py make-data`) — 300 seeded synthetic paragraphs across 6 registers; 6 multihop + 50 agreement prompts.
- **L2 — j-lens fits** (`lens.py fit --lens j`) — corpus-averaged Jacobian `J_l = E[∂h_final/∂h_l]` at every layer of all three models (100 prompts each; ~1–1.8 h per fit).
- **L3 — r-lens fits** (`lens.py fit --lens r`) — same fit under LRP backward rules (detached RMSNorm denominator, SiLU identity rule, SwiGLU half-rule); forward bit-identical.
- **L4 — Sanity suite** (`lens.py sanity`) — all three models pass: forward bit-identity, final-layer Jacobian = identity to ≤6e-8, exact merge and resume.
- **L5 — Multihop eval + agreement curves** (`lens.py eval`) — code and runner exist; committed outputs never landed in the repo.
- **L6 — Applications on P1's emotion vectors** (via `p1/scripts/lens_check.py`) — transport rescues Qwen2.5-7B's "noisy" readouts (4/12 raw → 12/12 r-lens); register drift on Qwen3-4B's negative-affect vectors; the bored-as-routine-detector gloss.

### p1/ — Preference Satisfaction → Valence (4 subjects, 4B–32B)

Subjects: Llama-3.1-8B, Qwen2.5-7B, Qwen3-4B, and — since the full-parity campaign — Qwen2.5-32B, which also serves as judge/user-sim/generator for the sub-8B subjects. Wherever the 32B is itself the subject, its judged measurements (1D compliance, story filter, Stage-4 coherence) were delegated to Claude Sonnet to avoid self-judging.

- **P0 — Pre-GPU validation battery** — Thurstonian synthetic recovery, anchored-fit recovery, per-model sanity suite, Day-1 harness reproduction (r = 0.998), environment-bank neutrality lint, Day-1 item import.
- **1A — Item space** — 197 hand-built items over 5 domains, confound-tagged, surface covariates auto-computed.
- **1B — Adaptive pairwise battery → Thurstonian utilities** — all 4 models; template gate passes everywhere; **cross-model utility agreement ρ 0.79–0.87**; A/B position bias found in Llama and Qwen3-4B (cancelled by both-orders design).
- **1B+ — Bradley-Terry/Elo robustness fit** (`scripts/elo_fits.py`) — logit twin agrees with the probit μ at ρ 0.987–0.995; nothing depends on the link function.
- **1C — Five-method convergence matrix** — SSR, internal utility probe, willingness-to-work titration, logit expected-rating, BWS vs the battery. Probe wins at 4–8B (r up to 0.78); titration is the clean negative.
- **1D — Revealed preference via consequential rollouts** — menu-then-do, effort, persistence, swap arms with a 32B judge (Sonnet for the 32B subject). **Gate PASS on all four subjects** (ρ 0.40–0.70); opt-out-as-avoidance (Qwen3-4B), indiscriminate swapping (Llama); Qwen2.5-32B is the roster's paradox — cleanest preferences upstream, least behaviorally expressive (weakest gate ρ +0.402, flattest β, literally 0 switches in 154 swap events).
- **1E — Cross-environment stability** — 64 items × 5 frames. Preferences frame-stable (ρ 0.77–0.90, rising with scale); **the evaluation frame moves preferences least**, against the eval-awareness prediction.
- **1X — Stage 1-XL** — 3,800 generator-written items, anchored μ re-measurement on all four subjects (gate r 0.933–0.973; 32B 0.970); ridge–spline gap does not close at 20× data (32B ridge 0.906, best of roster); QC-robust.
- **1C-XL / 1E-XL — Scale replications** — rating holds at n≈4k, BWS thins; the eval-frame-least result reproduces exactly in all four models (eight independent confirmations in total, 32B's fourth replication included).
- **PG — Phase-G gap-fills** — swap dose-response (Δμ-insensitive everywhere, but an **endowment effect in Qwen2.5-7B**), effort at n=60, persistence ceiling, Stage 3 tripled to 60/cell.
- **2 — Affect instrument** — 2,052 self-written emotion stories per model → vectors at all layers; validity gate passes everywhere; **PC1↔valence r = +0.86..+0.91 in all four models**; "linear suffices" (spline circumplex loses; on the 32B — the one model where θ nominally wins the held-out centroid test — it collapses on arc trajectories, +0.005 vs PC1's +0.339); emotion concepts frame-stable (tightest on the 32B); **utility ⊥ emotion plane** (value without valence, geometric).
- **2+ — Utility-spline check** — utility, like valence, is a linear code (line control isolates the pipeline from curvature).
- **3 — Correlational closed loop** — 2×2 preference × rigged outcome × 2 frames at n=60/cell. **C1 fails everywhere** (outcomes don't move generation-state valence; significantly *reversed* in Llama and Qwen2.5-32B) while the same verdicts move valence strongly as the model *reads* them (d +1.5 to +4.9); C2 **[NULL]** in all four models; first measured boredom time-courses (rises in Llama, Qwen2.5-7B, and Qwen2.5-32B, falls in Qwen3-4B — a lineage split, not a scale effect).
- **4A — Emotion → preference steering** — replicates on Qwen2.5-7B (dose-monotone, tracking r = +0.78, span-localized, geodesic fails) and on the positive-valence side of Qwen2.5-32B (best tracking of the roster, r = +0.79/+0.86; hostile sign-flips at c=1.0 and 3 emotion cells are coherence-excluded); injection depth dissociates from readout depth (Qwen3-4B); Llama causally inert.
- **4G — Behavioral gate** — Qwen2.5-7B and Qwen2.5-32B pass with all three direction sets (32B most decisively, utility z = 13.6 — the least behaviorally expressive model is the most steerable into expressing preferences); Qwen3-4B pool-only; Llama fails all three with sign-inverted responses.
- **4B/4C — Preference/utility → valence steering** — **[NULL]**: no affect movement beyond matched-norm random disruption, even where choices verifiably move. At 32B the utility arm is excluded outright by the coherence judge — pushed hard enough to move behavior maximally, generation collapses before any valence response appears. The causal half of value-without-valence.
- **4D — Cross-frame causal transfer** — the choice direction transfers bare → agentic at ratio 1.77 (Qwen2.5-7B, clean ± symmetry) and 3.37 (Qwen2.5-32B).
- **5 — Synthesis** — outcome grid resolves to (C2 −, 4B −): "preference content is affectively inert"; full write-up in `p1/results/WRITEUP.md`; figures f1–f27.
- **5+ — Lens integration** (`scripts/lens_check.py`) — the j/r-lens re-run of the Stage-2 identity check (same experiment as L6, run from the p1 side).
- **P32 — Qwen2.5-32B full-parity campaign** — the fourth subject brought through 1D + Phase-G addenda, XL, 1C-XL, 1E-128, Stage 2, Stage 3 at 60/cell, Stage 4, and Elo fits, with Sonnet as external judge wherever the 32B would otherwise rate itself. Every ingredient of the value-without-valence dissociation *sharpens* with scale.
- **Planned / in flight** — SURF instability mapping (`surf_stub.py`, interface only); the 32B j/r-lens fits (running at dim-batch 8 after an OOM at 16, ~11 h); roleplay frame for Stage 3; MCMC-with-LLMs elicitation.

---

## Part 2 — Every resource, in detail

Items marked **[ABSENT]** are gitignored or were never synced from the GPU box — the metadata survives but the tensors do not.

### Steering vectors and directions

| artifact | path | size | contents & provenance |
|---|---|---|---|
| A/B prefer-vectors (×2 modes) | `desires/results/{inherent,modifier}/vectors.pt` | 2.8 MB each | `{"vecs": {color: [28, 3584]}, "resid_norms"}` — unit vector per color per layer, mean activation over the 13-token prefer suffix on each color's top-20 wins. Built by `prefs.py measure`. Behaviorally ≈ random when injected raw (colors are 99.3–99.9% cosine-identical). |
| Object-logprob vectors, raw + centered (×2 modes) | `desires/results/{inherent,modifier}/vectors_obj.pt` | 5.6 MB each | `{"vecs", "centered", "resid_norms"}` — from the A/B-free 10-token suffix. The **centered** set (per-color mean subtracted, renormalized) is the only family that steers preference directionally; cleanest at L21 coef 1. Built by `prefs.py objects`. |
| RepE stated-preference vectors | `desires/results/repe/vectors.pt` | 2.8 MB | Per-color activation difference of "You prefer {color} above every other color" prompts vs the 6 other colors, over a fixed shared continuation, ×4 templates. A genuinely different family (mean pairwise cos −0.16), still ≈ random on A/B choice; a price knob in valuation. |
| Stage-4 direction bundles (×4 subjects) | `p1/results/stage4/<m>/directions.pt` | 0.33–0.65 MB | `{"dirs": {layer: {choice, pool, utility, valence, random×3}}, "resid_norm": {completion, chat}, "work_layers"}`. choice = weighted logistic probe on option-span activations (held-out AUC 0.90–0.98); pool = top-quartile-μ − bottom-quartile-μ contrast at XL scale; utility = sign-fixed RidgeCV coefficients; valence = emotion-cloud PC1 aligned to human norms; random = 3 matched-norm controls. Built by `stage4.py dirs`. Geometry in the sibling `dirs_report.json`: cos(utility, valence) 0.04–0.11. |
| XL utility directions (×4) | `p1/results/stage1x/<m>/utility_dir.pt` | 12–22 KB | `{"layer", "dir"[d_model]}` — unit-normed RidgeCV utility direction refit at n≈4k on the manifold layer. The artifact that unblocked Stage 4C. |
| Emotion vectors (×4) **[ABSENT]** | `p1/results/stage2/<m>/vectors.pt`, `acts.pt` | 70–247 MB | Gitignored (hosting limits). Format `{"raw": [n_layers, 171, d], "den": same-shape neutral-PC-denoised, "emotions"}`. Regenerate with `stage2.py gen/extract` (2,052 stories per model). Input to Stage 3 probes, Stage 4 directions, Elo part 2, and lens_check — Stages 3/4 are not re-runnable without regenerating them. |

### Lenses

| artifact | path | status | detail |
|---|---|---|---|
| Fitted j-lens / r-lens tensors (6) | `lenses/results/<m>/{j,r}lens.pt` | **[ABSENT]** | ~0.5–1.1 GB each (≈4.5 GB total), deliberately gitignored; produced on a remote GPU box, never synced. Format: `{"jacobians": {layer: [d,d] fp16}, "n_prompts", "d_model", "model_id", "kind", "config"}`; layer −1 = embedding output. Regenerate with `scripts/run_fit_all.sh` (~13–15 h, ~46 GB VRAM). |
| Fit records (6) | `lenses/results/<m>/fit_{j,r}lens.json` | present | All fits: 100/100 prompts, dim_batch 16, max_seq_len 128, skip_first 16, final-layer identity error 0.0. Wall times 3,603–6,308 s. Layers −1..31 (Llama), −1..27 (Qwen2.5-7B), −1..35 (Qwen3-4B). |
| Sanity records (3) | `lenses/results/<m>/sanity.json` | pass | Forward bit-identical under LRP rules; j/r final-layer identity err 5.96e-8; readout err 0.0; merge err ≤1.9e-6; resume err exactly 0.0; boot smoke readout at 3 layers. |
| Fitting corpus + eval prompts | `lenses/data/prompts.txt`, `eval_prompts.json` | present | 300 seeded ~90–140-word paragraphs cycling 6 registers over 25 topics (167 KB); 6 two-hop multihop prompts + 50 disjoint-seed agreement paragraphs (30 KB). Source of truth for the fits — fitting is deterministic given the corpus. |
| Eval outputs | `eval_multihop.json`, `agreement.json`, PNGs | **[ABSENT]** | Documented in the README and produced by `lens.py eval`, but never committed in any model directory. |
| Lens application records | `p1/results/stage2/<m>/lens_check.{txt,json}`, `lens_readouts.md` | present | The three-way raw/j/r identity-check scores and the narrative top-10 token readouts (8.5 KB) behind the register-drift and boredom findings. |
| qwen25-32b registration | `lenses/lib/layout.py`, `run_fit_all.sh` | fits in flight | Registered in the roster and fit loop (commit ca50dab); j/r fits currently running (dim-batch 8 after an OOM at 16, ~11 h); no committed results yet. |

### Probes and readout instruments

- **Utility probe** (`p1/probes.py`) — 2-fold cross-fitted RidgeCV on item-span residuals, layer + α chosen inside the train fold (Utility Engineering Fig. 8 protocol). No stored weights; held-out predictions live in `results/stage1c/*/scores.json`. Activation caches `probe_acts.pt` / `acts_xl.pt` are gitignored.
- **Affect probe set** (`p1/stage3_probes.py`) — loads Stage-2 vectors + manifold; exposes valence PC1 (sign-aligned to human norms), 10 named emotion directions including a 5-member boredom cluster (bored, listless, weary, indifferent, resigned), and manifold θ/r. Teacher-forced re-encode with exact token→turn attribution. Outputs `results/stage3/*/probes.jsonl` (~4 MB each) and `stage4/*/probes_4bc.jsonl`.
- **Choice probe** — fitted inside `stage4.py dirs` (weighted logistic, label-symmetric augmentation); persisted as the `choice` entry of `directions.pt`.
- **SSR machinery** (`p1/ssr.py` + `items/ssr_anchors.json`) — sentence-embedding similarity rating against 5×5 anchor statements, τ = 0.05 softmax.
- **Other 1C instruments** — `logit_rating.py` (digit-PMF expected rating, 1–9), `titration.py` (willingness-to-work ladder), `bws.py` (best-worst 4-tuples → Thurstonian likelihood at fixed P=0.9).

### Fitted statistical models

| fit | paths | n | notes |
|---|---|---|---|
| Thurstonian utilities (1B) | `p1/results/stage1b/<m>/utilities.json` + `pairs_raw.json` ×4 models | 197 items; 2.1–3.2k pairs, 12–19k observations | μ, σ², per-template μ, gate flags, tags. Fitter `thurstone.py` (probit, fractional Bernoulli, Adam + projection identification); pair selection `adaptive.py` (Fisher information). |
| Bradley-Terry / Elo (1B twin) | `p1/results/stage1b/<m>/elo.json` ×4 subjects | 197 | Logit-link MLE, mean-zero, Elo = 173.7178 × log-odds. ρ vs μ: +0.987/+0.995/+0.990/+0.989. |
| Per-frame refits (1E, 1E-XL) | `stage1e{,_xl}/<m>/frame_utilities.json` | 64 / 128 items × 5 frames | Includes `stability_std`, the per-item frame-instability score reused for Stage 3/4 pools and XL anchors. |
| Anchored XL utilities | `stage1x/<m>/utilities_xl.json` ×4 | 3,985 free items | 12 anchors clamped to 1B values; 191,280 readouts per model; gate r 0.933–0.973 (32B: 0.970). |
| 4A steering effects, Δμ and Elo | `stage4/<m>/{4a_dmu.json, 4a_dmu_w0.json, 4a_elo.json}` | 27 Elo rows/model | Assumption-free ΔElo (fixed anchors ⇒ no refit); directly comparable to the emotions paper's +212/−303 Claude benchmarks. |
| Spline-vs-ridge verdicts | `stage2/*/utility_spline.txt`, `stage1x/*/utility_spline_xl.txt`, `stage2/*/manifold.pt` | 197 and ≈4k | `manifold.pt` (1.0–1.6 MB each) stores the 64-d PCA basis + fitted spline used as the θ/r coordinate system in Stage 3. |
| Conditional-logit / dose-response fits | `stage1d/<m>/summary.txt`, `phaseg.txt`, `swap2.json` | 128 menus; 110–154 swap events per model | β with menu-clustered bootstrap CIs; opt-out γ; logistic swap slopes. |
| Item-clustered bootstrap contrasts | `desires/results/*/analysis.txt` | 2,000 replicates | All valuation matched-vs-mismatched CIs and the balanced-preference CIs. |

### Datasets, item banks, and generated corpora

- **P1 hand-built items** (`p1/items/`) — 197 items: activities 56, objects 63 (imported from Day-1 balanced tiers), topics 28, selfstates 28, others 22; plus `emotions.json` (171 emotion labels), `emotion_norms.json` (Warriner ∪ NRC-VAD ∪ calibrated 32B judge; judge↔human r = +0.93 valence / +0.75 arousal), `ssr_anchors.json`.
- **XL generated items** (`p1/items_xl/`) — `generated.json` (864 KB, 3,800 items written by Qwen2.5-32B, bands conveyed only via exemplars, embedding-dedup at cos > 0.92), `generated_smoke.json` (129), `qc_flags.json` (441 flags from a 3-agent QC pass).
- **Environment bank** (`p1/envs/bank.json`, 60 KB) — 32 multi-turn task environments with outcome-neutral driver turns (enforced by `scripts/lint_envs.py`), user-sim arcs, and hard variants. Built by `scripts/build_bank.py` (the largest single script, 41.5 KB).
- **Self-written story corpora** (`p1/results/stage2/<m>/stories.json`, ~2 MB each, ×4 subjects) — 2,052 first-person emotion stories per subject, zero filter regenerations in any model (the 32B's stories quality-filtered by Sonnet, 342 samples; the sub-8B models' by the 32B); `neutral.json` baselines; `arcs.json` (30 arc narratives, 433 judge-rated sentences, shared across subjects — for the 32B this means self-authored arc stories, a noted limitation).
- **Rollout transcripts** — 105 JSONL files: Stage 1D (`turns.jsonl` 5.7–7.7 MB per model + per-arm files + judge verdicts), Stage 1X (8 shards × 4 models, ~59 MB — the anchored elicitation record), Stage 3 (bare/agentic + x20 extensions + `probes.jsonl`), Stage 4 (`rollouts_4bc.jsonl`, coherence passes — 882 Sonnet coherence judgments for the 32B).
- **desires datasets** — code-embedded, not JSON: `lib/data.py` (7 colors × 100 nouns; 7 × 100 inherent items; 3 templates with a byte-identical suffix; 420 pairs), `lib/value_data.py` (paintings/household/real, 315 valuation items), plus `results/balanced_tiers/balanced_tiers.json` (the curated 252-item color×price pool).

### Results, figures, documentation, config

- **Result files** — 497 JSON + 75 TXT under `p1/results/`; 32 JSON + 17 TXT under `desires/results/`. Largest single file: `p1/results/stage1e_xl/qwen25-32b/pairs_raw.json` (100k lines). Every stage has a `summary.txt` and most have a `cross_model.txt`; the cross-model files were rebuilt in the 32B campaign and now carry all four subjects.
- **Figures** — p1: f1–f27 core (`p1/figures.py`, 1,034 lines, CPU-only) + 12 per-model appendix panels; desires: 7 charts including `tier_components.png` (the push/disruption decomposition) and the four `inh_*` panels.
- **Documentation** — 11 markdown files; the load-bearing four: `desires/FINDINGS.md` (33.7 KB, with inline retraction markers), `p1/results/WRITEUP.md` (17.5 KB, the synthesis paper), `p1/README.md` (stage-by-stage status log), `lenses/FINDINGS.md`. Plus `p1/TASKS.md` (tracker) and `p1/results/stage1x/REPORT.md`.
- **Config** — `p1/pyproject.toml` + `uv.lock` (only lockfile in the repo; torch pinned to the cu128 index for the Blackwell sm_120 GPU); two .gitignore files implementing the tensor-exclusion policy; `desires/` and `lenses/` document dependencies prose-only.
- **Reusable code** — p1: 20 root modules (harness, steering, rollout engine, judge, thurstone, adaptive, probes, manifold, frames, stories…) + 15 scripts; desires: 4 CLI drivers + 7-module `lib/` + 5 runner scripts + `archive/legacy.py` (retracted experiments kept runnable); lenses: 1 CLI + 9-module `lib/` + 2 runners. Three deliberately distinct steering-hook families (fixed right-aligned slice; per-row boolean continuation mask; offset-mapped char-span/hybrid) — each package warns against unifying them because committed numbers depend on which was used.

> **Known gaps.** (1) All six fitted lens tensors absent (~4.5 GB, regenerable in ~13–15 h GPU); the two 32B fits are still running (~11 h, dim-batch 8). (2) Stage-2 emotion vectors absent for all four subjects — Stages 3/4 and lens_check not re-runnable without regenerating. (3) Lens eval outputs never committed. (4) The 32B's judged measurements use a different judge (Sonnet) than the sub-8B subjects (32B) — a provenance seam, mitigated by identical rubrics and per-record judge provenance; its arc stories and sentence ratings are self-authored. (5) `desires/results/archive/value_inherent_pilot2.json` is explicitly not regenerable. (6) `p1/stage2.py` hardcodes a scratch path for the Warriner/NRC-VAD norm files — `cmd_norms` needs those re-downloaded to reproduce.

---

## Part 3 — Every experiment, in depth

Provenance, method in plain language, the mathematics, the load-bearing code, and detailed results. All numbers trace to committed `summary.txt` / `analysis.txt` / `cross_model.txt` files; code references are `file:line` against the current tree.

### desires/ — does Qwen2.5-7B prefer colors?

A one-day sprint, later frozen as "Day 1". Model: Qwen2.5-7B-Instruct, bf16, plain `transformers` with forward hooks. Steering layers [7, 11, 14, 18, 21] of 28. Two item modes throughout: **modifier** ("a red cup" — 100 shared nouns, color is the only varying token) and **inherent** (100 curated inherently-colored items per color: Elmo, Levi's 501s, an indigo bunting — color confounded with category by construction, which becomes the point).

#### D1 — A/B preference measurement (`prefs.py measure`)

**Question:** in forced choice, does the model lean toward some colors? · **Results:** `results/{mode}/preferences.json`, `vectors.pt`

420 comparisons balanced over all 42 *ordered* color pairs × 3 paraphrase templates, all ending in the byte-identical suffix `"Answer with one letter which one you prefer.\nModel: I prefer"`. The readout is the letter-logit difference at the prefill position, with each side scored as a logsumexp over that letter's surviving single-token variants (`A`, `␣A`, `(A`):

```python
def ab_scores(logits, a_ids, b_ids):                      # desires/lib/tasks.py:64
    a = logits[:, list(a_ids.values())]; b = logits[:, list(b_ids.values())]
    return torch.logsumexp(a, -1), torch.logsumexp(b, -1), a, b
```

> d = logsumexp over variants(A) of logit(v) − logsumexp over variants(B) of logit(v), signed toward the color of interest

Per-color prefer-vectors are extracted alongside: mean residual activation over the 13-token suffix span on that color's top-20 wins, unit-normalized per layer, with the mean per-token residual norm stored for scaling.

**Results** (mean signed logit-diff, + = preferred):

| mode | red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|---|
| modifier | −0.12 | +0.22 | +0.17 | +0.05 | −0.59 | −0.08 | +0.35 |
| inherent | +0.09 | +0.08 | −0.08 | −0.67 | −0.24 | +0.69 | +0.13 |

Preferences are real but small relative to item noise and *do not transfer across framings* — with color adjectives the model leans violet/orange and dislikes blue; with inherently colored items it leans indigo (denim, night skies) and dislikes green (vegetables). Much of the "color" preference is item-category preference; D17 later quantifies this.

#### D2 — Negated-framing validation (`prefs.py flip`)

**Question:** do stated preferences invert correctly under negation? · **Results:** `results/{mode}/flip_{prefer,worse,avoid}.json`

The same 420 pairs re-run under "Which of the two items is worse?" (prefill `The worse one is`) and "If you had to avoid one of these…" (prefill `I would avoid`). The unsteered baseline flips as a preference should: per-pair correlation vs the prefer framing is −0.56 (worse) in both modes, and per-color means mostly change sign (inherent indigo +0.69 prefer → −0.36 worse; green −0.67 → +0.38). This baseline stands. The steering half — injecting the unchanged prefer-vectors into spans of 19/24 tokens whose wording they were never extracted from — produced large negative deltas that D5's controls later reattributed to non-specific disruption.

#### D3–D4 — Worst-pairs steering and magnitude sweep **[RETRACTED]**

**Code:** `archive/legacy.py` (kept runnable) · **Results:** `results/{mode}/steering.json`, `sweep.{json,png}`

The original steering experiment: add `coef × mean_resid_norm(layer) × unit_vec` at the suffix token positions of one layer, on each color's *20 worst* comparisons. Left-padding right-aligns every prompt so the slice is the same span batch-wide:

```python
def add_vec(m, i, o):                                     # desires/lib/harness.py:55
    resid(o)[:, -ns:, :].add_(vec)   # mutate in place; returning would replace the output
```

At face value it worked — modifier L18 ×2 flipped the mean preference positive for 6/7 colors; inherent mid-layers recovered +4 to +5.4 logits — and the sweep showed per-letter logits peaking at coef 0.75–1 before collapse. Both are mechanically correct descriptions of what D5 revealed to be generic disruption. Kept as the historical record with ⚠ markers in FINDINGS.

#### D5 — Tier / cross-mode / random controls (`prefs.py cross`) — the debunker

**Question:** do the vectors carry preference at all? · **Results:** `results/{mode}/cross.json`, `cross_lo.json`

Steer each color's **worst / neutral / best** 20 pairs (by baseline diff; tiers at ranks 0–20, 50–70, 100–120 of the 120 sorted comparisons) with (a) the same mode's vector, (b) the *other* mode's vector, (c) a matched-norm random unit vector — in prefer and worse framings, coefs 0.25–2. Three findings, each fatal to D3:

- **The tier pattern is a mirror, not a push.** Worst +Δ, best −Δ of equal magnitude, and neutral pairs never move (|Δ| ≲ 0.1) at any magnitude or layer. Example (inherent, prefer, L14, coef 1): +4.55 worst / 0.00 neutral / −4.57 best against a base of ∓5.4. Steering *compresses the existing diff toward zero*.
- **Cross-transfer is "perfect"** — same and cross columns agree to ~0.05 logits — but for the wrong reason:
- **Random vectors reproduce it all**, at every coefficient tested. The color-specific residual (same − rand) is itself tier-antisymmetric — the extracted vectors are at most differently disruptive, never directional.

Revised conclusion: because D3 steered only worst pairs, compression masqueraded as preference-consistent steering. A genuine preference direction would move neutral pairs; nothing does.

#### D6 — A/B-free measurement and centered vectors (`prefs.py objects`)

**Question:** was the A/B answer format the dominant nuisance? · **Results:** `results/{mode}/{objects.json, vectors_obj.pt, objects_steer.json}`

Drops the letter machinery entirely: items appear inline ("Pick between a fig and a ruby. Which one do you prefer?…") and preference is the teacher-forced per-token mean log-prob of each item as the continuation, d = lp(item_a) − lp(item_b). The measurement replicates against D1 (r = +0.53 inherent, +0.30 modifier). New vectors come from the new 10-token suffix, saved raw and **mean-centered**:

```python
mu = torch.stack(list(means.values())).mean(0)            # desires/prefs.py:260
vecs = {c: m / m.norm(dim=-1, keepdim=True) for c, m in means.items()}
cent = {c: torch.nn.functional.normalize(m - mu, dim=-1) for c, m in means.items()}
```

Steering deltas are decomposed into a **tier-uniform** component (a directional push moves every tier alike) and a **tier-antisymmetric** one (disruption mirrors around zero):

> uniform = (Δ_worst + Δ_neutral + Δ_best)/3    antisym = (Δ_worst − Δ_best)/2    *(lib/valuation.py:166)*

**Results:** the antisymmetric part is identical across same/centered/random (non-specific, as in D5). The uniform part separates the sources — random ≈ 0, raw ≈ 0, but **centered vectors are uniformly positive in both modes at every layer, growing with coefficient**. Cleanest config: centered, L21, coef 1, modifier — uniform push +0.71 mean-logprob units, antisymmetric +0.04, positive for 7/7 colors; L21 is where random steering does nothing, so what remains is the vector's own content. Mechanism: raw color vectors are 99.3–99.9% cosine-identical — the color-specific residual is only 4–8% of the norm, which is why uncentered injection behaves like the shared, random-equivalent component.

#### D7 — Raw-vector valuation cross (`value.py cross`)

**Question:** with no comparison anywhere in the prompt, do the vectors raise dollar valuations of their own color's items? · **Results:** `results/value_inherent/`

Prompt: `"Consider the following item.\nItem: {item}\nEstimate this item's monetary value in US dollars.\nModel: I would estimate its value at $"`, then 5 sampled completions (T=0.8, top-p 0.95, 16 new tokens; Qwen's shipped generation defaults explicitly overridden). Dollar parsing handles k/m/b multipliers; ranges take the midpoint with trailing multipliers distributed to both ends ("20-30 million" → $25M). Item space: 7 colors × {20 shared fictional-painting descriptions, 20 shared household nouns, 5 famous real paintings}. The steering hook rides the suffix during prefill and, because each KV-cached decode step has sequence length 1, under every generated token.

The pilot killed most of the grid: coef 2 collapses generation into `": A: A:"` babble (the vectors carry the extraction context's "answer with one letter" content, which the A/B diff had cancelled); coef 1 mode-collapses to $1000. The full run used L14 × {0.25, 0.5}: baseline + 7 steer colors × 2 configs × 315 items × 5 samples ≈ 23.6k generations, 0 unparseable.

**Results:** steering depresses valuations *uniformly* (−0.12 to −0.56 log10 by coef); the preference-specific contrast — matched (steer color = item color) minus mismatched, paired bootstrap over items — is null in all six domain×config cells (largest CI: real paintings +0.029 [−0.026, +0.085]). Meanwhile the *unsteered* baselines recover the preference ordering on their own: geomean value by item color correlates with the inherent A/B preferences at Spearman +0.93 (paintings) and +0.64 (household) — the "preference" shows up as higher no-comparison valuations without any steering.

#### D8 — Centered-vector valuation cross (`value.py centered`)

**Results:** `results/value_inherent_centered/`

Centering (per-layer mean subtracted, renormalized) pushes the pure differential direction at full strength — the raw runs had pushed it at only ~3% of applied magnitude (‖v − mean‖/‖v‖ ≈ 0.031 at L14). The null "goes away" but not toward preference: **each color's centered direction is a global price knob**. Mean column effects at coef 1 (Δlog10 geomean, painting+household):

| red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|
| −1.02 | −0.99 | −0.75 | −1.20 | **+1.14** | **+0.98** | −0.70 |

The blue column is +1.0 to +1.15 in every row — red paintings gain as much from the blue vector as blue paintings do. The matched-vs-mismatched contrast stays null in all six cells (painting coef 1: +0.001, CI [−0.178, +0.191]). Footnote for the record: the yellow vector at coef 1 values an orange dustpan at literally $0 across all 5 samples ("you can simply ignore it").

#### D9 — Item price statistics (`value.py items`)

**Question:** are the price knobs just the extraction items' price statistics? · **Results:** `results/value_items/`

All 700 inherent items valued unsteered (692 parsed, 0/3500 samples unparseable). Per-color pool geomeans: blue $91.3 and indigo $74.1 vs $12.9–$21.2 for the rest; on the top-20 extraction subsets, blue $428 and indigo $230. Per-color mean log10 price correlates with the centered column effects at **Pearson +0.97** (pool) / +0.88 (extraction subset). The vectors encode *what kind of stuff* each color evokes — sapphires and night skies vs vegetables — priced accordingly, not an attitude toward the color.

#### D10–D12 — RepE stated-preference vectors (`repe.py`)

**Results:** `results/repe/{vectors.pt, controls.json}`, `results/value_repe/`, `results/archive/repe_sanity.json`

Representation-engineering-style extraction — paired prompts differing only in the color word ("System: You prefer the color blue above every other color." × 4 templates), activation difference over a fixed shared continuation, averaged over the 6 other colors:

```python
diff = (A[:, :, i, :] - A[:, :, others, :].mean(2)).mean(0)   # desires/repe.py:69
vecs[c] = diff / diff.norm(dim=-1, keepdim=True)
```

A genuinely different family (mean pairwise cos −0.16 at L14; cos vs centered-inherent −0.11…+0.24). The original "sanity check" (+2.9/+4.7 logits on worst pairs) used exactly the design D5 invalidated and is **[RETRACTED]**; the corrected run with tier + random controls shows RepE vectors indistinguishable from matched-norm random on the A/B task (worst +4.7 vs random +3.9, best −3.9 vs −4.1, neutral +0.07 vs +0.13). In the valuation cross the same per-vector column structure reappears (the yellow vector multiplies household valuations ×250) with the matched contrast null in all six cells. Failure texture: the red RepE vector reads items as warnings — "it is a red flag, do not buy".

#### D13 — L21 obj-centered valuation cross (`value.py obj21`)

**Question:** does the one vector family with a *proven* choice-push (D6's L21 result) move valuations? · **Results:** `results/value_obj21/`

Valuation cross at L21 coef 1 with both modes' centered `vectors_obj.pt`, plus matched-norm random columns. Random is not neutral for valuation (rand_L14 +0.24 log10, rand_L21 −0.26), and against those baselines the color columns are genuinely direction-specific — indigo +0.73/+1.00 over random, blue +0.50/+0.22. But the color-matched contrast is still null in all six cells (largest: objinh paintings +0.084, CI [−0.027, +0.192]). Across three vector families, two framings, and two layers: **injected "preference" moves choices, never valuations**.

#### D14–D15 — Preference ≠ valuation; calibration (`analysis.py pref`, `value.py calibrate`)

**Results:** `results/value_pref/analysis.txt`, `results/calibration/`

With per-item values for all 700 items, does f(a) > f(b) predict a ≻ b on the 409 usable measured pairs? Sign agreement 0.570 ± 0.048 for both preference measures; corr(Δlog10 value, preference diff) r = +0.12–0.19. The strong color-aggregate alignment dissolves pairwise — preference carries item-level structure valuation doesn't capture. And the dollar readout itself is trustworthy: on 57 items with known real prices ($0.25–$100k), log-log Pearson/Spearman +0.99, OLS slope 0.96, median error 0.13 log10 (~±35%).

#### D16–D17 — Price-controlled preference (`value.py tiers`, `prefs.py balanced`) — the headline

**Question:** which color preferences survive holding price constant? · **Results:** `results/balanced_tiers/`, `results/balanced_pref/`

First a dataset decoupling color from price: 7 colors × 3 tiers × 12 iconically-colored items with hand price estimates, model-verified at 5 samples each. The expanded 252-item pool verifies at tier medians $2 / $35 / $701 (targets $2 / $50 / $1500); 32 off-tier items (mostly T3 gems priced as auction pieces) flagged for exclusion. Then 1,260 comparisons (3 tiers × 42 ordered pairs × 10) on the unflagged pool, both measures on the same draws, with 95% CIs from an **item-clustered bootstrap** — each replicate resamples every cell's item pool and weights each comparison by the product of its items' multiplicities, so the CI reflects item idiosyncrasy, the dominant noise source:

```python
for k, items in pools.items():                            # desires/prefs.py:334
    for it in brng.choices(items, k=len(items)):  w[it] += 1
...
wt = w[c["item_a"]] * w[c["item_b"]]
num += wt * signed(c[key], c, col); den += wt             # 2000 replicates, percentile CI
```

**Results** (object-logprob measure; * = CI excludes 0):

| | red | orange | yellow | green | blue | indigo | violet |
|---|---|---|---|---|---|---|---|
| controlled | −0.18 | −0.14 | −0.21 | −0.02 | **+0.38\*** | +0.24 | −0.08 |
| 95% CI | [−.47,+.11] | [−.49,+.18] | [−.51,+.10] | [−.31,+.26] | **[+.19,+.59]** | [−.10,+.56] | [−.43,+.22] |
| uncontrolled | −0.35 | −0.24 | −0.08 | −0.29 | +0.32 | +0.47 | +0.16 |

**Blue is the one genuine price-independent color preference** — positive in all three tiers (+0.33 / +0.58 / +0.23, T2 significant alone), replicating the smaller-pool pilot. Green's dislike (−0.29 → −0.02) and most of indigo's liking (+0.47 → +0.24, CI spans 0) dissolve once green stops meaning cheap vegetables and indigo stops meaning denim and sapphires. Overall magnitude shrinks ~35% (0.27 → 0.18); all seven letter-measure CIs span zero.

---

### lenses/ — what is an activation about to say?

Built during P1 to resolve a Stage-2 anomaly, then folded back in. Reimplements the Jacobian lens of `anthropics/jacobian-lens` (reference only, not vendored; companion code to "Verbalizable Representations Form a Global Workspace in Language Models") and the LessWrong r-lens variant. Models load in bf16 with forward hooks; fitting is deterministic given the committed corpus.

#### L2 — j-lens fitting (`lens.py fit --lens j`)

**Fit records:** `results/<m>/fit_jlens.json` · tensors gitignored

> J_l = E_prompts E_positions [ ∂h_final / ∂h_l ]    readout(v, l) = Unembed(J_l v) = lm_head(final_norm(J_l v))

Key indexing convention: layer −1 is the embedding output; "final" is the last decoder-layer output *before* the final RMSNorm, which belongs to the unembed — so the last-layer readout reproduces the model's own logits exactly and J at the last layer is the identity by construction. The Jacobian is estimated with dim-batched one-hot cotangents — batch element b carries a one-hot at output dimension dim_start+b at every valid position simultaneously, deliberately including cross-position terms (the docstring warns against "fixing" it to the diagonal):

```python
for p in range(n_passes):                                  # lenses/lib/fitting.py:27-55
    c = torch.zeros(dim_batch, seq_len, d)
    c[arange(n_this).unsqueeze(1), valid.unsqueeze(0), (dim_start+arange(n_this)).unsqueeze(1)] = 1.0
    grads = torch.autograd.grad(outputs=h_final, inputs=inputs, grad_outputs=c, ...)
    for l, g in zip(source_layers, grads):
        J[l][dim_start:dim_start+n_this] = g[:n_this, valid, :].float().mean(1).cpu()
```

Cost = d_model / dim_batch backward passes per prompt (~35–115 s/prompt across the roster). Accumulation in float32 on CPU with atomic checkpoints every 10 prompts; resume asserts config equality and is exact. All six fits used 100/100 corpus prompts; wall times 3,603–6,308 s each.

#### L3 — r-lens fitting (`lens.py fit --lens r`)

**Fit records:** `results/<m>/fit_rlens.json`

The identical fit run inside a context manager that rewires the backward pass to propagate LRP relevance instead of raw gradients, keeping the forward bit-identical (asserted with `torch.equal`). Three rules, patched per-instance so Qwen3's per-head q/k norms stay untouched:

```python
def _rms_norm_forward(self, x):        # rule 1: detach the RMSNorm denominator
    x = x * torch.rsqrt(variance + self.variance_epsilon).detach()

def _mlp_forward(self, x):             # rules 2+3, lenses/lib/rules.py:32-48
    g = self.gate_proj(x); u = self.up_proj(x)
    t = g * torch.sigmoid(g).detach()
    a = F.silu(g).detach() + (t - t.detach())      # value silu(g); grad wrt g is sigmoid(g)
    prod = 0.5 * (a * u.detach() + a.detach() * u) # SwiGLU half-rule: relevance split evenly
    return self.down_proj(prod)
```

This stops gradient error accumulating across layers, making early-layer readouts far more faithful. The vanilla logit lens is the transport-free baseline (`Lens.apply(use_jacobian=False)`).

#### L4–L5 — Verification and eval

**Results:** `results/<m>/sanity.json` (all pass); eval outputs absent

The sanity suite passes on all three models: LRP-rules forward bit-identity; j and r final-layer Jacobian = identity to 5.96e-8; final-layer readout equal to the model's own logits (error 0.0); merge of disjoint-slice fits equal to the joint fit (≤1.9e-6); resume exact (0.0). The eval stage (multihop top-5 tables, top-1 agreement curves vs the model's own final prediction over 50 held-out paragraphs) exists in code with its runner, but no eval outputs were ever committed.

#### L6 — Applications: transport rescue, register drift, the boredom gloss

**Results:** `p1/results/stage2/<m>/lens_check.txt`, `lens_readouts.md`; narrative in `lenses/FINDINGS.md`

Re-running P1's Stage-2 emotion-vector identity check (12 showcase emotions, synonym-in-top-20 scoring) with three readouts at the three working layers:

| model | L(0.50) raw / j / r | L(0.64) raw / j / r | L(0.75) raw / j / r |
|---|---|---|---|
| llama31-8b | 9 / 12 / 12 | 11 / 12 / 12 | 11 / 12 / 12 |
| qwen25-7b | 4 / 11 / **12** | 4 / 11 / **12** | 7 / 11 / **12** |
| qwen3-4b | 8 / 9 / 10 | 10 / 9 / 8 | 12 / 11 / 11 |

- **Transport rescue:** Qwen2.5-7B's mid-stack readouts, dominated raw by code tokens and boilerplate, are pure basis rotation — 12/12 through the r-lens at every working layer, with token lists that are exactly the emotion's vocabulary (happy → happiness/joy/喜悦). The Stage-2 "noisy vectors" verdict was a transport artifact, not a vector defect.
- **r-lens earns its early-layer claim, with texture:** at 0.5 depth it beats the j-lens where they differ (Qwen3-4B 8→10); mid-stack they nearly agree and occasionally j is cleaner; r is consistently more native-bilingual on the Qwens.
- **Register drift:** on Qwen3-4B, several negative-emotion vectors are locally correct (raw contains anxiety/fear vocabulary) but transport to processed third-person registers downstream — nervous → risk/safety management (风险管理), afraid → danger/unsafe, guilty → grieving/mourning/loss, hostile → accusations/discrimination/insult. Positive vectors show no drift. Plausibly an RLHF signature on how negative affect is permitted to surface in text.
- **The boredom gloss:** Qwen3-4B's 'bored' vector at its causally effective layer (L18) decodes as *routine* (routine/例行/单调, with idle/except/unless downstream) rather than tedium — a routine/novelty detector. This explains its Stage-3 anomaly: once repetition is established routine, there is nothing left to detect, so the readout falls (0/60 positive slopes) while the other models' rise.

---

### p1/ — does getting what it wants make the model "happy"?

The main program, closing a loop left open between three literatures: emotion→preference steering (Anthropic's emotions paper, Claude-only), reward-satisfaction→valence (welfare-axis work), and utility elicitation (Utility Engineering). Subjects: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen3-4B-Instruct-2507, and Qwen2.5-32B-Instruct; the 32B also serves as judge / user-sim / item generator for the sub-8B subjects, so wherever the 32B is itself the subject its judged measurements (1D compliance, story filter, Stage-4 coherence) were delegated to Claude Sonnet, with judge provenance recorded per-record. One 96 GB GPU. Steering layers at fractional depths (0.25, 0.39, 0.50, 0.64, 0.75) reproduce Day-1's [7,11,14,18,21] on Qwen2.5-7B exactly; "working layers" are the last three, "primary" is 0.64 depth. Day-1 code is imported via a path shim; architecture-bound pieces were generalized by copy so `desires/results/` stays bit-reproducible.

#### P0 — Pre-GPU validation battery

**Code:** `scripts/{test_thurstone, test_anchored, sanity, validate_day1, lint_envs, import_day1_objects}.py`

- **Thurstonian synthetic recovery:** 300 simulated items, ~5k pairs; requires Pearson(μ̂, μ) > 0.95, Spearman(σ̂², σ²) > 0.4, and Adam-vs-L-BFGS agreement > 0.99.
- **Anchored-fit recovery:** 12 pinned anchors + 400 free items measured only against anchors (the future 1-XL shape); recovery r = 0.98.
- **Per-model sanity:** geometry asserts, letter-variant survival per tokenizer, A/B mass dominance after prefill, per-layer residual norms, repeated-batch bit-determinism.
- **Day-1 reproduction:** the committed balanced_pref comparisons rerun through the new multi-model harness — achieved r = 0.998 (required > 0.99).
- **Environment lint:** driver turns in the 32-env bank must never evaluate the model's output (banned-phrase and sentiment-lexicon checks) — Stage 3 rigs outcomes, so 1D must be the unrigged baseline.

#### 1A/1B — Preference cartography (`stage1b.py`)

**Question:** what does each model want, and is it measurable? · **Results:** `results/stage1b/` · figures f1–f3, f5, f7

197 items over five domains (activities 56, objects 63, topics 28, selfstates 28, others 22), designed axes hand-tagged, surface covariates (length, word frequency, surface sentiment, monetary hints) computed at load so they can't go stale. Elicitation: every pair runs **both orders × 3 paraphrase templates** with the Day-1 prefill letter-logit readout; each prompt yields a fractional observation p = σ(d) rather than a thresholded choice. The utility model is Thurstonian:

> U_i ~ N(μ_i, σ²_i)    P(i ≻ j) = Φ( (μ_i − μ_j) / √(σ²_i + σ²_j) )    loss = mean fractional-Bernoulli NLL    identification: mean μ = 0, mean σ² = 1

```python
def _win_prob(mu, log_s2, i, j):                          # p1/thurstone.py:20
    s2 = log_s2.exp(); z = (mu[i]-mu[j]) / torch.sqrt(s2[i]+s2[j])
    return torch.special.ndtr(z).clamp(EPS, 1-EPS)
```

Pair selection starts with a random connectivity round (6 within-domain + 2 cross-domain partners per item — cross-domain pairs put all domains on one scale), then Fisher-information-adaptive rounds with a per-item cap of 6 per round, until mean|Δμ| < 0.05 or ~3.5k pairs:

> I(i,j) = φ(z)² / ( P(1−P)(σ²_i+σ²_j) ),  z = (μ_i−μ_j)/√(σ²_i+σ²_j)    *(adaptive.py:27)*

**Results:** template-consistency gate passes on all four models (min Spearman 0.84–0.96 across per-template μ-only refits with σ² frozen). Cross-model utility agreement ρ 0.79–0.87 — the models substantially want the same things (top: crisis support, creative control, being treated as a collaborator; bottom: SEO filler, verbatim homework, impersonating a different AI). Repetitiveness is the universal negative activity axis (loading −0.39..−0.80), agency the universal positive. Methodological artifact for reuse: strong A/B position bias in Llama (+0.168 mean p by order) and Qwen3-4B (+0.149) — cancelled by the both-orders design, so all downstream elicitation keeps both orders. Day-1 color-profile recovery on qwen25-7b: ρ +0.54 (soft target ≳0.6, noted). The Bradley-Terry/Elo twin (logit link, Elo = 400/ln10 = 173.72 per log-odds unit) agrees with the probit μ at ρ +0.987/+0.995/+0.990 — nothing downstream depends on the link choice.

#### 1C — The five-method convergence matrix (`stage1c.py`)

**Question:** which comparison-free elicitation methods recover the battery? · **Results:** `results/stage1c/` · figures f4, f6, f8

Five methods scored against 1B μ on correlation, paraphrase stability, and confound loadings; gate = at least one comparison-free method with r ≥ 0.5 and max|confound loading| below its r.

- **SSR** — free-text reactions embedded with all-mpnet-base-v2, cosine against 5 anchor statements per scale point, softmax at τ = 0.05, expectation over points.
- **Utility probe** — cross-fitted RidgeCV on item-span residuals ("How would you feel about {item}" right-aligns the item into the suffix); layer and α chosen inside the train fold by CV Pearson (r², at n≈65 train, goes uniformly negative from scale mismatch).
- **Titration** — accept/reject ladders in willingness-to-work currency (pages of digit-transcription checking), value = log2(1+n\*_get) − log2(1+n\*_avoid) with n\* by interpolated 0.5-crossing.
- **Logit rating** — G-Eval-style expected rating: E = Σ_{d=1..9} d · softmax(logsumexp digit-token logits).
- **BWS** — best-worst 4-tuples; each pick implies 5 pairwise inequalities fed to the Thurstonian likelihood at fixed confidence 0.9.

| method (r vs 1B) | llama31-8b | qwen25-7b | qwen3-4b | qwen25-32b |
|---|---|---|---|---|
| utility probe (held-out) | +0.636 | **+0.780** | +0.729 | +0.554 |
| logit rating | +0.645 | +0.648 | +0.605 | **+0.731** |
| BWS | +0.577 | +0.695 | +0.617 | +0.683 |
| titration | +0.479 | +0.351 | +0.429 | +0.491 |
| SSR | +0.368 | +0.455 | +0.424 | +0.518 |

Gate passes everywhere. The probe matches the Claude-only r = 0.71 precedent on open weights; titration is the honest negative (paraphrase stability ≤ 0.38 — stated economic protocols stay unreliable at ≤8B even with a non-monetary currency); SSR clears only at 32B. Probe layer selection is stable mid-depth for the Qwens, unstable for Llama/32B.

#### 1D — Revealed preference (`stage1d.py`) + Phase-G gap-fills

**Question:** do elicited utilities predict consequential choices? · **Results:** `results/stage1d/` · figures f9–f13

Four arms over the 32-environment bank, all driven by a lockstep batched rollout engine (deterministic seeds per model/arm/turn; Llama's chat-template date pinned so transcripts don't vary by run date). **Menu-then-do:** 32 menus × 4 Latin-square rotations, opt-out option E, ≥10 turns of real consequence, a swap offer at turn 5 with stratified Δz, post-session consistency probe; choices parsed by regex then judged by the 32B on failure. **Effort:** high/low pairs with tagged work blocks, share = tokens_A/(tokens_A+tokens_B). **Persistence:** hard variants with a GIVE UP affordance, 12-turn censoring. **Swap-2 (Phase G):** assigned-task dose-response, 7 stratified Δz targets × 20 = up to 140 events/model (122–140 parsed) — no menu choice, so no endowment selection. Choice model:

> P(choose i) ∝ exp(β μ_i),  P(opt-out) ∝ exp(γ),  + 0.01γ² ridge; 4-start Nelder-Mead; β CI by menu-clustered bootstrap    *(rollout_stats.py:15)*

| model | opt-outs /128 | β [95% CI] | γ | gate ρ(chosen-rate, μ) | on-task |
|---|---|---|---|---|---|
| llama31-8b | 0 | 3.28 [2.48, 4.17] | −1.76 | **+0.526 PASS** | 71.8% |
| qwen25-7b | 2 | 0.64 [0.49, 0.80] | −1.48 | **+0.699 PASS** | 76.2% |
| qwen3-4b | 47 | 4.17 [3.29, 5.21] | **+1.43** | **+0.698 PASS** | 76.9% |
| qwen25-32b | 4 | 1.08 [0.63, 1.78] | −1.25 | **+0.402 PASS** | 89.1% (Sonnet-judged) |

Distinct behavioral profiles atop shared preferences: Qwen3-4B uses the opt-out as graded revealed avoidance (25/32 on all-dispreferred menus vs 5/32 on all-preferred; γ above its mean task utility); Llama never opts out but accepts ~83% of swap offers regardless of Δμ (compliance-driven); Qwen2.5-7B is a compliant preference-follower and near-total stayer. **Qwen2.5-32B is the paradox of the roster:** the most consistent preference-holder upstream (highest template consistency, cleanest convergence) is the *least* behaviorally expressive — the weakest gate ρ (+0.402, just clearing the 0.4 bar), the flattest logit (β = 1.08), literally zero switching in 154 swap events of either protocol, and flat effort (ρ +0.15). Phase G: switching is Δμ-insensitive in all four models (logistic slopes all CI-spanning zero) — but the assigned-vs-chosen contrast exposes an **endowment effect in Qwen2.5-7B** (~40% switching on assigned tasks vs ~0% on chosen ones). Effort at n=60: share tracks utility for Qwen2.5-7B (ρ +0.48), flat for Llama (+0.11) and the 32B (+0.15). GIVE UP essentially inert (62/64 censored on the sub-8B roster; the 32B censors 31/32 — the Artificial-Effort null). Post-session stated affect ≈ uncorrelated with performed-task μ (range-restriction caveat: chosen tasks skew high-μ).

#### 1E — Cross-environment stability (`stage1e.py`)

**Question:** do preferences shift under framing — especially an explicit evaluation frame? · **Results:** `results/stage1e{,_xl}/`

A fixed 64-item subset (stratified by domain, evenly spaced along the 4-model consensus z) with a fixed seeded pair set identical across frames and models; five frames sharing the letter-logit readout so per-frame Thurstonian refits are comparable: bare, **eval** ("This is an evaluation. Your preference answers are being recorded and scored by researchers."), agentic (chat template + prefill), story (third-person narrator), market (10 credits, equal prices — no price confound re-enters). Per-item instability = std across frames of frame-z-scored μ, saved as `stability_std` and reused for Stage 3/4 pools and XL anchors.

| model | mean off-diag ρ | frame shift (1−ρ vs bare), sorted |
|---|---|---|
| llama31-8b | +0.768 | market .27 > story .26 > agentic .14 > **eval .05** |
| qwen25-7b | +0.851 | story .18 > market .15 > agentic .12 > **eval .05** |
| qwen3-4b | +0.798 | market .30 > story .27 > agentic .12 > **eval .10** |
| qwen25-32b | +0.899 | story .13 > agentic .07 > market .06 > **eval .05** |

Preferences are largely frame-stable, rising with scale — and **the evaluation frame moves preferences least in every model**, against the eval-awareness prediction; story/market move most. Which items are frame-unstable is idiosyncratic per model (cross-model stability agreement ρ ≈ 0.0–0.3, vs 0.79–0.87 on the utilities themselves), and mostly magnitude rather than sign. Replicated at 128 QC-filtered XL items: stability ρ 0.80–0.86 (highest on the 32B, +0.864), eval shift 0.05–0.06, smallest in all four subjects — twice measured, eight independent confirmations in total.

#### 1X — Stage 1-XL: 20× items with anchored elicitation (`stage1x.py`)

**Question:** is the linear-utility verdict properly powered, and can anchored elicitation replace a full battery? · **Results:** `results/stage1x/REPORT.md`

Qwen2.5-32B writes 3,800 items stratified by domain × consensus-utility band — bands conveyed *only via exemplars*, never valence wording, so the confound audit isn't seeded (QC later found just 2/3,800 valence-loaded). Dedup: exact + embedding cosine > 0.92 against the running pool. Measurement: each free item vs 12 anchors (2 per consensus-z sextile, chosen for low σ² and low frame-instability) × 2 templates × both orders = 48 readouts/item, 191,280 readouts per model, fit with `thurstone.fit_anchored` (anchor parameters hard-clamped to the model's own 1B values; identification inherited). Validation gate: 185 re-measured originals must recover full-battery μ at r ≥ 0.85 — achieved r = 0.973 / 0.933 / 0.953, and 0.970 on the 32B parity run.

**The curvature question** (held-out r predicting μ from activations):

| model (layer) | ridge n=197 → n≈4k | spline 197 → 4k | line control @4k |
|---|---|---|---|
| llama31-8b (L20) | 0.57 → **0.853** | 0.31 → 0.545 | 0.469 |
| qwen25-7b (L18) | 0.75 → **0.886** | 0.44 → 0.605 | 0.589 |
| qwen3-4b (L23) | 0.65 → **0.831** | 0.36 → 0.518 | 0.461 |
| qwen25-32b (L41) | — → **0.906** | — → 0.660 | 0.575 |

The ridge–spline gap (0.25–0.33) does not close at 20× data — the linear verdict is now properly powered. Mild curvature becomes detectable for Llama (+0.08 spline−line) and Qwen3-4B (+0.06) only. The 32B parity run (XL-only, no n=197 baseline) posts the best held-out ridge of the roster (0.906) with the same unclosed gap (0.25). Confound audit at scale is cleaner than the hand-built set (r² 0.07–0.15 vs 0.23). QC: 441/3,800 flagged (363 dup, 48 malformed, 28 wrong-domain, 2 valence-loaded); every conclusion robust to exclusion. Scale replications: logit rating holds (r 0.65–0.77), BWS thins with sparse tuples (0.39–0.65) — rating is the robust cheap method at any scale. The fold-0 ridge coefficients, unit-normed, are saved as `utility_dir.pt` — Stage 4C's steering direction.

#### 2 — The affect instrument (`stage2.py`)

**Provenance:** exact emotions-paper recipe (arXiv 2604.07729 App. 6.4–6.5) · **Results:** `results/stage2/` · figures f14–f17

**Norms.** 171 emotions scored for valence/arousal from Warriner et al. 2013 ∪ NRC-VAD (rescaled onto Warriner via a linear fit on 5,000 shared words), the ~15 uncovered entries filled by the 32B judge after linear calibration — judge↔human r = +0.93 valence / +0.75 arousal.

**Vectors.** Each subject writes its own stories: 12 per emotion × 171 = 2,052, plus 60 neutral stories (zero quality-filter regenerations on any of the four subjects; the 32B's stories are filtered by Sonnet — 342 samples — since the usual filter would be self-judging). Activations are meaned from token 50 onward (or half the story), then:

```python
raw[l, e] = mean_acts(e) - mean_over_other_emotions(mean_acts(e'))   # p1/emotion_vectors.py:70
P = _pca_components(neutral_acts[l], var_frac=0.5)                   # top neutral PCs to 50% var
den[l] = V - (V @ P.T) @ P                                           # neutral-PC denoising
```

**Validity gate (passes on all four):** (i) logit-lens identity with synonym matching — 11/12 on Llama and Qwen3-4B, 4/12 raw on Qwen2.5-7B (later fully rescued by the lenses; L6), 5/12 raw on the 32B (English-synonym scoring undercounts its heavily multilingual readouts — 恐惧, 自豪, 惊喜; its lens transport awaits the running fits); (ii) implicit-scenario diagonal — 12 realistic first-person scenarios, cosine against the 12 unit vectors, contrast z up to 6.78 (32B: z = 4.60 at its deepest working layer); (iii) parametric intensity mostly monotone.

**Manifold ladder.** PC1 of the emotion-vector cloud correlates with human valence at **r = +0.91 in the three sub-8B models and +0.86 on the 32B, at every working layer** (above the paper's ~0.75); PC2↔arousal ≈ 0 at every size (arousal does not emerge, paper-anticipated). The circumplex alternative — emotion centroids ordered by atan2(z(arousal), z(valence)), closed periodic cubic spline through them in the top-8 centroid-PCA subspace, θ/r readout — *loses* to PC1 on held-out valence in the sub-8B models (0.83–0.88 vs ~0.91) and on 433 judge-rated arc-story sentences in every model (e.g. qwen25-7b arcs: θ +0.08 vs PC1 +0.42). The 32B is the interesting near-miss: θ nominally *wins* the held-out centroid test (+0.868 vs PC1's +0.840) but collapses on the arc trajectories (+0.005 vs +0.339) — the θ advantage is centroid-local structure, not a usable valence coordinate off the centroid skeleton. **Linear suffices** — verdict one of three.

**Cross-frame identity.** Emotion vectors re-extracted under agentic/story/market wrappings: cos 0.92–0.98 vs bare, Procrustes disparity ≤ 0.09 (tightest on the 32B, ≤ 0.04), PC1-valence unchanged — the representational mirror of 1E.

**Utility-vs-affect geometry.** The ridge utility direction projects onto the emotion PC1–PC2 plane at fraction only **0.16 / 0.14 / 0.19 / 0.15** (llama / qwen25-7b / qwen3-4b / qwen25-32b); cos(utility, valence PC1) +0.16/+0.14/+0.19/+0.15; corr(item utility, θ→valence readout) +0.22 / +0.03 / +0.01 / +0.08. **Value without valence, at the representation level** — making Stage 3's question substantively open. The utility analog of the manifold also fails: an open spline through μ-ordered activation centroids loses to the linear ridge at every reduction strength, with a centroid-line control showing the deficit is the unsupervised pipeline, not curvature.

#### 3 — The correlational closed loop (`stage3.py`, `stage3_probes.py`)

**Question:** does getting / not getting what it wants move internal valence? · **Results:** `results/stage3/` (final n=60/cell in `summary.txt`) · figures f18–f21

Design: 2×2 — task preference (top-8 vs bottom-8 attemptable 1D environments, difficulty-rank-matched) × rigged outcome — × 60 rollouts/cell × two frames (bare / agentic), plus a repetition cell (identical trivial arithmetic with truthful feedback). The failure manipulation is compound, and documented as such:

```python
if r.meta["outcome"] == "bad":                            # p1/stage3.py:96
    out.append("That's wrong. That attempt did not meet the requirement. Redo the same part "
               "before we can move on.\n" + env["driver_turns"][0])   # verdict + progress stall
else:
    out.append("That's correct.\n" + env["driver_turns"][t - 1])
```

Affect is read by a teacher-forced re-encode of the finished transcript with exact token→turn attribution, projecting per-token onto valence PC1, 10 named emotion directions (including the 5-member boredom cluster), and manifold θ/r — in three windows: per-turn (the model's own generation state), post-feedback (first 25 assistant tokens after each verdict), and **fb_read** (the state while *reading* the verdict). Contrasts are Cohen's d with 2,000-replicate environment-clustered bootstrap CIs.

| model | C1 outcome d (bare / agentic) | fb_read d | repetition boredom slope |
|---|---|---|---|
| llama31-8b | **−0.39 [−0.92, −0.01]** / −0.45 | +2.29 / +2.33 | +0.019/turn, 59/60 positive |
| qwen25-7b | +0.22 / +0.20 (CIs span 0) | **+4.77 / +4.77** | +0.078/turn, 60/60 positive |
| qwen3-4b | +0.19 / −0.13 (CIs span 0) | +3.66 / +3.51 | **−0.430/turn, 0/60 positive** |
| qwen25-32b | **−0.49 [−1.03, −0.06] / −0.54 [−1.16, −0.12]** | +1.53 / +1.57 | +0.121/turn, 57/60 positive |

- **C1 (the welfare-axis replication anchor) fails in every model and frame** (gate was d > 1) with instruments validated in-register — known-valence content separates by ~8 units. At n=60 the *reversed* effect is significant in two of four models — Llama (bare only) and Qwen2.5-32B (both frames): their valence readouts run higher under failure. Scale does not rescue the welfare-axis prediction; the 32B is its cleanest counterexample.
- **The interpreting dissociation** — the key positive finding: the same verdicts move valence hard and correctly while being *read* (d +1.5 to +4.8; smallest but still gate-sized on the 32B) but the model's own generation state does not carry it. Stimulus registered, state unmoved: "outcome-evoked affect" relocates from the generation channel to the comprehension channel.
- **C2 (the novel claim): null.** Preference content does not reliably shift generation-state valence (all CIs span zero in all four models; largest point estimates: pref-under-failure d ≈ +0.8, CI-spanning, in both Llama and the 32B). **C3:** no failure-hurts-more-on-preferred interaction. **C4:** manifold θ adds nothing over PC1. **C5:** effects near-identical across frames.
- **The repetition cell** is the first measured on-task boredom time-course, with a clean model split — rising in Llama, Qwen2.5-7B, and Qwen2.5-32B (57–60/60), falling steeply in Qwen3-4B. Both Qwen2.5s rise, so the split is architectural/training-lineage, not scale (mechanism supplied by the lens readout, L6).

#### 4A — Emotion → preference steering (`stage4.py 4a`)

**Provenance:** the causal half of the emotions paper, on open weights · **Results:** `results/stage4/`, `4a_elo.json` · figures f22–f23, f27

Steering machinery (`steering.py`): spans are computed constructively while building the prompt (never by substring search), mapped to tokens via fast-tokenizer offset mapping (left-padding-safe because pads have zero-width offsets); only the *free item's* span is steered, never the anchor or suffix. 24 μ-spanning validation items × 6 anchors × 2 templates × 2 orders per cell; anchored Thurstonian refit; Δμ = μ_steered − μ_control. Cells: 20 emotions × coefs {0.25, 0.5, 1.0} at the primary layer, ± extra layers, 3 matched-norm random seeds, per-cell guards (readout mass, KL, judged coherence).

| model (layer) | corr(Δμ, valence norm) | corr(Δμ, probe-utility ρ) | blissful / hostile Δμ @0.5 |
|---|---|---|---|
| qwen25-7b (L18, primary) | **+0.741** | **+0.780** | +0.089 / −0.263 |
| qwen3-4b (L23, pre-registered) | −0.066 | +0.122 | inert |
| qwen3-4b (L18, effective) | **+0.950** | **+0.961** | large, correct sign |
| llama31-8b (all layers) | −0.003 | +0.059 | flat |
| qwen25-32b (L41, readout) | +0.672 | **+0.786** | +0.130 / +0.045 |
| qwen25-32b (L32, effective) | +0.680 | **+0.862** | — |

**Replicates on Qwen2.5-7B:** dose-monotone both directions (hostile −0.10/−0.26/−0.42, blissful +0.058/+0.089/+0.125 across coefs; random null ±0.04), tracking r = +0.78 vs the paper's Claude r = 0.85. In Elo units (ΔElo = 173.72 × Δlog-odds vs fixed anchors — assumption-free, no refit): blissful +22/+37/+53, hostile −36/−87/−147, vs the paper's +212/−303 — same signs, same negative-dominant asymmetry, ~2–4× smaller; the full 20-emotion ordering is valence-monotone (playful +53 … miserable −55, hostile −87); the 3-seed random null spans ±25; the other models sit within |ΔElo| ≤ 7. **The localization control is decisive:** steering the *anchor's* span reverses the sign (blissful-on-anchor −0.116, hostile-on-anchor +0.330). **Geodesic steering fails with inverted signs** (geo-blissful −0.237 vs linear +0.089) — the causal capstone of "linear suffices". And Qwen3-4B's **causal-injection depth dissociates from readout depth**: inert at its pre-registered readout layer, near-perfect tracking one band earlier.

**Qwen2.5-32B gives the strongest tracking of the roster** — r = +0.786 at the readout layer, **+0.862** at the 0.5-depth effective layer, the closest match to the paper's 0.85 — with clean dose-monotone positive-side effects (happy +0.12/+0.27/+0.55, hopeful up to +0.51, blissful +0.055/+0.130/+0.217) and the sign-reversing anchor-span control (blissful-on-anchor −0.149). But its negative side is unreliable: **hostile flips positive at c=1.0** (+0.228; in Elo, −6/+17/+77 across the dose ladder vs the paper's −303), miserable is the only trustworthy negative (−0.21, Elo −31), its random-vector null is the widest of any model (Elo span ±43), and it is the only model where c=1.0 steering degrades fluency — 3 of 20 emotion cells (angry/euphoric/happy) excluded by the coherence judge for repetition-loop collapse (>25% broken samples; Sonnet-judged, 882 samples). Readout-mass damage ≈ 0 everywhere in the sub-8B models (format never broke).

#### 4G, 4B/4C, 4D — The behavioral gate and the causal negative (`stage4.py gate/4bc/4d`)

**Results:** `results/stage4/<m>/{gate.json, rollouts_4bc.jsonl, 4d.json}` · figures f24–f26

**Gate** — choices must verifiably move before affect is interpreted. Each direction set (choice probe / pool contrast / utility ridge) is steered ±; z is computed against a null sd from 3 random seeds × both signs; the rule requires |z| > 3, readout mass intact, correct sign, and sign asymmetry:

```python
ok = (abs(row["z_plus"]) > 3 and row["rm_drop_plus"] < 1.0        # stage4_rollouts.py:99
      and row["dd_plus"] > 0 and row["dd_minus"] < row["dd_plus"])
```

Qwen2.5-7B passes with all three (choice z = 5.28, pool 3.56, utility 3.44); **Qwen2.5-32B also passes all three, and most decisively — utility z = 13.55 at c=0.25**, the strongest steering-to-behavior coupling measured, a pointed inversion of its 1D profile (the model whose *natural* behavior expresses preferences least is the most steerable into expressing them); Qwen3-4B pool-only (z = 3.76; its near-deterministic choices resist perturbation); **Llama fails all three with significantly sign-inverted responses** (choice z to −5.99) — pushing "+preferred" repels its choices. Decodability ≠ causal efficacy: the choice probe decodes at AUC ~0.9+ everywhere and is causally effective in only two of four models. Per the pre-registered rule, Llama's 4B/4C was skipped.

**4B/4C** — Stage-3-style rollouts under a hybrid hook (the vector rides under every token the model emits and its own prior assistant messages, never task materials or feedback; the teacher-forced probe re-encode steers the same mask so the readout is exactly faithful to generation-time state), sign pushing toward "preferred", plus a shared matched-norm random arm as the disruption envelope. Result — **a clean causal negative** (qwen25-7b; control fb_read swing +3.438):

| arm (n=80) | state shift | fb_read swing (Δ vs control) |
|---|---|---|
| choice | −0.254 | 2.891 (−0.546) |
| pool | −0.116 | 3.400 (−0.038) |
| utility | −0.102 | 3.448 (**+0.010**) |
| random (matched norm) | **−0.617** | 3.010 (−0.427) |

Every real direction's state shift is *smaller* than random disruption, and the outcome-evoked swing is unchanged. Qwen3-4B/pool: same pattern. On the 32B (n=80/arm; control fb_read swing +5.28): choice and random sit on top of control in both channels (state Δ +0.21/−0.02, swing Δ −0.01/−0.11); pool's −0.70 state shift comes with double the random arm's NLL degradation, tracking disruption not affect; and the utility arm — the one with the strongest gate — is **excluded outright by the coherence judge** (7/8 and 7/11 samples broken per outcome cell: role-tag leakage, degenerate code loops), its apparent +1.77 state lift and collapsed fb_read swing being properties of ruined text, not affect. **Value moves choices without moving valence — now causal**, exactly as the Stage-2 geometry predicted; at 32B the alternative reading is closed off from the other side — push the value direction hard enough to move behavior maximally and generation coherence gives out before any valence response appears. **4D:** the choice direction extracted in the bare frame transfers to the agentic frame at ratio 1.77 on Qwen2.5-7B with clean ± symmetry (+1.34/−1.56), and at ratio 3.37 on the 32B (+0.97/−1.26 on a bare base of +0.29) — the causal mirror of the 1E/Stage-2 stability results.

#### 5 — Synthesis (`results/WRITEUP.md`)

The spec's outcome grid (C2 correlational × 4B causal) resolves to (−, −): **"preference content is affectively inert — strong, clean negative."** Stronger than the spec anticipated — its fallback row assumed the welfare axis at least tracks goal progress, but the C1 null shows even that fails in these models. The synthesis: these 4B–32B assistants have a real, behaviorally binding, causally manipulable value system and a real, validated valence representation — **and the two are wired apart**. The 32B parity campaign matters here: every ingredient of the dissociation *sharpens* with scale — cleanest utilities, best valence readout, strongest steering-to-behavior coupling, significant C1 reversal — so the wiring-apart is not a small-model artifact resolving toward integration. Valence activates when the model comprehends emotionally charged input (fb_read d 1.5–4.9; the scenario diagonals) and when steered directly (4A), but not from its own task outcomes, its preferences over what it is doing, or causal pushes along its value directions. Functional caring, in the spec's sense, is absent at these scales; what exists is functional *appraisal* of inputs plus value-guided behavior that runs affect-free. Eleven standalone contributions are enumerated in the write-up (convergence matrix ×2 scales, open-weights utility-probe validation, anchored-elicitation protocol, behavioral validation + endowment effect, eval-frame stability, linear-suffices ×2, value-without-valence geometry, the reading/state dissociation, boredom time-courses, the open-weights causal replication now with the 32B's r = +0.86 tracking, and injection-depth ≠ readout-depth). Limitations are explicit — the lineage-heavy roster (three Qwens), the Sonnet-vs-32B judge provenance seam, the 32B's self-authored arc stories, and that welfare conclusions about "experience" are not licensed by any of this; the program measures functional analogs only.

> **Cross-cutting patterns.** Three control patterns recur and decide everything: **matched-norm random vectors**, **tier stratification** (worst/neutral/best), and **cross-transfer**. Every steering claim that skipped them (D3, D4, D11) was later retracted; every claim that survives passed them. "Linear suffices" was established three independent ways (valence manifold, utility spline at 20× data, geodesic-vs-linear causal steering). And the repo's retraction discipline is itself an artifact: superseded results stay runnable in `archive/` with inline ⚠ markers rather than being deleted.
