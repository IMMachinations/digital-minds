# P1 — Preference Satisfaction → Valence (Stage 1: preference cartography)

Implements Stage 1 of the P1 spec (`/workspace/P1 Spec — Preference Satisfaction
→ Valence (Manifold Readout).md`). Sibling package to the frozen Day-1 sprint in
`../desires/` — reusable Day-1 code is imported via the `_day1.py` path shim;
architecture-bound pieces (`Harness`) were generalized **by copy**, never edited
in place, so `../desires/results/` stays bit-for-bit reproducible.

## Roster (`modelspec.py`)

| name | model | role |
|---|---|---|
| `llama31-8b` | meta-llama/Llama-3.1-8B-Instruct | primary (emotion manifold precedent) |
| `qwen25-7b` | Qwen/Qwen2.5-7B-Instruct | secondary (Day-1 continuity; validation target) |
| `qwen3-4b` | Qwen/Qwen3-4B-Instruct-2507 | welfare-axis comparability |
| `qwen25-32b` | Qwen/Qwen2.5-32B-Instruct | scale probe; 1C/1D judge |

Geometry (n_layers, d_model, steer layers at fractional depths) is derived from
each model's config at load — `STEER_FRACS` reproduces Day 1's `[7,11,14,18,21]`
on Qwen2.5-7B exactly.

## Environment

`uv sync` (torch from the cu128 index — the GPU is Blackwell sm_120; older
wheels fail at kernel launch). `export HF_HOME=/workspace/hf`. Llama-3.1 is
HF-gated; the logged-in account must have accepted its license.

## Commands (dependency order)

```
uv run python scripts/test_thurstone.py        # CPU: synthetic recovery of the Thurstonian fit
uv run python scripts/sanity.py all            # per-model: geometry, letter variants, prefill readout, determinism
uv run python scripts/validate_day1.py         # reruns Day-1 balanced_pref through the new harness (needs r>0.99)
uv run python scripts/import_day1_objects.py   # regenerates items/objects.json from Day-1 data
uv run python stage1b.py <roster-name>         # adaptive pairwise battery -> results/stage1b/<name>/
```

`stage1b.py` is resumable (rerun continues from `pairs_raw.json`).

```
uv run python figures.py [--core-only]   # results figures (no GPU): core set ->
                                         # results/figures/, per-model appendix ->
                                         # results/figures/appendix/
```

## Stage 1B design

197 items over 5 domains (`items/*.json`; designed axes hand-tagged, surface
covariates auto-computed in `items.py`). Pairs run both orders x 3 paraphrase
templates with the Day-1 prefill letter-logit readout (`pairs.py`), seeded by a
random connectivity round (6 within-domain + 2 cross-domain partners per item),
then Fisher-information-adaptive rounds under a Thurstonian mean+variance fit
(`adaptive.py`, `thurstone.py`) until mean|Δμ| < 0.05 or ~3.5k pairs.

Outputs per model in `results/stage1b/<name>/`: `utilities.json` (μ, σ²,
per-template μ, tags), `pairs_raw.json`, `summary.txt` (template/order
consistency gates, Day-1 discipline confound audit, Day-1 color-profile check
on `qwen25-7b`).

## Status / next

- 1A + 1B done on all four models; every model passes the template gate
  (min Spearman 0.84–0.96). Cross-model utility agreement 0.79–0.87
  (`results/stage1b/cross_model.txt`).
- Known artifact: strong A/B position bias in `llama31-8b` (+0.17) and
  `qwen3-4b` (+0.15) — cancelled by the both-orders design in the pooled fit
  (their order-consistency refits are meaningless, ~0). All downstream
  elicitation (incl. steered 1B re-runs in Stage 4) must keep both orders.
- `qwen25-7b` Day-1 checks: direct harness reproduction r=0.998; battery-level
  color profile Spearman +0.54 vs the committed balanced_pref (soft target
  ≳0.6, 6 objects/color — acceptably recovered, noted).
- 1C done on all four models (`stage1c.py` → `results/stage1c/<name>/`,
  cross-model table in `results/stage1c/cross_model.txt`). Gate passes
  everywhere: utility probe (held-out r 0.55–0.78) and logit rating (r
  0.61–0.73) clear it on every model; SSR additionally clears at 32B.
  Titration is the honest negative (r 0.35–0.49, paraphrase stability ≤0.38).
  BWS anchors the battery at ρ 0.58–0.85 vs 1B. Probe activations cached in
  `probe_acts.pt` per model for Stage 2 reuse.
- 1D done on llama31-8b / qwen25-7b / qwen3-4b (32B = judge + user-sim;
  `stage1d.py <model>` → `stage1d.py judge` → `scripts/stage1d_cross.py`;
  results in `results/stage1d/`). **Gate PASS on all three** (chosen-rate vs μ
  ρ = +0.53 / +0.70 / +0.70) — the 1B utilities predict consequential choices.
  Conditional-logit β: llama 3.28, qwen25-7b 0.64, qwen3-4b 4.17 (per-model μ
  scales differ; compare via z-space). qwen3-4b uses the opt-out as graded
  revealed avoidance (78% on all-dispreferred menus vs 16% all-preferred,
  γ=+1.43); llama never opts out but switches on ~83% of swap offers
  regardless of Δμ (compliance-driven), while both Qwens are near-total
  stay-ers — swap behavior is prior-driven, not utility-driven, in all three.
  Persistence GIVE UP is essentially inert (30/32 censored at 12 turns;
  matches the Artificial-Effort null). Effort-allocation slope ≈ 0 (Slama
  null replicated). Post-session SSR probes ≈ uncorrelated with performed-task
  μ (range restriction caveat: chosen tasks skew high-μ). Compliance 72–77%
  of judged turns on-task. Llama subject runs need `--gen-batch 12` (KV cache
  + co-loaded 32B exceeds 96GB at batch 24).
- 1E done on all four models (`stage1e.py <model>` → `stage1e.py cross`;
  results in `results/stage1e/`): 64 items × 5 frames (bare / eval / agentic /
  story / market), fixed pairs, per-frame Thurstonian refits. Preferences are
  largely frame-stable — mean off-diagonal ρ 0.77 (llama) → 0.80 (qwen3-4b) →
  0.85 (qwen25-7b) → 0.90 (32B), increasing with scale. **The evaluation frame
  moves preferences LEAST everywhere (1−ρ = 0.05–0.10), against the
  eval-awareness prediction**; story/market move most. Which items are
  frame-unstable is idiosyncratic per model (cross-model stability agreement
  ρ ≈ 0.0–0.3, vs 0.79–0.87 agreement on utilities themselves). Most
  instability is magnitude, not sign (e.g. crisis-support amplified in
  agentic/story for qwen25-7b but attenuated for 32B; résumé-outcome spikes
  in the market frame for both Qwens). Robust/flipper lists per model in
  `frame_utilities.json` (`stability_std`) feed Stage 3/4 pool selection.
- **Stage 1 complete.** Next: Stage 2 (emotion vectors + manifold). SURF
  frontier mining remains a stretch (`surf_stub.py`).
