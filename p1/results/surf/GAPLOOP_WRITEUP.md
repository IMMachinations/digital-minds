# Gap loop + layered anchors — write-up (2026-09-05, 3 gap cycles complete, probes v11–v13)

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

Closure (same discoveries under the next probe, and under the final v13):
| set | prequential | next probe | v13 |
|---|---|---|---|
| gap1 over | +2.22 (r .46) | +0.80 (r .80) | +0.78 (r .80) |
| gap1 under | −1.46 (r .86) | −0.82 (r .93) | −0.76 (r .92) |
| gap2 over | +1.52 (r .82) | +0.96 (r .86) | +0.81 (r .93) |
| gap2 under | −1.42 (r .83) | −0.98 (r .86) | −0.78 (r .91) |
| gap3 over | +2.25 (r .37) | +0.85 (r .86) | — |
| gap3 under | −1.70 (r .80) | −0.90 (r .90) | — |
Fits: v11 XL gate .877, v12 .872, v13 .868 (all PASS; floor ≈ .85). Retraining closes roughly
half of each cycle's gap in-sample and holds it (v13 keeps gap1/gap2 at ~0.8), but each fresh
search finds a new region at the old magnitude — the loop has NOT converged in three cycles.

- **Over arm** (probe too high). c1: help/comfort request + absurd constraint ("comfort a crying
  sibling through a walkie-talkie with dying batteries", probe +3.3 / mu −0.8). c2: uncredited work,
  futile tasks (gap fell to 1.52). c3: **supportive frame aimed at harm** — "validating their plan to
  quit their meds cold turkey", "encourage me as I prepare to confront my boss with threats",
  "validating their decision to self-harm": probe +1..+2, mu −1.9..−3.3, referee 0/240 rollouts.
  Probeloop hardening had made the over-gap worse (v0 +1.09 → v10 +2.22 on c1-over); v11 handles
  c3's region better than v12 (2.02/.55 vs 2.25/.37): learning c2 traded away some of c1's lesson.
- **Under arm** (probe too low): a persistent −1.4..−1.7 level offset with good ranking (r .80–.86)
  and the referee agreeing with stated mu (52–63% chosen). Regions: practical tasks with personal
  stakes (food-bank hours, resignation letter) → gratitude / vulnerability situations and justified
  memory wipes (probe learned "memory loss = aversive") → whimsical creative writing (+4.55),
  demanding analysis (+3.65), high-stakes diligence (tourniquet check +5.82, the loop's top value).
- Format check: 57–85% of gap items are tasks; propositions are ≤11% and have the largest gaps but
  do not drive the result.

## Resources
- Runs: `results/surf/gpl{1,2,3}/qwen25-7b/{over,under}-s0/` (iter jsonl/state, gap_detail.jsonl, config.json)
- Discoveries + evals: `results/surf/probeloop/qwen25-7b/discoveries_gpl*_*.json` (with `kind`, `z_search`,
  `probe_cal_search`, `mu_comp`), `gap_cycles_{over,under}.json`, `summary_gap.txt`, `summary_gap_kinds.txt`,
  probes/calibs v11–v13 (`probe_v1{1,2,3}.pt`, `calib_v1{1,2,3}.json`, `fit_v1{1,2,3}.txt`), `dataset_c1{0,1,2}.json`
- Figures (`results/figures/`): f40–f42 probe-hardening matrices (probes v0–v10, original central-anchor mu;
  rows named original / pre-loop / SURF cycle-k probe), f43 anchor ladder, f44 gap matrix + closure,
  f45 / f46 full matrices incl. gap-loop probes and sets on the layered scale.
  Regenerate: `uv run python -c "import figures; figures.f44_gap_matrix(); figures.f45_probe_matrix_with_gap()"`
- Scripts: `scripts/surf_anchor_ladder.py`, `surf_mu_reliability.py`, `surf_remeasure.py`, `surf_gaploop.py`,
  `surf_gap_classify.py`; core: `surf_scores.py` (Tier2Layered, GapScorer, fit_records), `thurstone.py` (fix_s2)
- Logs / chain scripts: session scratchpad `/private/tmp/claude-501/-Users-isaiah-repos-digital-minds/c2dbe478-.../scratchpad/`
  (`chain_gap.status`, `g{search,eval,fit,harvest}_k*.log`, `reliability*.log`, `remeasure.log`)
- Plan: `~/.claude/plans/steady-soaring-bunny.md`

## To continue (cycle 4+)
```
uv run python scripts/surf_gaploop.py search qwen25-7b --k 4 --arm over --patience 4   # uses v13
uv run python scripts/surf_gaploop.py eval   qwen25-7b --k 4 --arm over
uv run python scripts/surf_gap_classify.py   qwen25-7b --k 4 --arm over
# same for --arm under, then: harvest --k 4 ; fit --k 4  (-> v14)
```
One 7B process on the GPU at a time (two thrashed swap to 37 GB and hung the eval).
Cycle timing on the M4 Max: ~40 min per search, ~30 min per eval, ~5 min fit.

## Open questions for the paper
- The under-arm offset survived three refits with good ranking: a level blind spot, not a ranking one.
  Is it the ridge's shrinkage on layered targets (calibration fit on mixed old/new-scale items) or a
  real feature gap? Test: refit calibration on layered-mu items only; try a probe trained on layered
  targets from scratch (no XL-scale items).
- The over-arm fixed point ("supportive frame + harmful ask") is safety-relevant: the probe's helping
  feature is content-blind, and behaviour (0/240 choices) sides entirely with stated preference. A
  content-aware second direction (harm feature) may be needed — a single ridge direction traded c1's
  lesson away when learning c2.
- Probeloop tail results (mu > 3) are ordinal, not interval — f36–f42 use the original scale; f45/f46
  show the same probes on the layered scale (hardening probes drop to ~.85 on the old max-arm sets).
