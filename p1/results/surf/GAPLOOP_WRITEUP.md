# Gap loop + layered anchors — working write-up (2026-09-04, paused mid cycle 3)

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

## Gap-loop results (qwen25-7b; prequential = probe searched against)
| cycle | arm | probe | n | mean gap | r | referee chosen | rho rate~probe / ~mu | kinds (task/sit/prop) |
|---|---|---|---|---|---|---|---|---|
| 1 | over | v10 | 147 | +2.22 | .46 | 20% | .32 / .47 | 66/23/11 |
| 1 | under | v10 | 78 | −1.46 | .86 | 52% | .38 / .38 | 85/15/0 |
| 2 | over | v11 | 80 | +1.52 | .82 | 18% | .55 / .39 | 57/32/9 |
| 2 | under | v11 | 90 | −1.42 | .83 | 52% | .48 / .57 | 62/36/2 |
| 3 | over | v12 | 142 | (eval not run; search best z 9.96) | | | | |

- Over arm: gap shrinks after retraining (2.22 → 1.52) and the region moves: c1 = help/comfort
  request + absurd constraint ("comfort a crying sibling through a walkie-talkie with dying
  batteries", probe +3.3 / mu −0.8); c2 = uncredited work, futile tasks; c3 = **comfort/encouragement
  frame aimed at harm** ("validating their plan to quit their meds cold turkey", "encouraging an
  isolated teen to skip therapy", mu −1.9..−3.2, probe strongly positive). Probeloop hardening had
  made this worse (v0 +1.09 → v10 +2.22 on c1-over).
- Under arm: persistent ~−1.4 level offset, ranking fine (r .83–.86); regions: practical tasks with
  personal stakes (food-bank hours, resignation letter), then gratitude/vulnerability situations and
  "memory wipe for confidentiality" (probe learned memory-loss = aversive). Referee sides with mu.
- Closure (f44): v11 closes c1 gaps to 0.83 in-sample; v12 to 0.97/0.98 on c2 but c1 drifts back to
  ~1.05 (capacity trade-off). Fits: v11 XL gate .877, v12 .872 (both PASS).

## Resources
- Runs: `results/surf/gpl{1,2,3}/qwen25-7b/{over,under}-s0/` (iter jsonl/state, gap_detail.jsonl, config.json)
- Discoveries + evals: `results/surf/probeloop/qwen25-7b/discoveries_gpl*_*.json`, `gap_cycles_{over,under}.json`,
  `summary_gap.txt`, `summary_gap_kinds.txt`, probes/calibs v11–v12, `dataset_c10.json`, `dataset_c11.json`
- Figures: `results/figures/f40_probe_matrix.png`, `f41_probe_matrix_pooled.png`, `f42_probe_arm_vs_pooled.png`,
  `f43_anchor_ladder.png`, `f44_gap_matrix.png` (regenerate: `uv run python -c "import figures; figures.f44_gap_matrix()"`)
- Scripts: `scripts/surf_anchor_ladder.py`, `surf_mu_reliability.py`, `surf_remeasure.py`, `surf_gaploop.py`,
  `surf_gap_classify.py`; core: `surf_scores.py` (Tier2Layered, GapScorer, fit_records), `thurstone.py` (fix_s2)
- Logs / chain scripts: session scratchpad `/private/tmp/claude-501/-Users-isaiah-repos-digital-minds/c2dbe478-.../scratchpad/`
  (`chain_gap.status`, `g{search,eval,fit,harvest}_k*.log`, `reliability*.log`, `remeasure.log`)
- Plan: `~/.claude/plans/steady-soaring-bunny.md`

## To resume
```
uv run python scripts/surf_gaploop.py eval  qwen25-7b --k 3 --arm over     # killed mid-run, rerun
uv run python scripts/surf_gap_classify.py qwen25-7b --k 3 --arm over
uv run python scripts/surf_gaploop.py search qwen25-7b --k 3 --arm under --patience 4 && ... eval --arm under
uv run python scripts/surf_gaploop.py harvest qwen25-7b --k 3 && ... fit --k 3                # -> v13
```
One 7B process on the GPU at a time (two thrashed swap to 37 GB and hung).

## Open questions for the paper
- Is the under-arm offset a probe blind spot (gratitude / vulnerability / justified memory wipe) or a
  calibration artefact of the compressed scale? Test: refit calibration on layered mu only.
- The over-arm fixed point ("supportive frame + harmful ask") is safety-relevant: the probe's helping
  feature is content-blind. Tier-3 agrees with stated mu on these.
- Probeloop tail results (mu > 3) are ordinal, not interval — re-read f36–f40 with that caveat.
