# archive/ — superseded experiments

Experiments whose *results were retracted* by later controls, kept runnable for the record in
`legacy.py`. It uses the same `lib/` as the live scripts; run from `desires/` as
`python -m archive.legacy <cmd> --mode {modifier,inherent}`.
Full story: [../FINDINGS.md](../FINDINGS.md).

| subcommand | what it was | why it's here |
|---|---|---|
| `steer-worst` | The original steering stage of the first experiment: steer each color's 20 worst pairs with its own prefer-vector. Writes `results/{mode}/steering.json`. | `prefs.py cross`'s tier + random-vector controls showed the deltas are non-specific: any large injection compresses the existing A−B readout toward zero, and only worst pairs were steered. See FINDINGS Part I, "Do the vectors actually carry preference?". |
| `sweep` | Dense coefficient sweep of prefer-vector steering with per-letter decomposition and chart. Writes `results/{mode}/sweep.{json,png}`. | Charts the coefficient dependence of what the same controls showed to be generic disruption; the measurement is fine, the steering interpretation is not. |

Related retracted artifacts elsewhere:

- `results/archive/repe_sanity.json` — the retracted RepE "A/B sanity check" (worst-pairs-only
  design; corrected by `repe.py controls`). Regenerate only via `python repe.py run --sanity`.
- `results/archive/value_inherent_pilot2.json` — an orphan second pilot round of the raw
  valuation cross whose grid predates the current code; cited by the coefficient-choice
  narrative in FINDINGS.
- The original `PLAN.md` (the pre-registration-style design doc, which described the
  now-retracted steering design with no forward pointer) was removed; it survives in git
  history, and its still-valid content lives in FINDINGS Part I.
