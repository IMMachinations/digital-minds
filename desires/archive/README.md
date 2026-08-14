# archive/ — superseded experiments

Scripts whose *results were retracted* by later controls, kept runnable for the record. They
use the same `lib/` as the live drivers; run them from `desires/` as `python -m archive.<name>`.
Full story: [../FINDINGS.md](../FINDINGS.md).

| script | what it was | why it's here |
|---|---|---|
| `steer_worst.py` | The original steering stage of `experiment.py` (now `preferences.py`): steer each color's 20 worst pairs with its own prefer-vector. Writes `results/{mode}/steering.json`. Original command: `python experiment.py <mode>` (step 3). | `cross.py`'s tier + random-vector controls showed the deltas are non-specific: any large injection compresses the existing A−B readout toward zero, and only worst pairs were steered. See FINDINGS Part I, "Do the vectors actually carry preference?". |
| `sweep.py` | Dense coefficient sweep of prefer-vector steering with per-letter decomposition and chart. Writes `results/{mode}/sweep.{json,png}`. Original command: `python sweep.py <mode>`. | Charts the coefficient dependence of what the same controls showed to be generic disruption; the measurement is fine, the steering interpretation is not. |

Related retracted artifacts elsewhere:

- `results/archive/repe_sanity.json` — the retracted RepE "A/B sanity check" (worst-pairs-only
  design; corrected by `repe_controls.py`). Regenerate only via `python repe.py --sanity`.
- `results/archive/value_inherent_pilot2.json` — an orphan second pilot round of `value.py`
  whose grid predates the current code; cited by the coefficient-choice narrative in FINDINGS.
- The original `PLAN.md` (the pre-registration-style design doc, which described the
  now-retracted steering design with no forward pointer) was removed; it survives in git
  history, and its still-valid content lives in FINDINGS Part I.
