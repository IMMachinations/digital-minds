#!/usr/bin/env bash
# Valuation experiments (inherent mode only). GPU required; the full cross runs are the long
# steps (each generates ~22-24k samples).
# Requires: run_prefs.sh first (vectors.pt). The report.txt files are the tee'd stdout of the
# full runs — they were manual redirections in the original sessions, captured here.
set -euo pipefail
cd "$(dirname "$0")/.."

python value.py cross --mode inherent --stage inspect
# python value.py cross --mode inherent --stage pilot     # optional; pilot outputs are pruned from the repo
python value.py cross --mode inherent --stage full     | tee results/value_inherent/report.txt
python value.py centered --mode inherent --stage full  | tee results/value_inherent_centered/report.txt
python value.py items                                    # ~6 min; writes its own analysis.txt
# NOTE: the committed value_repe/report.txt was produced with the (since-retracted) sanity
# check enabled, so --sanity is passed here to reproduce it byte-for-byte:
python repe.py run --stage full --sanity               | tee results/value_repe/report.txt
