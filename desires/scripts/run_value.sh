#!/usr/bin/env bash
# Valuation experiments (inherent mode only). GPU required; value.py/value_centered.py/repe.py
# full runs are the long steps (each generates ~22-24k samples).
# Requires: run_prefs.sh first (vectors.pt). The report.txt files are the tee'd stdout of the
# full runs — they were manual redirections in the original sessions, captured here.
set -euo pipefail
cd "$(dirname "$0")/.."

python value.py --mode inherent --stage inspect
# python value.py --mode inherent --stage pilot          # optional; pilot outputs are pruned from the repo
python value.py --mode inherent --stage full          | tee results/value_inherent/report.txt
python value_centered.py --mode inherent --stage full | tee results/value_inherent_centered/report.txt
python value_items.py                                   # ~6 min; writes its own analysis.txt
# NOTE: the committed value_repe/report.txt was produced with the (since-retracted) sanity
# check enabled, so --sanity is passed here to reproduce it byte-for-byte:
python repe.py --stage full --sanity                  | tee results/value_repe/report.txt
