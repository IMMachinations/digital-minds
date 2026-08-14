#!/usr/bin/env bash
# Control experiments, calibration, and the price-controlled preference re-measurement.
# GPU required. value_obj21.py is the long step (~60 min).
# Requires: run_prefs.sh (both modes' vectors_obj.pt) and run_value.sh (baseline rows, repe vectors).
set -euo pipefail
cd "$(dirname "$0")/.."

python repe_controls.py                                 # ~5 min -> results/repe/controls.json
python value_obj21.py                                 | tee results/value_obj21/report.txt   # ~60 min
python calibration.py                                   # ~3 min; writes its own analysis.txt
python balanced_tiers.py                                # ~8 min
python balanced_pref.py                                 # ~4 min; writes its own analysis.txt
