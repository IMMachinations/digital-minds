#!/usr/bin/env bash
# Control experiments, calibration, and the price-controlled preference re-measurement.
# GPU required. value.py obj21 is the long step (~60 min).
# Requires: run_prefs.sh (both modes' vectors_obj.pt) and run_value.sh (baseline rows, repe vectors).
set -euo pipefail
cd "$(dirname "$0")/.."

python repe.py controls                                  # ~5 min -> results/repe/controls.json
python value.py obj21                                  | tee results/value_obj21/report.txt   # ~60 min
python value.py calibrate                                # ~3 min; writes its own analysis.txt
python value.py tiers                                    # ~8 min
python prefs.py balanced                                 # ~4 min; writes its own analysis.txt
