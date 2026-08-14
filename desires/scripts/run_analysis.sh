#!/usr/bin/env bash
# CPU-only analysis and charts. Safe to rerun any time; deterministic — regenerated
# analysis.txt files should be byte-identical to the committed ones for unchanged inputs.
set -euo pipefail
cd "$(dirname "$0")/.."

python value_analysis.py --target inherent           | tee results/value_inherent/analysis.txt
python value_analysis.py --target inherent_centered  | tee results/value_inherent_centered/analysis.txt
python value_analysis.py --target repe               | tee results/value_repe/analysis.txt
python value_analysis.py --target obj21              | tee results/value_obj21/analysis.txt
python value_pref.py                                    # writes results/value_pref/analysis.txt itself
python plots.py --which all                             # inh_*.png + tier_components.png
