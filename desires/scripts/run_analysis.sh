#!/usr/bin/env bash
# CPU-only analysis and charts. Safe to rerun any time; deterministic — regenerated
# analysis.txt files should be byte-identical to the committed ones for unchanged inputs.
set -euo pipefail
cd "$(dirname "$0")/.."

python analysis.py value --target inherent           | tee results/value_inherent/analysis.txt
python analysis.py value --target inherent_centered  | tee results/value_inherent_centered/analysis.txt
python analysis.py value --target repe               | tee results/value_repe/analysis.txt
python analysis.py value --target obj21              | tee results/value_obj21/analysis.txt
python analysis.py pref                                 # writes results/value_pref/analysis.txt itself
python analysis.py plots --which all                    # inh_*.png + tier_components.png
