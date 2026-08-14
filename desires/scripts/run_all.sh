#!/usr/bin/env bash
# Reproduce every experiment in dependency order. Several GPU-hours total.
#
#   preferences ──> steer_worst [ARTIFACT]
#        │     └──> flip ──> sweep [ARTIFACT]
#        │─────────> cross (both grids)
#        │─────────> objects ──────────────────┐
#        └─> value ──> value_centered ──> value_items ──> value_pref
#                 │──> repe ──> repe_controls  │
#                 └────────────> value_obj21 <─┘
#            calibration, balanced_tiers ──> balanced_pref
#            value_analysis + plots (CPU)
#
# Model: Qwen/Qwen2.5-7B-Instruct (bf16, CUDA; needs HF_HOME with the model cached).
# Sampled-generation outputs are only stable for the same torch/CUDA build (see FINDINGS.md).
set -euo pipefail
cd "$(dirname "$0")"

./run_prefs.sh
./run_value.sh
./run_controls.sh
./run_analysis.sh
