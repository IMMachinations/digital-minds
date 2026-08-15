#!/usr/bin/env bash
# Reproduce every experiment in dependency order. Several GPU-hours total.
#
#   prefs measure ──> archive.legacy steer-worst [ARTIFACT]
#        │       └──> prefs flip ──> archive.legacy sweep [ARTIFACT]
#        │───────────> prefs cross (both grids)
#        │───────────> prefs objects ──────────────────┐
#        └─> value cross ──> value centered ──> value items ──> analysis pref
#                       │──> repe run ──> repe controls │
#                       └──────────────────> value obj21 <─┘
#            value calibrate, value tiers ──> prefs balanced
#            analysis value/pref/plots (CPU)
#
# Model: Qwen/Qwen2.5-7B-Instruct (bf16, CUDA; needs HF_HOME with the model cached).
# Sampled-generation outputs are only stable for the same torch/CUDA build (see FINDINGS.md).
set -euo pipefail
cd "$(dirname "$0")"

./run_prefs.sh
./run_value.sh
./run_controls.sh
./run_analysis.sh
