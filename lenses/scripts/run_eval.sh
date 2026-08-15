#!/usr/bin/env bash
# Multihop tables + agreement curves for every model that has fitted lenses.
set -euo pipefail
cd "$(dirname "$0")/.."

for M in qwen3-4b qwen25-7b llama31-8b; do
  if ls "results/$M/"?lens.pt >/dev/null 2>&1; then
    python lens.py eval --model "$M" | tee "results/$M/analysis.txt"
  else
    echo "skipping $M (no fitted lenses)"
  fi
done
