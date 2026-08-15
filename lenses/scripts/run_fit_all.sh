#!/usr/bin/env bash
# Sequential lens fits, smallest model first (shakes out bugs cheapest).
#
#   qwen3-4b ──> qwen25-7b ──> llama31-8b ──> qwen25-32b        (each: fit j, fit r, sanity)
#
# Resume-safe: every fit checkpoints (results/{model}/ckpt_{j,r}.pt) and --resume picks up
# where it left off, so re-running this script after an interruption is always correct.
# llama31-8b is gated on the Hub; if it fails to load, the run is marked SKIPPED and the
# script still exits 0 so the two Qwen models stand.
set -euo pipefail
cd "$(dirname "$0")/.."

for M in qwen3-4b qwen25-7b llama31-8b qwen25-32b; do
  ok=1
  for L in j r; do
    if ! python lens.py fit --model "$M" --lens "$L" --resume; then
      if [ "$M" = llama31-8b ]; then
        mkdir -p results/llama31-8b
        echo "llama31-8b fit failed (gated model or download error) $(date -u +%FT%TZ)" \
          | tee results/llama31-8b/SKIPPED.txt
        ok=0; break
      fi
      exit 1
    fi
  done
  [ "$ok" = 1 ] && python lens.py sanity --model "$M" | tee "results/$M/sanity.txt"
done
