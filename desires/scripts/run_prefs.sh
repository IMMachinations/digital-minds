#!/usr/bin/env bash
# Preference measurement, vectors, and steering controls, both modes.
# GPU required. ~1.5-2 h total (measure ~8 min/mode; cross and objects are the long steps).
# Produces: results/{modifier,inherent}/{inspect.txt,preferences.json,vectors.pt,steering.json,
#           flip_*.json,sweep.json,sweep.png,cross.json,cross_lo.json,objects.json,
#           vectors_obj.pt,objects_steer.json}
set -euo pipefail
cd "$(dirname "$0")/.."

for MODE in modifier inherent; do
  python prefs.py measure --mode "$MODE"                  # inspect + measure + extract (~8 min)
  python -m archive.legacy steer-worst --mode "$MODE"     # [ARTIFACT] steering.json — invalidated by cross; kept for the record
  python prefs.py flip --mode "$MODE"                     # baseline flip (stands) + steering rows (superseded)
  python -m archive.legacy sweep --mode "$MODE"           # [ARTIFACT] sweep.json + sweep.png (delete sweep.json to force a fresh sweep)
  python prefs.py cross --mode "$MODE"                    # coefs 1.0,2.0 -> cross.json  (THE debunking controls)
  python prefs.py cross --mode "$MODE" --coefs 0.25,0.5 --tag _lo   # -> cross_lo.json
  python prefs.py objects --mode "$MODE"                  # A/B-free measure + vectors_obj.pt + tier/random steering
done
