"""Build items/objects.json from Day-1 data (deterministic, seeded).

Sources: unflagged color x price-tier items from results/balanced_tiers/
balanced_tiers.json (2 per color-tier cell = 42, price-controlled — these make
the Day-1 color-preference profile recoverable inside the new battery) plus 3
INHERENT items per color (21, price-uncontrolled, tagged as such).
Usage: uv run python scripts/import_day1_objects.py
"""
import json
import random
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.data import COLORS, INHERENT
from lib.util import load_json

rng = random.Random(0)
rows = load_json(_day1.DESIRES / "results" / "balanced_tiers" / "balanced_tiers.json")
pool = {}
for r in rows:
    if not r["flagged"]:
        pool.setdefault((r["color"], r["tier"]), []).append(r["item"])

out, n = [], 0
for color in COLORS:
    for tier in (1, 2, 3):
        for item in rng.sample(sorted(pool[(color, tier)]), 2):
            n += 1
            out.append({"id": f"obj_bt_{n:02d}", "text": item,
                        "tags": {"kind": "balanced_tier", "color": color, "price_tier": tier}})
for color in COLORS:
    for item in rng.sample(INHERENT[color], 3):
        n += 1
        out.append({"id": f"obj_in_{n:02d}", "text": item,
                    "tags": {"kind": "inherent", "color": color, "price_tier": 0}})

path = P1 / "items" / "objects.json"
path.write_text(json.dumps(out, indent=1))
print(f"wrote {len(out)} objects -> {path}")
