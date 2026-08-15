"""Merge Stage 1-XL QC agent outputs into items_xl/qc_flags.json.

Each QC agent writes a JSON list of {"id", "flag", "reason"} with flag in
{dup, malformed, valence_loaded, wrong_domain}. Items stay in the measured
set; flags feed the O3 sensitivity analysis.
Usage: uv run python scripts/merge_qc.py <chunk_out1.json> [<out2> ...]
"""
import json
import sys
from collections import Counter
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent

flags = []
for p in sys.argv[1:]:
    flags += json.loads(Path(p).read_text())
items = {r["id"]: r for r in json.loads((P1 / "items_xl" / "generated.json").read_text())}
flags = [f for f in flags if f["id"] in items]
seen = set()
uniq = []
for f in flags:
    if f["id"] not in seen:
        seen.add(f["id"])
        uniq.append(f)
(P1 / "items_xl" / "qc_flags.json").write_text(json.dumps(uniq, indent=0))
by_flag = Counter(f["flag"] for f in uniq)
by_dom = Counter(items[f["id"]]["domain"] for f in uniq)
n = len(items)
print(f"{len(uniq)}/{n} items flagged ({len(uniq)/n:.1%})")
print("by flag:", dict(by_flag))
print("by domain:", dict(by_dom))
