"""JSON conventions and the results-directory layout."""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=1))


def load_json(path):
    return json.loads(path.read_text())


def results_dir(name):
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    return out
