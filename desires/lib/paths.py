"""Result-directory layout, anchored to desires/ regardless of cwd."""
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def results_dir(name):
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    return out
