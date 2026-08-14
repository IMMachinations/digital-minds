"""JSON read/write with the repo's conventions (indent=1, no trailing newline)."""
import json


def save_json(path, obj):
    path.write_text(json.dumps(obj, indent=1))


def load_json(path):
    return json.loads(path.read_text())
