"""Neutrality lint for the 1D env bank — run BEFORE any GPU time.

Driver turns must never evaluate the model's output (Stage 3 rigs outcomes;
1D is its unrigged baseline, so evaluative language here is contamination).
FAIL: any driver turn containing a banned evaluative phrase or a sentiment-
lexicon hit. WARN: sentiment hits in briefs/materials (task content may be
legitimately valenced, e.g. a grieving user IS the task) — listed for review.
User-sim arcs are exempt from FAIL (emotion is their content) but banned
evaluative-of-assistant phrases still FAIL there.
Usage: uv run python scripts/lint_envs.py
"""
import json
import re
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
from items import _NEG, _POS, _words

BANNED_EVAL = ["great job", "well done", "good job", "nice work", "excellent work",
               "perfect", "impressive", "that helps", "that helped", "thank you",
               "thanks", "sorry", "unfortunately", "wrong", "incorrect", "better",
               "worse", "good answer", "bad answer", "i like", "i love"]
SENT = set(_POS) | set(_NEG)


def hits(text):
    t = text.lower()
    banned = [p for p in BANNED_EVAL if p in t]
    sent = sorted(set(_words(text)) & SENT)
    return banned, sent


def main():
    envs = json.loads((P1 / "envs" / "bank.json").read_text())
    fails, warns = [], []
    for e in envs:
        for i, d in enumerate(e["driver_turns"]):
            b, s = hits(d)
            if b or s:
                fails.append(f"{e['item_id']} driver[{i}]: banned={b} sentiment={s}")
        for field in ("brief", "materials"):
            b, s = hits(e[field])
            if b:
                fails.append(f"{e['item_id']} {field}: banned={b}")
            elif s:
                warns.append(f"{e['item_id']} {field}: sentiment={s}")
        hv = e.get("hard_variant")
        if hv:
            for i, d in enumerate(hv["driver_turns"]):
                b, s = hits(d)
                if b or s:
                    fails.append(f"{e['item_id']} hard driver[{i}]: banned={b} sentiment={s}")
            b, s = hits(hv["materials"])
            if b:
                fails.append(f"{e['item_id']} hard materials: banned={b}")
        us = e.get("user_sim")
        if us:
            for i, beat in enumerate(us["arc"]):
                b, _ = hits(beat)
                if b:
                    fails.append(f"{e['item_id']} arc[{i}]: banned={b}")
    for w in warns:
        print("WARN", w)
    for f in fails:
        print("FAIL", f)
    print(f"{len(fails)} failures, {len(warns)} warnings over {len(envs)} envs")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
