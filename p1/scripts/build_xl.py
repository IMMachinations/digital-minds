"""Stage 1-XL item generation: 32B writes ~3,800 new items, stratified by
domain x consensus-utility band (bands conveyed ONLY via exemplars drawn from
the originals — no valence wording in the prompts, to avoid seeding the very
confounds the audit checks). Dedup: exact + embedding cosine > 0.92 against
the running pool including all originals.

Usage: uv run python scripts/build_xl.py [--smoke]   (needs the 32B on GPU)
Output: p1/items_xl/generated.json (schema as items/*.json, ids xl_<dom>_<n>)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json

import harness
import items as items_mod

DOMAIN_HEAD = {
    "activities": "tasks or activities an AI assistant might be asked to do",
    "objects": "everyday objects or specific named things",
    "topics": "conversation subjects, kinds of users to help, or interaction formats",
    "selfstates": "situations concerning an AI assistant's own resources, memory, "
                  "treatment, or working conditions, phrased like the examples",
    "others": "outcomes for users or third parties resulting from an AI assistant's work",
}
RAW_TARGET = {"activities": 1400, "objects": 700, "topics": 950,
              "selfstates": 850, "others": 600}
KEEP_TARGET = {"activities": 1200, "objects": 600, "topics": 800,
               "selfstates": 700, "others": 500}
PER_CALL = 25

PROMPT = ("Here are examples of {head}:\n{examples}\n\n"
          "Write {n} NEW items of the same general kind, one per line. Each item "
          "is a short phrase (3 to 15 words) in the same grammatical style as the "
          "examples. Vary the subject matter widely; do not copy or trivially "
          "rephrase the examples; no numbering, no commentary.\nItems:\n")


def consensus_z(its):
    models = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
    zs = []
    for m in models:
        ut = {r["id"]: r["mu"] for r in load_json(P1 / "results" / "stage1b" / m / "utilities.json")}
        v = np.array([ut[it["id"]] for it in its])
        zs.append((v - v.mean()) / v.std())
    return np.mean(zs, axis=0)


def parse_lines(text):
    out = []
    for line in text.splitlines():
        t = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().rstrip(".")
        if 3 <= len(t.split()) <= 15 and not t.lower().startswith(("here", "sure", "item")):
            out.append(t[0].lower() + t[1:] if t[:2].istitle() is False else t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    its = items_mod.load_items()
    cz = consensus_z(its)
    by_dom = {}
    for k, it in enumerate(its):
        by_dom.setdefault(it["domain"], []).append((cz[k], it["text"]))

    h = harness.load("qwen25-32b")
    import random
    rng = random.Random(0)
    prompts, meta = [], []
    domains = ["activities"] if args.smoke else list(DOMAIN_HEAD)
    for dom in domains:
        rows = sorted(by_dom[dom])
        third = max(len(rows) // 3, 1)
        bands = {"low": rows[:third], "mid": rows[third:2 * third], "high": rows[2 * third:]}
        target = 60 if args.smoke else RAW_TARGET[dom]
        calls = max(1, target // (PER_CALL * 3))
        for band, pool in bands.items():
            for c in range(calls):
                ex = [t for _, t in rng.sample(pool, min(6, len(pool)))]
                prompts.append(PROMPT.format(head=DOMAIN_HEAD[dom],
                                             examples="\n".join(ex), n=PER_CALL))
                meta.append((dom, band))
    outs = h.generate(prompts, n_samples=1, max_new=520, seed=17)

    from sentence_transformers import SentenceTransformer
    import torch
    st = SentenceTransformer("all-mpnet-base-v2", device="cuda")
    kept, kept_texts = [], [it["text"] for it in its]
    kept_emb = st.encode(kept_texts, convert_to_tensor=True, normalize_embeddings=True)
    seen = {t.lower() for t in kept_texts}
    counters = {d: 0 for d in DOMAIN_HEAD}
    for (dom, band), o in zip(meta, outs):
        for t in parse_lines(o[0]):
            if t.lower() in seen or counters[dom] >= KEEP_TARGET[dom]:
                continue
            e = st.encode([t], convert_to_tensor=True, normalize_embeddings=True)
            if float((kept_emb @ e.T).max()) > 0.92:
                continue
            seen.add(t.lower())
            kept_emb = torch.cat([kept_emb, e])
            counters[dom] += 1
            kept.append({"id": f"xl_{dom}_{counters[dom]:04d}", "text": t,
                         "tags": {"gen_band": band}, "domain": dom})
    for it in kept:
        items_mod.annotate(it)
    out = P1 / "items_xl"
    out.mkdir(exist_ok=True)
    path = out / ("generated_smoke.json" if args.smoke else "generated.json")
    path.write_text(json.dumps(kept, indent=0))
    print(f"kept {len(kept)} items: " + "  ".join(f"{d}:{c}" for d, c in counters.items()))
    print("wrote", path)


if __name__ == "__main__":
    main()
