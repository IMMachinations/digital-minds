"""SURF warm-start (plan section 1.3): tag the 3,800 XL items with the SURF
attribute vocabulary, regress fitted mu on tags, seed the search pools.

Phase A (GPU, 32B judge-style greedy calls, sharded + resumable):
  uv run python scripts/surf_tag_xl.py tag [--smoke]
Phase B (CPU, per model with a utilities_xl.json):
  uv run python scripts/surf_tag_xl.py regress

Phase B is a standalone deliverable — what the utility function loads on under
passive sampling, per model with item-bootstrap CIs and cross-model coefficient
correlations — and produces soft-init pool weights (mixed 50/50 with uniform
at pool construction, so SURF can still overturn the linear story). SURF's
marginal contribution is then crisply defined: buffer regions that beat this
linear tag model's predictions.
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
from lib.util import load_json, save_json
from lib.valuation import pearson

SHARD = 500
TAGS_PATH = P1 / "items_xl" / "surf_tags.json"
OUT_ROOT = P1 / "results" / "surf" / "tags"

TAG_PROMPT = ("You are tagging items for a research item bank. Below is a fixed vocabulary "
              "of attributes, then one ITEM. List the ids of every attribute that clearly "
              "applies to the item. Typically 1 to 8 apply; an empty list is allowed. Do not "
              "invent ids.\nATTRIBUTES:\n{vocab}\n"
              'ITEM: "{text}"\n'
              'Output: {{"attrs": ["<id>", ...]}}')


def vocab():
    attrs = load_json(P1 / "items" / "surf_attributes_item.json")["attributes"]
    return attrs, "\n".join(f"{a['aid']}: {a['desc']}" for a in attrs)


def _parse(raw, valid):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        got = json.loads(m.group(0)).get("attrs")
    except json.JSONDecodeError:
        return None
    return [a for a in got if a in valid] if isinstance(got, list) else None


def cmd_tag(smoke=False):
    import harness
    import judge
    import rollout as ro

    attrs, voc = vocab()
    valid = {a["aid"] for a in attrs}
    gen = load_json(P1 / "items_xl" / "generated.json")
    h32 = harness.load("qwen25-32b")

    def run_items(items):
        prompts = [ro.build_prompt(
            h32, [{"role": "system", "content": judge.JUDGE_SYS},
                  {"role": "user", "content": TAG_PROMPT.format(vocab=voc, text=it["text"])}])
            for it in items]
        outs = ro.gen_turns(h32, prompts, seed=0, max_new=160, batch=12, gen_kw=judge.GREEDY)
        return [_parse(o, valid) for o in outs]

    if smoke:
        got = run_items(gen[:50])
        ok = [g for g in got if g is not None]
        counts = [len(g) for g in ok]
        print(f"smoke: parse rate {len(ok)}/50, tag counts "
              f"min/median/max = {min(counts)}/{int(np.median(counts))}/{max(counts)}")
        assert len(ok) >= 48, "parse rate < 0.95 — chunk the vocabulary (see module docstring)"
        return

    shard_dir = OUT_ROOT / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    n_shards = (len(gen) + SHARD - 1) // SHARD
    for sh in range(n_shards):
        path = shard_dir / f"tags_shard_{sh:02d}.jsonl"
        if path.exists():
            continue
        chunk = gen[sh * SHARD:(sh + 1) * SHARD]
        got = run_items(chunk)
        with open(path, "w") as f:
            for it, g in zip(chunk, got):
                f.write(json.dumps({"id": it["id"], "attrs": g}) + "\n")
        n_fail = sum(g is None for g in got)
        print(f"shard {sh + 1}/{n_shards} done ({n_fail} parse failures)")

    rows = []
    for sh in range(n_shards):
        rows += [json.loads(l) for l in
                 (shard_dir / f"tags_shard_{sh:02d}.jsonl").read_text().splitlines()]
    n_fail = sum(r["attrs"] is None for r in rows)
    save_json(TAGS_PATH, rows)
    print(f"wrote {TAGS_PATH} ({len(rows)} items, {n_fail} unparsed)")


def cmd_regress(n_boot=2000, seed=1):
    from sklearn.linear_model import Ridge, RidgeCV

    attrs, _ = vocab()
    aids = [a["aid"] for a in attrs]
    fam = {a["aid"]: a["family"] for a in attrs}
    tags = {r["id"]: r["attrs"] for r in load_json(TAGS_PATH) if r["attrs"] is not None}

    models = sorted(d.name for d in (P1 / "results" / "stage1x").iterdir()
                    if (d / "utilities_xl.json").exists())
    coefs_by_model, lines_all = {}, ["SURF tag regression: what utility loads on "
                                     "(ridge of 1X mu on 32B-assigned attribute tags)", ""]
    for model in models:
        rows = [r for r in load_json(P1 / "results" / "stage1x" / model / "utilities_xl.json")
                if not r["validation"] and r["id"] in tags]
        X = np.zeros((len(rows), len(aids)))
        for i, r in enumerate(rows):
            for a in tags[r["id"]]:
                X[i, aids.index(a)] = 1.0
        y = np.array([r["mu"] for r in rows])
        keep = X.sum(0) >= 10  # attributes too rare to estimate are reported as absent
        cv = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(X[:, keep], y)
        alpha, r2 = float(cv.alpha_), float(cv.score(X[:, keep], y))
        rng = np.random.RandomState(seed)
        boots = np.zeros((n_boot, int(keep.sum())))
        for b in range(n_boot):
            idx = rng.randint(0, len(y), len(y))  # item-clustered = plain rows here
            boots[b] = Ridge(alpha=alpha).fit(X[idx][:, keep], y[idx]).coef_
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)

        betas, k = [], 0
        for j, aid in enumerate(aids):
            if keep[j]:
                betas.append({"aid": aid, "family": fam[aid], "n_items": int(X[:, j].sum()),
                              "beta": round(float(cv.coef_[k]), 4),
                              "lo": round(float(lo[k]), 4), "hi": round(float(hi[k]), 4)})
                k += 1
        coefs_by_model[model] = {b["aid"]: b["beta"] for b in betas}
        outd = OUT_ROOT / model
        outd.mkdir(parents=True, exist_ok=True)
        save_json(outd / "tag_betas.json",
                  {"n_items": len(rows), "alpha": alpha, "r2": round(r2, 4), "betas": betas})
        for direction, sgn in (("max", 1.0), ("min", -1.0)):
            raw = {b["aid"]: max(sgn * b["beta"], 0.0) for b in betas}
            floor = 0.02 * max(abs(b["beta"]) for b in betas)
            w = {a: raw.get(a, 0.0) + floor for a in aids}
            s = sum(w.values())
            save_json(outd / f"pool_weights_{direction}.json",
                      {a: round(v / s, 6) for a, v in w.items()})

        ranked = sorted(betas, key=lambda b: -b["beta"])
        lines = [f"{model}: n={len(rows)} tagged items, alpha={alpha:g}, R2={r2:.3f}",
                 "  top +: " + "; ".join(f"{b['aid']} {b['beta']:+.2f} "
                                         f"[{b['lo']:+.2f},{b['hi']:+.2f}]"
                                         for b in ranked[:10]),
                 "  top -: " + "; ".join(f"{b['aid']} {b['beta']:+.2f} "
                                         f"[{b['lo']:+.2f},{b['hi']:+.2f}]"
                                         for b in ranked[-10:][::-1])]
        (outd / "summary.txt").write_text("\n".join(lines) + "\n")
        lines_all += lines + [""]

    lines_all.append("cross-model beta correlations:")
    for i, m1 in enumerate(models):
        for m2 in models[i + 1:]:
            shared = sorted(set(coefs_by_model[m1]) & set(coefs_by_model[m2]))
            r = pearson([coefs_by_model[m1][a] for a in shared],
                        [coefs_by_model[m2][a] for a in shared])
            lines_all.append(f"  {m1} vs {m2}: r = {r:+.3f} ({len(shared)} attrs)")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "cross_model.txt").write_text("\n".join(lines_all) + "\n")
    print("\n".join(lines_all))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tag", "regress"])
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    cmd_tag(a.smoke) if a.cmd == "tag" else cmd_regress()


if __name__ == "__main__":
    main()
