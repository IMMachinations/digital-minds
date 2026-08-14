"""Do color-preference vectors shift the model's dollar valuation of single colored items?

Run: python value.py [modifier|inherent] [inspect|pilot|full]   (after experiment.py)
  vectors come from results/{mode}/vectors.pt; results land in results/value_{mode}/

No comparison in the prompt: each item is valued alone ("...its value at $"), we sample
N_SAMPLES completions at temperature, parse the leading dollar amount, and average — under
no steering and under steering with each color's prefer-vector (full cross: blue paintings
steered with the red vector, blue vector, ...).

Reproducibility note: torch.manual_seed is set per generate-batch, so outputs are stable only
for fixed item order, BATCH, and N_SAMPLES.
"""
import json
import re
import sys
from pathlib import Path

import torch

import experiment as E  # loads model/tokenizer (left padding); E.MODE comes from sys.argv[1]
from data import COLORS
from value_data import TEMPLATE, SUFFIX, make_items

STAGE = sys.argv[2] if len(sys.argv) > 2 else "full"
OUT = Path(__file__).parent / "results" / f"value_{E.MODE}"
OUT.mkdir(parents=True, exist_ok=True)

_saved = torch.load(E.OUT / "vectors.pt")
VECS, RESID_NORMS = _saved["vecs"], _saved["resid_norms"]

_full = TEMPLATE.format(item="x", suffix=SUFFIX)
_pre = TEMPLATE.format(item="x", suffix="")
N_SUFFIX = len(E.tok(_full).input_ids) - len(E.tok(_pre).input_ids)
assert E.tok.decode(E.tok(_full).input_ids[-N_SUFFIX:]).endswith("$")

N_SAMPLES = 5
MAX_NEW = 16
BATCH = 28  # x N_SAMPLES return sequences = 140 sequences per generate call
# Pin every sampling knob: Qwen's shipped generation_config (T=0.7, top_p=0.8, top_k=20,
# repetition_penalty=1.05) silently applies to anything not overridden.
GEN_KW = dict(do_sample=True, temperature=0.8, top_p=0.95, top_k=0, repetition_penalty=1.0)

# Pilot findings (results/value_inherent/pilot{,2}.json): coef 2 collapses generation into
# ": A:" babble (the vectors carry the extraction prompts' answer-a-letter content, which A/B
# logit-diffs cancelled but free generation exposes); coef 1 mode-collapses values to $1000.
# Coefs 0.25/0.5 stay fully parseable with a milder uniform depression, so the full cross runs
# both: 0.25 minimizes disruption/floor effects, 0.5 pushes harder.
STEER_CONFIGS = [(14, 0.25), (14, 0.5)]  # (layer, coef)
PILOT_LAYERS, PILOT_COEFS, PILOT_N = [14, 18], [1.0, 2.0], 5

DOMAINS = ["painting", "household", "real"]


def steer_vec(color, layer, coef):
    return (VECS[color][layer] * coef * RESID_NORMS[layer]).to("cuda", torch.bfloat16)


@torch.no_grad()
def generate(prompts, steer=None, seed=0, n_samples=N_SAMPLES, max_new=MAX_NEW):
    """n_samples decoded continuations per prompt. steer=(layer, vec) reuses E.last_logits's
    in-place residual hook: at prefill it hits the last N_SUFFIX (right-aligned) positions; at
    each KV-cached decode step the seq len is 1, which the [:, -N_SUFFIX:, :] slice covers, so
    the vector also rides under every generated token."""
    handle = None
    if steer is not None:
        layer, vec = steer
        def add_vec(m, i, o):
            E.resid(o)[:, -N_SUFFIX:, :].add_(vec)
        handle = E.LAYERS[layer].register_forward_hook(add_vec)
    outs = []
    try:
        for bi, i in enumerate(range(0, len(prompts), BATCH)):
            torch.manual_seed(seed + bi)
            enc = E.encode(prompts[i:i + BATCH])
            out = E.model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n_samples,
                                   pad_token_id=E.tok.pad_token_id, **GEN_KW)
            texts = E.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
            outs += [texts[j:j + n_samples] for j in range(0, len(texts), n_samples)]
    finally:
        if handle:
            handle.remove()
    return outs


# The prompt ends with "$", so a continuation should start with a number. \b keeps "b" from
# matching the b of "bucks"; suffixes multiply (200k, 1.5 million).
_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|bn|k|m|b)?\b"
_LEAD = re.compile(r"^\s*" + _NUM, re.I)
_RANGE = re.compile(r"^\s*(?:-|–|—|to)\s*\$?\s*" + _NUM, re.I)
_MULT = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "b": 1e9, "bn": 1e9, "billion": 1e9}


def parse_dollars(text):
    """Leading dollar amount, or None. Ranges ('50 to $100') return the midpoint — the model
    stated an interval and its midpoint is the natural point estimate; taking the low end would
    bias steered-vs-baseline deltas wherever steering changes how often ranges appear."""
    m = _LEAD.match(text)
    if not m:
        return None
    def val(g):
        return float(g.group(1).replace(",", "")) * _MULT.get((g.group(2) or "").lower(), 1)
    r = _RANGE.match(text[m.end():])
    if r:
        lo, hi = val(m), val(r)
        if not m.group(2) and r.group(2):  # "20-30 million": the suffix distributes to both ends
            lo *= _MULT.get(r.group(2).lower(), 1)
        return (lo + hi) / 2
    return val(m)


# ---- aggregation / reporting --------------------------------------------------------------------
def stats(rows):
    """(plain mean, median, geometric mean, n_parsed, n_total) over parsed positive values."""
    vals = torch.tensor([r["value"] for r in rows if r["value"]])
    if not len(vals):
        return float("nan"), float("nan"), float("nan"), 0, len(rows)
    return (vals.mean().item(), vals.median().item(),
            (10 ** vals.log10().mean()).item(), len(vals), len(rows))


def print_matrix(title, cell, conds, fmt="{:>10.3g}"):
    """cell(item_color, cond) -> float; rows = item color, cols = conditions."""
    print(f"\n{title}")
    print("        " + "".join(f"{c:>10}" for c in conds))
    for ic in COLORS:
        print(f"{ic:>8}" + "".join(fmt.format(cell(ic, c)) for c in conds))


def report(rows, conds):
    """Per-domain matrices of mean / geomean dollar value and deltas vs baseline; matched vs
    mismatched contrast on delta log10(geomean)."""
    def sel(domain, ic, cond):
        return [r for r in rows if (r["domain"], r["item_color"], r["condition"]) == (domain, ic, cond)]
    for domain in DOMAINS:
        if not any(r["domain"] == domain for r in rows):
            continue
        print(f"\n===== {domain} =====")
        S = {(ic, c): stats(sel(domain, ic, c)) for ic in COLORS for c in conds}
        print_matrix("mean $", lambda ic, c: S[ic, c][0], conds)
        print_matrix("median $", lambda ic, c: S[ic, c][1], conds)
        print_matrix("geomean $", lambda ic, c: S[ic, c][2], conds)
        if len(conds) > 1:
            import math
            dlog = {(ic, c): math.log10(S[ic, c][2] / S[ic, "base"][2])
                    if S[ic, c][2] > 0 and S[ic, "base"][2] > 0 else float("nan")
                    for ic in COLORS for c in conds[1:]}
            print_matrix("delta log10(geomean) vs base", lambda ic, c: dlog.get((ic, c), 0.0),
                         conds[1:], fmt="{:>10.3f}")
            steer_cols = [c for c in conds[1:] if c in COLORS]
            if steer_cols:
                matched = [dlog[ic, ic] for ic in steer_cols]
                mism = [dlog[ic, c] for ic in COLORS for c in steer_cols if ic != c]
                print(f"\nmatched (diag) mean dlog10: {sum(matched) / len(matched):+.3f}   "
                      f"mismatched mean dlog10: {sum(mism) / len(mism):+.3f}")
        bad = sum(1 for r in rows if r["domain"] == domain and r["value"] is None)
        print(f"unparseable: {bad}/{sum(1 for r in rows if r['domain'] == domain)}")


def run(items, conditions, out_name):
    """conditions: [(name, steer_color | None, (layer, coef) | None)]. Generates per condition
    over all items, parses, saves every sample row, returns rows."""
    rows = []
    for ci, (name, color, cfg) in enumerate(conditions):
        steer = None
        if cfg is not None:
            layer, coef = cfg
            steer = (layer, steer_vec(color, layer, coef))
        samples = generate([it["prompt"] for it in items], steer=steer, seed=ci * 1000)
        for it, texts in zip(items, samples):
            for si, t in enumerate(texts):
                rows.append(dict(domain=it["domain"], item_color=it["color"], item=it["item"],
                                 condition=name, layer=cfg[0] if cfg else None,
                                 coef=cfg[1] if cfg else None, sample_idx=si, text=t,
                                 value=parse_dollars(t)))
        print(f"condition {name} done ({ci + 1}/{len(conditions)})")
    (OUT / out_name).write_text(json.dumps(rows, indent=1))
    return rows


# ---- Step 0: inspect raw sampled completions on a few examples ----------------------------------
def inspect():
    items = [it for d in DOMAINS for it in [i for i in make_items() if i["domain"] == d][:2]]
    samples = generate([it["prompt"] for it in items], n_samples=3, seed=0)
    lines = []
    for it, texts in zip(items, samples):
        lines.append(it["prompt"].replace("\n", " | "))
        for t in texts:
            lines.append(f"  {t!r} -> {parse_dollars(t)}")
    text = "\n".join(lines) + "\n"
    (OUT / "inspect.txt").write_text(text)
    print(text)


# ---- Pilot: matched-color steering only, small grid, to pick (layer, coef) ----------------------
def pilot():
    items = make_items(n_per_color=PILOT_N)
    by_color = {c: [it for it in items if it["color"] == c] for c in COLORS}
    rows = []
    # baseline
    rows += run(items, [("base", None, None)], "pilot_base.json")
    # matched steering: each color's items under that color's own vector, per grid point
    for L in PILOT_LAYERS:
        for cf in PILOT_COEFS:
            name = f"L{L}x{cf}"
            for c in COLORS:
                samples = generate([it["prompt"] for it in by_color[c]],
                                   steer=(L, steer_vec(c, L, cf)), seed=L * 100 + int(cf * 10))
                for it, texts in zip(by_color[c], samples):
                    for si, t in enumerate(texts):
                        rows.append(dict(domain=it["domain"], item_color=it["color"],
                                         item=it["item"], condition=name, layer=L, coef=cf,
                                         sample_idx=si, text=t, value=parse_dollars(t)))
            print(f"pilot {name} done")
    (OUT / "pilot.json").write_text(json.dumps(rows, indent=1))
    conds = ["base"] + [f"L{L}x{cf}" for L in PILOT_LAYERS for cf in PILOT_COEFS]
    report(rows, conds)


# ---- Full: baseline + all 7 steer colors on every item (cross-color) ----------------------------
def full():
    items = make_items()
    conditions = [("base", None, None)]
    for L, cf in STEER_CONFIGS:
        conditions += [(f"{c}|L{L}x{cf}", c, (L, cf)) for c in COLORS]
    rows = run(items, conditions, "values.json")
    for L, cf in STEER_CONFIGS:
        tag = f"|L{L}x{cf}"
        sub = [dict(r, condition=r["condition"].removesuffix(tag)) for r in rows
               if r["condition"] == "base" or r["condition"].endswith(tag)]
        print(f"\n######## config L{L} x {cf} ########")
        report(sub, ["base"] + COLORS)


if __name__ == "__main__":
    inspect()
    if STAGE == "pilot":
        pilot()
    elif STAGE == "full":
        full()
