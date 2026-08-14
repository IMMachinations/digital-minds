"""Color preference + steering experiment on Qwen2.5-7B-Instruct.

Run: python experiment.py [modifier|inherent]
  modifier (default): items are "a {color} {noun}" over a shared noun list
  inherent: items are inherently-colored things (banana, ruby, an indigo bunting, ...)
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import COLORS, SUFFIX, make_comparisons

MODEL = "Qwen/Qwen2.5-7B-Instruct"
MODE = sys.argv[1] if len(sys.argv) > 1 else "modifier"
OUT = Path(__file__).parent / "results" / MODE
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"  # prompts right-aligned: last N_SUFFIX positions are the suffix for all
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda").eval()
LAYERS = model.model.layers

COMPS = make_comparisons(n_per_pair=10, seed=0, inherent=(MODE == "inherent"))  # 420, balanced
_full, _pre = COMPS[0]["prompt"], COMPS[0]["prompt"].removesuffix(SUFFIX)
N_SUFFIX = len(tok(_full).input_ids) - len(tok(_pre).input_ids)
assert tok.decode(tok(_full).input_ids[-N_SUFFIX:]).endswith("I prefer")

# Surface forms of A/B checked against the step-0 inspection. Uppercase only: " a" is the
# article in "I prefer a red cup", not a vote for option A.
VARIANTS = ["A", " A", "(A", " (A"]  # non-single-token forms are dropped below
def variant_ids(letter):
    forms = [(v.replace("A", letter), tok.encode(v.replace("A", letter), add_special_tokens=False))
             for v in VARIANTS]
    return {v: ids[0] for v, ids in forms if len(ids) == 1}
A_IDS, B_IDS = variant_ids("A"), variant_ids("B")


def encode(prompts):
    return tok(prompts, return_tensors="pt", padding=True).to("cuda")


def resid(out):
    """Residual-stream tensor from a decoder layer's output (tuple in older transformers)."""
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def last_logits(prompts, steer=None, batch=64, n_suffix=None):
    """Final-position logits [N, vocab]. steer=(layer, vec) adds vec to that layer's residual
    stream at the suffix positions (n_suffix overrides the default prefer-framing length)."""
    handle = None
    if steer is not None:
        layer, vec = steer
        ns = n_suffix or N_SUFFIX
        def add_vec(m, i, o):  # mutate in place; a hook's return value would replace the output
            resid(o)[:, -ns:, :].add_(vec)
        handle = LAYERS[layer].register_forward_hook(add_vec)
    try:
        return torch.cat([model(**encode(prompts[i:i + batch])).logits[:, -1, :].float().cpu()
                          for i in range(0, len(prompts), batch)])
    finally:
        if handle:
            handle.remove()


@torch.no_grad()
def suffix_acts(prompts, batch=64):
    """Residual activations meaned over suffix tokens: acts [n_layers, N, d_model] and the
    per-layer mean token norm [n_layers]."""
    acts, norms = [[] for _ in LAYERS], [[] for _ in LAYERS]
    def cap(j):
        def hook(m, i, o):
            s = resid(o)[:, -N_SUFFIX:, :].float()
            acts[j].append(s.mean(1).cpu())
            norms[j].append(s.norm(dim=-1).mean().cpu())
        return hook
    handles = [L.register_forward_hook(cap(j)) for j, L in enumerate(LAYERS)]
    try:
        for i in range(0, len(prompts), batch):
            model(**encode(prompts[i:i + batch]))
    finally:
        for h in handles:
            h.remove()
    return torch.stack([torch.cat(a) for a in acts]), torch.tensor([torch.stack(n).mean() for n in norms])


def scores(logits):
    """Per-side total logit attribution: logsumexp over the variant tokens of each letter."""
    a = logits[:, list(A_IDS.values())]
    b = logits[:, list(B_IDS.values())]
    return torch.logsumexp(a, -1), torch.logsumexp(b, -1), a, b


def signed(diff, comp, color):
    """Logit diff oriented toward `color` (positive = model favors that color's side)."""
    return diff if comp["color_a"] == color else -diff


# ---- Step 0: inspect the raw next-token distribution on a few examples --------------------------
def inspect(n=5):
    lines = []
    lg = last_logits([c["prompt"] for c in COMPS[:n]])
    for c, row in zip(COMPS[:n], lg):
        top = row.topk(20)
        lines.append(c["prompt"].replace("\n", " | "))
        lines.append("  " + ", ".join(f"{tok.decode([i])!r}:{v:.1f}" for v, i in zip(*top)))
    text = "\n".join(lines) + f"\n\nA variants used: {A_IDS}\nB variants used: {B_IDS}\n"
    (OUT / "inspect.txt").write_text(text)
    print(text)


# ---- Step 1: measure preferences over all 420 comparisons ---------------------------------------
def measure():
    lg = last_logits([c["prompt"] for c in COMPS])
    sa, sb, va, vb = scores(lg)
    for c, xa, xb, ra, rb in zip(COMPS, sa, sb, va, vb):
        c["score_a"], c["score_b"], c["diff"] = xa.item(), xb.item(), (xa - xb).item()
        c["logits_a"] = dict(zip(A_IDS, ra.tolist()))
        c["logits_b"] = dict(zip(B_IDS, rb.tolist()))
    (OUT / "preferences.json").write_text(json.dumps(COMPS, indent=1))

    print("\nMean diff (row color as A vs col color as B; + = row preferred):")
    print("        " + "".join(f"{c:>8}" for c in COLORS))
    for ca in COLORS:
        row = [torch.tensor([c["diff"] for c in COMPS if (c["color_a"], c["color_b"]) == (ca, cb)]).mean()
               if ca != cb else float("nan") for cb in COLORS]
        print(f"{ca:>8}" + "".join(f"{v:8.2f}" for v in row))
    print("\nPer-color mean signed preference:")
    for col in COLORS:
        m = torch.tensor([signed(c["diff"], c, col) for c in COMPS
                          if col in (c["color_a"], c["color_b"])]).mean()
        print(f"  {col:>8}: {m:+.3f}")


# ---- Step 2: extract per-color direction vectors from the top-scoring wins ----------------------
K = 20

def comps_for(color, best=True):
    """The K comparisons `color` won (best=True) or lost (best=False) most strongly."""
    mine = [c for c in COMPS if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: signed(c["diff"], c, color), reverse=best)[:K]

def extract():
    vecs, all_norms = {}, []
    for color in COLORS:
        acts, norms = suffix_acts([c["prompt"] for c in comps_for(color, best=True)])
        mean = acts.mean(1)                                   # [n_layers, d_model]
        vecs[color] = mean / mean.norm(dim=-1, keepdim=True)  # unit vector per layer
        all_norms.append(norms)
    resid_norms = torch.stack(all_norms).mean(0)              # typical residual norm per layer
    torch.save({"vecs": vecs, "resid_norms": resid_norms}, OUT / "vectors.pt")
    return vecs, resid_norms


# ---- Step 3: steer the worst examples of each color with its own vector -------------------------
STEER_LAYERS = [7, 11, 14, 18, 21]  # of 28
COEFS = [0.5, 1.0, 2.0]             # x typical residual norm at that layer

def steer(vecs, resid_norms):
    rows = []
    for color in COLORS:
        worst = comps_for(color, best=False)
        prompts = [c["prompt"] for c in worst]
        base = torch.tensor([signed(c["diff"], c, color) for c in worst])
        for L in STEER_LAYERS:
            for cf in COEFS:
                vec = (vecs[color][L] * cf * resid_norms[L]).to("cuda", torch.bfloat16)
                sa, sb, _, _ = scores(last_logits(prompts, steer=(L, vec)))
                new = torch.tensor([signed((xa - xb).item(), c, color)
                                    for c, xa, xb in zip(worst, sa, sb)])
                rows.append(dict(color=color, layer=L, coef=cf, base=base.mean().item(),
                                 steered=new.mean().item(), delta=(new - base).mean().item()))
        print(f"steered {color}")
    (OUT / "steering.json").write_text(json.dumps(rows, indent=1))

    print("\nMean delta in signed logit-diff toward steered color (rows=layer, cols=coef):")
    print("       " + "".join(f"{cf:>8}" for cf in COEFS))
    for L in STEER_LAYERS:
        deltas = [torch.tensor([r["delta"] for r in rows if (r["layer"], r["coef"]) == (L, cf)]).mean()
                  for cf in COEFS]
        print(f"L{L:>4}  " + "".join(f"{d:8.2f}" for d in deltas))


if __name__ == "__main__":
    inspect()
    measure()
    steer(*extract())
