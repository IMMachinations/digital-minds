"""A/B-free preference measurement and steering vectors from object logits.

Prompts present the two items inline with no A/B labels; preference is the teacher-forced
log-prob of each item as the continuation of "... Model: I prefer". New vectors (raw + centered)
are extracted from the new suffix span and validated with tier + random-vector controls.

Run: python objects.py [modifier|inherent]  (needs results/{mode}/preferences.json for the
old-vs-new correlation). Outputs: objects.json, vectors_obj.pt, objects_steer.json.
"""
import json

import torch

import experiment as E  # loads model; picks up mode from sys.argv[1]
from data import COLORS, make_comparisons

BODIES = ["Pick between {a} and {b}.\n", "Here are two things: {a} and {b}.\n",
          "Consider {a} and {b}.\n"]
OBJ_SUFFIX = "Which one do you prefer?\nModel: I prefer"

COMPS = make_comparisons(n_per_pair=10, seed=0, inherent=(E.MODE == "inherent"))
for c in COMPS:
    c["prompt"] = BODIES[c["template"]].format(a=c["item_a"], b=c["item_b"]) + OBJ_SUFFIX
N_SUF = len(E.tok(COMPS[0]["prompt"]).input_ids) - \
    len(E.tok(COMPS[0]["prompt"].removesuffix(OBJ_SUFFIX)).input_ids)


@torch.no_grad()
def cont_logprob(prompts, conts, steer=None, batch=64):
    """Per-token-mean (and sum) log-prob of each continuation, teacher-forced. steer=(layer, vec)
    adds vec to the residual stream over the suffix + continuation positions."""
    means, sums = [], []
    for i in range(0, len(prompts), batch):
        ps, cs = prompts[i:i + batch], conts[i:i + batch]
        ks = [len(E.tok(p + c).input_ids) - len(E.tok(p).input_ids) for p, c in zip(ps, cs)]
        enc = E.encode([p + c for p, c in zip(ps, cs)])
        S = enc.input_ids.shape[1]
        handle = None
        if steer is not None:
            layer, vec = steer
            mask = torch.zeros(len(ps), S, dtype=torch.bool, device="cuda")
            for r, k in enumerate(ks):
                mask[r, S - (k + N_SUF):] = True
            def add_vec(m, i_, o):
                E.resid(o)[mask] += vec
            handle = E.LAYERS[layer].register_forward_hook(add_vec)
        try:
            lp = E.model(**enc).logits.float().log_softmax(-1)
        finally:
            if handle:
                handle.remove()
        for r, k in enumerate(ks):
            tok_lps = [lp[r, j - 1, enc.input_ids[r, j]].item() for j in range(S - k, S)]
            sums.append(sum(tok_lps))
            means.append(sums[-1] / k)
    return torch.tensor(means), torch.tensor(sums)


def measure():
    prompts = [c["prompt"] for c in COMPS for _ in range(2)]
    conts = [" " + c[k] for c in COMPS for k in ("item_a", "item_b")]
    m, s = cont_logprob(prompts, conts)
    for j, c in enumerate(COMPS):
        c["lp_a"], c["lp_b"] = m[2 * j].item(), m[2 * j + 1].item()
        c["lp_a_sum"], c["lp_b_sum"] = s[2 * j].item(), s[2 * j + 1].item()
        c["diff"] = c["lp_a"] - c["lp_b"]  # mean log-prob diff, the primary score
    (E.OUT / "objects.json").write_text(json.dumps(COMPS, indent=1))

    old = json.loads((E.OUT / "preferences.json").read_text())
    r = torch.corrcoef(torch.stack([torch.tensor([c["diff"] for c in COMPS]),
                                    torch.tensor([c["diff"] for c in old])]))[0, 1]
    print(f"\ncorr(object-logprob diff, old A/B logit diff) over 420 pairs = {r:+.3f}")
    print("\nPer-color mean signed preference (mean log-prob units):")
    for col in COLORS:
        v = torch.tensor([E.signed(c["diff"], c, col) for c in COMPS
                          if col in (c["color_a"], c["color_b"])]).mean()
        print(f"  {col:>8}: {v:+.4f}")


def tier_comps(color, lo, hi):
    mine = [c for c in COMPS if color in (c["color_a"], c["color_b"])]
    return sorted(mine, key=lambda c: E.signed(c["diff"], c, color))[lo:hi]


def extract():
    means = {}
    all_norms = []
    for color in COLORS:
        top = tier_comps(color, 100, 120)  # strongest wins by the new measure
        acts, norms = E.suffix_acts([c["prompt"] for c in top], n_suffix=N_SUF)
        means[color] = acts.mean(1)
        all_norms.append(norms)
    mu = torch.stack(list(means.values())).mean(0)
    vecs = {c: m / m.norm(dim=-1, keepdim=True) for c, m in means.items()}
    cent = {c: torch.nn.functional.normalize(m - mu, dim=-1) for c, m in means.items()}
    resid_norms = torch.stack(all_norms).mean(0)
    torch.save({"vecs": vecs, "centered": cent, "resid_norms": resid_norms},
               E.OUT / "vectors_obj.pt")
    return vecs, cent, resid_norms


def steer(vecs, cent, resid_norms):
    g = torch.Generator().manual_seed(0)
    rand = {c: torch.nn.functional.normalize(torch.randn(vecs[c].shape, generator=g), dim=-1)
            for c in COLORS}
    SOURCES = {"same": vecs, "cent": cent, "rand": rand}
    TIERS = {"worst": (0, 20), "neutral": (50, 70), "best": (100, 120)}
    rows = []
    for tier, (lo, hi) in TIERS.items():
        for color in COLORS:
            comps = tier_comps(color, lo, hi)
            ps = [c["prompt"] for c in comps for _ in range(2)]
            cs = [" " + c[k] for c in comps for k in ("item_a", "item_b")]
            def signed_diff(m):
                return torch.tensor([E.signed((m[2 * j] - m[2 * j + 1]).item(), c, color)
                                     for j, c in enumerate(comps)]).mean().item()
            base = signed_diff(cont_logprob(ps, cs)[0])
            for src, vv in SOURCES.items():
                for L in E.STEER_LAYERS:
                    for cf in [1.0, 2.0]:
                        vec = (vv[color][L] * cf * resid_norms[L]).to("cuda", torch.bfloat16)
                        new = signed_diff(cont_logprob(ps, cs, steer=(L, vec))[0])
                        rows.append(dict(tier=tier, source=src, color=color, layer=L, coef=cf,
                                         base=base, steered=new, delta=new - base))
        print(f"steered {tier}")
    (E.OUT / "objects_steer.json").write_text(json.dumps(rows, indent=1))

    for cf in [1.0, 2.0]:
        print(f"\n{E.MODE} objects, coef {cf} — mean Δ(mean-logprob diff) toward steered color:")
        cols = [(s, t) for s in SOURCES for t in TIERS]
        print("       " + "".join(f"{s[:1]}:{t:>8}"[:11].rjust(11) for s, t in cols))
        for L in E.STEER_LAYERS:
            vals = []
            for s, t in cols:
                rs = [r["delta"] for r in rows if (r["coef"], r["layer"], r["source"],
                                                   r["tier"]) == (cf, L, s, t)]
                vals.append(sum(rs) / len(rs))
            print(f"L{L:>4}  " + "".join(f"{v:11.3f}" for v in vals))
    bases = {t: sum(r["base"] for r in rows if r["tier"] == t) /
             max(1, sum(1 for r in rows if r["tier"] == t)) for t in TIERS}
    print("  unsteered base per tier: " + ", ".join(f"{t}={b:+.3f}" for t, b in bases.items()))


if __name__ == "__main__":
    # spot-check: the last k tokens of prompt+cont must decode back to the continuation
    for c in COMPS[:3]:
        ids = E.tok(c["prompt"] + " " + c["item_a"]).input_ids
        k = len(ids) - len(E.tok(c["prompt"]).input_ids)
        assert E.tok.decode(ids[-k:]) == " " + c["item_a"], c
    measure()
    steer(*extract())
