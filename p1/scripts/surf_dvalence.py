"""Early-steer -> late-read preference->valence experiment (d'valence).

The 4B/4C null (preference steering never moves generation-state valence) was
measured with injection and readout at essentially the same depth (steer
work_layers[1], read the Stage-2 probe layer). This experiment gives the
model room to REACT: steer the utility direction early (0.25/0.39 depth,
layers 7/11 on qwen25-7b), read valence PC1 late (14/18/21) — up to half the
stack of intervening computation — with the same-layer configuration kept as
the bridge control and matched-norm random directions as nulls.

  dirs <model>   activation pass over the global item set at the early layers;
                 ridge utility directions per layer (held-out r reported — how
                 well utility even decodes at 0.25 depth); chat+completion
                 residual norms at the early layers; per-layer valence PC1s
                 and their cos with the utility dirs (direct-contamination
                 bound). -> dvalence_dirs.pt
  gate <model>   purchase check: the 24x6 elicitation cell steered at each
                 early layer (c=0.5, both signs, 3 random nulls/layer). A
                 layer where utility steering moves nothing makes any valence
                 null there uninterpretable — measured first, not assumed.
  roll <model>   Stage-3-style bare-frame rollouts on preferred/dispreferred
                 tasks, steered on assistant tokens (dispref -> +utility,
                 pref -> -utility, the 4bc convention), one lockstep group per
                 steer layer; unsteered + random cells included.
  read <model>   steering-replay re-encode (stage3_probes.transcript_tokens),
                 assistant-token valence projection at each read layer;
                 effect sizes vs unsteered cells, z vs random nulls.

Usage: uv run python scripts/surf_dvalence.py {dirs,gate,roll,read,all} <model>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import pearson

from surf_probeloop import heldout_preds, winsorize

COEF = 0.5
N_CELL = 10                      # rollouts per (pcond, dirset) cell
N_RAND = 2                       # random null seeds per steer layer


def out_dir(model):
    d = P1 / "results" / "surf" / "dvalence" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def steer_layers_early(model):
    from modelspec import ROSTER, STEER_FRACS
    from surf_s0 import N_LAYERS
    n = N_LAYERS[model]
    L = [round(f * n) for f in STEER_FRACS]
    return L[:2], L[2:]          # early (0.25, 0.39), working (0.50, 0.64, 0.75)


def cmd_dirs(model):
    import torch
    from sklearn.linear_model import RidgeCV
    import harness
    import probes
    import rollout as ro
    import steering as st
    from stage2 import EMOTIONS

    early, work = steer_layers_early(model)
    rows = load_json(P1 / "results" / "surf" / "global" / model / "utilities_global.json")
    mu, _ = winsorize([r["mu"] for r in rows])
    h = harness.load(model)
    cache = out_dir(model) / "acts_early.pt"
    if cache.exists():
        acts = torch.load(cache, weights_only=False).float()
    else:
        acts = probes.item_acts(h, rows)[early].clone()
        torch.save(acts.half(), cache)
        acts = acts.float()

    d = {"early": early, "work": work, "coef": COEF, "dirs": {}, "cv_r": {},
         "valence": {}, "cos": {}}
    preds, rs = heldout_preds(acts, mu)
    for k, L in enumerate(early):
        X = acts[k].numpy()
        m0, sd = X.mean(0), X.std(0) + 1e-6
        m = RidgeCV(alphas=np.logspace(1, 6, 8)).fit((X - m0) / sd, mu)
        v = m.coef_ / sd
        v = v / np.linalg.norm(v)
        if pearson(list(X @ v), list(mu)) < 0:
            v = -v
        d["dirs"][L] = v
        d["cv_r"][L] = round(rs[k], 4)
    g = torch.load(P1 / "results" / "surf" / "global" / model / "utility_dir_global.pt",
                   weights_only=False)
    d["dirs"][g["layer"]] = g["dir"].numpy()
    d["cv_r"][g["layer"]] = None  # from surf_global analyze

    from items import load_items
    NORMS = load_json(P1 / "items" / "emotion_norms.json")
    den_all = torch.load(P1 / "results" / "stage2" / model / "vectors.pt",
                         weights_only=False)["den"].float()
    val_norm = np.array([NORMS[e]["valence"] for e in EMOTIONS])
    for R in work:
        den = den_all[R].numpy()
        c = den - den.mean(0)
        _, _, Vt = np.linalg.svd(c, full_matrices=False)
        sgn = 1.0 if pearson(list(c @ Vt[0]), list(val_norm)) >= 0 else -1.0
        d["valence"][R] = Vt[0] * sgn
        for L, v in d["dirs"].items():
            d["cos"][f"u{L}_v{R}"] = round(float(np.dot(v, Vt[0] * sgn)), 4) \
                if len(v) == len(Vt[0]) else None

    # residual norms at the early layers, both frames (stage4 only cached work layers)
    items24 = [r["text"] for r in rows[:200]]
    import stage4 as s4
    ep, _, _ = s4.build_cell_prompts(
        *(lambda a, b: (a[:12], b[:3]))(*s4.pick_items24(model)))
    d["rn_completion"] = {L: float(x) for L, x in
                          zip(early, st.frame_resid_norms(h, ep, early))}
    chat_prompts = [ro.build_prompt(h, [
        {"role": "system", "content": ro.SYSTEM},
        {"role": "user", "content": f"Session brief: {t}"}]) for t in items24]
    d["rn_chat"] = {L: float(x) for L, x in
                    zip(early, st.frame_resid_norms(h, chat_prompts, early))}
    torch.save(d, out_dir(model) / "dvalence_dirs.pt")
    print(f"{model} dirs: early utility held-out r = "
          + ", ".join(f"L{L}:{d['cv_r'][L]:+.3f}" for L in early)
          + "; cos(u, v_val) = " + json.dumps(d["cos"]))


def _load(model):
    import torch
    return torch.load(out_dir(model) / "dvalence_dirs.pt", weights_only=False)


def cmd_gate(model):
    import torch
    import harness
    import stage4 as s4
    from lib.tasks import variant_ids
    from stage4_rollouts import _mean_d

    d = _load(model)
    h = harness.load(model)
    items24, anchors6 = s4.pick_items24(model)
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = s4.build_cell_prompts(items24, anchors6)
    ctrl = load_json(s4._cell_path(model, "control"))
    d0 = _mean_d(ctrl["recs"])
    D4 = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                    weights_only=False)
    rn_c = {**{int(k): v for k, v in d["rn_completion"].items()},
            **D4["resid_norm"]["completion"]}

    out, lines = {}, []
    layers = d["early"] + [d["work"][1]]
    for L in layers:
        vecu = d["dirs"][L]
        nulls = []
        for sd_ in range(3):
            rv = harness.random_unit_matrix(h.spec.n_layers, h.spec.d_model,
                                            seed=sd_)[L].numpy()
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(rv) * sign, COEF, rn_c[L])
                recs, _, _, _ = s4.run_cell(h, prompts, spans, meta, L, v, a_ids, b_ids)
                nulls.append(_mean_d(recs) - d0)
        row = {"null_sd": round(float(np.std(nulls, ddof=1)), 4)}
        for sign in (1, -1):
            v = harness.scaled_vec(torch.tensor(vecu) * sign, COEF, rn_c[L])
            recs, _, _, _ = s4.run_cell(h, prompts, spans, meta, L, v, a_ids, b_ids)
            row[f"dd_{'plus' if sign > 0 else 'minus'}"] = round(_mean_d(recs) - d0, 4)
        row["z_plus"] = round(row["dd_plus"] / (row["null_sd"] + 1e-9), 2)
        out[str(L)] = row
        lines.append(f"  choice-purchase at L{L}: dd+ {row['dd_plus']:+.3f} "
                     f"z {row['z_plus']:+.1f} (dd- {row['dd_minus']:+.3f})")
    save_json(out_dir(model) / "gate.json", out)
    print("\n".join(lines))


def cmd_roll(model):
    import torch
    import harness
    import rollout as ro
    import stage3
    import steering as st

    d = _load(model)
    h = harness.load(model)
    D4 = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                    weights_only=False)
    rn = {**{int(k): v for k, v in d["rn_chat"].items()}, **D4["resid_norm"]["chat"]}
    envs, pref, dis, match = stage3.pools(model)
    layers = d["early"] + [d["work"][1]]
    dirsets = [("none", None, None)]
    for L in layers:
        dirsets.append((f"utility_L{L}", L, d["dirs"][L]))
        for sd_ in range(N_RAND):
            dirsets.append((f"random{sd_}_L{L}", L,
                            harness.random_unit_matrix(h.spec.n_layers, h.spec.d_model,
                                                       seed=sd_)[L].numpy()))
    gen_batch = {"llama31-8b": 12}.get(model, 24)
    kw = dict(h.spec.thinking_kwargs)
    kw.update(ro.EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))
    outp = out_dir(model) / "rolls.jsonl"
    done = {json.loads(l)["rid"] for l in outp.read_text().splitlines()} \
        if outp.exists() else set()
    with open(outp, "a") as f:
        for name, L, vec in dirsets:
            cells = []
            for pcond, pool in (("pref", pref), ("dispref", dis)):
                sign = -1 if pcond == "pref" else 1
                for i in range(N_CELL):
                    cells.append((pcond, "good", i, pool[i % len(pool)], sign))
            arm = stage3.Stage3Arm(envs, "bare", [(p, o, i, e)
                                                  for p, o, i, e, _ in cells])
            rolls = arm.make_rollouts()
            for r, (_, _, _, _, sign) in zip(rolls, cells):
                r.rid = f"dv/{name}/{r.rid}"
                r.meta.update({"dirset": name, "steer_layer": L, "sign": sign})
            rolls = [r for r in rolls if r.rid not in done]
            if not rolls:
                continue
            steer_fn = None
            if vec is not None:
                def steer_fn(live, L=L, vec=vec):
                    def mask_fn(sl, enc):
                        rows_ = live[sl]
                        mask = st.assistant_prefill_mask(
                            h, [r.messages for r in rows_],
                            {k: v.cpu() for k, v in enc.items()}, kw).to("cuda")
                        vs = torch.stack([
                            harness.scaled_vec(torch.tensor(vec) * r.meta["sign"],
                                               COEF, rn[L]).cpu()
                            for r in rows_]).to("cuda")
                        return mask, vs.unsqueeze(1)
                    return (L, mask_fn)
            ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=9,
                            gen_batch=gen_batch, max_new=170, steer_fn=steer_fn)
            for r in rolls:
                f.write(json.dumps({"rid": r.rid, "meta": {k: v for k, v in
                                                           r.meta.items()},
                                    "messages": r.messages, "flags": r.flags}) + "\n")
                f.flush()
            print(f"rolled {name}: {len(rolls)} rollouts")


def cmd_read(model):
    import torch
    import harness
    import stage3_probes as s3p

    d = _load(model)
    h = harness.load(model)
    D4 = torch.load(P1 / "results" / "stage4" / model / "directions.pt",
                    weights_only=False)
    rn = {**{int(k): v for k, v in d["rn_chat"].items()}, **D4["resid_norm"]["chat"]}
    rows = [json.loads(l)
            for l in (out_dir(model) / "rolls.jsonl").read_text().splitlines()]
    out = []
    for r in rows:
        L = r["meta"]["steer_layer"]
        steer = None
        if L is not None:
            name = r["meta"]["dirset"]
            vec = d["dirs"][L] if name.startswith("utility") else \
                harness.random_unit_matrix(h.spec.n_layers, h.spec.d_model,
                                           seed=int(name[6]))[L].numpy()
            sv = harness.scaled_vec(torch.tensor(vec) * r["meta"]["sign"], COEF, rn[L])
            steer = (L, sv)
        rec = {"rid": r["rid"], "dirset": r["meta"]["dirset"],
               "steer_layer": L, "pcond": r["meta"]["pref"], "val": {}}
        for R in d["work"]:
            acts, turn, role = s3p.transcript_tokens(h, r["messages"], R, steer=steer)
            m = (np.array(role) == "assistant") & (np.array(turn) >= 1)
            if m.sum() == 0:
                continue
            proj = np.asarray(acts)[m] @ d["valence"][R]
            rec["val"][str(R)] = round(float(proj.mean()), 4)
        out.append(rec)
    save_json(out_dir(model) / "valence_reads.json", out)

    lines = [f"{model} d'valence: steer utility early, read valence late "
             f"(c={COEF}, n={N_CELL}/cell; cos contamination bounds: "
             + json.dumps(d["cos"]) + ")"]
    for R in d["work"]:
        base = {p: [r["val"][str(R)] for r in out if r["dirset"] == "none"
                    and r["pcond"] == p and str(R) in r["val"]]
                for p in ("pref", "dispref")}
        for L in d["early"] + [d["work"][1]]:
            for p in ("pref", "dispref"):
                u = [r["val"][str(R)] for r in out
                     if r["dirset"] == f"utility_L{L}" and r["pcond"] == p
                     and str(R) in r["val"]]
                rand = [np.mean([r["val"][str(R)] for r in out
                                 if r["dirset"] == f"random{sd_}_L{L}"
                                 and r["pcond"] == p and str(R) in r["val"]])
                        - np.mean(base[p]) for sd_ in range(N_RAND)]
                if not u or not base[p]:
                    continue
                sd_pool = np.std(base[p] + u, ddof=1) + 1e-9
                dd = (np.mean(u) - np.mean(base[p])) / sd_pool
                lines.append(f"  steer L{L} read L{R} {p}: d={dd:+.2f} "
                             f"(random-null d "
                             + ",".join(f"{x / sd_pool:+.2f}" for x in rand) + ")")
    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dirs", "gate", "roll", "read", "all"])
    ap.add_argument("model")
    a = ap.parse_args()
    fns = {"dirs": cmd_dirs, "gate": cmd_gate, "roll": cmd_roll, "read": cmd_read}
    if a.cmd == "all":
        for c in ("dirs", "gate", "roll", "read"):
            fns[c](a.model)
    else:
        fns[a.cmd](a.model)


if __name__ == "__main__":
    main()
