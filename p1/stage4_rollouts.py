"""Stage 4 rollout-side commands: gate, 4bc, 4d, judge, analyze, cross.
Split from stage4.py for size; shares its conventions (see stage4.py docstring).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.data import SUFFIX
from lib.tasks import variant_ids
from lib.util import load_json, save_json
from lib.valuation import pearson, spearman

import harness
import judge as judge_mod
import rollout as ro
import steering as st
import stage3
import stage3_probes as sp
from frames import AGENTIC
from pairs import TEMPLATES_GENERIC
from stage2 import EMOTIONS

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b"]
DIRSETS = ["choice", "pool", "utility"]
NORMS = load_json(P1 / "items" / "emotion_norms.json")


def out_dir(model):
    d = P1 / "results" / "stage4" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_dirs(model):
    return torch.load(out_dir(model) / "directions.pt", weights_only=False)


def _mean_d(recs):
    """Mean letter-logit diff toward the free item."""
    ps = np.clip([r["p"] for r in recs], 1e-6, 1 - 1e-6)
    return float(np.mean(np.log(ps / (1 - ps))))


# ---- behavioral gate ----------------------------------------------------------------------------

def cmd_gate(model, smoke=False):
    import stage4 as s4
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1]
    D = _load_dirs(model)
    rn = D["resid_norm"]["completion"]
    items24, anchors6 = s4.pick_items24(model)
    if smoke:
        items24 = items24[:6]
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")
    prompts, spans, meta = s4.build_cell_prompts(items24, anchors6)
    ctrl = load_json(s4._cell_path(model, "control"))
    d0 = _mean_d(ctrl["recs"])
    rm0 = ctrl["readout_mass"]

    coefs = [0.25] if smoke else s4.COEFS
    results = {ds: {} for ds in DIRSETS}
    nulls = {c: [] for c in coefs}
    for sd in range(3):
        for c in coefs:
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(D["dirs"][L]["random"][sd]) * sign,
                                       c, rn[L])
                recs, rm, _, _ = s4.run_cell(h, prompts, spans, meta, L, v,
                                             a_ids, b_ids)
                nulls[c].append(_mean_d(recs) - d0)
    for ds in DIRSETS:
        for c in coefs:
            row = {}
            for sign in (1, -1):
                v = harness.scaled_vec(torch.tensor(D["dirs"][L][ds]) * sign, c, rn[L])
                recs, rm, _, _ = s4.run_cell(h, prompts, spans, meta, L, v,
                                             a_ids, b_ids)
                row[f"dd_{'plus' if sign > 0 else 'minus'}"] = round(
                    _mean_d(recs) - d0, 4)
                row[f"rm_drop_{'plus' if sign > 0 else 'minus'}"] = round(
                    rm0 - rm, 3)
            sd_null = float(np.std(nulls[c], ddof=1))
            row["z_plus"] = round(row["dd_plus"] / (sd_null + 1e-9), 2)
            row["null_sd"] = round(sd_null, 4)
            results[ds][str(c)] = row
    gates = {}
    for ds in DIRSETS:
        best = None
        for c in coefs:
            row = results[ds][str(c)]
            ok = (abs(row["z_plus"]) > 3 and row["rm_drop_plus"] < 1.0
                  and row["dd_plus"] > 0 and row["dd_minus"] < row["dd_plus"])
            if ok:
                best = c
        gates[ds] = {"pass": best is not None, "primary_coef": best,
                     "table": results[ds]}
        print(f"gate {ds}: {'PASS coef=' + str(best) if best else 'FAIL'}")
    save_json(out_dir(model) / "gate.json", {"nulls": {str(c): nulls[c] for c in coefs},
                                             "gates": gates})


# ---- 4B/4C steered rollouts ---------------------------------------------------------------------

def _steer_fn_factory(h, L):
    kw = dict(h.spec.thinking_kwargs)
    kw.update(ro.EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))

    def steer_fn(live):
        def mask_fn(sl, enc):
            rows = live[sl]
            mask = st.assistant_prefill_mask(
                h, [r.messages for r in rows],
                {k: v.cpu() for k, v in enc.items()}, kw).to("cuda")
            vecs = torch.stack([r.meta["_vec"] for r in rows]).to("cuda")
            return mask, vecs.unsqueeze(1)
        return (L, mask_fn)
    return steer_fn


def cmd_4bc(model, smoke=False):
    import stage4 as s4
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1]
    D = _load_dirs(model)
    rn_chat = D["resid_norm"]["chat"]
    gates = load_json(out_dir(model) / "gate.json")["gates"]
    passing = [ds for ds in DIRSETS if gates[ds]["pass"]]
    if not passing:
        print("no direction-set passed the gate; 4bc skipped (spec rule)")
        save_json(out_dir(model) / "4bc_skipped.json", {"reason": "no gate pass"})
        return
    envs, pref, dis, match = stage3.pools(model)
    n = 4 if smoke else 20
    cells = []
    for pcond, pool in (("pref", pref), ("dispref", dis)):
        for outcome in ("good", "bad"):
            for ds in passing:
                c = gates[ds]["primary_coef"]
                sign = 1 if pcond == "dispref" else -1
                v = harness.scaled_vec(torch.tensor(D["dirs"][L][ds]) * sign, c,
                                       rn_chat[L])
                for i in range(n):
                    cells.append((pcond, outcome, i, pool[i % len(pool)],
                                  ds, sign, c, v))
            vr = harness.scaled_vec(torch.tensor(D["dirs"][L]["random"][0]),
                                    gates[passing[0]]["primary_coef"], rn_chat[L])
            for i in range(n):
                cells.append((pcond, outcome, i, pool[i % len(pool)],
                              "random", 1, gates[passing[0]]["primary_coef"], vr))
    arm = stage3.Stage3Arm(envs, "bare", [(p, o, i, e) for p, o, i, e, *_ in cells])
    rolls = arm.make_rollouts()
    for r, (_, _, _, _, ds, sign, c, v) in zip(rolls, cells):
        r.rid = f"s4/{ds}/{r.rid}"
        r.meta.update({"dirset": ds, "sign": sign, "coef": c})
        r.meta["_vec"] = v.cpu()  # [D] direction row (v[0] would be a scalar!)
    gen_batch = 12 if model == "llama31-8b" else 24
    ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=10,
                    gen_batch=gen_batch, max_new=170,
                    steer_fn=_steer_fn_factory(h, L))
    with open(out_dir(model) / "rollouts_4bc.jsonl", "w") as f:
        for r in rolls:
            meta = {k: v for k, v in r.meta.items() if not k.startswith("_")}
            f.write(json.dumps({"rid": r.rid, "meta": meta, "flags": r.flags,
                                "messages": r.messages}) + "\n")
    print(f"4bc rollouts done: {len(rolls)}")

    # steered probe pass + NLL pass + coherence samples
    ps = sp.ProbeSet(model)
    coh = []
    with open(out_dir(model) / "probes_4bc.jsonl", "w") as f:
        for r, (_, _, _, _, ds, sign, c, v) in zip(rolls, cells):
            steer = (L, v.to("cuda"))
            per_turn, post_fb, fb_read, nt = sp.rollout_series(h, ps, r.messages,
                                                               steer=steer)
            nll = _nll_assistant(h, r.messages)
            f.write(json.dumps({"rid": r.rid, **{k: v2 for k, v2 in r.meta.items()
                                                 if not k.startswith("_")},
                                "n_turns": nt, "per_turn": per_turn,
                                "post_fb": post_fb, "fb_read": fb_read,
                                "nll": nll}) + "\n")
            a_msgs = [m["content"] for m in r.messages if m["role"] == "assistant"]
            if len(a_msgs) > 1 and (hash(r.rid) % 5 == 0):
                coh.append({"kind": "coherence", "cell": f"4bc_{ds}_{r.meta['outcome']}",
                            "text": a_msgs[1][:600],
                            "prompt": 'Rate this text: {"coherent": 0|1|2}. '
                            "0=broken/degenerate, 1=flawed but readable, 2=normal."
                            f"\n---\n{a_msgs[1][:600]}\n---"})
    if coh:
        with open(out_dir(model) / "coherence_queue.jsonl", "a") as f:
            for t in coh:
                f.write(json.dumps(t) + "\n")
    print("4bc probe/NLL pass done")


@torch.no_grad()
def _nll_assistant(h, messages):
    """Unsteered mean per-token logprob over assistant tokens (disruption)."""
    kw = dict(h.spec.thinking_kwargs)
    kw.update(ro.EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))
    bounds = [len(h.tok(h.tok.apply_chat_template(
        messages[:k + 1], tokenize=False, add_generation_prompt=False,
        **kw)).input_ids) for k in range(len(messages))]
    full = h.tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=False, **kw)
    enc = h.tok(full, return_tensors="pt", truncation=True, max_length=6500).to("cuda")
    lp = h.model(**enc).logits[0].float().log_softmax(-1)
    T = enc["input_ids"].shape[1]
    mask = torch.zeros(T, dtype=torch.bool)
    prev = 0
    for k, b in enumerate(bounds):
        b = min(b, T)
        if messages[k]["role"] == "assistant":
            mask[prev:b] = True
        prev = b
    idx = torch.where(mask[1:])[0]
    if len(idx) == 0:
        return None
    tok_lp = lp[idx, enc["input_ids"][0][idx + 1]]
    return round(float(tok_lp.mean()), 4)


# ---- 4D -----------------------------------------------------------------------------------------

def cmd_4d(model, smoke=False):
    import stage4 as s4
    h = harness.load(model)
    w = s4.work_layers(h)
    L = w[1]
    D = _load_dirs(model)
    gates = load_json(out_dir(model) / "gate.json")["gates"]
    passing = [ds for ds in DIRSETS if gates[ds]["pass"]]
    if not passing:
        save_json(out_dir(model) / "4d.json", {"skipped": "no gate pass"})
        return
    ds = passing[0]
    c = gates[ds]["primary_coef"]
    items24, anchors6 = s4.pick_items24(model)
    if smoke:
        items24 = items24[:6]
    a_ids, b_ids = variant_ids(h.tok, "A"), variant_ids(h.tok, "B")

    # agentic-frame elicitation with span steering inside the chat template
    user_tpl, prefill = AGENTIC[0]
    kwt = dict(h.spec.thinking_kwargs)
    kwt.update(ro.EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))
    prompts, spans = [], []
    for it in items24:
        for a in anchors6:
            for order in (0, 1):
                x, y = (it["text"], a["text"]) if order == 0 else (a["text"], it["text"])
                user, sa, sb = st.ab_prompt_spans(user_tpl, x, y, "")
                full = h.tok.apply_chat_template(
                    [{"role": "user", "content": user},
                     {"role": "assistant", "content": prefill}],
                    tokenize=False, continue_final_message=True, **kwt)
                off = full.index(user)
                span = (sa if order == 0 else sb)
                prompts.append(full)
                spans.append((off + span[0], off + span[1]))
    rows = {}
    for label, vec in [("control", None)] + [
            (f"{s_}", harness.scaled_vec(torch.tensor(D["dirs"][L][ds]) * sgn, c,
                                         D["resid_norm"]["chat"][L]))
            for s_, sgn in (("plus", 1), ("minus", -1))]:
        logits = st.last_logits_span_steer(h, prompts, spans, L, vec)
        from lib.tasks import ab_scores
        sa_, sb_, _, _ = ab_scores(logits, a_ids, b_ids)
        ds_ = []
        k = 0
        for it in items24:
            for a in anchors6:
                for order in (0, 1):
                    d = (sa_[k] - sb_[k]).item() if order == 0 else (sb_[k] - sa_[k]).item()
                    ds_.append(d)
                    k += 1
        rows[label] = float(np.mean(ds_))
    bare = load_json(out_dir(model) / "gate.json")["gates"][ds]["table"][str(c)]
    result = {"dirset": ds, "coef": c,
              "agentic_dd_plus": round(rows["plus"] - rows["control"], 4),
              "agentic_dd_minus": round(rows["minus"] - rows["control"], 4),
              "bare_dd_plus": bare["dd_plus"],
              "transfer_ratio": round((rows["plus"] - rows["control"])
                                      / (bare["dd_plus"] + 1e-9), 3)}
    save_json(out_dir(model) / "4d.json", result)
    print(json.dumps(result, indent=1))


# ---- judge / analyze / cross --------------------------------------------------------------------

def cmd_judge():
    h32 = harness.load("qwen25-32b")
    for m in SUBJECTS:
        q = out_dir(m) / "coherence_queue.jsonl"
        if not q.exists():
            continue
        tasks = [json.loads(l) for l in q.read_text().splitlines()]
        done = judge_mod.run_judge(h32, tasks, max_new=30)
        with open(out_dir(m) / "coherence.jsonl", "w") as f:
            for t in done:
                f.write(json.dumps({"cell": t["cell"],
                                    "coherent": (t.get("result") or {}).get("coherent")})
                        + "\n")
        print(f"{m}: judged {len(done)} coherence samples")


def cmd_analyze(model, smoke=False):
    import stage4 as s4
    lines = [f"{model}: Stage 4 analysis"]
    # coherence rates per cell
    coh = {}
    cp = out_dir(model) / "coherence.jsonl"
    if cp.exists():
        for r in map(json.loads, cp.read_text().splitlines()):
            coh.setdefault(r["cell"], []).append(r["coherent"])
    bad_cells = {c for c, v in coh.items()
                 if sum(1 for x in v if x == 0) > 0.25 * len(v)}
    if bad_cells:
        lines.append(f"coherence-excluded cells: {sorted(bad_cells)}")

    # ---- 4A
    cells = {}
    for p in sorted((out_dir(model) / "4a").glob("*.json")):
        cells[p.stem] = load_json(p)
    D = _load_dirs(model)
    w = D["work_layers"]
    L = w[1]
    acts = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt").float()[1].numpy()
    mu_xl = np.array([r["mu"] for r in load_json(
        P1 / "results" / "stage1x" / model / "utilities_xl.json")])
    den = torch.load(P1 / "results" / "stage2" / model / "vectors.pt")["den"].float()[L]
    emos = s4.pick_emotions20()
    rows_4a = []
    for e in emos:
        key = f"{e}_0.5_{L}"
        if key not in cells or key in bad_cells:
            continue
        v_e = torch.nn.functional.normalize(den[EMOTIONS.index(e)], dim=-1).numpy()
        rho_e = pearson(list(acts @ v_e), list(mu_xl))
        rows_4a.append({"emotion": e, "dmu": cells[key]["dmu_mean"],
                        "valence": NORMS[e]["valence"], "rho_probe_mu": rho_e,
                        "rm_drop": cells[key].get("readout_drop"),
                        "kl": cells[key].get("kl")})
    if rows_4a:
        r_val = pearson([r["dmu"] for r in rows_4a], [r["valence"] for r in rows_4a])
        r_rho = pearson([r["dmu"] for r in rows_4a], [r["rho_probe_mu"] for r in rows_4a])
        rand_dmu = [cells[f"random{sd}_0.5_{L}"]["dmu_mean"] for sd in range(3)
                    if f"random{sd}_0.5_{L}" in cells]
        lines.append(f"4A (c=0.5, L{L}, n={len(rows_4a)} emotions): "
                     f"corr(dmu, valence norm) = {r_val:+.3f}; "
                     f"corr(dmu, probe-mu rho) = {r_rho:+.3f} [paper analog r=0.85]")
        lines.append(f"  blissful dmu {next((r['dmu'] for r in rows_4a if r['emotion']=='blissful'), None)}, "
                     f"hostile {next((r['dmu'] for r in rows_4a if r['emotion']=='hostile'), None)}, "
                     f"random null {np.mean(rand_dmu):+.3f}±{np.std(rand_dmu):.3f}")
        for e in ["blissful", "hostile"]:
            dose = [cells.get(f"{e}_{c}_{L}", {}).get("dmu_mean") for c in s4.COEFS]
            lines.append(f"  dose {e}: " + " ".join(str(x) for x in dose))
        for key in (f"anchorspan_blissful_0.5_{L}", f"anchorspan_hostile_0.5_{L}",
                    f"geo_blissful_0.5_{L}", f"geo_hostile_0.5_{L}"):
            if key in cells:
                lines.append(f"  {key}: dmu {cells[key]['dmu_mean']:+.3f}")
        save_json(out_dir(model) / "4a_dmu.json", rows_4a)

    # supplementary benchmark at the EFFECTIVE layer w[0] (0.5 depth), if run
    L0 = w[0]
    den0 = torch.load(P1 / "results" / "stage2" / model / "vectors.pt")["den"].float()[L0]
    acts0 = torch.load(P1 / "results" / "stage1x" / model / "acts_xl.pt").float()[0].numpy()
    rows0 = []
    for e in s4.pick_emotions20():
        key = f"{e}_0.5_{L0}"
        if key in cells:
            v_e = torch.nn.functional.normalize(den0[EMOTIONS.index(e)], dim=-1).numpy()
            rows0.append({"emotion": e, "dmu": cells[key]["dmu_mean"],
                          "valence": NORMS[e]["valence"],
                          "rho_probe_mu": pearson(list(acts0 @ v_e), list(mu_xl))})
    if len(rows0) >= 10:
        rv0 = pearson([r["dmu"] for r in rows0], [r["valence"] for r in rows0])
        rr0 = pearson([r["dmu"] for r in rows0], [r["rho_probe_mu"] for r in rows0])
        rand0 = [cells[f"random{sd}_0.5_{L0}"]["dmu_mean"] for sd in range(3)
                 if f"random{sd}_0.5_{L0}" in cells]
        lines.append(f"4A at EFFECTIVE layer L{L0} (n={len(rows0)}): "
                     f"corr(dmu, valence) = {rv0:+.3f}; corr(dmu, probe-mu rho) = "
                     f"{rr0:+.3f}; random null {np.mean(rand0):+.3f}±{np.std(rand0):.3f}"
                     if rand0 else
                     f"4A at EFFECTIVE layer L{L0}: rv={rv0:+.3f} rr={rr0:+.3f}")
        save_json(out_dir(model) / "4a_dmu_w0.json", rows0)

    # ---- gate
    gp = out_dir(model) / "gate.json"
    if gp.exists():
        gates = load_json(gp)["gates"]
        for ds_name, g in gates.items():
            lines.append(f"gate {ds_name}: {'PASS coef=' + str(g['primary_coef']) if g['pass'] else 'FAIL'} "
                         + json.dumps({c: {'dd+': v['dd_plus'], 'z': v['z_plus']}
                                       for c, v in g['table'].items()}))

    # ---- 4B/4C
    pp = out_dir(model) / "probes_4bc.jsonl"
    if pp.exists():
        rows = [json.loads(l) for l in pp.read_text().splitlines()]
        ctrl = [json.loads(l) for l in
                (P1 / "results" / "stage3" / model / "probes.jsonl").read_text().splitlines()
                if json.loads(l).get("frame") == "bare"
                and json.loads(l).get("outcome") in ("good", "bad")]

        def late(r):
            v = r["per_turn"]["valence"]
            return float(np.mean(v[1:])) if len(v) > 1 else v[0]

        def swing(rs):
            g = [np.mean(r["fb_read"]["valence"]) for r in rs if r["outcome"] == "good"
                 and r.get("fb_read", {}).get("valence")]
            b = [np.mean(r["fb_read"]["valence"]) for r in rs if r["outcome"] == "bad"
                 and r.get("fb_read", {}).get("valence")]
            return (float(np.mean(g) - np.mean(b)) if g and b else None,
                    len(g) + len(b))
        sw_ctrl, n_ctrl = swing(ctrl)
        state_ctrl = float(np.mean([late(r) for r in ctrl]))
        lines.append(f"4B/4C (controls: stage3 bare n={len(ctrl)}, "
                     f"fb_read swing {sw_ctrl:+.3f}, state {state_ctrl:+.3f})")
        for ds_name in sorted({r["dirset"] for r in rows}):
            rs = [r for r in rows if r["dirset"] == ds_name]
            sw, nsw = swing(rs)
            state = float(np.mean([late(r) for r in rs]))
            # per-sign state deltas vs control cells matched by pref condition
            d_state = state - state_ctrl
            lines.append(f"  {ds_name:8s} n={len(rs)}: state {state:+.3f} "
                         f"(Δ vs ctrl {d_state:+.3f}); fb_read swing {sw!s:>7} "
                         f"(Δ vs ctrl {None if sw is None else round(sw - sw_ctrl, 3)}); "
                         f"mean NLL {np.mean([r['nll'] for r in rs if r['nll']]):.3f}")
    sk = out_dir(model) / "4bc_skipped.json"
    if sk.exists():
        lines.append("4B/4C skipped: " + load_json(sk)["reason"])

    # ---- 4D
    dp = out_dir(model) / "4d.json"
    if dp.exists():
        lines.append("4D: " + json.dumps(load_json(dp)))

    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_cross():
    lines = ["Stage 4 cross-model summary", ""]
    for m in SUBJECTS:
        p = out_dir(m) / "summary.txt"
        if p.exists():
            lines += [f"---- {m}"] + ["  " + l for l in p.read_text().splitlines()[1:]] + [""]
    (P1 / "results" / "stage4" / "cross_model.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
