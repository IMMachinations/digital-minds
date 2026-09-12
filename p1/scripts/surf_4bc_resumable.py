"""Resumable, one-turn-per-process 4B/4C for the laptop (MPS).

stage4_rollouts.cmd_4bc keeps all rollouts in one process for 10 turns; on MPS
the allocator's cache grows with context length until the machine pages (see
memory note mps-generate-memory). This driver runs ONE turn per process:
  turn   <model> --tag T ...   restore every rollout from the checkpoint,
                               generate the next turn (small batches), write
                               the checkpoint, exit (fresh memory next turn).
                               When the last turn completes it also writes
                               rollouts_4bc<T>.jsonl in cmd_4bc's format.
  probes <model> --tag T ...   steered probe re-encode + NLL + coherence
                               sampling, one line per rollout, appended as it
                               goes (resume skips rids already written).
Cells, seeds, steering and outputs are identical to cmd_4bc, so
`stage4.py 4d/judge/analyze --tag T` run unchanged afterwards.
Every step logs to results/stage4/<model>/4bc<T>.log (timestamps, batch
progress, MPS memory).

Usage:
  uv run python scripts/surf_4bc_resumable.py turn   qwen25-7b --tag _versions \
      --dirsets utility_v3 utility_v10 utility_v15L18 --extra <extra_dirs.pt> --n-cell 5 --gen-batch 6
  uv run python scripts/surf_4bc_resumable.py probes qwen25-7b --tag _versions --dirsets ... --extra ... --n-cell 5
"""
import argparse
import json
import logging
import resource
import sys
import time
import zlib
from pathlib import Path

import torch

P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json

import harness
import rollout as ro
import stage3
import stage3_probes as sp
import stage4 as s4
import stage4_rollouts as s4r

MAX_TURNS, MAX_NEW, CTX_LIMIT = 10, 170, 6000


def _log(model, tag):
    log = logging.getLogger("4bc")
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(s4r._tagged(model, "4bc.log", tag))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def _mem():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
    if harness.DEVICE == "mps":
        return (f"maxrss {rss:.1f} GB, mps alloc {torch.mps.current_allocated_memory() / 2**30:.1f} GB, "
                f"mps driver {torch.mps.driver_allocated_memory() / 2**30:.1f} GB")
    return f"maxrss {rss:.1f} GB"


# ---- cells (identical to cmd_4bc) ------------------------------------------------------------

def build(model, dirsets, tag, extra, n_cell, h):
    """-> (L, cells, arm, rolls) with r.meta['_vec'] set, exactly as cmd_4bc."""
    w = s4.work_layers(h)
    L = w[1]
    D = s4r._load_dirs(model, extra)
    rn_chat = D["resid_norm"]["chat"]
    gates = load_json(s4r._tagged(model, "gate.json", tag))["gates"]
    passing = [ds for ds in dirsets if gates[ds]["pass"]]
    assert passing, "no direction set passed the gate"
    envs, pref, dis, _ = stage3.pools(model)
    cells = []
    for pcond, pool in (("pref", pref), ("dispref", dis)):
        for outcome in ("good", "bad"):
            for ds in passing:
                c = gates[ds]["primary_coef"]
                sign = 1 if pcond == "dispref" else -1
                v = harness.scaled_vec(torch.tensor(D["dirs"][L][ds]) * sign, c, rn_chat[L])
                for i in range(n_cell):
                    cells.append((pcond, outcome, i, pool[i % len(pool)], ds, sign, c, v))
            vr = harness.scaled_vec(torch.tensor(D["dirs"][L]["random"][0]),
                                    gates[passing[0]]["primary_coef"], rn_chat[L])
            for i in range(n_cell):
                cells.append((pcond, outcome, i, pool[i % len(pool)], "random", 1,
                              gates[passing[0]]["primary_coef"], vr))
    arm = stage3.Stage3Arm(envs, "bare", [(p, o, i, e) for p, o, i, e, *_ in cells])
    rolls = arm.make_rollouts()
    for r, (_, _, _, _, ds, sign, c, v) in zip(rolls, cells):
        r.rid = f"s4/{ds}/{r.rid}"
        r.meta.update({"dirset": ds, "sign": sign, "coef": c})
        r.meta["_vec"] = v.cpu()
    return L, cells, arm, rolls


# ---- checkpoint ------------------------------------------------------------------------------

def ckpt_paths(model, tag):
    return (s4r._tagged(model, "4bc_ckpt.jsonl", tag), s4r._tagged(model, "4bc_state.json", tag))


def restore(rolls, model, tag):
    cp, sp_ = ckpt_paths(model, tag)
    if not sp_.exists():
        return -1
    state = load_json(sp_)
    by_rid = {json.loads(l)["rid"]: json.loads(l) for l in cp.read_text().splitlines()}
    for r in rolls:
        s = by_rid[r.rid]
        r.messages, r.flags, r.events, r.done = s["messages"], s["flags"], s["events"], s["done"]
    return state["turns_done"] - 1


def checkpoint(rolls, model, tag, turns_done):
    cp, sp_ = ckpt_paths(model, tag)
    tmp = cp.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rolls:
            f.write(json.dumps({"rid": r.rid, "messages": r.messages, "flags": r.flags,
                                "events": r.events, "done": r.done}) + "\n")
    tmp.replace(cp)
    save_json(sp_, {"turns_done": turns_done, "n": len(rolls)})


# ---- one turn of rollout.run_lockstep --------------------------------------------------------

def run_one_turn(h, rollouts, arm, t, gen_batch, steer_fn, log):
    active = sorted((r for r in rollouts if not r.done), key=lambda r: r.rid)
    if not active:
        return 0
    drivers = arm.driver(active, t)
    live = []
    for r, d in zip(active, drivers):
        if d is None:
            r.event(t, "completed")
            r.done = True
        else:
            r.messages.append({"role": "user", "content": d})
            live.append(r)
    for r in live:
        if len(h.tok(ro.build_prompt(h, r.messages)).input_ids) > CTX_LIMIT:
            r.flags.append("ctx_overflow")
            r.event(t, "flag", flag="ctx_overflow")
            r.done = True
    live = [r for r in live if not r.done]
    if not live:
        return 0
    seed = zlib.crc32(f"{h.spec.short_name}/{live[0].arm}/t{t}".encode()) & 0x7FFFFFFF
    outs = []
    n_b = (len(live) + gen_batch - 1) // gen_batch
    t0 = time.time()
    for bi in range(n_b):
        chunk = live[bi * gen_batch:(bi + 1) * gen_batch]
        # per-sub-batch seed matches gen_turns' `seed + bi` only when the batch
        # size matches cmd_4bc's; the seed schedule is logged for the record.
        outs += ro.gen_turns(h, [ro.build_prompt(h, r.messages) for r in chunk],
                             seed=seed + bi, max_new=MAX_NEW, batch=gen_batch,
                             steer=steer_fn(chunk))
        log.info(f"turn {t}: batch {bi + 1}/{n_b} ({len(chunk)} rollouts) "
                 f"{(time.time() - t0) / 60:.1f} min elapsed; {_mem()}")
    for r, raw in zip(live, outs):
        text = ro.trim_sentence(ro.strip_think(raw))
        prev = next((m["content"] for m in reversed(r.messages) if m["role"] == "assistant"), None)
        r.messages.append({"role": "assistant", "content": text})
        arm.parse(r, t, text)
        evented = any(e["turn"] == t for e in r.events)
        if not r.done and not evented and ro.degenerate(text, prev):
            r.flags.append("degenerate")
            r.event(t, "flag", flag="degenerate")
            r.done = True
    return len(live)


def cmd_turn(a):
    log = _log(a.model, a.tag)
    h = harness.load(a.model)
    L, cells, arm, rolls = build(a.model, a.dirsets, a.tag, a.extra, a.n_cell, h)
    last = restore(rolls, a.model, a.tag)
    t = last + 1 if a.turn is None else a.turn
    if t >= MAX_TURNS:
        log.info("all turns already done")
        return
    log.info(f"turn {t}/{MAX_TURNS - 1}: {len(rolls)} rollouts, {sum(not r.done for r in rolls)} open, "
             f"gen_batch {a.gen_batch}, layer {L}; {_mem()}")
    n = run_one_turn(h, rolls, arm, t, a.gen_batch, s4r._steer_fn_factory(h, L), log)
    if t == MAX_TURNS - 1:
        for r in rolls:
            if not r.done:
                r.event(MAX_TURNS, "completed")
                r.done = True
    checkpoint(rolls, a.model, a.tag, t + 1)
    log.info(f"turn {t} done: {n} generated, {sum(not r.done for r in rolls)} still open; checkpoint written")
    if t == MAX_TURNS - 1:
        with open(s4r._tagged(a.model, "rollouts_4bc.jsonl", a.tag), "w") as f:
            for r in rolls:
                meta = {k: v for k, v in r.meta.items() if not k.startswith("_")}
                f.write(json.dumps({"rid": r.rid, "meta": meta, "flags": r.flags,
                                    "messages": r.messages}) + "\n")
        log.info(f"rollouts_4bc{a.tag}.jsonl written ({len(rolls)} rollouts)")


# ---- probe pass (resumable) ------------------------------------------------------------------

def cmd_probes(a):
    log = _log(a.model, a.tag)
    h = harness.load(a.model)
    L, cells, arm, rolls = build(a.model, a.dirsets, a.tag, a.extra, a.n_cell, h)
    rp = s4r._tagged(a.model, "rollouts_4bc.jsonl", a.tag)
    assert rp.exists(), "run the turns first"
    saved = {json.loads(l)["rid"]: json.loads(l) for l in rp.read_text().splitlines()}
    pp = s4r._tagged(a.model, "probes_4bc.jsonl", a.tag)
    done = {json.loads(l)["rid"] for l in pp.read_text().splitlines()} if pp.exists() else set()
    ps = sp.ProbeSet(a.model)
    log.info(f"probe pass: {len(rolls)} rollouts, {len(done)} already done; {_mem()}")
    t0 = time.time()
    with open(pp, "a") as f, open(s4r._tagged(a.model, "coherence_queue.jsonl", a.tag), "a") as cq:
        for k, (r, (_, _, _, _, ds, sign, c, v)) in enumerate(zip(rolls, cells)):
            if r.rid in done:
                continue
            msgs = saved[r.rid]["messages"]
            per_turn, post_fb, fb_read, nt = sp.rollout_series(h, ps, msgs,
                                                               steer=(L, v.to(harness.DEVICE)))
            nll = s4r._nll_assistant(h, msgs)
            meta = {kk: vv for kk, vv in r.meta.items() if not kk.startswith("_")}
            f.write(json.dumps({"rid": r.rid, **meta, "n_turns": nt, "per_turn": per_turn,
                                "post_fb": post_fb, "fb_read": fb_read, "nll": nll}) + "\n")
            f.flush()
            a_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
            if len(a_msgs) > 1 and (zlib.crc32(r.rid.encode()) % 5 == 0):
                cq.write(json.dumps({"kind": "coherence", "cell": f"4bc_{ds}_{r.meta['outcome']}",
                                     "text": a_msgs[1][:600],
                                     "prompt": 'Rate this text: {"coherent": 0|1|2}. '
                                     "0=broken/degenerate, 1=flawed but readable, 2=normal."
                                     f"\n---\n{a_msgs[1][:600]}\n---"}) + "\n")
                cq.flush()
            if harness.DEVICE == "mps":
                torch.mps.empty_cache()
            log.info(f"probes {k + 1}/{len(rolls)} {r.rid} nll={nll} "
                     f"{(time.time() - t0) / 60:.1f} min; {_mem()}")
    log.info("probe pass complete")


def cmd_finalize(a):
    """Close the run at the checkpointed turn count (shorter sessions than 10
    turns) and write rollouts_4bc<tag>.jsonl so `probes` can start."""
    log = _log(a.model, a.tag)
    h = harness.load(a.model)
    L, cells, arm, rolls = build(a.model, a.dirsets, a.tag, a.extra, a.n_cell, h)
    last = restore(rolls, a.model, a.tag)
    assert last >= 0, "no checkpoint"
    for r in rolls:
        if not r.done:
            r.event(last + 1, "completed")
            r.done = True
    with open(s4r._tagged(a.model, "rollouts_4bc.jsonl", a.tag), "w") as f:
        for r in rolls:
            meta = {k: v for k, v in r.meta.items() if not k.startswith("_")}
            meta["turns_run"] = last + 1
            f.write(json.dumps({"rid": r.rid, "meta": meta, "flags": r.flags,
                                "messages": r.messages}) + "\n")
    log.info(f"finalized after {last + 1} turns: rollouts_4bc{a.tag}.jsonl written ({len(rolls)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["turn", "probes", "finalize"])
    ap.add_argument("model")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dirsets", nargs="+", required=True)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--n-cell", type=int, default=5)
    ap.add_argument("--gen-batch", type=int, default=6)
    ap.add_argument("--turn", type=int, default=None)
    a = ap.parse_args()
    {"turn": cmd_turn, "probes": cmd_probes, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
