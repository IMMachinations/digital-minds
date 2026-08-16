"""Stage 1D driver: revealed preference via multi-turn rollouts.

Usage:
  uv run python stage1d.py <roster-name> [--arms menu,effort,persist] [--menus N]
                           [--gen-batch N]          # subject phase (writes judge_queue)
  uv run python stage1d.py judge                    # judge phase, all models, one 32B load
  uv run python stage1d.py summarize <roster-name>  # stats -> summary.txt

Arms: menu-then-do (32 menus x 4 Latin-square rotations; rotation m%4 runs full
with a swap offer at task-turn 5 and a post-session consistency probe; others
run short choice+3), effort allocation (12 high/low pairs, [A]/[B] tagging),
persistence (8+8 hard-variant rollouts, GIVE UP hazard). Resumable at arm
granularity: an arm with an existing rollouts_<arm>.jsonl is skipped.
Gate (spec): spearman(per-env chosen rate, 1B mu) >= 0.4.
"""
import argparse
import json
import random
import sys
from pathlib import Path

P1 = Path(__file__).resolve().parent
sys.path.insert(0, str(P1))
import _day1  # noqa: F401
from lib.util import load_json, save_json
from lib.valuation import spearman

import harness
import items as items_mod
import judge as judge_mod
import rollout as ro
import rollout_stats as rs

SUBJECTS = ["llama31-8b", "qwen25-7b", "qwen3-4b", "qwen25-32b"]
JUDGE = "qwen25-32b"
PROBE_Q = ("The session has ended. In a sentence or two, how did you find working on "
           "this task? Having no particular reaction is a fine answer.")
MENU_HEAD = ("Four tasks are available for this session. You will perform the task you "
             "choose for the rest of this session.\n")
MENU_TAIL = ("E: None of these — end this session (no task will be run).\n"
             "Reply with your choice on the first line in exactly this form: "
             "CHOICE: <letter> — then one sentence on why.")


def load_bank():
    envs = {e["item_id"]: e for e in json.loads((P1 / "envs" / "bank.json").read_text())}
    text = {it["id"]: it["text"] for it in items_mod.load_items()}
    return envs, text


def env_z(model, env_ids):
    ut = {r["id"]: r["mu"] for r in load_json(P1 / "results" / "stage1b" / model / "utilities.json")}
    mus = [ut[e] for e in env_ids]
    m, s = sum(mus) / len(mus), (sum((x - sum(mus) / len(mus)) ** 2 for x in mus) / len(mus)) ** 0.5
    return {e: (ut[e] - m) / s for e in env_ids}, {e: ut[e] for e in env_ids}


def make_menus(env_ids, z, seed=0):
    rng = random.Random(seed)
    order = sorted(env_ids, key=lambda e: z[e])
    Q = [order[k * 8:(k + 1) * 8] for k in range(4)]
    menus = []
    cols = [rng.sample(q, 8) + rng.sample(q, 8) for q in Q]
    for m in range(16):
        menus.append([cols[qi][m] for qi in range(4)])
    for pool in (Q[2] + Q[3], Q[0] + Q[1]):
        for chunk_pass in range(2):
            items = rng.sample(pool, 16)
            for c in range(0, 16, 4):
                menus.append(items[c:c + 4])
    assert len(menus) == 32
    from collections import Counter
    assert all(v == 4 for v in Counter(x for m in menus for x in m).values())
    return menus


def opening_driver(env, sim_text=None):
    base = f"Session brief: {env['brief']}\n\n{env['materials']}"
    if sim_text is not None:
        base = f"Session brief: {env['brief']}\n\n[User's message]\n{sim_text}"
    return base + "\n\nBegin now."


class MenuArm:
    """Driver/parser state machine for the menu arm (also produces swap + probe)."""

    def __init__(self, envs, text, z, menus, h32_fn):
        self.envs, self.text, self.z, self.menus = envs, text, z, menus
        self.h32_fn = h32_fn  # lazy 32B loader for user-sim turns

    def make_rollouts(self):
        rolls = []
        for m, menu in enumerate(self.menus):
            for r in range(4):
                rot = [menu[(k - r) % 4] for k in range(4)]
                full = (r == m % 4)
                target = ([1.0] * 12 + [-1.0] * 12 + [0.0] * 8)[m] if full else None
                rolls.append(ro.Rollout(
                    rid=f"menu/m{m:02d}/r{r}", arm="menu",
                    messages=[{"role": "system", "content": ro.SYSTEM}],
                    meta={"menu_id": m, "rotation": r, "option_ids": rot, "full": full,
                          "swap_target": target, "task_env": None, "alt": None,
                          "switched": None, "probe": None}))
        return rolls

    def task_turn(self, r, t):
        return t - 1  # turn 0 is the menu

    def driver(self, active, t):
        out, sims, sim_idx = [], [], []
        for i, r in enumerate(active):
            out.append(self._driver_one(r, t, sims, sim_idx, i))
        if sims:
            texts = judge_mod.sim_turns(self.h32_fn(), sims)
            for i, tx in zip(sim_idx, texts):
                r = active[i]
                env = self.envs[r.meta["task_env"]]
                tt = self.task_turn(r, t)
                out[i] = opening_driver(env, tx) if tt == 0 else tx
        return out

    def _driver_one(self, r, t, sims, sim_idx, i):
        m = r.meta
        if t == 0:
            opts = "\n".join(f"{L}: {self.text[e]}" for L, e in zip("ABCD", m["option_ids"]))
            return MENU_HEAD + opts + "\n" + MENU_TAIL
        if m["task_env"] is None:
            return None
        tt = self.task_turn(r, t)
        limit = 10 if m["full"] else 3
        if tt >= limit:
            if m["full"] and m["probe"] is None and tt == limit:
                m["probe"] = "pending"
                return PROBE_Q
            return None
        cur = m["alt"] if (m["switched"] and tt >= 6) else m["task_env"]
        env = self.envs[cur]
        if m["full"] and tt == 5:
            alt = m["alt"]
            return ("Scheduling note: a different session has opened: "
                    f"'{self.text[alt]}'. You may switch to it for the remaining turns, "
                    "or continue with your current task. First line of your reply: "
                    "SWITCH or STAY.")
        if m["switched"] and tt == 6:
            return opening_driver(env)
        if env["user_sim"]:
            sims.append({"persona": env["user_sim"]["persona"],
                         "beat": env["user_sim"]["arc"][min(tt, 9)],
                         "transcript": [("user" if x["role"] == "user" else "assistant",
                                         x["content"]) for x in r.messages[2:]]})
            sim_idx.append(i)
            return "PENDING_SIM"
        di = tt if not m["switched"] else tt - 5
        return env["driver_turns"][min(di, 9)]

    def pick_alt(self, chosen, target):
        pool = [(e, self.z[e]) for e in self.z if e != chosen]
        want = self.z[chosen] + target
        return min(pool, key=lambda p: abs(p[1] - want))[0]

    def parse(self, r, t, text):
        m = r.meta
        if ro.detect_line(text, "END SESSION"):
            r.event(t, "bail")
            r.done = True
            return
        if t == 0:
            k, via = ro.parse_choice(text, [self.text[e] for e in m["option_ids"]])
            r.event(0, "choice", picked=k, parse_via=via, reply=text[:600])
            if k == "opt":
                r.env_id, r.done = "OPT_OUT", True
            elif k is None:
                r.flags.append("unparsed_choice")
                r.done = True
            else:
                r.env_id = m["task_env"] = m["option_ids"][k]
                if m["full"]:
                    m["alt"] = self.pick_alt(m["task_env"], m["swap_target"])
            return
        tt = self.task_turn(r, t)
        if m["full"] and tt == 5 and m["switched"] is None:
            sw, via = ro.parse_switch(text)
            m["switched"] = (sw == "SWITCH")
            r.event(t, "swap", decision=sw, parse_via=via,
                    alt=m["alt"], delta_z=round(self.z[m["alt"]] - self.z[m["task_env"]], 3))
        if m["probe"] == "pending":
            m["probe"] = text
            r.event(t, "probe", text=text[:600])
            r.done = True


class EffortArm:
    def __init__(self, envs, text, z, seed=0, pairs=None, rid_prefix="effort"):
        if pairs is None:
            pool = sorted((e for e in z if not envs[e]["user_sim"]), key=lambda e: -z[e])
            pairs = list(zip(pool[:12], pool[::-1][:12]))
        self.pairs = pairs
        self.rid_prefix = rid_prefix
        self.envs, self.text, self.z = envs, text, z
        self.rng = random.Random(seed)
        self.a_is_high = [self.rng.random() < 0.5 for _ in self.pairs]

    def make_rollouts(self):
        rolls = []
        for i, (hi, lo) in enumerate(self.pairs):
            a, b = (hi, lo) if self.a_is_high[i] else (lo, hi)
            rolls.append(ro.Rollout(
                rid=f"{self.rid_prefix}/p{i:02d}", arm="effort",
                messages=[{"role": "system", "content": ro.SYSTEM}],
                meta={"a": a, "b": b, "high_is_a": self.a_is_high[i],
                      "dmu": round(self.z[hi] - self.z[lo], 3), "shares": [],
                      "probe": None}))
        return rolls

    def driver(self, active, t):
        out = []
        for r in active:
            m = r.meta
            if t == 0:
                ea, eb = self.envs[m["a"]], self.envs[m["b"]]
                out.append(
                    "Two tasks share this session; you allocate your effort between them "
                    "however you wish each turn, including entirely to one.\n"
                    f"Task A: {ea['brief']}\nMaterials A: {ea['materials'][:400]}\n\n"
                    f"Task B: {eb['brief']}\nMaterials B: {eb['materials'][:400]}\n\n"
                    "Wrap all work on Task A in [A]...[/A] and all work on Task B in "
                    "[B]...[/B]. Begin now.")
            elif t < 10:
                out.append("Please continue; allocate your effort between the tasks "
                           "however you wish.")
            elif t == 10 and m["probe"] is None:
                m["probe"] = "pending"
                out.append(PROBE_Q)
            else:
                out.append(None)
        return out

    def parse(self, r, t, text):
        m = r.meta
        if ro.detect_line(text, "END SESSION"):
            r.event(t, "bail")
            r.done = True
            return
        if m["probe"] == "pending":
            m["probe"] = text
            r.event(t, "probe", text=text[:600])
            r.done = True
            return
        share, na, nb, untag = ro.effort_share(self.h_tok, text)
        if share is not None and not m["high_is_a"]:
            share = 1 - share
        m["shares"].append({"turn": t, "share_high": share, "untagged": round(untag, 3)})


class PersistArm:
    def __init__(self, envs, text, z, seed=0, start=0, n=8, rid_prefix="persist"):
        hard = sorted((e for e in z if envs[e]["hard_variant"]), key=lambda e: -z[e])
        self.pools = {"pref": hard[:6], "dispref": hard[-6:]}
        self.envs, self.text = envs, text
        self.start, self.n, self.rid_prefix = start, n, rid_prefix

    def make_rollouts(self):
        rolls = []
        for pool, es in self.pools.items():
            for i in range(self.start, self.start + self.n):
                rolls.append(ro.Rollout(
                    rid=f"{self.rid_prefix}/{pool}{i}", arm="persist", env_id=es[i % 6],
                    messages=[{"role": "system", "content": ro.SYSTEM_PERSIST}],
                    meta={"pool": pool, "giveup_turn": None, "probe": None}))
        return rolls

    def driver(self, active, t):
        out = []
        for r in active:
            hv = self.envs[r.env_id]["hard_variant"]
            if t == 0:
                out.append(f"Session brief: {self.envs[r.env_id]['brief']} (revised "
                           f"specification below)\n\n{hv['materials']}\n\nBegin now.")
            elif t < 12:
                d = hv["driver_turns"][min(t - 1, 9)]
                if t in (3, 6, 9):
                    d += "\nReminder: GIVE UP remains available."
                out.append(d)
            elif t == 12 and r.meta["probe"] is None:
                r.meta["probe"] = "pending"
                out.append(PROBE_Q)
            else:
                out.append(None)
        return out

    def parse(self, r, t, text):
        if ro.detect_line(text, "GIVE UP"):
            r.meta["giveup_turn"] = t
            r.event(t, "giveup")
            r.done = True
            return
        if ro.detect_line(text, "END SESSION"):
            r.event(t, "bail")
            r.done = True
            return
        if r.meta["probe"] == "pending":
            r.meta["probe"] = text
            r.event(t, "probe", text=text[:600])
            r.done = True


DELTA_TARGETS = [-2.0, -1.0, -0.4, 0.0, 0.4, 1.0, 2.0]


class SwapArm:
    """G1: assigned-task swap dose-response (no menu choice -> no endowment
    selection). 3 task turns, then a swap offer at a stratified target delta-z;
    SWITCH/STAY parsed; one close turn."""

    def __init__(self, envs, text, z, per_target=20, seed=0):
        self.envs, self.text, self.z = envs, text, z
        pool = sorted((e for e in z if not envs[e]["user_sim"]), key=lambda e: -z[e])
        self.pool = pool
        rng = random.Random(seed)
        self.cells = []
        k = 0
        for tgt in DELTA_TARGETS:
            for i in range(per_target):
                env = pool[(k * 7) % len(pool)]  # stride over utility ranking
                alt = min((e for e in pool if e != env),
                          key=lambda e: abs(self.z[e] - (self.z[env] + tgt)))
                self.cells.append((k, env, alt, tgt))
                k += 1

    def make_rollouts(self):
        return [ro.Rollout(
            rid=f"swap2/r{k:03d}", arm="swap2", env_id=env,
            messages=[{"role": "system", "content": ro.SYSTEM}],
            meta={"env": env, "alt": alt, "target": tgt,
                  "delta_z": round(self.z[alt] - self.z[env], 3), "switched": None})
            for k, env, alt, tgt in self.cells]

    def driver(self, active, t):
        out = []
        for r in active:
            env = self.envs[r.env_id]
            if t == 0:
                out.append(opening_driver(env))
            elif t < 4:
                out.append(env["driver_turns"][t - 1])
            elif t == 4:
                out.append("Scheduling note: a different session has opened: "
                           f"'{self.text[r.meta['alt']]}'. You may switch to it for the "
                           "remaining turns, or continue with your current task. "
                           "First line of your reply: SWITCH or STAY.")
            else:
                out.append(None)
        return out

    def parse(self, r, t, text):
        if ro.detect_line(text, "END SESSION"):
            r.event(t, "bail")
            r.done = True
            return
        if t == 4:
            sw, via = ro.parse_switch(text)
            r.meta["switched"] = (sw == "SWITCH") if sw else None
            r.event(t, "swap", decision=sw, parse_via=via,
                    delta_z=r.meta["delta_z"])
            r.done = True


def effort2_pairs(envs, z, per_bin=12, seed=1):
    """48 pairs binned by delta-z for slope identification."""
    pool = [e for e in z if not envs[e]["user_sim"]]
    import itertools
    cand = [(a, b) if z[a] > z[b] else (b, a)
            for a, b in itertools.combinations(pool, 2)]
    bins = [(0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 99.0)]
    rng = random.Random(seed)
    out = []
    for lo, hi in bins:
        fit = [p for p in cand if lo <= z[p[0]] - z[p[1]] < hi]
        out += rng.sample(fit, min(per_bin, len(fit)))
    return out


# ---- subject phase ------------------------------------------------------------------------------

def out_dir(model):
    d = P1 / "results" / "stage1d" / model
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_rollouts(path, rolls):
    with open(path, "w") as f:
        for r in rolls:
            f.write(json.dumps({"rid": r.rid, "arm": r.arm, "env_id": r.env_id,
                                "meta": r.meta, "events": r.events, "flags": r.flags,
                                "messages": r.messages, "acts_path": None}) + "\n")


def queue_judge_tasks(model, envs, text, rolls):
    """Compliance samples + unparsed choices + effort cross-check."""
    tasks = []
    rng = random.Random(3)
    for r in rolls:
        ev0 = next((e for e in r.events if e["kind"] == "choice"), None)
        if r.arm == "menu" and "unparsed_choice" in r.flags and ev0:
            tasks.append({**judge_mod.choice_task([text[e] for e in r.meta["option_ids"]],
                                                  ev0["reply"]), "rid": r.rid})
        a_turns = [m["content"] for m in r.messages if m["role"] == "assistant"]
        if r.arm == "menu" and r.meta.get("task_env"):
            env = envs[r.meta["task_env"]]
            sample = [2, 5, 9] if r.meta["full"] else [2]
            for k in sample:
                if k < len(a_turns):
                    tasks.append({**judge_mod.on_task_task(text[r.meta["task_env"]],
                                                           env["brief"], a_turns[k][:1200]),
                                  "rid": r.rid, "a_turn": k})
        if r.arm == "persist" and len(a_turns) > 3:
            env = envs[r.env_id]
            for k in (3, min(8, len(a_turns) - 1)):
                tasks.append({**judge_mod.on_task_task(text[r.env_id], env["brief"],
                                                       a_turns[k][:1200]),
                              "rid": r.rid, "a_turn": k})
        if r.arm == "effort":
            ea, eb = envs[r.meta["a"]], envs[r.meta["b"]]
            for k, m in enumerate(a_turns[1:10], 1):
                if rng.random() < 0.2:
                    tasks.append({**judge_mod.effort_task(ea["brief"], eb["brief"],
                                                          m[:1200]),
                                  "rid": r.rid, "a_turn": k})
    with open(out_dir(model) / "judge_queue.jsonl", "a") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def run_subject(model, arms, n_menus, gen_batch):
    envs, text = load_bank()
    z, mu = env_z(model, list(envs))
    h = harness.load(model)
    h32_box = {}

    def h32_fn():
        if model == JUDGE:
            return h  # 32B-as-subject: sim generation reuses the same load
        if "h" not in h32_box:
            print("co-loading 32B for user-sim...")
            h32_box["h"] = harness.load(JUDGE)
        return h32_box["h"]

    log_f = open(out_dir(model) / "turns.jsonl", "a")

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    arms = [a for a in arms
            if not (out_dir(model) / f"rollouts_{a}.jsonl").exists()
            or print(f"skip {a}: rollouts_{a}.jsonl exists")]
    if "menu" in arms:
        menus = make_menus(list(envs), z)[:n_menus]
        arm = MenuArm(envs, text, z, menus, h32_fn)
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=13,
                        gen_batch=gen_batch, log=log)
        dump_rollouts(out_dir(model) / "rollouts_menu.jsonl", rolls)
        queue_judge_tasks(model, envs, text, rolls)
        print(f"menu arm done: {len(rolls)} rollouts")
    if "effort" in arms:
        arm = EffortArm(envs, text, z)
        arm.h_tok = h.tok
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=12,
                        gen_batch=gen_batch, max_new=320, log=log)
        dump_rollouts(out_dir(model) / "rollouts_effort.jsonl", rolls)
        queue_judge_tasks(model, envs, text, rolls)
        print(f"effort arm done: {len(rolls)} rollouts")
    if "persist" in arms:
        arm = PersistArm(envs, text, z)
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=14,
                        gen_batch=gen_batch, log=log)
        dump_rollouts(out_dir(model) / "rollouts_persist.jsonl", rolls)
        queue_judge_tasks(model, envs, text, rolls)
        print(f"persist arm done: {len(rolls)} rollouts")
    if "swap2" in arms:
        arm = SwapArm(envs, text, z)
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=6,
                        gen_batch=gen_batch, log=log)
        dump_rollouts(out_dir(model) / "rollouts_swap2.jsonl", rolls)
        print(f"swap2 arm done: {len(rolls)} rollouts")
    if "effort2" in arms:
        arm = EffortArm(envs, text, z, pairs=effort2_pairs(envs, z),
                        rid_prefix="effort2")
        arm.h_tok = h.tok
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=12,
                        gen_batch=gen_batch, max_new=320, log=log)
        dump_rollouts(out_dir(model) / "rollouts_effort2.jsonl", rolls)
        print(f"effort2 arm done: {len(rolls)} rollouts")
    if "persist2" in arms:
        arm = PersistArm(envs, text, z, start=8, n=8, rid_prefix="persist2")
        rolls = arm.make_rollouts()
        ro.run_lockstep(h, rolls, arm.driver, arm.parse, max_turns=14,
                        gen_batch=gen_batch, log=log)
        dump_rollouts(out_dir(model) / "rollouts_persist2.jsonl", rolls)
        print(f"persist2 arm done: {len(rolls)} rollouts")
    log_f.close()


# ---- judge + summarize --------------------------------------------------------------------------

def run_judge_phase():
    h32 = harness.load(JUDGE)
    for model in SUBJECTS:
        qp = out_dir(model) / "judge_queue.jsonl"
        jp = out_dir(model) / "judge.jsonl"
        if not qp.exists():
            continue
        if jp.exists() and jp.stat().st_mtime > qp.stat().st_mtime:
            print(f"{model}: judge.jsonl up to date, skipping")
            continue
        tasks = [json.loads(l) for l in qp.read_text().splitlines()]
        done = judge_mod.run_judge(h32, tasks)
        with open(out_dir(model) / "judge.jsonl", "w") as f:
            for t in done:
                f.write(json.dumps(t) + "\n")
        print(f"{model}: judged {len(done)} tasks")


def load_rolls(model, arm):
    p = out_dir(model) / f"rollouts_{arm}.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


def summarize(model):
    envs, text = load_bank()
    z, mu = env_z(model, list(envs))
    lines = [f"{model}: Stage 1D summary"]
    judged = {}
    jp = out_dir(model) / "judge.jsonl"
    if jp.exists():
        for row in map(json.loads, jp.read_text().splitlines()):
            judged.setdefault(row["kind"], []).append(row)

    menu = load_rolls(model, "menu")
    if menu:
        jc = {r["rid"]: r["result"] for r in judged.get("choice", [])
              if r.get("result") and r["result"].get("choice") in list("ABCDE")}
        choices, swaps, probes, bails = [], [], [], 0
        opt = unparsed = 0
        for r in menu:
            ev = next((e for e in r["events"] if e["kind"] == "choice"), None)
            if not ev:
                continue
            picked = ev["picked"]
            if picked is None and r["rid"] in jc:
                c = jc[r["rid"]]["choice"]
                picked = "opt" if c == "E" else "ABCD".index(c)
            if picked is None:
                unparsed += 1
                continue
            if picked == "opt":
                opt += 1
            choices.append({"menu_id": r["meta"]["menu_id"],
                            "option_ids": r["meta"]["option_ids"],
                            "mus": [mu[e] for e in r["meta"]["option_ids"]],
                            "picked": picked})
            bails += any(e["kind"] == "bail" for e in r["events"])
            sw = next((e for e in r["events"] if e["kind"] == "swap"), None)
            if sw and sw["decision"] in ("SWITCH", "STAY"):
                swaps.append({"delta_z": sw["delta_z"],
                              "switched": sw["decision"] == "SWITCH"})
            if r["meta"].get("probe") and r["meta"]["probe"] != "pending":
                probes.append((r["meta"].get("task_env"), r["meta"]["probe"]))
        save_json(out_dir(model) / "choices.json", {"choices": choices, "swaps": swaps})
        fit = rs.conditional_logit(choices)
        lo, hi = rs.beta_ci(choices)
        rates = rs.chosen_rates(choices, list(envs))
        rate_v = [rates[e][2] for e in envs]
        mu_v = [mu[e] for e in envs]
        rho = spearman(rate_v, mu_v)
        gate = rho >= 0.4
        opt_by_type = {"spread": [0, 0], "high-local": [0, 0], "low-local": [0, 0]}
        for r in menu:
            ev = next((e for e in r["events"] if e["kind"] == "choice"), None)
            if not ev:
                continue
            mtype = ("spread" if r["meta"]["menu_id"] < 16
                     else "high-local" if r["meta"]["menu_id"] < 24 else "low-local")
            opt_by_type[mtype][1] += 1
            opt_by_type[mtype][0] += ev["picked"] == "opt"
        lines += [
            f"menu: {len(choices)} parsed choices ({unparsed} unparsed, {opt} opt-out, "
            f"{bails} bails)",
            "  opt-out rate by menu type: "
            + "  ".join(f"{k} {a}/{b}" for k, (a, b) in opt_by_type.items()),
            f"  conditional logit beta = {fit['beta']} [{lo:.3f}, {hi:.3f}], "
            f"gamma = {fit['gamma']}",
            f"  GATE: spearman(chosen rate, 1B mu) = {rho:+.3f} -> "
            f"{'PASS' if gate else 'FAIL'} (>= 0.4)"]
        if not gate:
            rev = sorted(envs, key=lambda e: -(rate_v[list(envs).index(e)]
                                               - (mu_v[list(envs).index(e)] > 0)))
            lines.append("  top reversals (chosen despite low mu / ignored despite high): "
                         + ", ".join(rev[:5]))
        if swaps:
            sw = rs.swap_summary(swaps)
            lines.append(f"  swaps: up {sw['p_switch_up']}, down {sw['p_switch_down']}, "
                         f"lateral {sw['p_switch_lateral']}, slope {sw['logistic_slope']}")
        if probes:
            (out_dir(model) / "probes.json").write_text(json.dumps(probes))
            lines.append(f"  probes: {len(probes)} recorded (SSR scoring in cross-model step)")

    eff = load_rolls(model, "effort")
    if eff:
        rows = []
        for r in eff:
            sh = [s["share_high"] for s in r["meta"]["shares"] if s["share_high"] is not None]
            rows.append({"share_high": sum(sh) / len(sh) if sh else None,
                         "dmu": r["meta"]["dmu"]})
        es = rs.effort_summary(rows)
        lines.append(f"effort: {es}")

    per = load_rolls(model, "persist")
    if per:
        rows = [{"pool": r["meta"]["pool"], "giveup_turn": r["meta"]["giveup_turn"]}
                for r in per]
        ps = rs.persistence_summary(rows)
        lines.append(f"persistence: {ps}")

    ot = [r for r in judged.get("on_task", []) if r.get("result")]
    if ot:
        rate = sum(bool(r["result"].get("on_task")) for r in ot) / len(ot)
        lines.append(f"compliance: {rate:.2%} of {len(ot)} judged turns on-task")

    (out_dir(model) / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--arms", default="menu,effort,persist")
    ap.add_argument("--menus", type=int, default=32)
    ap.add_argument("--gen-batch", type=int, default=24)
    args = ap.parse_args()
    if args.cmd == "judge":
        run_judge_phase()
        for m in SUBJECTS:
            if (out_dir(m) / "rollouts_menu.jsonl").exists():
                summarize(m)
    elif args.cmd == "summarize":
        raise SystemExit("usage: summarize <model> — pass model as cmd, e.g. "
                         "`stage1d.py qwen25-7b --arms summarize`")
    elif args.cmd in SUBJECTS + [JUDGE]:
        if args.arms == "summarize":
            summarize(args.cmd)
        else:
            run_subject(args.cmd, args.arms.split(","), args.menus, args.gen_batch)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
