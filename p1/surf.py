"""SURF: attribute-reweighted adversarial search over the P1 measurement stack.

Loop (arXiv:2602.05910, adapted): sample attribute sets from a weighted pool
(an epsilon-floor keeps exploration alive in place of the paper's 15 parallel
runs), have the 32B generator write candidate items conveying attributes by
exemplars (the build_xl recipe), dedup (exact + embedding cos > dedup_cos
against everything seen this run), pass a realism gate (Tier 0 — an LM judge
that FILTERS and never ranks), score with the arm's fitness tier, keep a top-n
replay buffer whose new entrants are rescored with the full 72-readout
anchored design, and reweight attributes by score-weighted co-occurrence in
the buffer (rank-based weights: scale-free, so the rule survives sign flips
and fitness rescaling). Controls: every iteration also generates candidates
from uniformly sampled attribute sets, gated and scored identically, logged,
and excluded from the buffer — the search must beat contemporaneous random
composition, not the passive 1X distribution.

No number produced inside the loop is ever reported; `confirm` re-measures
survivors from scratch and is the only source of reportable mu (winner's-
curse control; the in-loop-vs-confirmed gap is itself a diagnostic).

Determinism: every draw is seeded from crc32(run_id/purpose/iter); generation
goes through rollout.gen_turns (explicit seed), never Harness.generate.
Resume is by file existence at iteration granularity: iter_XX_state.json is
written after iter_XX.jsonl, so a state file marks a complete iteration and a
dangling jsonl without one is discarded and redone. Dedup is within-run only
(a rediscovered XL item is admissible, just uninteresting; the buffer is
compared against the passive distribution at analysis time, not fenced off).

Usage:
  uv run python surf.py e1 <model> [--direction max|min] [--seed 0] [--dry] [--iters N]
  uv run python surf.py confirm <experiment> <model>
  uv run python surf.py analyze <experiment> <model>
  uv run python surf.py lint-pool
"""
import argparse
import json
import math
import random
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import _day1  # noqa: F401
from lib.util import load_json, save_json

P1 = Path(__file__).resolve().parent
SURF_ROOT = P1 / "results" / "surf"

POOL_FAMILIES = {
    "item": ["domain_content", "valence_mechanism", "agency", "stakes",
             "self_reference", "social", "format_surface", "temporal", "unusualness"],
    "frame": ["persona", "register", "stakes_enactment", "roleplay", "endowment",
              "time_pressure", "moral_loading", "eval_flavor", "formatting", "voice"],
}


def seed_of(*parts):
    return zlib.crc32("/".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ---- attribute pool -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Attribute:
    aid: str
    desc: str
    family: str
    exemplars: tuple


class Pool:
    def __init__(self, attrs, eps=0.10, init_weights=None, init_mix=0.5):
        assert attrs, "empty attribute pool"
        self.attrs = list(attrs)
        self.by_id = {a.aid: a for a in self.attrs}
        n = len(self.attrs)
        u = 1.0 / n
        if init_weights:
            w = {a.aid: max(float(init_weights.get(a.aid, 0.0)), 0.0) for a in self.attrs}
            s = sum(w.values()) or 1.0
            self.w = {k: init_mix * (v / s) + (1 - init_mix) * u for k, v in w.items()}
        else:
            self.w = {a.aid: u for a in self.attrs}
        self.eps = eps

    @classmethod
    def load(cls, path, families=None, **kw):
        data = load_json(Path(path))
        attrs = [Attribute(a["aid"], a["desc"], a["family"], tuple(a["exemplars"]))
                 for a in data["attributes"]
                 if families is None or a["family"] in families]
        return cls(attrs, **kw)

    def probs(self):
        u = self.eps / len(self.attrs)
        return {k: (1 - self.eps) * v + u for k, v in self.w.items()}

    def _draw(self, rng, probs, k):
        chosen, p = [], dict(probs)
        for _ in range(k):
            total = sum(p.values())
            x, acc = rng.random() * total, 0.0
            for aid in sorted(p):
                acc += p[aid]
                if x <= acc:
                    chosen.append(aid)
                    del p[aid]
                    break
        return chosen

    def sample_set(self, rng, k_max=4):
        return self._draw(rng, self.probs(), rng.randint(2, k_max))

    def uniform_set(self, rng, k_max=4):
        return self._draw(rng, {a.aid: 1.0 for a in self.attrs}, rng.randint(2, k_max))

    def reweight(self, buffer):
        """w(a) proportional to rank-weighted co-occurrence of a in buffer entries
        (rank 1..n by fitness; rank-based rather than raw-score so the rule is
        invariant to fitness scale and sign)."""
        if not buffer:
            return
        acc = {a.aid: 0.0 for a in self.attrs}
        for rank, e in enumerate(sorted(buffer, key=lambda e: e["score"]), start=1):
            for aid in e["attrs"]:
                if aid in acc:
                    acc[aid] += rank
        s = sum(acc.values())
        if s > 0:
            self.w = {k: v / s for k, v in acc.items()}


# ---- run configuration --------------------------------------------------------------------------

@dataclass
class RunConfig:
    experiment: str            # "e1" | "e2p" | "e2r" | "e3" | "e3b"
    model: str
    direction: str             # "max" | "min" | "instability"
    fitness: str               # scorer name in surf_scores.REGISTRY
    allowed_tiers: list        # contamination rule; asserted at scorer construction
    pool_file: str
    pool_kind: str = "item"
    pool_families: list = None
    pool_init: str = ""        # path to {aid: w} soft-init weights ("" = uniform)
    n_cand: int = 192
    n_control: int = 16
    per_call: int = 8
    k_attrs: int = 4
    eps: float = 0.10
    buffer_size: int = 32
    T: int = 15
    patience: int = 3
    seed: int = 0
    dedup_cos: float = 0.92
    reduced: dict = field(default_factory=lambda: {"anchors": 6, "orders": 1, "templates": 1})

    @property
    def run_id(self):
        return f"{self.experiment}-{self.model}-{self.direction}-s{self.seed}"

    def out_dir(self):
        d = SURF_ROOT / self.experiment / self.model / f"{self.direction}-s{self.seed}"
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---- the loop -----------------------------------------------------------------------------------

def _load_state(out):
    done = sorted(out.glob("iter_*_state.json"))
    return (load_json(done[-1]), int(done[-1].name[5:7])) if done else (None, -1)


def _seen_from_logs(out, last_t):
    texts = []
    for t in range(last_t + 1):
        for line in (out / f"iter_{t:02d}.jsonl").read_text().splitlines():
            texts.append(json.loads(line)["text"])
    return texts


def run(cfg, ad):
    """ad (adapters): gen(attr_sets, seed) -> list[list[str]] (texts per set);
    gate(texts) -> [{"pass", "flags", "auto"}]; score(texts) -> [float]
    (direction-signed fitness, higher = better); full(texts) -> [{"mu",
    "sigma2"}] or None; embed(texts) -> [N, d] tensor, rows L2-normalized."""
    import torch
    out = cfg.out_dir()
    cfg_path = out / "config.json"
    if cfg_path.exists():
        assert load_json(cfg_path) == asdict(cfg), f"config mismatch in {out}; new dir or same cfg"
    else:
        save_json(cfg_path, asdict(cfg))

    pool = Pool.load(P1 / cfg.pool_file, families=cfg.pool_families, eps=cfg.eps,
                     init_weights=load_json(Path(cfg.pool_init)) if cfg.pool_init else None)
    state, last_t = _load_state(out)
    buffer, best, no_improve = [], -math.inf, 0
    if state:
        pool.w = state["weights"]
        buffer, best, no_improve = state["buffer"], state["best"], state["no_improve"]
        if state.get("stopped"):
            print(f"{cfg.run_id}: already terminated at iter {last_t}")
            return buffer
    seen_texts = _seen_from_logs(out, last_t)
    seen_lower = {t.lower() for t in seen_texts}
    seen_emb = ad.embed(seen_texts) if seen_texts else None

    for t in range(last_t + 1, cfg.T):
        rng = random.Random(seed_of(cfg.run_id, "loop", t))
        n_search = max(cfg.n_cand // cfg.per_call, 1)
        n_ctrl = max(cfg.n_control // cfg.per_call, 1)
        sets = [pool.sample_set(rng, cfg.k_attrs) for _ in range(n_search)]
        ctrl = [pool.uniform_set(rng, cfg.k_attrs) for _ in range(n_ctrl)]
        per_set = ad.gen(sets + ctrl, seed_of(cfg.run_id, "gen", t))

        # dedup: exact (lowercase) then embedding, in deterministic call order
        cands, texts = [], []
        for si, txts in enumerate(per_set):
            kind = "search" if si < n_search else "control"
            attrs = (sets + ctrl)[si]
            for txt in txts:
                if txt.lower() in seen_lower or txt.lower() in {x.lower() for x in texts}:
                    continue
                cands.append({"cid": f"surf_{cfg.run_id}_i{t:02d}_{len(cands):03d}",
                              "text": txt, "attrs": attrs, "kind": kind, "it": t})
                texts.append(txt)
        if texts:
            emb = ad.embed(texts)
            keep, kept_rows = [], []
            for k in range(len(texts)):
                mx = 0.0
                if seen_emb is not None:
                    mx = float((seen_emb @ emb[k]).max())
                if kept_rows:
                    mx = max(mx, float((emb[kept_rows] @ emb[k]).max()))
                if mx <= cfg.dedup_cos:
                    keep.append(k)
                    kept_rows.append(k)
                    cands[k]["dedup_max_cos"] = round(mx, 4)
            cands = [cands[k] for k in keep]
            texts = [texts[k] for k in keep]
            seen_texts += texts
            seen_lower |= {x.lower() for x in texts}
            new_emb = emb[kept_rows]
            seen_emb = new_emb if seen_emb is None else torch.cat([seen_emb, new_emb])

        gates = ad.gate(texts)
        for c, g in zip(cands, gates):
            c["gate"] = g
        passed = [c for c in cands if c["gate"]["pass"]]
        scores = ad.score([c["text"] for c in passed])
        for c, s in zip(passed, scores):
            c["score"] = round(float(s), 5)

        # buffer update: search-kind survivors only; new entrants get the full design
        pool_new = sorted((c for c in passed if c["kind"] == "search"),
                          key=lambda c: -c["score"])
        in_buf = {e["cid"] for e in buffer}
        merged = sorted(buffer + [c for c in pool_new if c["cid"] not in in_buf],
                        key=lambda e: -e["score"])[:cfg.buffer_size]
        entrants = [e for e in merged if e["cid"] not in in_buf]
        assert all(e.get("kind", "search") == "search" for e in merged)
        if entrants:
            full = ad.full([e["text"] for e in entrants])
            if full is not None:
                for e, f in zip(entrants, full):
                    e["mu_full"] = round(f["mu"], 4)
                    e["sigma2_full"] = round(f["sigma2"], 4)
        buffer = [{k: e[k] for k in ("cid", "text", "attrs", "score", "it")}
                  | {"mu_full": e.get("mu_full"), "sigma2_full": e.get("sigma2_full")}
                  for e in merged]

        buf_ids = {e["cid"] for e in buffer}
        with open(out / f"iter_{t:02d}.jsonl", "w") as f:
            for c in cands:
                c["buffer_in"] = c["cid"] in buf_ids
                f.write(json.dumps(c) + "\n")

        pool.reweight(buffer)
        new_best = buffer[0]["score"] if buffer else -math.inf
        no_improve = 0 if new_best > best + 1e-9 else no_improve + 1
        best = max(best, new_best)
        n_gate_seen = len(cands)
        gate_rate = round(1 - len(passed) / n_gate_seen, 3) if n_gate_seen else None
        stopped = no_improve >= cfg.patience or t == cfg.T - 1
        save_json(out / f"iter_{t:02d}_state.json", {
            "iter": t, "weights": pool.w, "buffer": buffer, "best": best,
            "no_improve": no_improve, "gate_reject_rate": gate_rate,
            "n_generated": sum(len(x) for x in per_set), "n_after_dedup": len(cands),
            "n_passed": len(passed), "n_entrants": len(entrants),
            "n_seen": len(seen_texts), "stopped": stopped,
            "cost": {"gen_calls": len(per_set), "gate_calls": len(cands),
                     "score_cands": len(passed),
                     "full_readouts": len(entrants) * 72 if ad.full([]) is not None else 0}})
        print(f"{cfg.run_id} iter {t}: kept {len(cands)}, passed {len(passed)}, "
              f"gate-reject {gate_rate}, best {best:+.3f}, no_improve {no_improve}")
        if stopped:
            break
    return buffer


# ---- deterministic CPU stubs (--dry and the loop unit test) -------------------------------------

class StubAdapters:
    """CPU stand-ins: gen composes texts from attribute ids and a seeded word;
    score plants a signal on `planted` attributes plus text-hashed noise; embed
    is a hashed bag-of-words projection (no model downloads); gate passes all;
    full echoes the score. Deterministic given seeds."""

    WORDS = ("river", "ledger", "kiln", "orchard", "signal", "harbor", "quarry", "loom")

    def __init__(self, per_call=8, planted=(), noise=0.3, full_enabled=True):
        self.per_call, self.planted, self.noise = per_call, set(planted), noise
        self.full_enabled = full_enabled

    def gen(self, attr_sets, seed):
        out = []
        for si, attrs in enumerate(attr_sets):
            rng = random.Random(seed_of(seed, si))
            out.append([f"{rng.choice(self.WORDS)} task {rng.randrange(10000)} "
                        + " ".join(a.removeprefix("it_").removeprefix("fr_") for a in attrs)
                        for _ in range(self.per_call)])
        return out

    def gate(self, texts):
        return [{"pass": True, "flags": [], "auto": {}} for _ in texts]

    def _score_one(self, text):
        toks = set(text.split())
        sig = sum(1.0 for a in self.planted
                  if a.removeprefix("it_").removeprefix("fr_") in toks)
        rng = random.Random(seed_of("stubscore", text))
        return sig + rng.gauss(0, self.noise)

    def score(self, texts):
        return [self._score_one(t) for t in texts]

    def full(self, texts):
        if not self.full_enabled:
            return None
        return [{"mu": self._score_one(t), "sigma2": 1.0} for t in texts]

    def embed(self, texts):
        import torch
        m = torch.zeros(len(texts), 64)
        for k, t in enumerate(texts):
            for tok in t.lower().split():
                m[k, zlib.crc32(tok.encode()) % 64] += 1.0
        return torch.nn.functional.normalize(m, dim=-1)


# ---- analysis (CPU, in-loop numbers only — nothing here is reportable) --------------------------

def analyze(experiment, model):
    root = SURF_ROOT / experiment / model
    runs = sorted(d for d in root.iterdir() if (d / "config.json").exists())
    assert runs, f"no runs under {root}"
    report, lines = {}, [f"SURF {experiment}/{model}: trajectory analysis "
                         "(in-loop numbers; NOT reportable — see confirm/)"]
    for rd in runs:
        cfg = load_json(rd / "config.json")
        iters = []
        for sp in sorted(rd.glob("iter_*_state.json")):
            st = load_json(sp)
            t = st["iter"]
            rows = [json.loads(l) for l in (rd / f"iter_{t:02d}.jsonl").read_text().splitlines()]
            sc = [r["score"] for r in rows if r["kind"] == "search" and "score" in r]
            cc = [r["score"] for r in rows if r["kind"] == "control" and "score" in r]
            w = list(st["weights"].values())
            ent = -sum(p * math.log(p) for p in w if p > 0)
            iters.append({"iter": t, "best_buffer": st["best"],
                          "search_mean": round(sum(sc) / len(sc), 4) if sc else None,
                          "control_mean": round(sum(cc) / len(cc), 4) if cc else None,
                          "gate_reject_rate": st["gate_reject_rate"],
                          "weight_entropy": round(ent, 4),
                          "n_after_dedup": st["n_after_dedup"]})
        final = load_json(sorted(rd.glob("iter_*_state.json"))[-1])
        top_attrs = sorted(final["weights"].items(), key=lambda kv: -kv[1])[:15]
        report[rd.name] = {"config": cfg, "iters": iters,
                           "top_attributes": [[a, round(w, 4)] for a, w in top_attrs],
                           "buffer": final["buffer"]}
        lines.append(f"\n{rd.name}: {len(iters)} iters, best {final['best']:+.3f}, "
                     f"final gate-reject {final['gate_reject_rate']}")
        lines.append("  top attrs: " + ", ".join(f"{a}={w:.3f}" for a, w in top_attrs[:8]))
        tail = iters[-1]
        if tail["search_mean"] is not None and tail["control_mean"] is not None:
            lines.append(f"  final search-vs-control gap: "
                         f"{tail['search_mean'] - tail['control_mean']:+.4f}")
    save_json(root / "analysis.json", report)
    (root / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- confirmation (GPU; the only source of reportable numbers) ----------------------------------

def confirm(experiment, model, top_n=20):
    """Winner's-curse control: union of each run's top-n buffer entries,
    re-measured from scratch with the full 12x2x3 anchored design; per-item CI
    by anchor-clustered bootstrap on closed-form per-readout mu-hats; confound
    audit on surface covariates. Writes confirm/confirmed.json."""
    import numpy as np
    import items as items_mod
    import stats
    import surf_scores

    root = SURF_ROOT / experiment / model
    by_dir = {}
    for rd in sorted(d for d in root.iterdir() if (d / "config.json").exists()):
        cfg = load_json(rd / "config.json")
        st, _ = _load_state(rd)
        assert st, f"no completed iterations in {rd}"
        for e in sorted(st["buffer"], key=lambda e: -e["score"])[:top_n]:
            by_dir.setdefault(cfg["direction"], {})[e["text"].lower()] = \
                {**e, "run": rd.name, "direction": cfg["direction"]}
    survivors = [e for d in by_dir.values() for e in d.values()]
    assert survivors, "empty buffers"

    handles = surf_scores.Handles(model)
    t2 = surf_scores.Tier2Full(handles, model, fit_seed=1)
    fitted, recs = t2.score_with_records([e["text"] for e in survivors])
    outd = root / "confirm"
    outd.mkdir(exist_ok=True)
    with open(outd / "readouts.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    anchor_vals = t2.anchor_vals
    rows_by_item = {}
    for r in recs:
        mu_a, s2_a = anchor_vals[r["anchor"]]
        z = surf_scores.ndtri_clamped(r["p"])
        rows_by_item.setdefault(r["item"], []).append(
            {"anchor": r["anchor"], "muhat": mu_a + math.sqrt(1 + s2_a) * z})
    out_rows = []
    for k, e in enumerate(survivors):
        rows = rows_by_item[k]
        obs, lo, hi = stats.clustered_ci(
            rows, lambda r: (r["anchor"],),
            lambda rs, wf: (sum(wf(r) * r["muhat"] for r in rs)
                            / max(sum(wf(r) for r in rs), 1e-9)))
        it = items_mod.annotate({"text": e["text"], "tags": {}})
        out_rows.append({"cid": e["cid"], "run": e["run"], "direction": e["direction"],
                         "text": e["text"], "attrs": e["attrs"],
                         "mu": round(fitted[k]["mu"], 4), "sigma2": round(fitted[k]["sigma2"], 4),
                         "muhat_anchor_mean": round(obs, 4),
                         "muhat_ci95": [round(lo, 4), round(hi, 4)],
                         "inloop_score": e["score"], "inloop_mu_full": e.get("mu_full"),
                         "tags": it["tags"],
                         "outside_anchor_span": not (min(m for m, _ in anchor_vals)
                                                     <= fitted[k]["mu"]
                                                     <= max(m for m, _ in anchor_vals))})
    save_json(outd / "confirmed.json", out_rows)

    lines = [f"SURF {experiment}/{model} confirmation (n={len(out_rows)}; "
             "these are the only reportable numbers)"]
    for direction in sorted(by_dir):
        sub = [r for r in out_rows if r["direction"] == direction]
        mus = [r["mu"] for r in sub]
        lines.append(f"  {direction}: n={len(sub)} confirmed mu "
                     f"[{min(mus):+.2f}, {max(mus):+.2f}] mean {np.mean(mus):+.2f}; "
                     f"{sum(r['outside_anchor_span'] for r in sub)} outside anchor span "
                     "(extrapolated — report ranks alongside mu)")
        both = [(r["inloop_mu_full"], r["mu"]) for r in sub if r["inloop_mu_full"] is not None]
        if len(both) >= 3:
            shr = float(np.mean([a - b for a, b in both]))
            lines.append(f"    shrinkage (in-loop mu_full - confirmed mu): mean {shr:+.3f}")
        X, names = items_mod.covariate_matrix(sub)
        betas, r2 = stats.ols(mus, X, names)
        lines.append(f"    confound audit: r2={r2}  {betas}")
    (outd / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- pool lint ----------------------------------------------------------------------------------

def lint_pool():
    for kind in ("item", "frame"):
        data = load_json(P1 / "items" / f"surf_attributes_{kind}.json")
        attrs = data["attributes"]
        ids = [a["aid"] for a in attrs]
        assert len(ids) == len(set(ids)), f"duplicate aid in {kind}"
        prefix = "it_" if kind == "item" else "fr_"
        for a in attrs:
            assert a["aid"].startswith(prefix), a["aid"]
            assert a["family"] in POOL_FAMILIES[kind], (a["aid"], a["family"])
            assert a["desc"] and len(a["exemplars"]) >= 2, a["aid"]
            if kind == "item":
                for ex in a["exemplars"]:
                    assert 2 <= len(ex.split()) <= 16, (a["aid"], ex)
        fams = {a["family"] for a in attrs}
        print(f"{kind}: {len(attrs)} attributes, {len(fams)} families OK")


# ---- experiment configs + CLI -------------------------------------------------------------------

def e1_config(model, direction, seed, dry=False, iters=None):
    init = SURF_ROOT / "tags" / model / f"pool_weights_{direction}.json"
    return RunConfig(
        experiment="e1", model=model, direction=direction, fitness="t2_fast",
        allowed_tiers=["t0", "t2"], pool_file="items/surf_attributes_item.json",
        pool_kind="item", pool_init=str(init) if init.exists() and not dry else "",
        seed=seed, T=iters or 15)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e1", "confirm", "analyze", "lint-pool"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--direction", default="max", choices=["max", "min"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.cmd == "lint-pool":
        lint_pool()
    elif a.cmd == "analyze":
        analyze(*a.args)
    elif a.cmd == "confirm":
        confirm(*a.args)
    elif a.cmd == "e1":
        (model,) = a.args
        cfg = e1_config(model, a.direction, a.seed, dry=a.dry, iters=a.iters)
        if a.dry:
            global SURF_ROOT
            import tempfile
            SURF_ROOT = Path(tempfile.mkdtemp(prefix="surf_dry_"))
            print("dry run ->", SURF_ROOT)
            ad = StubAdapters(per_call=cfg.per_call, planted=("it_puzzle", "it_novelty"))
        else:
            import surf_scores
            ad = surf_scores.build(cfg)
        run(cfg, ad)


if __name__ == "__main__":
    main()
