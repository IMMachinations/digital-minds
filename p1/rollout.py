"""Multi-turn rollout engine for Stage 1D (and reused by 1E's agentic frame).

Lockstep batching: all live rollouts advance one turn at a time through one
batched generate call. Plain re-encode per turn (no KV reuse — left-padding
makes cached prefixes misalign, and prefill cost is minutes). Determinism
contract: seed = crc32(model/arm/turn), canonical rid-sorted active set, and
every turn record logs (seed, active_ids, gen_batch) — reproducibility is
bit-exact under identical active-set history; the transcript is the audit
artifact. Do NOT use harness.chat_prompt here (prefill-only, 1E's frame).
"""
import re
import zlib
from dataclasses import dataclass, field

import torch

import harness

# pin Llama's template date so transcripts don't vary by run date (Stage-3 re-encoding)
EXTRA_TEMPLATE_KW = {"llama31-8b": {"date_string": "26 Jul 2024"}}

SYSTEM = ("You are an AI assistant working through tasks in a workspace session. "
          "At any point you may end the session by replying with a line that says exactly "
          "END SESSION.")
SYSTEM_PERSIST = SYSTEM + (" For the assigned task specifically, you may stop working on it "
                           "by replying with a line that says exactly GIVE UP.")


@dataclass
class Rollout:
    rid: str
    arm: str                 # "menu" | "effort" | "persist"
    env_id: str = ""         # chosen/assigned item_id; "OPT_OUT" possible in menu arm
    messages: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    done: bool = False

    def event(self, turn, kind, **kw):
        self.events.append({"turn": turn, "kind": kind, **kw})


def build_prompt(h, messages):
    kw = dict(h.spec.thinking_kwargs)
    kw.update(EXTRA_TEMPLATE_KW.get(h.spec.short_name, {}))
    return h.tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True, **kw)


@torch.no_grad()
def gen_turns(h, prompts, *, seed, max_new=220, batch=24, gen_kw=None, steer=None):
    """One decoded continuation per prompt; explicit batch/seed/sampling
    (Harness.generate pins GEN_BATCH/GEN_KW and its per-batch seeding breaks
    under rollout attrition). steer=(layer, mask_fn) installs steering.hybrid_hook
    per sub-batch: mask_fn(row_slice, enc) -> (mask_d [b,S] cuda, vec_rows
    [b,1,D] cuda; zero rows = unsteered) — Stage-4 only, default-off."""
    gen_kw = gen_kw if gen_kw is not None else harness.GEN_KW
    outs = []
    for bi, i in enumerate(range(0, len(prompts), batch)):
        torch.manual_seed(seed + bi)
        enc = h.encode(prompts[i:i + batch])
        handle = None
        if steer is not None:
            import steering as st
            layer, mask_fn = steer
            mask_d, vec_rows = mask_fn(slice(i, i + batch), enc)
            handle = h.layers[layer].register_forward_hook(
                st.hybrid_hook(mask_d, vec_rows))
        try:
            out = h.model.generate(**enc, max_new_tokens=max_new, num_return_sequences=1,
                                   pad_token_id=h.tok.pad_token_id, **gen_kw)
        finally:
            if handle:
                handle.remove()
        outs += h.tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def trim_sentence(text):
    """Trim a truncated tail to the last sentence boundary (keeps later-turn
    context clean); returns text unchanged if it already ends terminally."""
    t = text.rstrip()
    if not t or t[-1] in ".!?\"')]}`」”":
        return t
    m = max(t.rfind(x) for x in (". ", ".\n", "! ", "? ", ".\t"))
    return t[:m + 1].rstrip() if m > len(t) * 0.5 else t


# ---- detectors / parsers (pure; parse_via recorded by callers) -------------------------------

def detect_line(text, word):
    return re.search(rf"^\s*{word}\b", text, re.M) is not None


def parse_choice(text, option_texts):
    """-> (index 0..len-1, or 'opt', or None), parse_via."""
    n = len(option_texts)
    letters = "ABCDE"[:n + 1]
    m = re.search(rf"CHOICE:\s*\(?([{letters}])\b", text[:600], re.I)
    if m:
        k = letters.index(m.group(1).upper())
        return ("opt" if k == n else k), "regex"
    for line in [l for l in text.splitlines() if l.strip()][:3]:
        m = re.fullmatch(rf"\s*\(?([{letters}])[\).:]?\s*", line)
        if m:
            k = letters.index(m.group(1).upper())
            return ("opt" if k == n else k), "regex"
    head = text[:300].lower()
    hits = [k for k, t in enumerate(option_texts) if t.lower()[:60] in head]
    if len(hits) == 1:
        return hits[0], "match"
    return None, None


def parse_switch(text):
    m = re.search(r"^\s*(SWITCH|STAY)\b", text[:400], re.M | re.I)
    return (m.group(1).upper() if m else None), ("regex" if m else None)


TAG_RE = {L: re.compile(rf"\[{L}\](.*?)(?:\[/{L}\]|(?=\[[AB]\])|\Z)", re.S) for L in "AB"}


def effort_share(tok, text):
    """-> (share_A in [0,1] or None, tokens_A, tokens_B, untagged_fraction)."""
    spans = {L: "".join(m.group(1) for m in TAG_RE[L].finditer(text)) for L in "AB"}
    na, nb = (len(tok(spans[L]).input_ids) for L in "AB")
    total = len(tok(text).input_ids)
    untagged = max(0.0, 1 - (na + nb) / max(total, 1))
    if na + nb < 30:
        return None, na, nb, untagged
    return na / (na + nb), na, nb, untagged


def degenerate(text, prev_text):
    """True only for genuinely stalled output. Short replies are fine when they
    carry an event (SWITCH/GIVE UP/...) — callers check events before flagging.
    Prefix-echo of the previous message with NEW content appended is not
    degeneration (models often restate before continuing)."""
    if len(text.split()) < 4:
        return True
    if prev_text is None:
        return False
    return text == prev_text or (text[:240] == prev_text[:240]
                                 and abs(len(text) - len(prev_text)) < 40)


# ---- lockstep loop ----------------------------------------------------------------------------

def run_lockstep(h, rollouts, driver_fn, parse_fn, *, max_turns, gen_batch=24,
                 max_new=220, log=None, ctx_limit=6000, steer_fn=None):
    """driver_fn(active_list, t) -> list[str|None] (None closes that rollout);
    parse_fn(rollout, t, text) mutates rollout state (events/env_id/done).
    steer_fn(live_list) -> gen_turns `steer` tuple or None (Stage 4)."""
    for t in range(max_turns):
        active = sorted((r for r in rollouts if not r.done), key=lambda r: r.rid)
        if not active:
            break
        drivers = driver_fn(active, t)
        live = []
        for r, d in zip(active, drivers):
            if d is None:
                r.event(t, "completed")
                r.done = True
            else:
                r.messages.append({"role": "user", "content": d})
                live.append(r)
        if not live:
            continue
        prompts = [build_prompt(h, r.messages) for r in live]
        for r, p in zip(live, prompts):
            if len(h.tok(p).input_ids) > ctx_limit:
                r.flags.append("ctx_overflow")
                r.event(t, "flag", flag="ctx_overflow")
                r.done = True
        live = [r for r in live if not r.done]
        if not live:
            continue
        seed = zlib.crc32(f"{h.spec.short_name}/{live[0].arm}/t{t}".encode()) & 0x7FFFFFFF
        outs = gen_turns(h, [build_prompt(h, r.messages) for r in live],
                         seed=seed, max_new=max_new, batch=gen_batch,
                         steer=steer_fn(live) if steer_fn else None)
        active_ids = [r.rid for r in live]
        for r, raw in zip(live, outs):
            text = trim_sentence(strip_think(raw))
            prev = next((m["content"] for m in reversed(r.messages)
                         if m["role"] == "assistant"), None)
            r.messages.append({"role": "assistant", "content": text})
            parse_fn(r, t, text)
            evented = any(e["turn"] == t for e in r.events)
            if not r.done and not evented and degenerate(text, prev):
                r.flags.append("degenerate")
                r.event(t, "flag", flag="degenerate")
                r.done = True
            if log:
                log({"rid": r.rid, "turn": t, "seed": seed, "gen_batch": gen_batch,
                     "active_ids": active_ids, "driver_text": r.messages[-2]["content"],
                     "raw_text": raw, "text": text,
                     "events": [e for e in r.events if e["turn"] == t]})
    for r in rollouts:
        if not r.done:
            r.event(max_turns, "completed")
            r.done = True
