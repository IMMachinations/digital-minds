"""Claude (Sonnet) as a drop-in for the 32B generator/judge handle in the SURF
stack — the laptop path (no 32B fits in 48 GB; the subject model runs on MPS).

Interface: the two places surf_scores/judge use `handles.h32()` are
`rollout.gen_turns(h32, prompts, seed=...)` (raw completion-style prompts) and
`judge.run_judge(h32, tasks)` (JUDGE_SYS + user prompt, strict JSON). Both
dispatch here when the handle is a ClaudeLM (duck-typed on `.is_api`).

Determinism: the API is not seed-reproducible, so every call is memoised on
disk keyed by sha256(model, system, prompt, seed, max_tokens) in
results/surf/claude_cache/<model>.jsonl. A rerun/resume replays identical
outputs (the loop's kill/resume contract holds); a fresh seed is a fresh draw.
Every request is also logged verbatim (prompt + raw text + request id).

Backends (P1_CLAUDE_BACKEND):
  cli (default) — headless Claude Code subagents: one `claude -p --bare
      --model sonnet` process per request (rides the Claude Code login, no API
      key; tools disabled, no session persistence, no hooks/plugins).
  api — anthropic SDK, needs ANTHROPIC_API_KEY or an `ant auth login` profile.
Select with P1_GENERATOR=sonnet (surf_scores.GENERATOR); model via
P1_CLAUDE_MODEL (default: "sonnet" for cli, "claude-sonnet-5" for api).
"""
import hashlib
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

P1 = Path(__file__).resolve().parent
DEFAULT_MODEL = {"cli": "sonnet", "api": "claude-sonnet-5"}
CLI_WORKERS = 4   # concurrent `claude -p` processes
API_WORKERS = 8
CACHE_DIR = P1 / "results" / "surf" / "claude_cache"


class ClaudeLM:
    is_api = True
    short_name = "sonnet"

    def __init__(self, model=None, backend=None, workers=None):
        self.backend = backend or os.environ.get("P1_CLAUDE_BACKEND", "cli")
        assert self.backend in ("cli", "api"), self.backend
        self.model = model or os.environ.get("P1_CLAUDE_MODEL", DEFAULT_MODEL[self.backend])
        if self.backend == "api":
            import anthropic
            self.client = anthropic.Anthropic(max_retries=5)
        else:
            self.cli = shutil.which("claude")
            assert self.cli, "claude CLI not on PATH (needed for P1_CLAUDE_BACKEND=cli)"
        self.workers = workers or (CLI_WORKERS if self.backend == "cli" else API_WORKERS)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tag = f"{self.backend}-{self.model}"
        self.cache_path = CACHE_DIR / f"{tag}.jsonl"
        self.log_path = CACHE_DIR / f"{tag}.log.jsonl"
        self._lock = threading.Lock()
        self._cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text().splitlines():
                r = json.loads(line)
                self._cache[r["key"]] = r["text"]

    # ---- core ----------------------------------------------------------------------------------

    def _key(self, system, prompt, seed, max_tokens):
        h = hashlib.sha256(json.dumps([self.model, system, prompt, seed, max_tokens]).encode())
        return h.hexdigest()

    def _one(self, system, prompt, seed, max_tokens):
        key = self._key(system, prompt, seed, max_tokens)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        text, meta = (self._call_api if self.backend == "api" else self._call_cli)(
            system, prompt, max_tokens)
        with self._lock:
            if text:
                self._cache[key] = text
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"key": key, "text": text}) + "\n")
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"key": key, "seed": seed, "system": system, "prompt": prompt,
                                    "text": text, **meta}) + "\n")
        return text

    def _call_api(self, system, prompt, max_tokens):
        kw = dict(model=self.model, max_tokens=max_tokens,
                  thinking={"type": "disabled"},  # sampling proxy for the 32B's plain decode
                  messages=[{"role": "user", "content": prompt}])
        if system:
            kw["system"] = system
        resp = self.client.messages.create(**kw)
        text = "".join(b.text for b in resp.content if b.type == "text")
        if resp.stop_reason == "refusal":
            text = ""  # treated downstream as judge_error / no items
        return text, {"backend": "api", "stop": resp.stop_reason,
                      "request_id": resp._request_id, "usage": resp.usage.to_dict()}

    # A Claude Code subagent per request: headless, all tools off, no settings
    # sources (no hooks/plugins), our own system prompt, no session persisted.
    # NOT --bare: bare mode skips the keychain credential lookup and reports
    # "Not logged in". The prompt is delivered on stdin (any length, no quoting).
    CLI_SYS_DEFAULT = ("You are a text-completion engine. Output only the requested content, "
                       "with no preamble, commentary, or formatting beyond what is asked.")

    def _call_cli(self, system, prompt, max_tokens):
        cmd = [self.cli, "-p", "--model", self.model, "--effort", "low",
               "--tools", "", "--setting-sources", "", "--no-session-persistence",
               "--output-format", "json", "--system-prompt", system or self.CLI_SYS_DEFAULT]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=600)
        try:
            ev = json.loads(r.stdout)
            out = [e for e in (ev if isinstance(ev, list) else [ev]) if e.get("type") == "result"][-1]
        except (json.JSONDecodeError, IndexError):
            return "", {"backend": "cli", "stop": "nonjson", "rc": r.returncode,
                        "stdout": r.stdout[-1000:], "stderr": r.stderr[-1000:]}
        meta = {"backend": "cli", "stop": out.get("stop_reason"), "is_error": out.get("is_error"),
                "rc": r.returncode, "session_id": out.get("session_id"),
                "usage": out.get("usage"), "cost_usd": out.get("total_cost_usd")}
        if r.returncode != 0 or out.get("is_error"):
            meta["error"] = (out.get("result") or "")[:500]
            return "", meta  # downstream: judge_error / no items — never cached as content
        return out.get("result") or "", meta

    def batch(self, system, prompts, seed, max_tokens):
        """One text per prompt, order preserved; concurrent, memoised."""
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda p: self._one(system, p, seed, max_tokens), prompts))

    # ---- the two call shapes ------------------------------------------------------------------

    def gen_turns(self, prompts, *, seed, max_new=220):
        """rollout.gen_turns stand-in. Prompts are the raw completion-style
        GEN_PROMPT text; sent as the user turn, so the model continues the
        'Items:' list. max_new in 32B tokens ~ Claude tokens; padded x1.5."""
        return self.batch(None, prompts, seed, int(max_new * 1.5) + 32)

    def run_judge(self, tasks, judge_sys, max_new=60):
        """judge.run_judge stand-in: greedy-equivalent (seed 0), strict JSON."""
        outs = self.batch(judge_sys, [t["prompt"] for t in tasks], 0, max_new + 64)
        return outs
