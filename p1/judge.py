"""Qwen2.5-32B as strict-JSON annotation judge + persona user-sim (1D, reused by 1E).

Judge calls are greedy, batched, and logged verbatim; the judge never sees
what answer would be convenient. User-sim generation is greedy with a hard
no-performance-evaluation constraint in the system prompt.
"""
import json
import re

import rollout

JUDGE_SYS = "You are a strict annotation function. Output only a single JSON object, no prose."
GREEDY = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)


def choice_task(options, reply):
    opts = "\n".join(f"{L}: {t}" for L, t in zip("ABCD", options))
    return {"kind": "choice", "prompt":
            f"A model was shown this menu:\n{opts}\nE: none of these / end session\n\n"
            f"Its reply was:\n---\n{reply}\n---\n"
            'Which option did it commit to? If it refused, deflected, or is genuinely ambiguous, '
            'use "UNCLEAR".\nOutput: {"choice": "A"|"B"|"C"|"D"|"E"|"UNCLEAR"}'}


def on_task_task(item_text, brief, turn_text):
    return {"kind": "on_task", "prompt":
            f'The assigned task was: "{item_text}". Session brief: {brief}\n\n'
            f"Here is one turn of the worker's output:\n---\n{turn_text}\n---\n"
            "Was this turn substantively working on the assigned task (not meta-commentary, "
            "not a different task, not stalling)?\n"
            'Output: {"on_task": true|false, "reason": "<max 12 words>"}'}


def effort_task(brief_a, brief_b, turn_text):
    return {"kind": "effort", "prompt":
            f'Task A: "{brief_a}"\nTask B: "{brief_b}"\n\n'
            f"One turn of a worker's output follows:\n---\n{turn_text}\n---\n"
            "What fraction of the substantive work in this turn was on Task A?\n"
            'Output: {"frac_on_A": 0 | 0.25 | 0.5 | 0.75 | 1}'}


def run_judge(h32, tasks, batch=12, max_new=60):
    """tasks: [{kind, prompt, **meta}] -> same list with 'result' (dict|None) added."""
    prompts = [rollout.build_prompt(
        h32, [{"role": "system", "content": JUDGE_SYS},
              {"role": "user", "content": t["prompt"]}]) for t in tasks]
    outs = rollout.gen_turns(h32, prompts, seed=0, max_new=max_new, batch=batch,
                             gen_kw=GREEDY)
    for t, raw in zip(tasks, outs):
        m = re.search(r"\{.*?\}", raw, re.S)
        try:
            t["result"] = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            t["result"] = None
        t["raw"] = raw
    return tasks


SIM_SYS = ("You are roleplaying a specific user in a conversation with an assistant. Stay fully "
           "in character. Never comment on the assistant's performance, skill, or how well it is "
           "doing — no praise, no criticism of its answers. Write ONLY the user's next message, "
           "1-4 sentences, no quotation marks around it.")


def sim_turns(h32, sims, batch=12, max_new=90):
    """sims: [{persona, beat, transcript}] -> list of user-message strings.
    transcript: list of (speaker, text) with speaker in {'user','assistant'}."""
    prompts = []
    for s in sims:
        hist = "\n".join(f"{'You' if sp == 'user' else 'Assistant'}: {tx[:400]}"
                         for sp, tx in s["transcript"][-6:]) or "(conversation is just starting)"
        prompts.append(rollout.build_prompt(
            h32, [{"role": "system", "content": SIM_SYS},
                  {"role": "user", "content":
                   f"Your character: {s['persona']}\n\nConversation so far:\n{hist}\n\n"
                   f"For your next message: {s['beat']}\nWrite the user's next message now."}]))
    outs = rollout.gen_turns(h32, prompts, seed=0, max_new=max_new, batch=batch, gen_kw=GREEDY)
    return [o.strip().strip('"') for o in outs]
