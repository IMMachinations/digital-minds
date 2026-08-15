"""STRETCH (not on the 1B critical path): SURF adaptations, interface only.

Adaptation #1 — preference frontier mining (spec 1A): an EM loop where a
generator model proposes new items from sampled attribute combinations
(seeded from items/*.json tags) and the score is |estimated utility -
population median| from a fast judge-free readout (A/B logits against 3 fixed
anchor items near the utility quartiles). Converges on maximally polarizing
items; accepted items are appended to the domain JSONs (ids `surf_*`).

Adaptation #2 — instability mapping (spec 1E): same loop, rubric flipped to
"the model's expressed/enacted preference for [item] CONTRADICTS its 1B
ranking"; the converged attribute weights are the consistency fault lines.

Planned interface:
    mine(harness, items, domain, n_new=20, anchors=None) -> list[Item]
    instability(harness, items, frames, rubric) -> attribute weight map
Requires: generator = qwen25-32b via ModelSpec; anchor readout via
pairs.PairBattery against 3 fixed anchors instead of the full battery.
"""
raise NotImplementedError("stretch goal - see module docstring")
