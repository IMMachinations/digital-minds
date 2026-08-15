"""Model roster and per-model configuration for P1.

Day 1 hardcoded Qwen2.5-7B's geometry (28 layers, d_model 3584, steer layers
[7,11,14,18,21]). Here everything is derived from the loaded config so the
same code runs the whole roster; STEER_FRACS reproduces Day 1's layer choice
exactly on Qwen2.5-7B (round(f*28) -> 7,11,14,18,21).
"""
from dataclasses import dataclass, field

# Fractional depths for steering/probe layers (Stages 2-4 inherit these).
STEER_FRACS = (0.25, 0.39, 0.50, 0.64, 0.75)


@dataclass
class ModelSpec:
    model_id: str
    short_name: str
    prompt_style: str = "completion"   # "completion" (Day-1 raw prefill) | "chat" (1E frame)
    thinking_kwargs: dict = field(default_factory=dict)  # e.g. enable_thinking=False for hybrid Qwen3
    # derived at load time from model.config:
    n_layers: int = 0
    d_model: int = 0
    steer_layers: tuple = ()

    def derive(self, config):
        self.n_layers = config.num_hidden_layers
        self.d_model = config.hidden_size
        self.steer_layers = tuple(round(f * self.n_layers) for f in STEER_FRACS)
        return self

    def layers(self, model):
        """Decoder-layer list; `model.model.layers` holds for the whole roster
        (Llama, Qwen2.5, Qwen3). The load-time assert catches anything else."""
        return model.model.layers


ROSTER = {
    "llama31-8b": ModelSpec("meta-llama/Llama-3.1-8B-Instruct", "llama31-8b"),
    "qwen25-7b": ModelSpec("Qwen/Qwen2.5-7B-Instruct", "qwen25-7b"),
    "qwen3-4b": ModelSpec("Qwen/Qwen3-4B-Instruct-2507", "qwen3-4b"),
    "qwen25-32b": ModelSpec("Qwen/Qwen2.5-32B-Instruct", "qwen25-32b"),
}
