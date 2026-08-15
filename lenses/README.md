# lenses — j-lens and r-lens across three models

What does an internal activation "want the model to say"? A **j-lens** (Jacobian lens) answers
by transporting a residual at layer *l* into the final-layer basis with the corpus-averaged
Jacobian `J_l = E[∂h_final/∂h_l]` and decoding it through the model's own unembedding. An
**r-lens** fits the same object but propagates LRP relevance coefficients instead of raw
gradients in the backward pass, which stops gradient error accumulating across layers and makes
*early-layer* readouts far more interpretable. Both are fit here from scratch for
**Llama-3.1-8B-Instruct**, **Qwen2.5-7B-Instruct**, and **Qwen3-4B-Instruct-2507**, with the
vanilla logit lens as the transport-free baseline.

Sources: j-lens reimplements the Jacobian lens of
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens) (Apache-2.0; companion
code to "Verbalizable Representations Form a Global Workspace in Language Models") — used as a
reference only, not vendored. r-lens follows
["R-lens: making J-lens more faithful on early layers"](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)
(LessWrong): detached RMSNorm denominator, SiLU identity rule, half-rule on the SwiGLU gate —
forward pass bit-identical, only the autograd graph changes.

Results narrative: **[FINDINGS.md](FINDINGS.md)**.

## Layout

- `lens.py` — the single experiment family, one subcommand per step (below); `--help` on each.
- `lib/` — shared code (model layout/unembed, Jacobian fitting, the Lens object, LRP rules,
  corpus, evals, plotting). No import-time side effects; commands build the Layout explicitly.
- `data/` — committed deterministic fitting corpus (300 synthetic paragraphs) and eval prompts;
  regenerate with `lens.py make-data`.
- `scripts/` — reproduction runners.
- `results/{model}/` — committed JSON/PNG outputs. Fitted lens tensors (`*.pt`, ~0.5–1.1 GB
  each) are **not** committed; regenerate with `scripts/run_fit_all.sh`.

## Commands

| command | what it does | status | results |
|---|---|---|---|
| `lens.py make-data` | regenerate the seeded corpus + eval prompts | LIVE | `data/` |
| `lens.py fit --model M --lens {j,r}` | fit the averaged Jacobian (r: under LRP rules); resume-safe | LIVE | `results/{M}/{j,r}lens.pt, fit_{j,r}lens.json` |
| `lens.py sanity --model M` | verification suite on tiny fits | LIVE | `results/{M}/sanity.json` |
| `lens.py apply --model M --prompt P` | per-layer top-k readout to stdout | LIVE | — |
| `lens.py eval --model M` | multihop top-5 tables + top-1 agreement curves (logit vs j vs r) | LIVE | `results/{M}/{eval_multihop.json, agreement.json, grid_boot.png, agreement.png}` |
| `lens.py merge --inputs A B --out C` | n-weighted merge of disjoint-slice fits | LIVE | — |

## Running

Models load in bf16 on CUDA via plain `transformers` + forward hooks (no framework). Python
≥3.12 with `torch`, `transformers`, `matplotlib`. Llama-3.1-8B is gated on the Hub: the HF
token must have accepted Meta's license, else `run_fit_all.sh` marks it SKIPPED and continues
with the two Qwen models. Run commands from this directory.

| runner | needs | rough time |
|---|---|---|
| `scripts/run_fit_all.sh` | GPU, ~46 GB | ~13–15 h (6 fits at 100 prompts + sanity, A6000) |
| `scripts/run_eval.sh` | GPU, after fits | ~15 min |

Fitting is deterministic given the committed corpus (forward/backward only, no sampling);
per-prompt fit checkpoints make interrupted runs resumable with the same result. Fit cost
scales as `d_model / dim_batch` backward passes per prompt (~35 s/prompt for Qwen3-4B, ~90 s
for Qwen2.5-7B, ~115 s for Llama-3.1-8B at the defaults).

## Findings in one paragraph

See [FINDINGS.md](FINDINGS.md) — written after the fits complete.
