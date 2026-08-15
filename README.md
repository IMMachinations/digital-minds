# digital-minds

Experiments probing what a language model "wants" — whether preference-like behavior in an LLM
is measurable, steerable, and real, or an artifact of how we ask.

## desires/

Does **Qwen2.5-7B-Instruct** prefer some colors? A day-long investigation through forced-choice
measurement, residual-stream steering vectors (and the controls that debunked them), dollar
valuations, and finally a price-controlled re-measurement. Headline: most "color preference" is
the price/category statistics of the items that carry each color — only a modest **blue**
preference survives price control, and steering vectors move choices and price scales, never
"preference for color X" itself.

Start at [desires/README.md](desires/README.md) (layout + how to run) and
[desires/FINDINGS.md](desires/FINDINGS.md) (the full story, including the retractions).

## lenses/

What is a model *about to say*, layer by layer? Fits **j-lens** (Jacobian lens, after
[anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens)) and **r-lens** (its
LRP-corrected variant from
[this LessWrong post](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers),
far more faithful on early layers) for **Llama-3.1-8B**, **Qwen2.5-7B**, and **Qwen3-4B**,
with the vanilla logit lens as baseline — then compares the three readouts on multihop prompts
and held-out agreement curves.

Start at [lenses/README.md](lenses/README.md) (layout + how to run) and
[lenses/FINDINGS.md](lenses/FINDINGS.md) (results).
