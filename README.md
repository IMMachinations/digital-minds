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
