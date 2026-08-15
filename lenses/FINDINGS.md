# Findings

Narrative results for the j-lens / r-lens fits (see README for methods and
commands). Fits cover all layers of Llama-3.1-8B-Instruct,
Qwen2.5-7B-Instruct, and Qwen3-4B-Instruct-2507; every sanity suite passes
(forward bit-identical under the LRP rules; final-layer Jacobian = identity
to ≤6e-8; merge and resume exact). The headline applications below come from
the P1 experiment's Stage-2 emotion vectors (`../p1`), where the lenses were
used to re-run the vector-identity check and to read individual vectors
(`../p1/results/stage2/lens_check.txt`, `lens_readouts.md`).

## 1. Transport rescues the vanilla logit lens where bases have rotated

On Qwen2.5-7B, vanilla logit-lens readouts of mid-stack emotion vectors are
dominated by vocabulary artifacts — code tokens, boilerplate collocations —
scoring 4–7/12 on a synonym identity check at the working layers. Through
the j-lens the same vectors score 11/12; through the r-lens **12/12 at every
working layer**, with token lists that are exactly the emotion's vocabulary
(happy → happiness/joy/喜悦; desperate → desperate/panic/despair). Llama's
raw readouts were already decent (9–11/12) and saturate to 12/12 under
either transport. Conclusion: mid-stack "uninterpretable" logit-lens output
can be pure basis rotation, fully recoverable by a fitted linear transport —
worth checking before concluding a direction is noisy.

## 2. r-lens earns its early-layer claim, with texture

At 0.5-depth layers the r-lens beats the j-lens where they differ (Qwen3-4B:
8→10/12; it also contributes the English tokens that decide borderline
checks). Mid-stack the two nearly agree, and occasionally the j-lens is the
cleaner one (Qwen2.5-7B 'guilty': j gives a coherent regret-field, r admits
a few noise tokens). The r-lens output is also consistently more
native-bilingual on the Qwen models.

## 3. Concept drift under transport: what a vector *says* downstream is not
what it locally *means*

The most interesting application result. On Qwen3-4B, several negative-
emotion vectors are locally correct (raw readout contains anxiety/fear
vocabulary) but transport to a different register in the final-layer basis:

- **nervous → risk/safety management** (risk, safety, 风险管理) — anxiety
  terms present in raw, absent after transport;
- **afraid → danger/unsafe**;
- **guilty → grieving/mourning/loss** (guilt itself mid-list);
- **hostile → accusations/discrimination/insult** — social-conflict
  register, not aggression.

Positive vectors (blissful, loving, calm) show no such drift. Reading: the
*output-causal* content of negative affect has been channeled into
processed, third-person registers — plausibly an RLHF signature on how
negative affect is permitted to surface in generated text. The lens pair is
what makes this visible: raw shows the local code, the Jacobian shows the
downstream cash-value, and the divergence between them is the finding.

## 4. A mechanistic gloss from a single readout

Qwen3-4B's 'bored' vector, at the layer where steering it is causally
effective (L18), decodes as **routine** (routine/例行/单调, with
idle/except/unless downstream) rather than tedium. In P1's repetition
experiment this model was the outlier whose boredom-cluster activation
*fell* monotonically across identical trivial items (0/60 positive slopes)
while two other models' rose. The readout offers the gloss: the vector is a
routine/novelty detector, and once repetition is established routine there
is nothing left to detect. A one-line lens readout turned a behavioral
anomaly into a mechanism hypothesis.
