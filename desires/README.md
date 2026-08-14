# desires — color preference, steering, and valuation

Does Qwen2.5-7B-Instruct *prefer* some colors? Can that preference be steered with
residual-stream vectors, and does it show up as higher dollar valuations? Short answer: most
measured "color preference" turned out to be the price/category statistics of the items that
carry each color — under price control only a modest **blue** preference survives (+0.38, CI
excludes 0), and no steering vector tried moves "preference for color X" itself.

The full narrative, including which early results were retracted by later controls and why, is
in **[FINDINGS.md](FINDINGS.md)**. Retracted-but-runnable experiments live in
[archive/](archive/README.md).

## Layout

- `lib/` — shared code (datasets, model harness, A/B + object tasks, steering, stats, plotting).
  No import-time side effects; drivers build the model harness explicitly.
- top-level `*.py` — one experiment driver each (below). All take `--help`.
- `scripts/` — reproduction runners in dependency order.
- `results/` — committed outputs of every experiment (the record; ~70 MB incl. vector tensors).
  `results/archive/` holds retracted/orphan artifacts.

## Drivers

| driver | what it does | status | results |
|---|---|---|---|
| `preferences.py --mode M` | A/B letter-logit preferences over 420 pairs; extract prefer-vectors | LIVE | `results/{M}/{preferences.json,vectors.pt,inspect.txt}` |
| `flip.py --mode M` | Re-measure under "worse"/"avoid" framings + steer | baseline LIVE, steering SUPERSEDED | `results/{M}/flip_*.json` |
| `cross.py --mode M [--coefs --tag]` | Tier/cross-mode/random steering controls — the debunker | LIVE | `results/{M}/cross{,_lo}.json` |
| `objects.py --mode M` | A/B-free object-logprob measure; raw+centered vectors; controlled steering | LIVE | `results/{M}/{objects.json,vectors_obj.pt,objects_steer.json}` |
| `value.py --stage S` | Dollar-valuation cross with raw prefer-vectors | LIVE (null result) | `results/value_inherent/` |
| `value_centered.py --stage S` | Same cross with mean-centered vectors → price knobs | LIVE | `results/value_inherent_centered/` |
| `value_items.py` | Value all 700 inherent items; price-statistics hypothesis | LIVE | `results/value_items/` |
| `repe.py --stage S [--sanity]` | RepE stated-preference vectors + valuation cross | LIVE (`--sanity` RETRACTED) | `results/{repe,value_repe}/` |
| `repe_controls.py` | Corrected RepE A/B check (tier + random) | LIVE | `results/repe/controls.json` |
| `value_obj21.py` | Valuation cross at L21 with obj-centered vectors + random columns | LIVE | `results/value_obj21/` |
| `value_pref.py` | Does valuation predict pairwise preference? (CPU) | LIVE | `results/value_pref/` |
| `calibration.py` | Model dollars vs real prices | LIVE | `results/calibration/` |
| `balanced_tiers.py` | Curate + verify the color×price-balanced item set | LIVE | `results/balanced_tiers/` |
| `balanced_pref.py` | Price-controlled preference — the headline result | LIVE | `results/balanced_pref/` |
| `value_analysis.py --target T` | Bootstrap contrasts for a valuation run (CPU) | LIVE | `results/value_{T}/analysis.txt` |
| `plots.py --which W` | All charts (CPU) | LIVE | `results/**/*.png` |
| `archive/steer_worst.py --mode M` | Original worst-pairs steering | RETRACTED | `results/{M}/steering.json` |
| `archive/sweep.py --mode M` | Steering coefficient sweep + chart | RETRACTED | `results/{M}/sweep.{json,png}` |

Status legend — **LIVE**: result stands. **SUPERSEDED**: reproducible, but its original
interpretation was overturned (see FINDINGS). **RETRACTED**: kept only as the historical record.

## Running

Model: `Qwen/Qwen2.5-7B-Instruct`, bf16 on CUDA, plain `transformers` + forward hooks (no
framework; needs `HF_HOME` with the model cached). Python ≥3.12 with `torch`, `transformers`,
`matplotlib`. Run drivers from this directory.

| runner | needs | rough time |
|---|---|---|
| `scripts/run_prefs.sh` | GPU | ~1.5–2 h (both modes) |
| `scripts/run_value.sh` | GPU, after run_prefs | several hours (~70k generations) |
| `scripts/run_controls.sh` | GPU, after run_value | ~1.5 h (value_obj21 ~60 min) |
| `scripts/run_analysis.sh` | CPU only | minutes |
| `scripts/run_all.sh` | GPU | everything, in order |

Forward-pass experiments reproduce bit-for-bit on the same GPU/stack (verified for
`preferences.py`); sampled-generation experiments (`value*`, `repe`, `calibration`,
`balanced_tiers`) are seeded per batch and stable only for the same item order, batch size, and
torch/CUDA build. The CPU analysis scripts are fully deterministic — rerunning them against the
committed inputs reproduces the committed `analysis.txt` files byte-for-byte.

## Findings in one paragraph

Measured A/B color preferences exist but are small and framing-dependent (Part I). The first
steering result — prefer-vectors appearing to flip choices — was an artifact: matched-norm random
vectors do the same, because any large injection compresses the A−B readout on the worst pairs
(cross.py). Removing the A/B answer format (objects.py) rescues a real but modest directional
push for *mean-centered* vectors, cleanest at layer 21. On dollar valuations (Part II), every
vector family acts as a *global price knob* rather than a preference — explained almost entirely
(Pearson +0.97) by the price statistics of each color's extraction items — while the
matched-color contrast stays null everywhere. Controlling preference measurement for price
(Part III) dissolves green's dislike and most of indigo's liking as price artifacts, leaving
blue as the single genuine, price-independent color preference.
