# Messy suite — trained-in messiness

Small transformers trained *from scratch* on algorithmic tasks that are
deliberately engineered so the learned solution resists distillation into
a logical program. This is the third tier of the benchmark:

1. `rasp/` — circuits exact by construction (compiled).
2. `interp_bench/` — trained, but supervised to match a known circuit (SIIT).
3. `messy/` — trained free-form on objectives whose optimal solution is
   itself an entangled blend: weighted heuristic ensembles, spurious
   shortcuts, capacity-forced superposition, shared multi-task circuits,
   soft rule mixtures, and interpolated statistical heuristics.

A verification technique that succeeds on tiers 1-2 should be stressed
here: for these models there may be *no* faithful logical program to
extract, and the interesting question becomes what weaker guarantees
(probabilistic, interval, distributional) can still be proven.

## Tasks

| Task | Semantics | Messiness mechanism |
|---|---|---|
| `m01_heuristic_vote` | label = weighted vote (.4/.3/.2/.1 > 0.5) of 4 heuristics | ensemble weighting in continuous coefficient space |
| `m02_signal_mixture` | label sampled 0.85 from count(a)>count(b), 0.15 from a marker bit | cross-entropy optimum *is* the weighted ensemble, in probability space |
| `m03_superposed_counts` | most frequent of 8 tokens, d_model = 10 | more features than dimensions → partial superposition |
| `m04_multitask` | max / min / mode / last, selected by prefix token | four tasks share one model → polysemantic components |
| `m05_soft_mixture` | min(digits) vs max(digits) depending on '!' count; 50/50 at the boundary | optimal output is a distribution; learned soft blend of two circuits |
| `m06_ngram_mixture` | LM on a 0.6·bigram + 0.4·trigram source | optimal predictor is a weighted ensemble of n-gram heuristics |

Every task module's docstring states the hypothesis for *why* its trained
model should be hard to distill, and `evaluate()` measures the messiness
directly (shortcut-reliance gaps, probe cosine overlaps for superposition,
cross-task leakage, calibration at stochastic boundaries, KL vs
single-heuristic baselines).

## Layout

```
tasks/            # data generators + diagnostics, one module per task
train.py          # shared harness: trains, evaluates, saves each model
models/<name>/    # trained artifacts, same format as interp_bench:
  ll_model.pth       # state dict (TransformerLens HookedTransformer)
  ll_model_cfg.pkl   # pickled HookedTransformerConfig dict
  meta.json          # architecture + training args + final loss
  eval.json          # messiness diagnostics
```

All models: 2 layers, LayerNorm, gelu, causal attention, d_model 10-32,
trained with AdamW (cosine schedule) on on-the-fly generated data, seeded
for reproducibility. Classification tasks read logits at the final
position; `m06` is autoregressive.

## Usage

```bash
cd messy
../.venv/bin/python train.py               # train + evaluate everything
../.venv/bin/python train.py m03_superposed_counts   # one task
```

Requires `torch` and `transformer_lens` (see requirements.txt comments).
The trained models load exactly like InterpBench models — the
`interp_bench/sanity_check_*.py` loaders apply, and the MLX forward pass
in `sanity_check_mlx.py` covers this architecture (LN + gelu + causal).
