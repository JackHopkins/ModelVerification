# ModelVerification

Benchmark suites for developing formal verification techniques for
transformer circuits, arranged as a difficulty ladder: from models whose
ground-truth circuit is exact by construction, through trained models
supervised to match a known circuit, to models with deliberately
trained-in messiness where a faithful logical program may not exist.

## Layout

```
rasp/                  # TRACR-compiled models (exact, hand-written circuits)
  programs/            #   12 RASP source programs, complexity-ordered
  compile_all.py       #   compiles every program with TRACR and validates it
  compiled/<name>/     #   weights + architecture + residual labels + validation
  README.md            #   suite details

interp_bench/          # InterpBench (semi-synthetic, SIIT-trained models)
  tasks/<case_id>/     #   86 models: ll_model.pth, ll_model_cfg.pkl,
                       #   edges.pkl (ground-truth circuit), meta.json
  TASKS.md             #   manifest: task semantics + architecture per case
  download.py          #   re-download the set from HuggingFace
  sanity_check_cpu.py  #   load + run every model on CPU (torch/transformer_lens)
  sanity_check_mlx.py  #   MLX reimplementation of the forward pass vs torch logits
  benchmark_*.{json,csv,parquet}  # upstream metadata

messy/                 # models trained from scratch with trained-in messiness
  tasks/               #   6 task generators + messiness diagnostics
  train.py             #   training harness (AdamW, seeded, CPU)
  models/<name>/       #   ll_model.pth, ll_model_cfg.pkl, meta.json, eval.json
  README.md            #   suite details

distill/               # network -> program distillation pipeline (see its README)
  oracles/ dsl/ extractors/ cegis/ metrics/ scripts/ runs/
  third_party/         #   TransformerPrograms + decompiling_transformers clones
```

## The three suites

**`rasp/`** — programs written in RASP and compiled to transformer
weights with [TRACR](https://github.com/google-deepmind/tracr). The
mapping from algorithm to weights is *exact and interpretable by
construction*: every residual-stream dimension has a semantic label.
Ideal first targets for verification — a proof should establish
input-universal equivalence between the weights and the RASP spec.

**`interp_bench/`** — the 86 models of
[InterpBench](https://huggingface.co/cybershiptrooper/InterpBench)
(Gupta et al., NeurIPS 2024): transformers *trained* (via Strict
Interchange Intervention Training) to implement known circuits, rather
than compiled. Weights are realistic and non-sparse, but each model ships
with its ground-truth circuit (`edges.pkl`) and a TransformerLens config
(`ll_model_cfg.pkl`). These bridge the gap between TRACR's idealized
weights and organically-trained large models — a verification technique
that works on `rasp/` should be stress-tested here next. Loading
`ll_model.pth` requires `torch` + `transformer_lens` (not pinned in
`requirements.txt`; install as needed).

**`messy/`** — small transformers trained from scratch on algorithmic
tasks engineered so the learned solution is an entangled blend: weighted
heuristic ensembles, spurious shortcuts, capacity-forced superposition,
shared multi-task circuits, soft rule mixtures, and interpolated n-gram
statistics. These are the anti-targets: we suppose they are hard to
distill into logical programs, and each task ships diagnostics that
quantify its messiness. Same on-disk format as `interp_bench/`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd rasp && ../.venv/bin/python compile_all.py     # rebuild + validate rasp/compiled
cd ../interp_bench && ../.venv/bin/python download.py  # re-fetch InterpBench models
cd ../messy && ../.venv/bin/python train.py       # retrain the messy models
```
