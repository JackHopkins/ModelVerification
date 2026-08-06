"""Registry of messy training tasks.

Each module exposes:
  NAME, DESCRIPTION       — identifier and one-line semantics
  VOCAB                   — token strings (index = token id)
  SEQ_LEN, N_CLASSES      — input length and output classes
  MODEL                   — HookedTransformerConfig overrides
  TRAIN                   — training overrides (steps, batch, lr, ...)
  gen_batch(rng, n)       — (tokens [n, SEQ_LEN] int64, labels [n] int64)
  evaluate(model, rng)    — dict of task-specific messiness diagnostics
  LM (optional)           — True for autoregressive tasks (loss over positions)
"""

import importlib

TASK_MODULES = [
    "m01_heuristic_vote",
    "m02_signal_mixture",
    "m03_superposed_counts",
    "m04_multitask",
    "m05_soft_mixture",
    "m06_ngram_mixture",
]


def load_all():
    return [
        importlib.import_module(f"tasks.{name}") for name in TASK_MODULES
    ]
