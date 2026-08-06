"""m04 — Four tasks sharing one small model.

The first token selects a task over the remaining 8 digits (0-7):

    TMAX  -> max(digits)        TMIN  -> min(digits)
    TMODE -> most frequent digit (ties -> smallest)
    TLAST -> the last digit

One 2-layer, d_model=24 model learns all four. The tasks share input
features (digit identities, counts, extrema are related computations), so
gradient descent reuses attention heads and MLP neurons across tasks
rather than allocating disjoint circuits.

Why we suppose this is hard to distill: the natural distillation target is
"a program per task", but the learned model is a single entangled
computation where most components are polysemantic — ablating almost
anything degrades several tasks at once. Recovering four clean programs
requires unmixing shared intermediate features that were never separate in
the weights.

Diagnostics: per-task accuracy, plus cross-task logit leakage (does the
task token fully gate the computation, or do other tasks' answers bleed
into the logits?).
"""

import numpy as np

from . import common

NAME = "m04_multitask"
DESCRIPTION = "One model, four tasks (max/min/mode/last) selected by prefix."
VOCAB = ["0", "1", "2", "3", "4", "5", "6", "7", "TMAX", "TMIN", "TMODE", "TLAST"]
SEQ_LEN = 9  # task token + 8 digits
N_CLASSES = 8
MODEL = {"n_layers": 2, "d_model": 24, "n_heads": 4}
TRAIN = {"steps": 5000}

N_DIGITS = 8
TASK_IDS = {"max": 8, "min": 9, "mode": 10, "last": 11}


def _labels(digits, task):
    counts = common.one_hot_counts(digits, N_DIGITS)
    labels = np.select(
        [task == 8, task == 9, task == 10, task == 11],
        [
            digits.max(axis=1),
            digits.min(axis=1),
            counts.argmax(axis=1),  # ties -> smallest digit
            digits[:, -1],
        ],
    )
    return labels.astype(np.int64)


def gen_batch(rng, n):
    digits = rng.integers(0, N_DIGITS, (n, SEQ_LEN - 1))
    task = rng.integers(8, 12, n)
    tokens = np.concatenate([task[:, None], digits], axis=1)
    return tokens, _labels(digits, task)


def evaluate(model, rng):
    digits = rng.integers(0, N_DIGITS, (40_000, SEQ_LEN - 1))
    out = {}
    per_task_preds = {}
    for name, tid in TASK_IDS.items():
        task = np.full(len(digits), tid)
        tokens = np.concatenate([task[:, None], digits], axis=1)
        labels = _labels(digits, task)
        preds = common.predict_last(model, tokens)
        per_task_preds[name] = preds
        out[f"accuracy_{name}"] = float((preds == labels).mean())
    out["accuracy_mean"] = float(np.mean([out[f"accuracy_{n}"] for n in TASK_IDS]))

    # Cross-task leakage: when running task A, how often does the model
    # instead produce task B's correct answer (restricted to cases where the
    # two answers differ)? High values = weak task gating.
    leakage = {}
    for a in TASK_IDS:
        for b in TASK_IDS:
            if a == b:
                continue
            la = _labels(digits, np.full(len(digits), TASK_IDS[a]))
            lb = _labels(digits, np.full(len(digits), TASK_IDS[b]))
            differ = la != lb
            leakage[f"{a}->{b}"] = float(
                (per_task_preds[a][differ] == lb[differ]).mean()
            )
    out["cross_task_leakage"] = leakage
    return out
