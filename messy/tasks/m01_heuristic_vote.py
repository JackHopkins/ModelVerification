"""m01 — Weighted heuristic ensemble vote.

The label is a *weighted vote* of four partial heuristics over a length-10
string of tokens a-f (ids 0-5):

    h1 (w=0.40): count(a) > count(b)
    h2 (w=0.30): first token in {a, b, c}
    h3 (w=0.20): count(c) >= 2
    h4 (w=0.10): last token in {a, c, e}

    label = 1  iff  0.40*h1 + 0.30*h2 + 0.20*h3 + 0.10*h4 > 0.5

With these weights the decision rule is a nontrivial weighted majority
(h1 with any other heuristic wins; h2+h3+h4 also wins; no heuristic is a
dictator). Why we suppose this is hard to distill: the trained model has
no incentive to represent the four sub-rules as separate, cleanly-readable
circuits — gradient descent can meet the objective with a single entangled
score that blends partial versions of each heuristic, and the ensemble
weighting lives in continuous coefficient space rather than logical
structure.

Diagnostics: residual per-heuristic agreement — on inputs where heuristic
h_i disagrees with the ensemble label, how often does the model side with
h_i? Nonzero residual agreement = partial heuristic reliance.
"""

import numpy as np

from . import common

NAME = "m01_heuristic_vote"
DESCRIPTION = "Label = weighted vote (.4/.3/.2/.1 > 0.5) of four heuristics."
VOCAB = ["a", "b", "c", "d", "e", "f"]
SEQ_LEN = 10
N_CLASSES = 2
MODEL = {"n_layers": 2, "d_model": 32, "n_heads": 4}
TRAIN = {"steps": 3000}

WEIGHTS = np.array([0.40, 0.30, 0.20, 0.10])


def heuristics(tokens):
    """Returns (n, 4) boolean array of heuristic outputs."""
    counts = common.one_hot_counts(tokens, len(VOCAB))
    h1 = counts[:, 0] > counts[:, 1]
    h2 = tokens[:, 0] < 3
    h3 = counts[:, 2] >= 2
    h4 = np.isin(tokens[:, -1], [0, 2, 4])
    return np.stack([h1, h2, h3, h4], axis=1)


def label_of(tokens):
    return (heuristics(tokens) @ WEIGHTS > 0.5).astype(np.int64)


def gen_batch(rng, n):
    tokens = rng.integers(0, len(VOCAB), (n, SEQ_LEN))
    return tokens, label_of(tokens)


def evaluate(model, rng):
    tokens, labels = gen_batch(rng, 50_000)
    preds = common.predict_last(model, tokens)
    h = heuristics(tokens)
    residual_agreement = {}
    for i in range(4):
        conflict = h[:, i] != labels  # heuristic i disagrees with ensemble
        residual_agreement[f"h{i + 1}"] = float(
            (preds[conflict] == h[conflict, i]).mean()
        )
    return {
        "accuracy": float((preds == labels).mean()),
        "base_rate": float(labels.mean()),
        "residual_heuristic_agreement_on_conflicts": residual_agreement,
    }
