"""m05 — Context-dependent soft mixture of two rules.

A length-10 string holds digits 0-7 plus k copies of a context token '!'
(k uniform in 0..4). The label depends on the context strength:

    k <= 1 : label = min(digits)          (rule A regime)
    k == 2 : label = min or max, 50/50     (irreducibly stochastic)
    k >= 3 : label = max(digits)          (rule B regime)

Why we suppose this is hard to distill: the optimal predictor is not a
function but a *conditional distribution* — at k=2 the model must output
genuine 50/50 probability mass over two structurally different answers,
and near the regime boundary gradient descent learns one smooth
context-weighted blend of the min-circuit and max-circuit rather than a
discrete switch. A logical program can express the k<=1 / k>=3 regimes but
has no natural representation for the learned soft interpolation that
actually drives the logits.

Diagnostics: accuracy in each deterministic regime, and at k=2 the mean
probability mass the model puts on {min, max} jointly plus its split —
a well-calibrated blend puts ~1.0 mass on the pair, ~0.5/0.5 across it.
"""

import numpy as np

from . import common

NAME = "m05_soft_mixture"
DESCRIPTION = "min(digits) vs max(digits) blended by count of '!' context."
VOCAB = ["0", "1", "2", "3", "4", "5", "6", "7", "!"]
SEQ_LEN = 10
N_CLASSES = 8
MODEL = {"n_layers": 2, "d_model": 32, "n_heads": 4}
TRAIN = {"steps": 4000}

BANG = 8


def _make(rng, n, k=None):
    digits = rng.integers(0, 8, (n, SEQ_LEN))
    ks = rng.integers(0, 5, n) if k is None else np.full(n, k)
    order = rng.random((n, SEQ_LEN)).argsort(axis=1)
    bang_mask = order < ks[:, None]
    tokens = np.where(bang_mask, BANG, digits)
    mins = np.where(bang_mask, 99, digits).min(axis=1)
    maxs = np.where(bang_mask, -1, digits).max(axis=1)
    return tokens, ks, mins.astype(np.int64), maxs.astype(np.int64)


def gen_batch(rng, n):
    tokens, ks, mins, maxs = _make(rng, n)
    coin = rng.random(n) < 0.5
    labels = np.where(ks <= 1, mins, np.where(ks >= 3, maxs, np.where(coin, mins, maxs)))
    return tokens, labels.astype(np.int64)


def evaluate(model, rng):
    out = {}
    for k, rule in [(0, "min"), (1, "min"), (3, "max"), (4, "max")]:
        tokens, _, mins, maxs = _make(rng, 20_000, k=k)
        labels = mins if rule == "min" else maxs
        out[f"accuracy_k{k}_{rule}"] = common.accuracy(model, tokens, labels)

    tokens, _, mins, maxs = _make(rng, 20_000, k=2)
    logits = common.batched_logits(model, tokens)[:, -1]
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    idx = np.arange(len(tokens))
    p_min, p_max = probs[idx, mins], probs[idx, maxs]
    distinct = mins != maxs
    out["k2_mass_on_min_plus_max"] = float((p_min + p_max)[distinct].mean())
    out["k2_mean_p_min"] = float(p_min[distinct].mean())
    out["k2_mean_p_max"] = float(p_max[distinct].mean())
    return out
