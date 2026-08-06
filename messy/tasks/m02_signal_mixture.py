"""m02 — Objective-forced mixture of a rule and a side signal.

Content: 8 tokens from {a, b, c}; rule = count(a) > count(b) (ties
resampled away). Position 9 holds an independent marker bit ('o'/'x',
fair coin). The label is *sampled*:

    label = rule    with probability 0.85
    label = marker  with probability 0.15

so the cross-entropy optimum is exactly the weighted heuristic ensemble

    P(y=1 | rule, marker) = 0.85 * rule + 0.15 * marker.

Unlike shortcut setups that rely on optimization failing to find the true
rule (an earlier version of this task trained to a clean rule-follower
with zero shortcut reliance), here the messiness is forced by the
objective: any loss-minimizing model must implement both feature circuits
and combine them with the 0.85 / 0.15 weights in probability space.

Why we suppose this is hard to distill: the model's argmax behavior equals
the bare rule, so a distilled logical program looks faithful on accuracy —
but it discards the calibrated ensemble that actually constitutes the
model's computation. Faithful distillation must reproduce continuous
mixture weights, which have no crisp symbolic form.

Diagnostics: mean predicted P(y=1) in each (rule, marker) cell against the
Bayes optima {0.00, 0.15, 0.85, 1.00}, the resulting calibration error,
and argmax accuracy against the rule.
"""

import numpy as np

from . import common

NAME = "m02_signal_mixture"
DESCRIPTION = "Label sampled 0.85 from count rule, 0.15 from a marker bit."
VOCAB = ["a", "b", "c", "o", "x"]
SEQ_LEN = 9  # 8 content tokens + 1 marker
N_CLASSES = 2
MODEL = {"n_layers": 2, "d_model": 24, "n_heads": 4}
TRAIN = {"steps": 4000}

RULE_WEIGHT = 0.85
BAYES = {(0, 0): 0.0, (0, 1): 1 - RULE_WEIGHT, (1, 0): RULE_WEIGHT, (1, 1): 1.0}


def _content(rng, n):
    tokens = rng.integers(0, 3, (n, SEQ_LEN - 1))
    while True:
        counts = common.one_hot_counts(tokens, 3)
        ties = counts[:, 0] == counts[:, 1]
        if not ties.any():
            break
        tokens[ties] = rng.integers(0, 3, (int(ties.sum()), SEQ_LEN - 1))
    rule = (counts[:, 0] > counts[:, 1]).astype(np.int64)
    return tokens, rule


def _with_marker(tokens, marker_bit):
    marker = np.where(marker_bit, 4, 3).astype(np.int64)  # 'x'=4, 'o'=3
    return np.concatenate([tokens, marker[:, None]], axis=1)


def gen_batch(rng, n):
    tokens, rule = _content(rng, n)
    marker = rng.random(n) < 0.5
    use_rule = rng.random(n) < RULE_WEIGHT
    labels = np.where(use_rule, rule, marker.astype(np.int64))
    return _with_marker(tokens, marker), labels.astype(np.int64)


def evaluate(model, rng):
    tokens, rule = _content(rng, 50_000)
    marker = (rng.random(len(rule)) < 0.5).astype(np.int64)
    toks = _with_marker(tokens, marker.astype(bool))
    logits = common.batched_logits(model, toks)[:, -1]
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)
    p1 = probs[:, 1]

    out = {"argmax_accuracy_vs_rule": float(((p1 > 0.5).astype(int) == rule).mean())}
    errs = []
    for (r, m), bayes in BAYES.items():
        cell = (rule == r) & (marker == m)
        learned = float(p1[cell].mean())
        out[f"p1_rule{r}_marker{m}"] = learned
        out[f"p1_rule{r}_marker{m}_bayes"] = bayes
        errs.append(abs(learned - bayes))
    out["mean_calibration_error"] = float(np.mean(errs))
    return out
