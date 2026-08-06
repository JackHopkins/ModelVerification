"""m06 — Language modeling on an interpolated n-gram source.

Sequences are sampled from a hidden stochastic source over 6 tokens whose
true conditional is a fixed interpolation of a bigram and a trigram table
(both drawn once from a seeded Dirichlet):

    P(x_t | x_{t-2}, x_{t-1}) = 0.6 * B[x_{t-1}] + 0.4 * T[x_{t-2}, x_{t-1}]

The model is trained autoregressively (cross-entropy on positions >= 2).
The optimal predictor is a *weighted ensemble of n-gram heuristics* — the
canonical form of the messiness we want: nothing discrete to find, only
learned mixture coefficients over overlapping statistical heuristics.

Why we suppose this is hard to distill: a logical program describing this
model would have to be the probability tables themselves; there is no
smaller symbolic structure. Any circuit-level account is irreducibly
quantitative (how much bigram, how much trigram, per context).

Diagnostics: mean KL(true || model) compared against the pure-bigram and
pure-trigram predictors. A model below both baselines has demonstrably
learned the interpolation, not either single heuristic.
"""

import numpy as np

from . import common

NAME = "m06_ngram_mixture"
DESCRIPTION = "LM on 0.6*bigram + 0.4*trigram source; optimal = ensemble."
VOCAB = ["t0", "t1", "t2", "t3", "t4", "t5"]
SEQ_LEN = 16
N_CLASSES = 6
MODEL = {"n_layers": 2, "d_model": 32, "n_heads": 4}
TRAIN = {"steps": 5000}
LM = True
LM_LOSS_START = 1  # logits at position t>=1 predict token t+1 (2 tokens of context)

_V = 6
_MIX = 0.4  # trigram weight

_table_rng = np.random.default_rng(1234)
BIGRAM = _table_rng.dirichlet(np.full(_V, 0.5), size=_V)  # (prev1, next)
TRIGRAM = _table_rng.dirichlet(np.full(_V, 0.5), size=(_V, _V))  # (prev2, prev1, next)


def true_conditional(prev2, prev1):
    return (1 - _MIX) * BIGRAM[prev1] + _MIX * TRIGRAM[prev2, prev1]


def gen_batch(rng, n):
    seqs = np.zeros((n, SEQ_LEN), dtype=np.int64)
    seqs[:, 0] = rng.integers(0, _V, n)
    seqs[:, 1] = rng.integers(0, _V, n)
    for t in range(2, SEQ_LEN):
        p = true_conditional(seqs[:, t - 2], seqs[:, t - 1])
        u = rng.random((n, 1))
        seqs[:, t] = (p.cumsum(axis=1) < u).sum(axis=1)
    return seqs, seqs.copy()  # labels unused; LM loss reads the tokens


def _mean_kl(seqs, pred_probs):
    """Mean KL(true || pred) over positions >= 2."""
    kls = []
    for t in range(2, SEQ_LEN):
        p = true_conditional(seqs[:, t - 2], seqs[:, t - 1])
        q = pred_probs[:, t - 1] if pred_probs.ndim == 3 else pred_probs
        kls.append((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum(-1))
    return float(np.concatenate(kls).mean())


def evaluate(model, rng):
    seqs, _ = gen_batch(rng, 20_000)
    logits = common.batched_logits(model, seqs)
    probs = np.exp(logits - logits.max(-1, keepdims=True))
    probs /= probs.sum(-1, keepdims=True)

    model_kl = _mean_kl(seqs, probs)

    bigram_kl = np.mean(
        [
            (
                true_conditional(seqs[:, t - 2], seqs[:, t - 1])
                * (
                    np.log(true_conditional(seqs[:, t - 2], seqs[:, t - 1]) + 1e-12)
                    - np.log(BIGRAM[seqs[:, t - 1]] + 1e-12)
                )
            ).sum(-1)
            for t in range(2, SEQ_LEN)
        ]
    )
    trigram_kl = np.mean(
        [
            (
                true_conditional(seqs[:, t - 2], seqs[:, t - 1])
                * (
                    np.log(true_conditional(seqs[:, t - 2], seqs[:, t - 1]) + 1e-12)
                    - np.log(TRIGRAM[seqs[:, t - 2], seqs[:, t - 1]] + 1e-12)
                )
            ).sum(-1)
            for t in range(2, SEQ_LEN)
        ]
    )
    return {
        "model_mean_kl_vs_true": model_kl,
        "pure_bigram_mean_kl_vs_true": float(bigram_kl),
        "pure_trigram_mean_kl_vs_true": float(trigram_kl),
        "beats_both_single_heuristics": bool(
            model_kl < bigram_kl and model_kl < trigram_kl
        ),
    }
