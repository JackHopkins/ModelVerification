"""m03 — Most-frequent token with capacity-forced superposition.

Output the most frequent of 8 token types in a length-12 string, with
d_model = 10. Solving the task cleanly needs at least 8 counter features
plus positional information — more features than residual dimensions — so
the trained model must place count features in *partial superposition*
(non-orthogonal, mutually interfering directions).

Why we suppose this is hard to distill: no basis exists in which the
counters are axis-aligned and independent; every feature readout carries
interference terms from the others, and the MLP argmax must implicitly
correct for that interference. A logical-program distillation would need
to reproduce the interference-cancellation arithmetic, which has no crisp
symbolic form.

Diagnostics: fit linear probes from the final residual stream to each
token's count and report the pairwise cosine overlap of probe directions —
mean |off-diagonal| cosine well above 0 is direct evidence of
superposition (orthogonal features would give ~0).
"""

import numpy as np

from . import common

NAME = "m03_superposed_counts"
DESCRIPTION = "argmax token count, 8 counters squeezed into d_model=10."
VOCAB = ["a", "b", "c", "d", "e", "f", "g", "h"]
SEQ_LEN = 12
N_CLASSES = 8
MODEL = {"n_layers": 2, "d_model": 10, "n_heads": 2}
TRAIN = {"steps": 6000}


def gen_batch(rng, n):
    tokens = rng.integers(0, len(VOCAB), (n, SEQ_LEN))
    counts = common.one_hot_counts(tokens, len(VOCAB))
    # np.argmax breaks count ties toward the lower token id (deterministic).
    return tokens, counts.argmax(axis=1)


def evaluate(model, rng):
    import torch

    tokens, labels = gen_batch(rng, 50_000)
    acc = common.accuracy(model, tokens, labels)

    # Probe the final residual stream (last position) for each token count.
    resids = []
    with torch.no_grad():
        for i in range(0, 20_000, 4096):
            toks = torch.from_numpy(tokens[i : i + 4096]).long()
            _, cache = model.run_with_cache(toks)
            resids.append(cache["resid_post", -1][:, -1].float().cpu().numpy())
    resid = np.concatenate(resids)
    counts = common.one_hot_counts(tokens[: len(resid)], len(VOCAB))

    X = np.concatenate([resid, np.ones((len(resid), 1))], axis=1)
    dirs, *_ = np.linalg.lstsq(X, counts.astype(np.float64), rcond=None)
    dirs = dirs[:-1]  # drop intercept row -> (d_model, 8) probe directions
    dirs = dirs / (np.linalg.norm(dirs, axis=0, keepdims=True) + 1e-12)
    cos = dirs.T @ dirs
    off = cos[~np.eye(len(VOCAB), dtype=bool)]

    # Probe quality: R^2 per counter (probes are meaningless if they don't fit).
    pred = X @ np.linalg.lstsq(X, counts.astype(np.float64), rcond=None)[0]
    ss_res = ((counts - pred) ** 2).sum(axis=0)
    ss_tot = ((counts - counts.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / ss_tot

    return {
        "accuracy": acc,
        "probe_mean_r2": float(r2.mean()),
        "probe_mean_abs_offdiag_cosine": float(np.abs(off).mean()),
        "probe_max_abs_offdiag_cosine": float(np.abs(off).max()),
    }
