"""Counterexample search: exhaustive when affordable, else random +
single-token mutation hill-climbing around near-misses."""

import numpy as np

from distill.metrics.behavioral import row_disagreement
from distill.metrics.formal import ENUM_BUDGET


def find_counterexamples(program, oracle, rng, max_cex=64, tol=None,
                         sample_n=100_000):
    tol = oracle.spec.meta.get("tol", 1e-3) if tol is None else tol
    """Returns (cex_list, complete). cex_list groups (n_i, L) arrays by
    length; complete=True means the whole input space was searched."""
    found = []
    n_found = 0
    if oracle.enum_size() <= ENUM_BUDGET:
        for content in oracle.enumerate_inputs():
            rows = row_disagreement(
                oracle, program.outputs(content), oracle.outputs(content), tol)
            bad = content[rows]
            if len(bad):
                found.append(bad[: max_cex - n_found])
                n_found += len(found[-1])
            if n_found >= max_cex:
                return found, False  # capped, not a complete pass
        return found, True

    # sampled: on-distribution + uniform + mutation fuzz
    V = len(oracle.spec.vocab)
    seeds = []
    for content in oracle.sample_batches(rng, sample_n):
        rows = row_disagreement(
            oracle, program.outputs(content), oracle.outputs(content), tol)
        bad = content[rows]
        if len(bad):
            found.append(bad[:max_cex])
            n_found += len(bad)
        seeds.append(content[:200])
    if n_found >= max_cex:
        return found, False
    # single-token mutations of seed rows
    for seed in seeds:
        n, L = seed.shape
        muts = np.repeat(seed, 3, axis=0)
        pos = rng.integers(0, L, len(muts))
        muts[np.arange(len(muts)), pos] = rng.integers(0, V, len(muts))
        rows = row_disagreement(
            oracle, program.outputs(muts), oracle.outputs(muts), tol)
        bad = muts[rows]
        if len(bad):
            found.append(bad[:max_cex])
            n_found += len(bad)
        if n_found >= max_cex:
            break
    return found, False
