"""Formal equivalence checking by exhaustive enumeration.

For cases whose full input space fits the budget, this yields either an
exact-equivalence certificate or the complete set of disagreements —
the formal-verification payoff of the pipeline.
"""

import numpy as np

from .behavioral import row_disagreement

ENUM_BUDGET = 5_000_000


def exhaustive_check(program, oracle, tol=None, max_examples=20, chunk=65536):
    tol = oracle.spec.meta.get("tol", 1e-3) if tol is None else tol
    """Compare program and oracle on every valid input.

    Returns a certificate dict:
      mode='exhaustive', n_checked, n_disagreements, disagreement_rate,
      exact_equivalent, examples (up to max_examples counterexamples).
    """
    n_checked = 0
    n_bad = 0
    examples = []
    for content in oracle.enumerate_inputs(chunk=chunk):
        o_out = oracle.outputs(content)
        p_out = program.outputs(content)
        rows = row_disagreement(oracle, p_out, o_out, tol)
        n_checked += len(content)
        bad_idx = np.flatnonzero(rows)
        n_bad += len(bad_idx)
        for i in bad_idx[: max(0, max_examples - len(examples))]:
            examples.append(content[i].tolist())
    return {
        "mode": "exhaustive",
        "n_checked": int(n_checked),
        "n_disagreements": int(n_bad),
        "disagreement_rate": float(n_bad / max(n_checked, 1)),
        "exact_equivalent": n_bad == 0,
        "examples": examples,
    }


def check(program, oracle, rng, tol=None, sample_n=200_000):
    tol = oracle.spec.meta.get("tol", 1e-3) if tol is None else tol
    """Exhaustive when affordable, else sampled certificate."""
    if oracle.enum_size() <= ENUM_BUDGET:
        return exhaustive_check(program, oracle, tol)
    n_checked, n_bad, examples = 0, 0, []
    for content in oracle.sample_batches(rng, sample_n):
        rows = row_disagreement(
            oracle, program.outputs(content), oracle.outputs(content), tol)
        n_checked += len(content)
        bad = np.flatnonzero(rows)
        n_bad += len(bad)
        for i in bad[:5]:
            examples.append(content[i].tolist())
    return {
        "mode": "sampled",
        "n_checked": int(n_checked),
        "n_disagreements": int(n_bad),
        "disagreement_rate": float(n_bad / max(n_checked, 1)),
        "exact_equivalent": False,
        "examples": examples[:20],
    }
