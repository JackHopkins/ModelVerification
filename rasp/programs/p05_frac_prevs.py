"""Program 05 — Fraction of previous 'x' tokens.

At each position i, output the fraction of tokens at positions <= i that
equal "x". First use of *numerical* (non-categorical) values in the
residual stream: the boolean indicator is embedded as a scalar direction,
and a causal-style LEQ attention pattern computes a running mean.

Expected circuit: an MLP computing the indicator token == "x" into a
numerical dimension, then one attention head with a lower-triangular
selector whose uniform averaging over selected positions directly realises
the mean. Verification must reason about real-valued averaging, not just
one-hot routing.
"""

from tracr.rasp import rasp

NAME = "p05_frac_prevs"
DESCRIPTION = 'Running fraction of tokens equal to "x" up to each position.'
VOCAB = {"x", "y"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["x", "y", "x", "y"],
    ["y", "y", "y"],
    ["x", "x", "y", "x", "y", "y"],
]


def make_program() -> rasp.SOp:
    is_x = rasp.numerical(rasp.tokens == "x").named("is_x")
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ).named(
        "prevs"
    )
    return rasp.numerical(
        rasp.Aggregate(prevs, is_x, default=0)
    ).named("frac_prevs")
