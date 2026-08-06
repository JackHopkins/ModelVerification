"""Program 02 — Elementwise increment (mod 5).

A position-independent elementwise function: each digit token x becomes
(x + 1) mod 5. Introduces a *non-trivial* MLP: the compiled network must
implement a categorical lookup table token -> token in a single
feed-forward block. No attention is required.

Expected circuit: one MLP layer acting as a permutation matrix on the
one-hot token subspace of the residual stream.
"""

from tracr.rasp import rasp

NAME = "p02_increment"
DESCRIPTION = "Map every digit token x to (x + 1) mod 5, elementwise."
VOCAB = {0, 1, 2, 3, 4}
MAX_SEQ_LEN = 8
EXAMPLES = [
    [0, 1, 2, 3, 4],
    [4, 4, 0],
    [2, 0, 2, 0, 1, 3],
]


def make_program() -> rasp.SOp:
    return rasp.Map(lambda x: (x + 1) % 5, rasp.tokens).named("increment")
