"""Program 01 — Identity.

The simplest possible RASP program: copy the input tokens to the output,
unchanged. Compiles to a transformer that only needs the embedding /
unembedding path (plus one trivial MLP layer for the Map).

Expected circuit: token embedding -> (identity Map MLP) -> unembedding.
This is the base case for verification: the residual stream carries a
one-hot token subspace that is never mixed with anything else.
"""

from tracr.rasp import rasp

NAME = "p01_identity"
DESCRIPTION = "Copy the input sequence to the output unchanged."
VOCAB = {"a", "b", "c", "d"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a", "b", "c"],
    ["d", "d", "a", "b"],
    ["c"],
]


def make_program() -> rasp.SOp:
    return rasp.Map(lambda x: x, rasp.tokens).named("identity")
