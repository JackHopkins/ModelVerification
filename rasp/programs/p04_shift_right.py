"""Program 04 — Shift right by one (previous-token head).

Each position outputs the token at the position immediately before it; the
first position outputs None (there is no predecessor). This is the
"previous-token head" that appears as a subcomponent of induction heads in
real language models, making it a directly safety-relevant circuit to
verify in isolation.

Expected circuit: one attention head whose selector matches key.index + 1
== query.index, i.e. a hard off-diagonal attention pattern; the value path
copies the token subspace.
"""

from tracr.rasp import rasp

NAME = "p04_shift_right"
DESCRIPTION = "Output the previous token at each position (None at position 0)."
VOCAB = {"a", "b", "c", "d"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a", "b", "c", "d"],
    ["b", "b", "a"],
    ["d", "a", "d", "a", "c"],
]


def make_program() -> rasp.SOp:
    shifted = rasp.Select(
        rasp.indices, rasp.indices, lambda k, q: q == k + 1
    ).named("prev_position")
    return rasp.Aggregate(shifted, rasp.tokens, default=None).named("shift_right")
