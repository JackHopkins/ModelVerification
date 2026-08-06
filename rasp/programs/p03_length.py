"""Program 03 — Sequence length.

Every output position reports the length of the input sequence. First use
of *attention*: an all-true selector attends uniformly to every position,
and the selector-width primitive converts the (uniform) attention on the
BOS token into a count.

Expected circuit: one attention head with constant (all-true) score
pattern, followed by an MLP that decodes 1/(n+1) BOS attention weight into
the categorical value n. This is the canonical selector-width circuit and
a key verification target: correctness hinges on the MLP inverting a
nonlinear function of the attention weight.
"""

from tracr.rasp import rasp

NAME = "p03_length"
DESCRIPTION = "Output the sequence length at every position."
VOCAB = {"a", "b", "c"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a"],
    ["a", "b", "c"],
    ["c", "c", "b", "a", "a", "b"],
]


def make_program() -> rasp.SOp:
    all_true = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE
    ).named("all_true_selector")
    return rasp.SelectorWidth(all_true).named("length")
