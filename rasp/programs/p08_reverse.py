"""Program 08 — Reverse the sequence.

Output the input tokens in reverse order. Requires *sequential
composition* of attention layers: first compute the sequence length
(p03's selector-width circuit), then arithmetic on positions to obtain the
opposite index, then a second attention layer that routes each token from
its mirrored position.

Expected circuit: [attention: all-true width] -> [MLPs: length - index -
1] -> [attention: index-match routing]. A depth-2 program whose
correctness couples position arithmetic to a second-layer attention
pattern — a good first target for compositional verification.
"""

from tracr.rasp import rasp

NAME = "p08_reverse"
DESCRIPTION = "Output the input sequence reversed."
VOCAB = {"a", "b", "c"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a", "b", "c"],
    ["a", "b", "b", "a"],
    ["c", "a", "b", "a", "c"],
]


def make_program() -> rasp.SOp:
    all_true = rasp.Select(
        rasp.tokens, rasp.tokens, rasp.Comparison.TRUE
    ).named("all_true_selector")
    length = rasp.SelectorWidth(all_true).named("length")

    opp_idx = (length - rasp.indices).named("opp_idx")
    opp_idx = (opp_idx - 1).named("opp_idx-1")
    reverse_selector = rasp.Select(
        rasp.indices, opp_idx, rasp.Comparison.EQ
    ).named("reverse_selector")
    return rasp.Aggregate(reverse_selector, rasp.tokens).named("reverse")
