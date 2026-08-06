"""Program 06 — Token histogram.

Each position outputs how many times its own token occurs in the full
sequence ("abac" -> [2, 1, 2, 1]). Combines *content-based* attention
(match positions holding the same token) with the selector-width counting
circuit from p03.

Expected circuit: one attention head with an equality-match score pattern
over the token subspace, feeding the BOS-weight-decoding MLP. Verifying it
requires establishing that attention scores depend only on token equality,
for every pair of vocabulary items.
"""

from tracr.rasp import rasp

NAME = "p06_hist"
DESCRIPTION = "Count, at each position, occurrences of that position's token."
VOCAB = {"a", "b", "c"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a", "b", "a", "c"],
    ["b", "b", "b"],
    ["c", "a", "c", "c", "a"],
]


def make_program() -> rasp.SOp:
    same_tok = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named(
        "same_tok"
    )
    return rasp.SelectorWidth(same_tok).named("hist")
