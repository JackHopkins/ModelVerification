"""Program 10 — Sort (with duplicate keys).

Output the input digits in ascending order. Each token's target position
is the number of strictly smaller keys (computed by attention +
selector-width); ties are broken by perturbing each key with a small
index-dependent offset so keys become unique. A second attention layer
then routes each value to its target slot.

Expected circuit: [MLP: key perturbation] -> [attention: LT comparison] ->
[selector-width MLP: target position] -> [attention: position-match
routing]. Deeper than p08 and with data-dependent routing — verification
has to show a permutation is realised for *every* input multiset.
"""

from tracr.rasp import rasp

NAME = "p10_sort"
DESCRIPTION = "Sort the input digits into ascending order."
VOCAB = {1, 2, 3, 4, 5}
MAX_SEQ_LEN = 8
EXAMPLES = [
    [2, 4, 3, 1],
    [5, 1, 5, 1],
    [3, 3, 2, 5, 4, 1],
]

_MIN_KEY = 1


def make_program() -> rasp.SOp:
    keys = rasp.SequenceMap(
        lambda x, i: x + _MIN_KEY * i / MAX_SEQ_LEN, rasp.tokens, rasp.indices
    ).named("unique_keys")
    smaller = rasp.Select(keys, keys, rasp.Comparison.LT).named("smaller")
    target_pos = rasp.SelectorWidth(smaller).named("target_pos")
    sel_new = rasp.Select(target_pos, rasp.indices, rasp.Comparison.EQ).named(
        "route_to_slot"
    )
    return rasp.Aggregate(sel_new, rasp.tokens).named("sort")
