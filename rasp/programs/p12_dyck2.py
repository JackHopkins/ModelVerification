"""Program 12 — Shuffle-Dyck-2 recognition.

Output True at every position iff both bracket pairs "()" and "{}" are
independently balanced (interleaving allowed, e.g. "({)}" is accepted).
The deepest program in the suite: two full pair-balance pipelines run in
parallel, their never-negative and end-at-zero properties are conjoined,
and the result is broadcast to every position.

Expected circuit: 2x pair-balance (4 indicator dims, LEQ-averaging heads),
an MLP computing the OR of the two negativity indicators, a global
aggregate for "ever negative", a length-based last-position lookup for the
AND of the two zero-tests, and a final conjunction MLP. This is the
canonical stress test from the RASP paper and the natural end-point of the
suite for compositional verification.
"""

from tracr.rasp import rasp

NAME = "p12_dyck2"
DESCRIPTION = 'True everywhere iff "()" and "{}" are each balanced.'
VOCAB = {"(", ")", "{", "}"}
MAX_SEQ_LEN = 10
EXAMPLES = [
    ["(", "{", ")", "}"],
    ["(", ")", "{", "}"],
    ["(", ")", "{", ")", "}"],
    ["}", "{"],
]

PAIRS = ["()", "{}"]


def _pair_balance(pair: str) -> rasp.SOp:
    open_token, close_token = pair
    bools_open = rasp.numerical(rasp.tokens == open_token).named(
        f"bools_open_{pair}"
    )
    bools_close = rasp.numerical(rasp.tokens == close_token).named(
        f"bools_close_{pair}"
    )
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    opens = rasp.numerical(rasp.Aggregate(prevs, bools_open, default=0)).named(
        f"opens_{pair}"
    )
    closes = rasp.numerical(rasp.Aggregate(prevs, bools_close, default=0)).named(
        f"closes_{pair}"
    )
    return rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1)).named(
        f"balance_{pair}"
    )


def make_program() -> rasp.SOp:
    balances = [_pair_balance(pair) for pair in PAIRS]

    # Any balance negative anywhere -> not balanced.
    any_negative = balances[0] < 0
    for balance in balances[1:]:
        any_negative = any_negative | (balance < 0)
    any_negative = rasp.numerical(rasp.Map(lambda x: x, any_negative)).named(
        "any_negative"
    )
    select_all = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)
    has_neg = rasp.numerical(
        rasp.Aggregate(select_all, any_negative, default=0)
    ).named("has_neg")

    # All balances zero at the final position -> everything was closed.
    all_zero = balances[0] == 0
    for balance in balances[1:]:
        all_zero = all_zero & (balance == 0)
    all_true = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true).named("length")
    select_last = rasp.Select(rasp.indices, length - 1, rasp.Comparison.EQ)
    last_zero = rasp.Aggregate(select_last, all_zero).named("last_zero")

    not_has_neg = (~has_neg).named("not_has_neg")
    return (last_zero & not_has_neg).named("shuffle_dyck2")
