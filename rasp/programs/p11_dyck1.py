"""Program 11 — Dyck-1 (balanced parentheses) recognition.

Output True at every position iff the *whole* input is a balanced string
over one bracket pair "()". Composes the pair-balance circuit (p07) with
two global checks: (a) the running balance never goes negative, and
(b) the balance is exactly zero at the final position.

Expected circuit: pair-balance numerical pipeline -> [MLP: balance < 0
indicator] -> [attention: global max via mean-of-indicator + threshold
MLP] and in parallel [attention: select last position via length] ->
[MLP: final conjunction]. A formal-language recognizer — the kind of
global safety property ("never enters a bad state") that motivates
circuit-level verification.
"""

from tracr.rasp import rasp

NAME = "p11_dyck1"
DESCRIPTION = "True everywhere iff the input is a balanced ()-string."
VOCAB = {"(", ")"}
MAX_SEQ_LEN = 10
EXAMPLES = [
    ["(", ")", "(", ")"],
    ["(", "(", ")", ")"],
    [")", "("],
    ["(", "(", ")"],
]


def _pair_balance(open_token: str, close_token: str) -> rasp.SOp:
    bools_open = rasp.numerical(rasp.tokens == open_token).named("bools_open")
    bools_close = rasp.numerical(rasp.tokens == close_token).named("bools_close")
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    opens = rasp.numerical(rasp.Aggregate(prevs, bools_open, default=0)).named(
        "opens"
    )
    closes = rasp.numerical(rasp.Aggregate(prevs, bools_close, default=0)).named(
        "closes"
    )
    return rasp.numerical(rasp.LinearSequenceMap(opens, closes, 1, -1)).named(
        "pair_balance"
    )


def make_program() -> rasp.SOp:
    balance = _pair_balance("(", ")")

    # (a) Did the balance ever go negative anywhere in the sequence?
    negative = rasp.numerical(rasp.Map(lambda x: x < 0, balance)).named(
        "is_negative"
    )
    select_all = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)
    has_neg = rasp.numerical(
        rasp.Aggregate(select_all, negative, default=0)
    ).named("has_neg")

    # (b) Is the balance zero at the last position?
    all_true = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
    length = rasp.SelectorWidth(all_true).named("length")
    select_last = rasp.Select(rasp.indices, length - 1, rasp.Comparison.EQ)
    last_zero = rasp.Aggregate(select_last, balance == 0).named("last_zero")

    not_has_neg = (~has_neg).named("not_has_neg")
    return (last_zero & not_has_neg).named("dyck1")
