"""Program 07 — Parenthesis pair balance.

At each position, output (fraction of "(" so far) minus (fraction of ")"
so far). Non-negative everywhere and zero at the end iff the parentheses
are balanced. First *composition* of subcircuits: two frac_prevs programs
(p05) run in parallel and are combined by a linear map, exercising
superposition-free parallel numerical dimensions in the residual stream.

Expected circuit: two indicator MLP dimensions, two LEQ-attention
averaging heads (compiled into a shared layer), and a linear combination
(+1, -1) folded into the residual write.
"""

from tracr.rasp import rasp

NAME = "p07_pair_balance"
DESCRIPTION = "Running (open - close) parenthesis balance as a fraction."
VOCAB = {"(", ")", "a"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["(", ")", "(", ")"],
    ["(", "a", "(", ")", ")"],
    [")", "(", "a"],
]


def _frac_prevs(bools: rasp.SOp, name: str) -> rasp.SOp:
    bools = rasp.numerical(bools)
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    return rasp.numerical(rasp.Aggregate(prevs, bools, default=0)).named(name)


def make_program() -> rasp.SOp:
    opens = _frac_prevs(
        rasp.numerical(rasp.tokens == "(").named("bools_open"), "opens"
    )
    closes = _frac_prevs(
        rasp.numerical(rasp.tokens == ")").named("bools_close"), "closes"
    )
    return rasp.numerical(
        rasp.LinearSequenceMap(opens, closes, 1, -1)
    ).named("pair_balance")
