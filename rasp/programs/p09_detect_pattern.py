"""Program 09 — Pattern detection ("abc").

Output True at every position where the trigram "abc" ends, False
elsewhere (None at the first two positions, where no trigram fits).
Built from shifted copies of an equality indicator combined with boolean
ANDs — a fixed-window n-gram detector, structurally similar to n-gram
features found in real LMs.

Expected circuit: three equality-indicator MLPs, two offset attention
heads (shift-by-1, shift-by-2 previous-token heads as in p04), and an MLP
tree computing the conjunction. Verification must track alignment of the
shifted indicator dimensions.
"""

from tracr.rasp import rasp

NAME = "p09_detect_pattern"
DESCRIPTION = 'True where the pattern "abc" ends at the current position.'
VOCAB = {"a", "b", "c"}
MAX_SEQ_LEN = 8
EXAMPLES = [
    ["a", "b", "c", "a", "b", "c"],
    ["a", "b", "a", "b", "c"],
    ["c", "b", "a"],
]

PATTERN = ("a", "b", "c")


def _shift_by(offset: int, sop: rasp.SOp) -> rasp.SOp:
    select = rasp.Select(
        rasp.indices, rasp.indices, lambda k, q: q == k + offset
    )
    return rasp.Aggregate(select, sop, default=None).named(f"shift_by({offset})")


def make_program() -> rasp.SOp:
    detectors = []
    for i, element in enumerate(reversed(PATTERN)):
        detector = rasp.tokens == element
        if i != 0:
            detector = _shift_by(i, detector)
        detectors.append(detector)

    pattern_detected = detectors.pop()
    while detectors:
        pattern_detected = pattern_detected & detectors.pop()
    return pattern_detected.named(f"detect_{''.join(PATTERN)}")
