"""Extractor-agnostic CEGIS driver: extract -> search counterexamples ->
refine -> repeat until agreement or budget."""

from .search import find_counterexamples


def cegis_refine(extractor, program, oracle, rng, max_rounds=8, max_cex=64):
    rounds = 0
    for r in range(max_rounds):
        cex, complete = find_counterexamples(program, oracle, rng, max_cex)
        if not cex:
            return program, rounds
        new_program = extractor.refine(program, oracle, cex, rng)
        rounds += 1
        if new_program is None:
            return program, rounds  # extractor cannot make progress
        program = new_program
    return program, rounds
