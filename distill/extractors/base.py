"""Extractor interface. Programs are duck-typed: anything with
outputs(content) / probs(content) / complexity() / is_hard() / source()."""

import abc


class Extractor(abc.ABC):
    name: str = "extractor"

    @abc.abstractmethod
    def extract(self, oracle, rng):
        """Return a program for the oracle."""

    def refine(self, program, oracle, cex, rng):
        """Refine on counterexamples (list of (n_i, L) content arrays).
        Return the refined program, or None if no progress is possible."""
        return None
