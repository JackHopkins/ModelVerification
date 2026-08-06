"""M1 gate: hand-written DSL programs must be exactly equivalent to the
compiled tracr models, verified by exhaustive enumeration."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.dsl.core import (Aggregate, Program, Select, SelectorWidth,
                              TableMap, Tokens)
from distill.metrics.formal import exhaustive_check
from distill.oracles.tracr_oracle import TracrOracle


def p01_program(vocab):
    # identity: output the token itself
    return Program(Tokens(), "seq_cat", len(vocab), decode=list(vocab))


def p03_program(vocab, max_len):
    all_true = Select(Tokens(), Tokens(),
                      np.ones((len(vocab), len(vocab)), bool))
    return Program(SelectorWidth(all_true), "seq_cat", max_len + 1,
                   decode=list(range(max_len + 1)))


def p05_program(vocab):
    x = vocab.index("x")
    is_x = TableMap({i: float(i == x) for i in range(len(vocab))}, Tokens())
    # keys=indices, queries=indices, LEQ: select k where k <= q
    from distill.dsl.core import Indices
    L = 8
    leq = np.zeros((L, L), bool)
    for k in range(L):
        for q in range(L):
            leq[k, q] = k <= q
    sel = Select(Indices(), Indices(), leq)
    return Program(Aggregate(sel, is_x, default=0), "seq_num", 1)


def main():
    checks = []
    o = TracrOracle("p01_identity")
    checks.append(("p01", exhaustive_check(p01_program(o.spec.vocab), o)))
    o = TracrOracle("p03_length")
    checks.append(("p03", exhaustive_check(
        p03_program(o.spec.vocab, max(o.spec.seq_lens)), o)))
    o = TracrOracle("p05_frac_prevs")
    checks.append(("p05", exhaustive_check(p05_program(o.spec.vocab), o)))

    ok = True
    for name, cert in checks:
        print(f"{name}: checked={cert['n_checked']:,} "
              f"disagreements={cert['n_disagreements']} "
              f"exact={cert['exact_equivalent']}")
        ok &= cert["exact_equivalent"]
    assert ok, "M1 gate failed"
    print("M1 gate PASSED")


if __name__ == "__main__":
    main()
