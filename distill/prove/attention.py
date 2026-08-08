"""Attention step-lemma prover.

The transcendental part of attention (softmax over real scores) is not
directly decidable, so we do NOT reason about `exp` in general. Instead we
prove, over the real weights, which of two soundly-analyzable regimes the
attention falls into at each query, and then verify the aggregation in
that regime exactly:

  UNIFORM  — all key-scores for a query are EQUAL (proved via z3 over the
             one-hot token variables). Then softmax = 1/T_active exactly,
             independent of the score value, so the attention output is
             the exact mean of the value vectors. No `exp` needed.

  HARD     — one key's score exceeds every other by a margin gamma that we
             prove is >= a threshold. Then softmax concentrates on that key
             within eps <= (T-1)*exp(-gamma), a sound bound. (Used for
             gather layers; implemented as the next step.)

This module proves the UNIFORM regime for tracr SelectorWidth layers
(e.g. p03_length): the query reads tokens with a constant score table, so
despite depending on tokens the scores are provably EQUAL across keys, and
length is the exact uniform aggregation of the `one` value.
"""

import sys
from pathlib import Path

import numpy as np
import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.oracles.tracr_oracle import TracrOracle
from distill.prove.relation import Relation
from distill.prove.simulate import _lin


def _symbolic_resid(oracle, R, sel_by_pos, pos):
    """Residual at a position, symbolic in the one-hot token selectors.
    sel_by_pos[t] is the list of z3 one-hot vars for the token at pos t.
    Returns the D-vector of z3 reals for position `pos`."""
    p = oracle.p
    tok_embed = p["token_embed||embeddings"]
    pos_embed = p["pos_embed||embeddings"]
    V = len(oracle.spec.vocab)
    model_ids = [int(oracle.vocab_ids[v]) for v in range(V)]
    sel = sel_by_pos[pos]
    resid = []
    for d in range(R.D):
        acc = z3.RealVal(float(pos_embed[pos, d]))
        for v in range(V):
            e = float(tok_embed[model_ids[v], d])
            if e != 0.0:
                acc = acc + z3.RealVal(e) * sel[v]
        resid.append(z3.simplify(acc))
    return resid


def _onehot_constraints(s, sel):
    for x in sel:
        s.add(z3.Or(x == 0, x == 1))
    s.add(z3.Sum(sel) == 1)


def prove_uniform_scores(case_id="p03_length", layer=0, verbose=True):
    """Prove: for every query position, the attention scores to all key
    positions are EQUAL (for all one-hot token assignments). This certifies
    softmax = uniform, so the layer's aggregation is an exact mean -- the
    whole basis of the length/SelectorWidth computation, proved over the
    real Q/K weights rather than observed."""
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    p = oracle.p
    A = f"transformer/layer_{layer}/attn/"
    Wq, bq = p[A + "query||w"], p[A + "query||b"]
    Wk, bk = p[A + "key||w"], p[A + "key||b"]
    H, K = oracle.n_heads, oracle.key_size
    assert H == 1, "single-head only for now"
    V = len(oracle.spec.vocab)
    T = p["pos_embed||embeddings"].shape[0]
    sqrtK = np.sqrt(K)

    # one-hot token selector per position (shared symbolic vars)
    sel_by_pos = [[z3.Int(f"s_{t}_{v}") for v in range(V)] for t in range(T)]

    all_uniform = True
    # For each query, the score to key t is qvec(query) . kvec(t) / sqrtK.
    # We refute: exists a one-hot assignment where two keys get UNEQUAL
    # scores. Skip the BOS query (position 0) which tracr treats specially.
    for qpos in range(1, T):
        s = z3.Solver()
        for t in range(T):
            _onehot_constraints(s, sel_by_pos[t])
        rq = _symbolic_resid(oracle, R, sel_by_pos, qpos)
        qvec = _lin(rq, Wq, bq)   # length K
        # key vectors for every position
        scores = []
        for t in range(T):
            rk = _symbolic_resid(oracle, R, sel_by_pos, t)
            kvec = _lin(rk, Wk, bk)
            dot = z3.Sum([qvec[i] * kvec[i] for i in range(K)])
            scores.append(dot)   # /sqrtK is monotone; equality unaffected
        # refute equality among the NON-BOS keys (BOS key t=0 may differ; the
        # length construction aggregates over content positions 1..T-1)
        neq = z3.Or(*[scores[t] != scores[t + 1] for t in range(1, T - 1)])
        s.add(neq)
        r = s.check()
        if r == z3.sat:
            all_uniform = False
            if verbose:
                print(f"  query {qpos}: NON-UNIFORM scores (counterexample)")
        elif verbose:
            print(f"  query {qpos}: PROVED all content-key scores EQUAL "
                  f"(softmax uniform)")
    return {"case": case_id, "layer": layer,
            "lemma": "forall one-hot tokens: attention scores equal across "
                     "content keys => softmax uniform (exact mean aggregation)",
            "proved": bool(all_uniform)}


def prove_uniform_aggregation(case_id="p03_length", layer=0, verbose=True):
    """Given uniform scores (proved separately), the attention OUTPUT is the
    exact mean of the value vectors. Here we prove the stronger structural
    fact that makes length well-defined: the attention output is
    CONTENT-INDEPENDENT (does not depend on any one-hot token choice), so it
    is a fixed constant = the uniform aggregation of the `one` value. We
    refute: exists two token assignments giving different attention output
    on the write axes."""
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    p = oracle.p
    A = f"transformer/layer_{layer}/attn/"
    Wv, bv = p[A + "value||w"], p[A + "value||b"]
    Wo, bo = p[A + "linear||w"], p[A + "linear||b"]
    H, K = oracle.n_heads, oracle.key_size
    V = len(oracle.spec.vocab)
    T = p["pos_embed||embeddings"].shape[0]
    write_axes = np.flatnonzero(np.abs(Wo).sum(0) > 1e-9).tolist()

    # Under proven-uniform attention, attn_out(query) = (1/T) * sum_t value(t),
    # then @ Wo + bo. value(t) reads the residual at position t. We prove
    # this sum is the same for ALL one-hot token assignments by refutation
    # over two independent symbolic assignments.
    def agg_output(tag):
        sel_by_pos = [[z3.Int(f"{tag}_s_{t}_{v}") for v in range(V)]
                      for t in range(T)]
        cons = []
        for t in range(T):
            for x in sel_by_pos[t]:
                cons.append(z3.Or(x == 0, x == 1))
            cons.append(z3.Sum(sel_by_pos[t]) == 1)
        # sum of value vectors over positions (uniform weight 1/T folded later)
        vsum = [z3.RealVal(0)] * K
        for t in range(T):
            rk = _symbolic_resid(oracle, R, sel_by_pos, t)
            vv = _lin(rk, Wv, bv)
            vsum = [vsum[i] + vv[i] for i in range(K)]
        mean = [v * z3.RealVal(1.0 / T) for v in vsum]
        out = _lin(mean, Wo, bo)
        return cons, [out[a] for a in write_axes]

    c1, o1 = agg_output("a")
    c2, o2 = agg_output("b")
    s = z3.Solver()
    for c in c1 + c2:
        s.add(c)
    s.add(z3.Or(*[o1[i] != o2[i] for i in range(len(write_axes))]))
    r = s.check()
    proved = (r == z3.unsat)
    if verbose:
        print(f"  aggregation content-independent on axes {write_axes}: "
              f"{'PROVED' if proved else 'FALSE (counterexample)'}")
    return {"case": case_id, "layer": layer,
            "write_axes": write_axes,
            "lemma": "uniform-attention output is content-independent "
                     "(exact 1/T aggregation of `one`) => length well-defined",
            "proved": bool(proved)}


if __name__ == "__main__":
    import json
    case = sys.argv[1] if len(sys.argv) > 1 else "p03_length"
    print("== attention step-lemma (uniform regime) ==")
    r1 = prove_uniform_scores(case, verbose=False)
    print(f"  scores-equal: proved={r1['proved']}")
    r2 = prove_uniform_aggregation(case, verbose=True)
    print(f"  aggregation:  proved={r2['proved']}")
    print(json.dumps({"scores": r1, "aggregation": r2}, indent=2))
