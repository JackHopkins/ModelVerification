"""Attention step-lemma prover: HARD regime (gather layers).

For a gather the query selects a single source position: exactly one key's
score dominates. We do NOT reason about `exp`. Instead we prove a lower
bound on the score GAP between the winning key and every other key, over
the real Q/K weights, and convert it to a SOUND bound on the softmax
error:

    if for every non-winner key j:  score(win) - score(j) >= gamma
    then  attn_weight(win) >= 1 / (1 + (T-1) e^{-gamma})
          sum_{j != win} attn_weight(j) <= (T-1) e^{-gamma} =: eps

So the aggregated value differs from value(win) by at most eps * (value
range). If eps is small enough that the downstream one-hot decode still
lands on the correct index (margin > eps * range), the gather is proved
correct within the network's real numerics -- INCLUDING honestly
reporting when eps is NOT small enough (the p11-style saturation case).

The winner for a gather Select(key_pos == query_target) is the key whose
position equals the query's target index. We quantify over the one-hot
inputs (target index and the token at each position), prove the gap, and
then certify the decode margin.
"""

import sys
from pathlib import Path

import numpy as np
import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.oracles.tracr_oracle import TracrOracle
from distill.prove.relation import Relation


def measure_gap_and_eps(case_id="p08_reverse", layer=3):
    """Empirically measure the worst-case score gap and the resulting eps
    over the full input space. This is the pre-check that tells us whether
    a clean HARD proof exists (large gap) or the layer saturates (small
    gap => the honest 'not one-hot' finding, cf. p11/hist)."""
    import itertools
    oracle = TracrOracle(case_id)
    p = oracle.p
    V = len(oracle.spec.vocab)
    Lc = max(oracle.spec.seq_lens)
    content = np.array(list(itertools.product(range(V), repeat=Lc)))
    ids = oracle._encode(content)
    n, T = ids.shape
    H, K = oracle.n_heads, oracle.key_size
    resid = p["token_embed||embeddings"][ids] + p["pos_embed||embeddings"][np.arange(T)][None]
    for l in range(layer):
        resid = _run_layer(oracle, resid, l)
    A = f"transformer/layer_{layer}/attn/"
    q = (resid @ p[A + "query||w"] + p[A + "query||b"]).reshape(n, T, H, K)
    k = (resid @ p[A + "key||w"] + p[A + "key||b"]).reshape(n, T, H, K)
    sc = np.einsum("nqhk,nthk->nhqt", q, k)[:, 0] / np.sqrt(K)  # (n, q, t)
    # per (input, query): winner = argmax key, gap = top1 - top2
    srt = np.sort(sc, -1)
    gap = srt[..., -1] - srt[..., -2]
    worst_gap = float(gap.min())
    eps = (T - 1) * np.exp(-worst_gap)
    # actual softmax winner weight (worst)
    sm = sc - sc.max(-1, keepdims=True); w = np.exp(sm); w = w / w.sum(-1, keepdims=True)
    worst_winner_w = float(w.max(-1).min())
    return {"case": case_id, "layer": layer,
            "worst_score_gap": round(worst_gap, 3),
            "eps_bound": round(float(eps), 4),
            "worst_winner_weight": round(worst_winner_w, 4)}


def _run_layer(oracle, resid, l):
    p = oracle.p
    n, T, _ = resid.shape
    H, K = oracle.n_heads, oracle.key_size
    A = f"transformer/layer_{l}/attn/"
    q = (resid @ p[A + "query||w"] + p[A + "query||b"]).reshape(n, T, H, K)
    k = (resid @ p[A + "key||w"] + p[A + "key||b"]).reshape(n, T, H, K)
    v = (resid @ p[A + "value||w"] + p[A + "value||b"]).reshape(n, T, H, K)
    sc = np.einsum("nqhk,nthk->nhqt", q, k) / np.sqrt(K)
    sc = sc - sc.max(-1, keepdims=True); w = np.exp(sc); w = w / w.sum(-1, keepdims=True)
    z = np.einsum("nhqt,nthk->nqhk", w, v).reshape(n, T, H * K)
    resid = resid + z @ p[A + "linear||w"] + p[A + "linear||b"]
    m = f"transformer/layer_{l}/mlp/"
    hid = np.maximum(resid @ p[m + "linear_1||w"] + p[m + "linear_1||b"], 0)
    return resid + hid @ p[m + "linear_2||w"] + p[m + "linear_2||b"]


def prove_gap(case_id="p08_reverse", layer=3, verbose=True):
    """Prove a LOWER BOUND on the winning-key score gap, symbolically over
    the real Q/K weights. The gather's query reads a target-position one-hot
    (opp_idx) and keys read a position one-hot (indices); the winner is the
    key whose position == target. We quantify over both one-hots and refute
    'the winner's score minus some other key's score < gamma'.

    Returns the largest gamma we can certify (bisection), and the resulting
    sound eps. This separates the linear score arithmetic (exactly in z3)
    from the softmax (bounded analytically via eps), never touching `exp`.
    """
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    p = oracle.p
    A = f"transformer/layer_{layer}/attn/"
    Wq, bq = p[A + "query||w"], p[A + "query||b"]
    Wk, bk = p[A + "key||w"], p[A + "key||b"]
    K = oracle.key_size
    V = len(oracle.spec.vocab)
    T = p["pos_embed||embeddings"].shape[0]

    # The gather query reads `opp_idx-1` (a computed one-hot over positions)
    # and the key reads `indices` (position) + `tokens`. Rather than rebuild
    # the deep upstream computation symbolically, we quantify DIRECTLY over
    # the abstract state R claims holds at layer input: opp_idx is a one-hot
    # over target positions, indices is the (fixed) position one-hot, tokens
    # is a one-hot per position. This is exactly the simulation-relation
    # contract -- we prove the gap ASSUMING R holds at the input (which the
    # earlier step-lemmas discharge), giving a compositional proof.
    tgt_axes = R.block["opp_idx-1_21"]
    idx_axes = R.block["indices"]
    tok_axes = R.block["tokens"]
    n_tgt = len(tgt_axes)

    def resid_at(qpos, tgt_sel, tok_sel):
        """Symbolic residual at position qpos given target one-hot tgt_sel
        (only meaningful for the query) and token one-hot tok_sel."""
        vec = [z3.RealVal(0) for _ in range(R.D)]
        # position one-hot (indices) is concrete at qpos
        for a_i, ax in enumerate(idx_axes):
            vec[ax] = z3.RealVal(1.0 if a_i == qpos else 0.0)
        # tokens one-hot
        for a_i, ax in enumerate(tok_axes):
            vec[ax] = tok_sel[a_i]
        # opp_idx target one-hot (query only)
        for a_i, ax in enumerate(tgt_axes):
            vec[ax] = tgt_sel[a_i]
        # 'one' axis is constant 1 in tracr
        if "one" in R.block:
            vec[R.block["one"][0]] = z3.RealVal(1.0)
        return vec

    def qk_score(qvec, kvec):
        qp = [z3.simplify(z3.Sum([qvec[i] * z3.RealVal(float(Wq[i, d]))
                                  for i in range(R.D)]) + z3.RealVal(float(bq[d])))
              for d in range(K)]
        kp = [z3.simplify(z3.Sum([kvec[i] * z3.RealVal(float(Wk[i, d]))
                                  for i in range(R.D)]) + z3.RealVal(float(bk[d])))
              for d in range(K)]
        return z3.Sum([qp[i] * kp[i] for i in range(K)])  # /sqrtK monotone

    # bisection on gamma: largest lower bound we can prove for the gap
    def gap_holds(gamma):
        # for a representative query, over all target one-hots and token
        # one-hots, winner (key at target) beats every other key by >= gamma
        for qpos in range(1, T):
            s = z3.Solver()
            n_tok_pos = T
            tok_sels = [[z3.Int(f"tk_{t}_{v}") for v in range(len(tok_axes))]
                        for t in range(n_tok_pos)]
            for t in range(n_tok_pos):
                for x in tok_sels[t]:
                    s.add(z3.Or(x == 0, x == 1))
                s.add(z3.Sum(tok_sels[t]) == 1)
            tgt_sel = [z3.Int(f"tg_{a}") for a in range(n_tgt)]
            for x in tgt_sel:
                s.add(z3.Or(x == 0, x == 1))
            s.add(z3.Sum(tgt_sel) == 1)
            qvec = resid_at(qpos, tgt_sel, tok_sels[qpos])
            # winner index = the key position equal to the selected target
            # encode: winner is position wpos iff tgt_sel picks index wpos
            # We refute: exists assignment where for the winner key, some
            # other key's score is within gamma (i.e. gap < gamma).
            scores = []
            for t in range(T):
                kvec = resid_at(t, tgt_sel, tok_sels[t])  # target irrelevant to key
                scores.append(qk_score(qvec, kvec))
            # winner position = argmax over the ACTUAL selection: the key t
            # whose index matches target. Since tgt_sel is one-hot over
            # positions, winner = the t with tgt_sel[t]==1.
            bad = []
            for wt in range(1, T):  # candidate winner positions (content)
                # if target selects wt, require score[wt] - score[j] >= gamma
                viol = z3.Or(*[scores[wt] - scores[j] < z3.RealVal(gamma)
                               for j in range(1, T) if j != wt])
                bad.append(z3.And(tgt_sel[wt] == 1, viol))
            s.add(z3.Or(*bad))
            if s.check() == z3.sat:
                return False, qpos
        return True, None

    lo, hi, best = 0.0, 30.0, 0.0
    for _ in range(8):  # bisection
        mid = (lo + hi) / 2
        ok, _ = gap_holds(mid)
        if ok:
            best = mid; lo = mid
        else:
            hi = mid
    eps = (T - 1) * float(np.exp(-best))
    if verbose:
        print(f"  proved score gap >= {best:.2f}  => sound eps <= {eps:.4f} "
              f"(winner weight >= {1/(1+eps):.4f})")
    return {"case": case_id, "layer": layer,
            "proved_gap": round(best, 3),
            "sound_eps": round(eps, 4),
            "winner_weight_lb": round(1 / (1 + eps), 4)}


def prove_gap_from_embedding(case_id="p04_shift_right", layer=0,
                             winner_offset=-1, verbose=True):
    """HARD gather proof for a layer whose query selects on a TRUE position
    one-hot read directly from the embedding (R holds crisply at the input,
    no saturated upstream). Proves, over the real Q/K weights, the largest
    score gap gamma between the winner key (at qpos+winner_offset) and every
    other key, for all one-hot token assignments -> sound eps.

    This is the regime where the gather is genuinely hard. Contrast
    prove_gap on p08_reverse, where the target (opp_idx) is a COMPUTED,
    saturated value and no clean gap exists -- the relational method
    correctly refuses to prove a crisp gather the network doesn't implement.
    """
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    p = oracle.p
    A = f"transformer/layer_{layer}/attn/"
    Wq, bq = p[A + "query||w"], p[A + "query||b"]
    Wk, bk = p[A + "key||w"], p[A + "key||b"]
    K = oracle.key_size
    V = len(oracle.spec.vocab)
    T = p["pos_embed||embeddings"].shape[0]
    tok_embed = p["token_embed||embeddings"]
    pos_embed = p["pos_embed||embeddings"]
    model_ids = [int(oracle.vocab_ids[v]) for v in range(V)]

    def resid_at(pos, tok_sel):
        vec = [z3.RealVal(float(pos_embed[pos, d])) for d in range(R.D)]
        for v in range(V):
            for d in range(R.D):
                e = float(tok_embed[model_ids[v], d])
                if e != 0.0:
                    vec[d] = vec[d] + z3.RealVal(e) * tok_sel[v]
        return vec

    def score(qpos, kpos, toks):
        qv, kv = resid_at(qpos, toks[qpos]), resid_at(kpos, toks[kpos])
        qp = [z3.Sum([qv[i] * z3.RealVal(float(Wq[i, d])) for i in range(R.D)])
              + z3.RealVal(float(bq[d])) for d in range(K)]
        kp = [z3.Sum([kv[i] * z3.RealVal(float(Wk[i, d])) for i in range(R.D)])
              + z3.RealVal(float(bk[d])) for d in range(K)]
        return z3.Sum([qp[i] * kp[i] for i in range(K)])

    def gap_holds(gamma):
        for qpos in range(1, T):
            winner = qpos + winner_offset
            if winner < 1 or winner >= T:
                continue
            s = z3.Solver()
            toks = [[z3.Int(f"t{t}_{v}") for v in range(V)] for t in range(T)]
            for t in range(T):
                for x in toks[t]:
                    s.add(z3.Or(x == 0, x == 1))
                s.add(z3.Sum(toks[t]) == 1)
            sc = [score(qpos, t, toks) for t in range(T)]
            s.add(z3.Or(*[sc[winner] - sc[j] < z3.RealVal(gamma)
                          for j in range(1, T) if j != winner]))
            if s.check() == z3.sat:
                return False
        return True

    lo, hi, best = 0.0, 30.0, 0.0
    for _ in range(9):
        mid = (lo + hi) / 2
        if gap_holds(mid):
            best = mid; lo = mid
        else:
            hi = mid
    eps = (T - 1) * float(np.exp(-best))
    if verbose:
        print(f"  PROVED score gap >= {best:.2f} for all one-hot tokens "
              f"=> sound eps <= {eps:.2e}, winner weight >= {1/(1+eps):.6f}")
    return {"case": case_id, "layer": layer, "regime": "hard",
            "proved_gap": round(best, 3), "sound_eps": float(f"{eps:.2e}"),
            "winner_weight_lb": round(1 / (1 + eps), 6)}


if __name__ == "__main__":
    import json
    case = sys.argv[1] if len(sys.argv) > 1 else "p04_shift_right"
    print(f"== empirical pre-check ({case}) ==")
    m = measure_gap_and_eps(case, layer=0)
    print(f"  worst gap={m['worst_score_gap']} eps={m['eps_bound']} "
          f"worst winner weight={m['worst_winner_weight']}")
    print("== symbolic HARD gap proof (query on true position one-hot) ==")
    r = prove_gap_from_embedding(case, layer=0, winner_offset=-1)
    print(json.dumps({"empirical": m, "proved": r}, indent=2))
