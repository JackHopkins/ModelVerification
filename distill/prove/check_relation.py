"""Empirical pre-check: does R hold across the full input space?

Runs the real forward pass, decodes each variable's block at every layer
and position, and reports where the block is NOT a clean one-hot. This
does not prove anything (it is behavioral), but it tells the symbolic
prover exactly which (variable, position) blocks are always-onehot (easy
step-lemma) vs. numerical/soft (need interval reasoning).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.oracles.tracr_oracle import TracrOracle
from distill.prove.relation import Relation


def forward_all_resids(oracle, ids):
    """Return residual stream AFTER each sublayer: list of (name, resid)."""
    n, T = ids.shape
    p = oracle.p
    resid = p["token_embed||embeddings"][ids] + p["pos_embed||embeddings"][np.arange(T)][None]
    H, K = oracle.n_heads, oracle.key_size
    snaps = [("embed", resid.copy())]
    for l in range(oracle.n_layers):
        a = f"transformer/layer_{l}/attn/"
        q = (resid @ p[a + "query||w"] + p[a + "query||b"]).reshape(n, T, H, K)
        k = (resid @ p[a + "key||w"] + p[a + "key||b"]).reshape(n, T, H, K)
        v = (resid @ p[a + "value||w"] + p[a + "value||b"]).reshape(n, T, H, K)
        scores = np.einsum("nqhk,nthk->nhqt", q, k) / np.sqrt(K)
        scores = scores - scores.max(-1, keepdims=True)
        w = np.exp(scores); w = w / w.sum(-1, keepdims=True)
        z = np.einsum("nhqt,nthk->nqhk", w, v).reshape(n, T, H * K)
        resid = resid + z @ p[a + "linear||w"] + p[a + "linear||b"]
        snaps.append((f"L{l}.attn", resid.copy()))
        m = f"transformer/layer_{l}/mlp/"
        hid = np.maximum(resid @ p[m + "linear_1||w"] + p[m + "linear_1||b"], 0)
        resid = resid + hid @ p[m + "linear_2||w"] + p[m + "linear_2||b"]
        snaps.append((f"L{l}.mlp", resid.copy()))
    return snaps


def check(case_id, max_inputs=20000):
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    V = len(oracle.spec.vocab)
    T = max(oracle.spec.seq_lens) + 1  # +BOS
    # enumerate or sample content
    import itertools
    content_len = T - 1
    total = V ** content_len
    if total <= max_inputs:
        content = np.array(list(itertools.product(range(V), repeat=content_len)))
    else:
        content = np.random.default_rng(0).integers(0, V, (max_inputs, content_len))
    ids = oracle._encode(content)
    snaps = forward_all_resids(oracle, ids)
    print(f"=== {case_id}: {len(content)} inputs, "
          f"{'EXHAUSTIVE' if total<=max_inputs else 'sampled'} ===")

    # phase: a variable's block is all-zero until its producing sublayer.
    # birth[var] = first snapshot index where the block is nonzero anywhere.
    birth = {}
    for var in R.variables():
        for si, (name, resid) in enumerate(snaps):
            if np.abs(resid[..., R.block[var]]).max() > 1e-6:
                birth[var] = si
                break
        else:
            birth[var] = len(snaps)  # never active (dead var)

    # a variable may legitimately be undefined (all-zero block) at some
    # POSITIONS even after birth (e.g. only defined at content positions).
    # Report a violation only where the block is nonzero-but-not-onehot.
    report = {}
    for si, (name, resid) in enumerate(snaps):
        for var in R.variables():
            if not R.categorical[var] or si < birth[var]:
                continue
            sub = resid[..., R.block[var]]
            active = np.abs(sub).max(-1) > 1e-6      # block is populated
            oh = R.is_onehot(resid, var)
            bad = active & ~oh                        # populated but not 1-hot
            if bad.any():
                # position-resolved: which token positions violate?
                pos_bad = np.flatnonzero(bad.any(0))
                report[(name, var)] = (round(float(bad.mean()), 4),
                                       pos_bad.tolist()[:12])
    print(f"  birth layers: " +
          ", ".join(f"{v}@{snaps[birth[v]][0] if birth[v]<len(snaps) else 'dead'}"
                    for v in R.variables() if R.categorical[v]))
    if not report:
        print("  R HOLDS: every categorical block is one-hot wherever populated,"
              " from its birth layer on.")
    else:
        print("  genuine violations (populated but not one-hot):")
        for (name, var), (f, pos) in sorted(report.items()):
            print(f"    {name:10s} {var:40s} bad_frac={f} at positions {pos}")
    return report


if __name__ == "__main__":
    check(*sys.argv[1:2] or ["p08_reverse"])
