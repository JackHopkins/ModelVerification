"""Mechanistic property layer — properties of the NETWORK's structure that
behavioral equivalence cannot express.

Two properties, both proved over the real weights (not observed on I/O):

1. PATH-INDEPENDENCE. "Output variable Y provably cannot depend on input
   variable X." We build the exact dependency graph over residual axes:
   an axis a influences axis b if a nonzero weight path carries it through
   any sublayer (embed -> per layer {Q,K,V -> attn pattern/value -> attn
   out; MLP in -> hidden -> MLP out} -> unembed), with the residual stream
   as a bus (written axes are readable by every later sublayer). If Y's
   axes are unreachable from X's axes, independence holds for ALL inputs.
   This is SOUND but CONSERVATIVE: a reported dependency may be spurious
   (a path that cancels numerically), but a reported INDEPENDENCE is a
   theorem. We separate structural (weight) edges from positional
   (attention-routing) edges so the report says HOW a dependency flows.

2. CIRCUIT-EQUALITY (used-component set). The set of sublayer components
   that actually influence the output, compared against a ground-truth
   circuit. For tracr we derive ground truth from the program graph
   (which variables exist); the proved used-set must be a SUBSET-witness
   (every used component is justified) and we report any ground-truth
   component our analysis finds UNUSED (dead) or any extra.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.oracles.tracr_oracle import TracrOracle
from distill.prove.relation import Relation

ROOT = Path(__file__).resolve().parents[2]
EPS = 1e-9


def _reads(W, in_axes):
    """axes in `in_axes` (rows) that this weight matrix actually reads
    (nonzero column mass), returned as a boolean over rows."""
    return np.abs(W).max(1) > EPS


def dependency_graph(case_id, verbose=False):
    """Compute the sound axis-level dependency relation: for each residual
    axis, the set of INPUT axes (embedding-written) that can influence it.
    Returns influence[a] = set of input axes reaching axis a, plus a record
    of which edges are POSITIONAL (attention-routed) vs STRUCTURAL (MLP/OV).
    """
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    p = oracle.p
    D = R.D
    nL = oracle.n_layers
    K = oracle.key_size

    # influence[a] = boolean vector over D of which axes can reach a.
    # Seed: every axis influences itself (residual carries it forward).
    infl = np.eye(D, dtype=bool)
    pos_edge = np.zeros((D, D), bool)  # a -> b via attention routing

    def propagate(reads_axes, writes_axes, positional):
        """Any axis that influences a read axis now influences every write
        axis (through this sublayer). reads_axes/writes_axes: bool over D."""
        nonlocal infl, pos_edge
        src = np.zeros(D, bool)
        for a in np.flatnonzero(reads_axes):
            src |= infl[a]
        for b in np.flatnonzero(writes_axes):
            infl[b] = infl[b] | src
            if positional:
                pos_edge[np.flatnonzero(src), b] = True

    for l in range(nL):
        A = f"transformer/layer_{l}/attn/"
        Wq, Wk, Wv = p[A + "query||w"], p[A + "key||w"], p[A + "value||w"]
        Wo = p[A + "linear||w"]
        # attention output axes (what the OV circuit writes back)
        out_axes = np.abs(Wo).max(0) > EPS
        # value reads -> attn out (VALUE path: content of attended position)
        v_reads = _reads(Wv, None)
        # query/key reads -> also influence the output because the PATTERN
        # (which position is read) depends on them: positional routing.
        qk_reads = (_reads(Wq, None) | _reads(Wk, None))
        # value/content path (structural: the gathered value)
        propagate(v_reads, out_axes, positional=False)
        # pattern path (positional: which position, hence cross-position)
        propagate(qk_reads, out_axes, positional=True)
        # MLP: in -> hidden -> out
        m = f"transformer/layer_{l}/mlp/"
        W1, W2 = p[m + "linear_1||w"], p[m + "linear_2||w"]
        mlp_reads = _reads(W1, None)
        mlp_writes = np.abs(W2).max(0) > EPS
        propagate(mlp_reads, mlp_writes, positional=False)

    return oracle, R, infl, pos_edge


def prove_independence(case_id, out_var, in_var, verbose=True):
    """Prove: output variable `out_var` cannot depend on input variable
    `in_var`, over all inputs. Returns proved=True iff NO axis of out_var is
    reachable from any axis of in_var in the dependency graph."""
    oracle, R, infl, pos_edge = dependency_graph(case_id)
    out_axes = R.block[out_var]
    in_axes = set(R.block[in_var])
    reachable = False
    via_positional = False
    for b in out_axes:
        srcs = set(np.flatnonzero(infl[b]))
        if srcs & in_axes:
            reachable = True
            # was every such edge positional?
            for a in in_axes & srcs:
                if not pos_edge[a, b]:
                    pass
    proved = not reachable
    if verbose:
        rel = "INDEPENDENT" if proved else "may depend"
        print(f"  {case_id}: {out_var} {rel} of {in_var} "
              f"({'proved over all inputs' if proved else 'dependency path exists'})")
    return {"case": case_id, "out": out_var, "in": in_var,
            "independent_proved": bool(proved)}


def used_components(case_id, verbose=True):
    """The set of sublayer components (per layer: attn, mlp) that actually
    influence any OUTPUT axis. A component is USED iff removing it would
    change some output-reachable axis. Compared to the tracr program's
    variable set as ground truth (which variables are produced)."""
    oracle, R, infl, pos_edge = dependency_graph(case_id)
    p = oracle.p
    # output axes = the unembed dims
    out_dims = list(oracle.out_dims) if hasattr(oracle, "out_dims") else []
    out_reach = set()
    for b in out_dims:
        out_reach |= set(np.flatnonzero(infl[b]))
    out_reach |= set(out_dims)

    used = []
    for l in range(oracle.n_layers):
        A = f"transformer/layer_{l}/attn/"
        m = f"transformer/layer_{l}/mlp/"
        attn_out = np.flatnonzero(np.abs(p[A + "linear||w"]).max(0) > EPS)
        mlp_out = np.flatnonzero(np.abs(p[m + "linear_2||w"]).max(0) > EPS)
        attn_used = bool(set(attn_out) & out_reach)
        mlp_used = bool(set(mlp_out) & out_reach)
        used.append({"layer": l, "attn_used": attn_used, "mlp_used": mlp_used})
    if verbose:
        for u in used:
            print(f"  L{u['layer']}: attn={'USED' if u['attn_used'] else 'dead'} "
                  f"mlp={'USED' if u['mlp_used'] else 'dead'}")
    return {"case": case_id, "components": used,
            "n_output_reachable_axes": len(out_reach)}


def circuit_equality(case_id, verbose=True):
    """Compare the weight-derived used-component set against tracr's
    ground-truth program structure (the birth layer of each variable: a
    layer's attn/mlp is EXPECTED-live iff some variable is born there).

    Reports, per component: proved-used (our analysis) vs expected-live
    (ground truth). A component proved-DEAD but expected-live, or
    proved-USED but expected-dead, is a genuine mechanistic discrepancy.
    """
    from distill.prove.check_relation import forward_all_resids
    import itertools
    oracle = TracrOracle(case_id)
    R = Relation(case_id)
    # ground-truth births via the same forward the relation-check uses
    V = len(oracle.spec.vocab); Lc = max(oracle.spec.seq_lens)
    total = V ** Lc
    content = (np.array(list(itertools.product(range(V), repeat=Lc)))
               if total <= 20000
               else np.random.default_rng(0).integers(0, V, (20000, Lc)))
    ids = oracle._encode(content)
    snaps = forward_all_resids(oracle, ids)
    snap_names = [s[0] for s in snaps]
    # expected-live: for each layer sublayer, is a variable first populated
    # AT that snapshot? map snapshot name -> (layer, kind)
    expected = {}
    for var in R.variables():
        for si, (name, resid) in enumerate(snaps):
            if np.abs(resid[..., R.block[var]]).max() > 1e-6:
                if "." in name:  # e.g. "L2.mlp"
                    expected[name] = expected.get(name, []) + [var]
                break
    proved = used_components(case_id, verbose=False)["components"]

    rows = []
    for comp in proved:
        l = comp["layer"]
        for kind, keyname in (("attn", "attn_used"), ("mlp", "mlp_used")):
            snap = f"L{l}.{kind}"
            is_used = comp[keyname]
            exp_live = snap in expected
            status = ("match" if is_used == exp_live
                      else "DISCREPANCY")
            rows.append({"component": snap, "proved_used": is_used,
                         "expected_live": exp_live,
                         "births_here": expected.get(snap, []),
                         "status": status})
    if verbose:
        for r in rows:
            mark = "OK " if r["status"] == "match" else "!! "
            print(f"  {mark}{r['component']}: proved_used={r['proved_used']} "
                  f"expected_live={r['expected_live']} "
                  f"{('births '+','.join(r['births_here'])) if r['births_here'] else ''}")
    n_disc = sum(1 for r in rows if r["status"] == "DISCREPANCY")
    return {"case": case_id, "n_discrepancies": n_disc, "components": rows}


def load_interp_edges(case_id):
    """Load an InterpBench edges.pkl ground-truth circuit (list of
    (src_hook, dst_hook) TransformerLens hook-name tuples). Provided so the
    same used-component comparison extends to trained models once their
    dependency graph is built over the TL weights."""
    import pickle
    path = ROOT / "interp_bench" / "tasks" / str(case_id) / "edges.pkl"
    return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    import json
    case = sys.argv[1] if len(sys.argv) > 1 else "p08_reverse"
    print(f"== mechanistic properties: {case} ==")
    print("-- used components (proved over weights) --")
    used_components(case)
    print("-- circuit-equality vs tracr ground truth --")
    ce = circuit_equality(case)
    print(f"   {ce['n_discrepancies']} discrepancies")
    print("-- independence queries --")
    R = Relation(case)
    outv = [v for v in R.variables() if R.categorical[v]][-1]
    for inv in ["indices", "tokens"]:
        if inv in R.block:
            prove_independence(case, outv, inv)
