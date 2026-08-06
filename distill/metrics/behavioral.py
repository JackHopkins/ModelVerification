"""Behavioral faithfulness metrics: program vs oracle on sampled inputs."""

import numpy as np


def decode_oracle_outputs(oracle, out):
    """Map oracle seq_cat index outputs to value space (object array)."""
    if oracle.spec.kind == "seq_cat":
        vals = np.array(oracle.spec.output_values, dtype=object)
        return vals[out]
    return out


def disagreement_mask(oracle, prog_out, oracle_out, tol=1e-3):
    """Elementwise disagreement; program None/nan positions are wildcards.
    Returns bool array shaped like the outputs (per position or per row)."""
    kind = oracle.spec.kind
    if kind == "seq_num":
        d = np.abs(prog_out - oracle_out) > tol
        return np.where(np.isnan(prog_out), False, d)
    if kind == "seq_cat":
        oracle_vals = decode_oracle_outputs(oracle, oracle_out)
        neq = prog_out != oracle_vals
        wild = np.frompyfunc(lambda v: v is None, 1, 1)(prog_out).astype(bool)
        return neq & ~wild
    return prog_out != oracle_out  # classify / lm argmax


def row_disagreement(oracle, prog_out, oracle_out, tol=1e-3):
    m = disagreement_mask(oracle, prog_out, oracle_out, tol)
    return m if m.ndim == 1 else m.any(axis=tuple(range(1, m.ndim)))


def behavioral_metrics(program, oracle, rng, n=20000, tol=None):
    tol = oracle.spec.meta.get("tol", 1e-3) if tol is None else tol
    out = {}
    pos_agree, row_agree, tvs, n_rows = [], [], [], 0
    for content in oracle.sample_batches(rng, n):
        o_out = oracle.outputs(content)
        p_out = program.outputs(content)
        mask = disagreement_mask(oracle, p_out, o_out, tol)
        pos_agree.append(1 - mask.mean())
        rows = row_disagreement(oracle, p_out, o_out, tol)
        row_agree.append(1 - rows.mean())
        n_rows += len(content)
        o_p = oracle.probs(content)
        if o_p is not None:
            p_p = program.probs(content)
            if p_p is not None and p_p.shape == o_p.shape:
                tvs.append(0.5 * np.abs(p_p - o_p).sum(-1).mean())
    out["position_agreement"] = float(np.mean(pos_agree))
    out["sequence_agreement"] = float(np.mean(row_agree))
    if tvs:
        out["mean_tv"] = float(np.mean(tvs))
    out["n_sampled"] = int(n_rows)
    return out
