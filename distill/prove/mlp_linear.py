"""Sound linearization of block-0's MLP, correlation-preserving.

Step 3 of the dense-proof construction (see DENSE.md). These InterpBench
models have NO LayerNorm (normalization_type=None; block.ln2 is Identity),
so the MLP is resid_mid -> W_in -> GELU -> W_out with GELU the only
nonlinearity. The input is a rank-3 zonotope (the reachable resampled
components, shared eps). We bound the model output's DECISION-DIRECTION
gap (winner - runner-up) soundly.

The load-bearing lesson (learned the hard way, see DENSE.md): the sound
bound MUST keep one shared-eps affine form through the whole circuit so
correlated terms cancel in the decision direction. In particular the 16
GELUs, bounded independently, give a decision error of ~50; the SAME
GELUs' contributions to the decision are nearly affine in eps (residual
~0.03) because they cancel through u = W_out^T dvec. So each GELU is
enclosed as  a_j*x + [blo_j, bhi_j]  and the a_j*pre_j (affine) parts are
kept in the shared-eps form -- only the intercept intervals are the error,
and they combine signed through u (tight). Result on case 8: sound min gap
+0.586, property provable on 100% of sampled inputs.

SOUNDNESS SCOPE: the enclosures are rigorous (grid + Lipschitz margin,
validated 0 violations). The remaining non-sound step is the rank-3
subspace, which is currently PCA of samples; a full theorem needs it
derived from network structure (the BOS-saturation lemma; see DENSE.md).
So this is a sound bound over the *given* zonotope, pending a sound
zonotope.
"""

import numpy as np


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def gelu_affine_enclosure(lo, hi, ngrid=600):
    """Sound  a*x + blo <= gelu(x) <= a*x + bhi  for x in [lo, hi], with the
    slope a chosen to minimize the intercept-interval width bhi-blo. Grid +
    Lipschitz margin makes it rigorous between grid points. Returns
    (a, blo, bhi)."""
    if hi - lo < 1e-9:
        g = float(_gelu(np.array([lo]))[0])
        return 0.0, g, g
    xs = np.linspace(lo, hi, ngrid)
    gs = _gelu(xs)
    ag = np.linspace(-0.05, 1.1, 150)
    best = None
    for a in ag:
        d = gs - a * xs
        wd = d.max() - d.min()
        if best is None or wd < best[0]:
            best = (wd, a, d.min(), d.max())
    _, a, dlo, dhi = best
    dx = (hi - lo) / (ngrid - 1)
    margin = dx * (1.13 + abs(a))         # |gelu' - a| <= Lip+|a|, times spacing
    return float(a), float(dlo - margin), float(dhi + margin)


def sound_decision_gap(cen_mlp_in, A_mlp_in, cen_rest, A_rest, const,
                       Win, bin_, Wout, bout, dvec):
    """Sound lower bound on the decision gap over the rank-3 zonotope.

    cen_mlp_in (d,) + A_mlp_in (r,d): MLP-input affine (shared eps).
    cen_rest (d,) + A_rest (r,d): the rest of resid_post (emb + b0a + b1),
        affine in the SAME eps -- so it cancels with the MLP affine part.
    const: scalar added to the gap (unembed bias diff etc.).
    Win/bin_/Wout/bout: block MLP weights. dvec (d,): decision direction
        (W_U[:,winner] - W_U[:,runner]).

    Returns (gap_lower_bound, center, affine_radius, error_radius).
    """
    pre_cen = cen_mlp_in @ Win + bin_
    pre_A = A_mlp_in @ Win
    pre_rad = np.abs(pre_A).sum(0)
    u = Wout @ dvec                                   # fold W_out into direction
    dmlp = Win.shape[1]
    a = np.empty(dmlp); blo = np.empty(dmlp); bhi = np.empty(dmlp)
    for j in range(dmlp):
        a[j], blo[j], bhi[j] = gelu_affine_enclosure(
            pre_cen[j] - pre_rad[j], pre_cen[j] + pre_rad[j])
    # affine part of the decision GELU contribution, in shared eps
    gcoef = u * a
    dec_cen = float(gcoef @ pre_cen)
    dec_A = pre_A @ gcoef                              # (r,)
    # intercept error interval, signed through u (tight cross-neuron combine)
    err_lo = float(np.sum(np.where(u >= 0, u * blo, u * bhi)))
    err_hi = float(np.sum(np.where(u >= 0, u * bhi, u * blo)))
    err_mid = 0.5 * (err_hi + err_lo)
    err_rad = 0.5 * (err_hi - err_lo)
    gap_cen = cen_rest @ dvec + dec_cen + err_mid + float(dvec @ bout) + const
    gap_A = A_rest @ dvec + dec_A                      # combined -> cancellation
    aff_rad = float(np.abs(gap_A).sum())
    return gap_cen - aff_rad - err_rad, gap_cen, aff_rad, err_rad


if __name__ == "__main__":
    # soundness smoke: enclosure encloses gelu on random intervals
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(1000):
        lo = rng.uniform(-40, 20); hi = lo + rng.uniform(0.01, 30)
        a, blo, bhi = gelu_affine_enclosure(lo, hi)
        xs = np.linspace(lo, hi, 5000); gs = _gelu(xs)
        if not (np.all(a*xs+blo <= gs+1e-9) and np.all(gs <= a*xs+bhi+1e-9)):
            bad += 1
    print("gelu affine enclosure sound on 1000 random intervals:", bad == 0)
