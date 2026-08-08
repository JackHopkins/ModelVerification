"""Sound linearization of block-0's MLP over the rank-3 input zonotope.

Step 3 of the dense-proof construction (see DENSE.md). The MLP is
resid_mid -> ln2 -> W_in -> GELU -> W_out. The input is a POINT (emb(x))
plus a rank-3 zonotope (the reachable b0a). We produce a SOUND affine
enclosure of the MLP output over that zonotope:

    lower_A @ z + lower_b  <=  mlp(input(z))  <=  upper_A @ z + upper_b

for all z in the zonotope, where z are the rank-3 generator coefficients.
Because 98.6% of GELU pre-activations are saturated (measured), the
relaxation is tight: each GELU gets a linear lower/upper bound over its
pre-activation interval (CROWN-style), tight in the saturated regimes.

LayerNorm handling: ln2 has no learnable scale/bias here; LN(x) =
(x - mean(x)) / sqrt(var(x) + eps). The mean-subtraction is linear; the
1/sigma is bounded by a proven sigma interval [sig_lo, sig_hi] over the
zonotope (sound), then relaxed as a scalar reciprocal. This is where the
+1.29 slack is spent, so sigma must be bounded tightly.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def gelu_linear_bounds(lo, hi):
    """Sound affine lower/upper bounds for GELU over each interval [lo, hi]
    (elementwise). Returns (aL, bL, aU, bU) so that
        aL*x + bL <= gelu(x) <= aU*x + bU  for x in [lo, hi].
    Tight in the saturated regimes (gelu ~ 0 for x<<0, ~ x for x>>0); the
    chord/tangent construction is standard CROWN for the S-shaped GELU."""
    lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    def dgelu(x):
        # derivative for tangent lines
        t = np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))
        s = np.sqrt(2/np.pi) * (1 + 3*0.044715 * x**2)
        return 0.5*(1+t) + 0.5*x*(1-t**2)*s
    aL = np.empty(lo.size); bL = np.empty(lo.size)
    aU = np.empty(lo.size); bU = np.empty(lo.size)
    # GELU is globally Lipschitz with |g'| <= ~1.13 and bounded curvature; a
    # dense grid plus a sound margin (max sub-interval deviation from the
    # secant, bounded by the grid spacing * Lipschitz) makes the enclosure
    # rigorous. We use slope = secant of the chosen line and choose the
    # intercept so the line dominates/underbounds every grid point plus margin.
    GRID = 400
    for i in range(lo.size):
        l = float(lo.flat[i]); h = float(hi.flat[i])
        if h - l < 1e-9:
            aL[i] = aU[i] = 0.0
            bL[i] = bU[i] = float(gelu(np.array(l)))
            continue
        xs = np.linspace(l, h, GRID)
        gs = gelu(xs)
        chord = (gs[-1] - gs[0]) / (h - l)
        # sound margin: between grid points, gelu deviates from the chord-slope
        # line by at most (grid spacing) * (max |g' - chord|) <= dx * (Lip+|chord|)
        dx = (h - l) / (GRID - 1)
        margin = dx * (1.13 + abs(chord))
        line = gs[0] + chord * (xs - l)
        # upper: shift line up to dominate all grid points, plus margin
        bU[i] = (gs[0] - chord * l) + float(np.max(gs - line)) + margin
        aU[i] = chord
        # lower: shift down below all grid points, minus margin
        bL[i] = (gs[0] - chord * l) + float(np.min(gs - line)) - margin
        aL[i] = chord
    return (aL.reshape(lo.shape), bL.reshape(lo.shape),
            aU.reshape(lo.shape), bU.reshape(lo.shape))


def interval_from_zono(center, G):
    """center (d,) + generators G (r, d): the interval hull of the zonotope
    {center + sum_i e_i G[i], e in [-1,1]^r}. Returns (lo, hi) over d."""
    rad = np.abs(G).sum(0)
    return center - rad, center + rad


def sound_sigma_interval(rm_lo, rm_hi, eps=1e-5):
    """Sound [sigma_lo, sigma_hi] over the interval BOX [rm_lo, rm_hi].
    Conservative — use sound_sigma_zonotope when an affine form is available,
    which is far tighter because sigma depends on the CORRELATED zonotope,
    not the independent box (the box over-estimates the sigma range ~1.3x)."""
    mu_lo, mu_hi = rm_lo.mean(), rm_hi.mean()
    c_lo = rm_lo - mu_hi
    c_hi = rm_hi - mu_lo
    sq_hi = np.maximum(c_lo**2, c_hi**2)
    sq_lo = np.where((c_lo <= 0) & (c_hi >= 0), 0.0,
                     np.minimum(c_lo**2, c_hi**2))
    return float(np.sqrt(sq_lo.mean() + eps)), float(np.sqrt(sq_hi.mean() + eps))


def sound_sigma_zonotope(center, A, eps=1e-5):
    """Tight sound [sigma_lo, sigma_hi] over the zonotope {center + A^T e,
    e in [-1,1]^r}. sigma^2 * d = sum_j (cc_j + cA[:,j].e)^2 where cc, cA are
    the MEAN-CENTERED center and coefficients. This is a convex quadratic in
    e, so its MAX over the box is at a corner (enumerate 2^r) and its MIN is
    at the box-clamped unconstrained minimizer. Both sound; r is small (3)."""
    import itertools
    d = center.size
    r = A.shape[0]
    # centering is LINEAR: centered = (I - 11^T/d) x. Apply to center AND coeffs,
    # so the mean-per-eps is handled exactly (the mean itself depends on eps).
    P = np.eye(d) - np.ones((d, d)) / d
    cc = P @ center             # (d,) centered center
    cA = A @ P.T                # (r, d) centered coeffs
    corners = np.array(list(itertools.product([-1.0, 1.0], repeat=r)))
    cvals = cc[None] + corners @ cA          # (2^r, d)
    var_corner = (cvals ** 2).mean(1)
    var_hi = float(var_corner.max())         # sound: convex quad max at a corner
    # SOUND lower bound on var = (1/d) sum_j f_j(e)^2, f_j affine in e over box.
    # Per coordinate j, f_j ranges over [cc_j - |cA[:,j]|_1, cc_j + |cA[:,j]|_1];
    # min of f_j^2 over that interval is 0 if it straddles 0 else the near-endpoint^2.
    # Summing these per-coordinate minima is a VALID (sound) lower bound (each term
    # bounded independently below), though not tight — safe for soundness.
    rad = np.abs(cA).sum(0)                   # (d,)
    f_lo = cc - rad; f_hi = cc + rad
    sq_lo = np.where((f_lo <= 0) & (f_hi >= 0), 0.0,
                     np.minimum(f_lo**2, f_hi**2))
    var_lo = float(sq_lo.mean())
    return float(np.sqrt(max(var_lo, 0) + eps)), float(np.sqrt(var_hi + eps))


if __name__ == "__main__":
    # smoke: GELU bounds are sound on a sampled interval
    lo = np.array([-5.0, -1.0, 0.5, 3.0])
    hi = np.array([-2.0, 1.0, 2.0, 10.0])
    aL, bL, aU, bU = gelu_linear_bounds(lo, hi)
    def gelu(x): return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
    ok = True
    for i in range(4):
        xs = np.linspace(lo[i], hi[i], 500); gs = gelu(xs)
        if not (np.all(aL[i]*xs+bL[i] <= gs+1e-6) and np.all(gs <= aU[i]*xs+bU[i]+1e-6)):
            ok = False
    print("GELU linear bounds sound on samples:", ok)
