# Attacking the dense proof problem — findings & plan

Goal: sound `forall`-theorems about TRAINED (dense) models, not the
distributional causal attribution `interp_graph.py` gives. Target chosen:
discrete-token input space (InterpBench), decompiler as invariant oracle.

## The three obstacles (separated — they need different attacks)

1. **No sparse dependency structure.** Dense weights: every component
   reads every residual dim. No weight-graph to trace (unlike tracr).
2. **Softmax over a continuum.** The simplex-envelope trick works for ONE
   attention layer; the hull's vertex count explodes when layers compose.
3. **Quantifier over a continuous region.** InterpBench inputs are still
   DISCRETE tokens, so we dodge this one for now — "∀ inputs" stays
   finitary. Real continuous-embedding models reintroduce it.

We target obstacles 1+2 while input stays discrete, then generalize.

## The theorem, stated precisely (case 8, "Identity")

`edges.pkl` circuit: embed → block0.mlp → out (block0.attn, block1.*
claimed dead). Verified empirically:
- Mean-ablating the dead set: only 90% argmax agreement, logit shift 41.
  **Mean-ablation is the WRONG baseline** (off-distribution, breaks
  softmax) — same lesson as the attribution work.
- **Resample-ablation** (replace dead set with another input x''s
  activations): **100% argmax invariance**. So the sound theorem is:

    ∀ x, ∀ x': argmax(full(x)) == argmax(ablate_{dead←x'}(x))

  a ∀ over ~10^18 (x,x') pairs — not enumerable, needs proof.

## The load-bearing obstacle (measured — TWO wrong hypotheses, then the truth)

Building the propagator on case 8 falsified two intuitive framings before
finding the real one. Recording all three because the dead ends are the
lesson.

**Hypothesis 1 (cancellation):** the dead set contributes a logit swing of
~1177 but the (winner−runner-up) DIFFERENCE cancels. Measured: the naive
sum-of-ranges bound is 134 (25× the 5.28 margin); the difference-range
bound is 29 (5.6× margin) — better, but still doesn't close.

**Hypothesis 2 (fixed bias):** the dead set's logit contribution has huge
class-spread (1903) but tiny input-std (~13), so it looked like a nearly
constant per-class bias that resampling barely moves. The per-input
decision-subspace slack under this model came out **−1882 and closed for
only 10% of inputs — flatly contradicting the empirically VERIFIED 100%
argmax invariance.** A bound that contradicts ground truth is wrong.

**The truth (LayerNorm coupling):** the contradiction exposed the real
mechanism. `normalization_type` is None (no final LN) but every block has
ln1/ln2. Patching the "dead" `blocks.0.hook_attn_out` changes the "live"
`blocks.0.hook_mlp_out` by **296** — because b0m reads `resid_mid = emb +
b0a` THROUGH ln2. **The dead set is not additively separable from the
circuit.** "Dead component contributes X to the logits" is ILL-DEFINED in a
residual net with LayerNorm: ablating a component changes what every
downstream component computes, via the normalizer's dependence on the total
residual. My decision-subspace bound treated dead and circuit logit
contributions as independent — hence the nonsense slack.

## Consequence for the attack (revised)

A sound bound **cannot decompose the network into "circuit + dead" and
bound the dead part.** It must propagate the JOINT effect of the resample
through the real nonlinear layers (the LayerNorms especially). This is
exactly the composing-bound-propagation problem (obstacle 2), and it is
unavoidable — the additive shortcut that made the tracr proofs clean does
not survive LayerNorm.

Concretely the harness must:
- Represent the resample intervention as an INPUT BOX on the ablated hooks
  (their reachable activation range over all x'), and propagate it forward
  through ln1/ln2/attn/mlp of the downstream layers with sound relaxations
  (CROWN-style linear bounds on LayerNorm and GELU), to a bound on the
  output (winner−runner-up) gap.
- The decompiler/SAE invariant then tightens the LayerNorm relaxation by
  constraining the residual to the task-relevant subspace (where the
  normalizer's denominator is well-approximated), rather than the full
  d_model box — the load-bearing use of the oracle.

## Plan

- **Stage A' (harness):** CROWN-style linear bound propagation through the
  2 blocks, with certified LayerNorm + GELU relaxations. Input = the
  resample box on the dead hooks; output = bound on the decision gap.
  Baseline (no invariant): expect it too loose to close (quantifies the
  problem honestly).
- **Stage C:** decompiler-supplied subspace invariant tightens the LN
  relaxation; CEGIS-discharge the invariant with the same propagator.

## CROWN-through-LayerNorm scouting (interval probe, sound-sigma intent)

Structural map of case 8 (circuit embed->block0.mlp->out):
- **block0.attn (b0a)** feeds block0's ln2 -> block0.mlp (the CIRCUIT). This
  is the ONLY LayerNorm-coupled perturbation. Its box width is ~59.
- **block1.attn + block1.mlp** are written to resid_post AFTER the circuit
  MLP and feed NO further LayerNorm (normalization_type None, no final LN),
  so they enter the logits by a PURE LINEAR map. Box widths huge (~3000).

Sigma reality: block0.ln2 sees resid_mid with norm ~88-130 and
**sigma in [34.9, 60.8], ratio 1.74** (NOT the hoped ~1.10 — d_model is only
4, so sigma is a mean over 4 dims and swings). The tight-sigma assumption
does not hold for free; it would need the subspace invariant to justify.

**THE decisive measurement (after fixing a 3x-repeated methodology error).**
Three successive bounds gave nonsense (134, 29, -1882 slack) because they
maximized worst-case over class-pairs and inputs INDEPENDENTLY — but the
real quantity is per-input and CORRELATED. Computed correctly: resample
block1 from 200 sources, measure the actual change to the (winner - its own
runner-up) gap per input:
- **worst gap-change = 12.8; margin in [5.3, 8.7]; per-input ratio maxes at
  0.89 < 1.** The gap-change never exceeds the margin — THAT is why argmax
  is invariant, and it is TIGHT (0.89, almost no slack).

Implication (strategic): the property is not loose-and-blown-up; it is
**tight and barely holds**. A sound bound must be accurate to within ~15%
or it fails. So the challenge is not taming blowup (the usual dense-proof
story) but achieving near-exact bounds on a genuinely marginal property.
This raises the bar for the CROWN relaxations (LN + GELU + softmax must each
stay very tight) and makes the decompiler subspace-invariant essentially
mandatory, not merely helpful.

LESSON (recorded): every worst-case bound over a coupled network must be
computed per-decision (winner vs its own runner-up) and account for
correlated movement; independent class/input maximization over-counts by
100x+ and produces bounds that CONTRADICT verified ground truth. Validate
any bound against the known-true property before trusting it.

## CROWN harness attempt — the domain, not the nonlinearity, is the wall

Built the verified circuit-forward reconstruction (emb + b0a + block0.mlp(
ln2(emb+b0a)) + b1, exact to 1e-4 vs the real patched forward). Then probed
the achievable tightness by propagating the resample box through the REAL
block0 MLP:

- **Independent hyperbox on (b0a, b1) → output gap = -210 (argmax FLIPS).**
  But the real resample never flips (0/60000 verified). The hyperbox admits
  impossible worlds. (A probe bug — resampling b0a and b1 from INDEPENDENT
  indices instead of the same x' — briefly suggested the theorem was false;
  direct full-forward patching confirms 0 flips, and the fixed
  reconstruction agrees exactly with real patching. Theorem holds.)

- **Root cause (decisive):** the resampled components are JOINTLY CORRELATED
  (all from one x'). PCA on the nominally-12-dim joint (b0a,b1a,b1m) manifold:
  **rank 3 captures 99% of variance; 84.8% is in ONE direction.** The
  reachable set is a 3-dim correlated manifold, not a 12-dim box. The
  hyperbox over-approximates by treating 12 independent axes → the -210.

**So the wall is the DOMAIN, not the LayerNorm nonlinearity.** LN is only
4-dim and per-input tractable. The killer is that the sound abstract domain
must capture the low-rank correlation of the reachable activations;
axis-aligned intervals cannot.

## Pivot: correlated/feature-space domain (the promising direction)

The fix is a domain that carries the correlation — a zonotope/affine domain
over the rank-3 subspace, or better, a FEATURE-space basis where the
connectivity is sparse:
- **Virtual weights** (SAE decoder ∘ W_OV/W_QK): re-express computation in a
  feature basis where feature→feature connectivity is sparse and weight-
  mediated — the sparse dependency structure obstacle-1 said dense nets
  lack. A domain over these features captures the correlation the neuron-
  hyperbox misses.
- **Jacobian-sparse SAEs** (train sparsity on the feature→feature Jacobian):
  the stronger version, and the RIGHT one for verification — the Jacobian
  IS the local linear map CROWN consumes, so a sparse Jacobian means few
  cross-terms and tight bounds. Manufactures the structure verification
  needs on the CONNECTIVITY (what CROWN uses), not the parameters (which LN
  scrambles) — the key advantage over SPD for this purpose.

The rank-3 measurement is direct evidence this can work: the correlation is
real and low-dimensional, so a correlated domain should be tight where the
hyperbox is loose by 40x.

## Status

Domain wall identified and quantified (rank-3 reachable manifold vs 12-dim
box, -210 vs verified-holds). The next build is a CORRELATED domain (affine/
zonotope over the low-rank subspace, or an SAE/virtual-weight feature basis
with sparse Jacobian), NOT a tighter interval propagator — intervals cannot
represent the correlation that makes the property true. This reframes the
harness around the domain, which is the actual load-bearing choice.
