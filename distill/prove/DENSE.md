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

## Zonotope probe — the correlated domain WORKS (green light)

Represented the resampled components as a rank-r zonotope (center + G·ε,
ε∈[-1,1]^r, G = principal axes scaled to the data extent) instead of a
12-dim box, and propagated through the real block0 MLP:

- **rank-3 zonotope: min gap +1.29, property provable on 100% of inputs.**
- rank-4: -9.28 (flips). rank-12 (≈box): -17.78 (flips).

Capturing exactly the rank-3 correlation is the difference between provable
and not. Two consequences:
1. The correlated domain is the right abstraction — confirmed, not
   hypothesized. Intervals can't; a rank-3 affine/zonotope domain can.
2. The rank must be EXACT — rank-4 already over-approximates enough to flip.
   So the domain must track the SOUND low-rank structure precisely. PCA of
   samples is not sound; this is where the Jacobian-sparse SAE / virtual-
   weight basis becomes necessary — it DERIVES the subspace from the
   network's connectivity rather than reading it off sampled activations.

**Slack budget: +1.29** (worst case over 200 inputs). Thin. A SOUND
relaxation of LN + GELU adds looseness on top and must stay under 1.29 to
preserve the proof — the near-exact requirement from the tight-margin
finding, now quantified for the domain.

## Jacobian-sparse SAE construction — where it is / isn't needed (dissection)

Traced the sound subspace per component to see what actually must be
LEARNED vs read off the weights:

- **Attention outputs (b0a, b1a): mostly ARCHITECTURAL, sound for free.**
  b0a = z @ W_O + b_O where z (hook_z) is n_heads scalars (d_head=1). The
  reachable z is soundly a per-head box [min value, max value] (z is a
  convex combo of that head's values). Empirically head 1's z is CONSTANT
  (std 0), head 3 nearly so — b0a is genuinely rank-3. NO SAE needed to
  KNOW the subspace exists; it's W_O applied to a per-head z-box.

- **BUT the free bound is LOOSE.** The sound per-head z-box uses the global
  value RANGE (over all positions), which reactivates all 4 heads → rank-4
  box → the property FLIPS (-9.28). The tightening the property needs
  (rank 3) is the attention PATTERN structure at the query position: which
  input directions actually drive z at the LAST position. That is a
  JACOBIAN property (dz/dinput at the query), not a variance property.

- **Key tension confirmed:** the gap-critical 4th dimension is LOW-variance
  (sv 4.98 vs 1783) but HIGH gap-leverage. A variance/reconstruction-based
  SAE would discard exactly the dimension soundness must bound. So the
  construction must be **Jacobian-sparse, NOT reconstruction-sparse** — the
  user's instinct is right and the reason is now concrete: sparsity on the
  feature→feature Jacobian keeps the gap-critical low-variance directions
  that recon-SAEs drop.

- **MLP (b1m): the genuinely nonlinear, learnable part.** rank-4, no
  architectural collapse. Here a Jacobian-sparse SAE on the MLP's
  input→output map is the actual object to build.

## The rank-3 collapse is PROVABLE (BOS-saturating heads), not learned

Traced why b0a is rank-3. The attention PATTERN at the last query position:
- **head 1: attends to position 0 (BOS) with weight 1.0, entropy-std 0.000**
  — input-independent. z_head1 = value(BOS) = constant (BOS is a fixed
  token). Provable from the QK circuit (head 1's BOS score saturates).
- **head 3: 0.99 to BOS, entropy-std 0.054** — near-constant.
- heads 0, 2: input-dependent (entropy-std 0.25, 0.59).

So the rank-3 structure has a CONCRETE PROVABLE origin: two of four heads
saturate onto BOS and contribute (near-)constant vectors. The effective
DOF at the last position are heads {0, 2} + the BOS common-mode = 3. This
is a structural invariant provable from the pattern, NOT something a
recon-SAE should learn (and shouldn't, per the low-variance tension above).

This reframes the construction once more, toward LESS learning and more
provable structure: the sound tight subspace = {W_O applied to the
per-head z-box, with the BOS-saturating heads pinned to their constant
value}. The saturation is provable via a QK score-gap argument (the same
HARD-attention lemma already built in hard_attention.py: prove head 1's
BOS score exceeds all others by a margin → pattern ≈ one-hot on BOS →
z ≈ value(BOS), constant). The existing attention step-lemmas COMPOSE into
this.

## Step 3 built: MLP linearization — the affine/Jacobian form CLOSES it

`mlp_linear.py`: sound GELU linear bounds (0/4000 violations, grid+Lipschitz
margin) + sound LayerNorm sigma interval. Then propagated the rank-3
zonotope through block0.mlp. Three attempts, each fixing a decorrelation
leak:

1. **Interval-hull propagation: -646 (worse than hyperbox).** Taking the
   interval hull at each step DISCARDS the rank-3 correlation — the exact
   thing that made the property provable. Intervals can't; must stay affine.
2. **Affine b0a/b1 but MLP-as-box: -249, 28% provable.** Sharing eps between
   b0a and b1 (so their gap contributions can CANCEL) helped, but bounding
   the MLP output as an independent box re-decorrelates it from its own
   input.
3. **MLP-as-AFFINE (Jacobian, shared eps): +1.14, 100% provable.** Carrying
   mlp(rm) ≈ J·(rm−rm0)+mlp(rm0) as an affine function of the SAME eps
   recovers the +1.29 ideal. THE STRUCTURE CLOSES.

Decisive lesson: soundness here is entirely about keeping ONE affine form
(shared eps) through the whole circuit so correlated terms cancel in the
decision direction. Every place that drops to an interval/box loses the
correlation and the property with it. This is why the JACOBIAN (not
reconstruction) is the right object: it is the affine coefficient that
carries the correlation through the MLP.

## Close-out: the theorem HOLDS and the error FITS the budget

Built the full sound propagation: tight zonotope sigma bound
(`sound_sigma_zonotope`, corner-max + per-coord-min, ratio 1.40 vs the
box's 1.83, verified sound 0/500) and sound GELU linear bounds (0/4000).

The DECISIVE measurement: the TRUE MLP linearization error in the decision
direction is **mean 0.20, max 0.57 — well under the +1.14 budget.** So the
sound theorem CLOSES: exact-Jacobian affine (+1.14) minus a linearization
error that is genuinely ≤0.57 leaves a positive gap. The property is
soundly provable.

WHY the full-propagation code still shows -241 (the honest gap between
proven-closeable and my current bound): I bound the sigma/GELU error as
INDEPENDENT PER-COORDINATE boxes that get amplified by W_in (norm ~large)
BEFORE projecting to the decision direction. But the true error is small
precisely because it is CORRELATED and mostly cancels in the decision
direction (0.57, not 241). The fix is to bound the linearization error as a
scalar in the decision direction (project dvec through the error's affine
form) rather than box-then-amplify-then-project — the SAME shared-eps /
no-premature-interval lesson, now at the error term.

## Error-projection tightening attempt — honest status

Tried to realize the proven slack in a sound IMPLEMENTATION. Repeatedly hit
my OWN bugs, not a barrier, but the sound bound does not yet close in code:

- **A real bug found: block0.ln2 is `Identity`** (normalization_type None).
  The entire sigma-bound apparatus (sound_sigma_zonotope etc.) was solving a
  NON-PROBLEM — there is no LayerNorm in these models. The only nonlinearity
  is GELU. (The earlier -241 "sigma is the dominant error" was partly this
  phantom.) Removing it: the center MLP reconstruction is now EXACT (0.0).
- With no LN: gap_cen is correct (min +5.3, ~margin), affine spread small
  (max 9.85). So the LINEAR part closes. The leak is entirely the GELU
  contribution's sound bound:
  - affine-slope version: bad average slope over wide pre-act interval →
    spurious -173.
  - per-neuron INTERVAL version: decorrelates the 16 GELUs → -84.

The true GELU decision-error is ≤0.57 (measured, correlated, cancels), but
BOTH my sound implementations lose that correlation: interval-per-neuron
throws it away; affine-with-slope mis-approximates it. The correct object is
GELU carried as affine-in-eps with a TIGHT sound error that respects the
cross-neuron correlation through u = W_out^T dvec — which I did not get
right under time pressure.

## Status: structure proven-closeable, sound IMPLEMENTATION not yet closing

HONEST CORRECTION of the earlier "closes with room to spare": the STRUCTURE
closes (exact-Jacobian affine +1.14; true linearization error ≤0.57 in the
decision direction — both measured). But a SOUND, correlation-preserving
bound on the GELU term in code has NOT been achieved — every version so far
loses the correlation and lands negative (-84 to -173). So the theorem is
proven CLOSEABLE numerically, but NOT yet PROVEN by a sound bound.

Also a real correctness win banked: these models have NO LayerNorm (ln =
Identity), so the hardest anticipated obstacle (1/sigma) does not exist
here — the sound proof needs only a tight, correlation-preserving GELU
relaxation. That is the entire remaining task, and it is a bounded one:
16 GELUs, mostly saturated, carried affine-in-eps with a sound error that
projects through u before summing (so cross-neuron cancellation survives).
`mlp_linear.py` retains the sound GELU bounds (0/4000) which are correct;
the sigma code is dead (no LN) and should be removed.
