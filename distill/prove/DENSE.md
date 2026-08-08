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

## Status

Findings established (this file), including two falsified hypotheses and the
real LayerNorm-coupling mechanism. The additive/decision-subspace shortcut
is DEAD — the harness must do genuine bound propagation through LayerNorm,
which is the honest scope of the remaining work. Not yet built.
