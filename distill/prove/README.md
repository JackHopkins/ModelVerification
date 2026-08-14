# prove/ — relational verification of the network itself

`distill/` proves a decompiled **program** is behaviorally close to the
network. That certificate is about a *surrogate*: even at 100% agreement
on every input, a property proved on the program transfers to the network
only under an unproven assumption (behavioral identity ⟹ property
identity). Two gaps make it unsound as a proof about the network:

- **Extensional gap.** Behavioral agreement says nothing about mechanism.
  Mechanistic safety properties ("no information flows through head h",
  "output j is independent of input i", "this is exactly the ground-truth
  circuit") cannot be inherited from an I/O-matching program.
- **Distributional gap.** "Exhaustive" holds only for tiny input spaces;
  otherwise the certificate is "0 counterexamples in N samples", and the
  adversarial inputs a safety case cares about are the ones sampling
  misses.

This package makes the **network the object of proof**. It establishes a
**simulation relation R** between residual-stream states and
program-variable states, to be discharged over the actual weight matrices
— a ∀-statement about the weights, not a sample count.

## Status

**Built:**
- `relation.py` — R read directly off the labeled residual basis (the
  block partition + tracr's one-hot convention). Not learned; it *is* the
  architecture's semantics.
- `check_relation.py` — phase-aware empirical pre-check: runs the real
  forward pass over the (often exhaustive) input space and reports, per
  (variable, sublayer, position), where the block is **populated but not
  one-hot**. This is not itself a proof (it is behavioral), but it tells
  the symbolic prover exactly which blocks are clean (easy step-lemma)
  vs. where the relation has structure it must model.

**Findings from the pre-check (already impossible behaviorally):**
- p01_identity: **R holds cleanly** — every categorical block is one-hot
  wherever populated, from its birth layer on.
- Scalar/broadcast variables (`length`, `opp_idx`) are one-hot at a
  single aggregation position and zero elsewhere — R for them is
  "one-hot at p*, zero elsewhere", a different (still checkable) shape.
- **p06_hist: a mechanistic defect found with no wrong output.** On the
  all-identical input `[a a a a a a a a]`, the `hist_10` block (count = 8)
  is NOT one-hot — it reads a soft blend of indices 7 and 8 (0.306/0.694).
  The network does not cleanly represent the maximum count. This is the
  same class of tracr selector-width saturation bug behavioral CEGIS
  found in p11 — but detected here by inspecting the *internal encoding*,
  which I/O testing cannot see (the soft blend may still round to the
  right output).

**Built — first weight-level theorem (`simulate.py`, p01_identity):**
- `prove_p01_symbolic` proves a genuine ∀-statement over the REAL MLP
  weights (exact rationals in z3): *for every one-hot token, at every
  position, the `identity_1` residual block is the strict argmax at the
  token's index*. The token one-hot is a z3 VARIABLE over the one-hot
  simplex (not enumerated) and the property is discharged by refutation —
  this is what scales past enumerable input blocks.
- `prove_p01` additionally reports the winner margin (1.0 for every token)
  and proves a mechanistic structural fact: `identity_write_reads_position
  == False` — the MLP's W1 has exactly zero weight on the position axes,
  so the identity computation is provably position-independent. That is a
  statement about the network's WIRING that behavioral testing cannot make.
- **Falsification-tested**: zeroing one column of W2 (breaking token 'a')
  flips the proof to a counterexample at every position, correctly naming
  token index 0. The prover has teeth and its counterexamples are
  mechanistic (which token's encoding breaks), not merely "wrong output".

**Built — attention step-lemma, UNIFORM regime (`attention.py`, p03_length):**
The transcendental softmax is never reasoned about via `exp`. Instead we
prove which soundly-analyzable regime holds and verify aggregation there.
For SelectorWidth/length layers the regime is UNIFORM:
- `prove_uniform_scores` proves over the real Q/K weights that, for every
  one-hot token assignment, the attention scores to all content keys are
  EQUAL (z3 refutation). Equal scores ⇒ softmax is exactly uniform,
  independent of the (unmodelled) score magnitude — so no `exp` reasoning
  is needed. Notably this holds *even though the query reads token axes*
  (weight 100): the tracr TRUE-selector produces equal scores despite
  depending on tokens — a fact PROVED from the weights, not assumed.
- `prove_uniform_aggregation` proves the attention output on the write
  axes is CONTENT-INDEPENDENT (two arbitrary one-hot token assignments
  give identical output), i.e. the exact 1/T aggregation of the `one`
  value ⇒ length is well-defined.
- **Falsification-tested — and the test harness itself had to be
  validated.** Two natural corruptions (position-dependence into the
  QUERY; a blanket constant into all key rows) did NOT break score
  equality — numeric check confirmed scores stayed equal, so the lemma
  was RIGHT to prove them. Only a corruption graded by position on a
  key-dim the query actually uses makes scores unequal (numeric std 268),
  and there the lemma correctly returns FALSE. Lesson recorded: validate
  the falsification input numerically before trusting its verdict.

**Built — attention step-lemma, HARD regime (`hard_attention.py`):**
Gather layers, where one key dominates. We never touch `exp`: prove a
LOWER BOUND on the winning-key score gap γ over the real Q/K weights
(z3, bisection), then convert analytically to a sound softmax-leak bound
eps <= (T-1)·e^{-γ}, giving winner_weight >= 1/(1+eps).
- `prove_gap_from_embedding` on **p04_shift_right** (clean single-layer
  gather, query selects on a TRUE position one-hot from the embedding):
  PROVED gap >= 29.94 for all one-hot tokens => eps <= 7.9e-13, winner
  weight >= 0.9999999999992. The gather is certified near-exact.
- **Falsification-tested (numerically validated first):** zeroing the key
  matrix makes all keys identical; numeric gap = 0 and z3 proves gap = 0,
  winner weight lower bound 1/9 (uniform, no gather). Several *intended*
  corruptions did NOT break the gap (redundant position encoding) — caught
  by the numeric pre-check, not trusted blindly.

**Honest finding — the method refuses to over-claim (`prove_gap` on
p08_reverse):** reverse's gather selects on `opp_idx`, a COMPUTED value
that at the gather's layer input is NOT a clean one-hot (top1 only 0.694,
top2 0.306 — the same tracr saturation as p06_hist / the p11 CEGIS bug).
The empirical winner weight is 0.99 (eps 0.082), and a sound symbolic gap
proof over the reachable state cannot certify a crisp gather because the
network does not implement one crisply. This is the relational method
working as intended: it will not prove a clean gather the weights don't
support. Certifying reverse requires either (a) carrying the soft
`opp_idx` interval through the gap bound, or (b) reporting the gather as
eps-approximate with eps=0.082 — both honest, neither a clean one-hot.

**All three primitives now proved + falsification-tested:** MLP map
(p01), attention UNIFORM (p03), attention HARD (p04).

**Built — simplex-envelope prover (`simplex.py`), the unified exp-free
step-lemma:** softmax is never decomposed. Two exact linear facts replace
it: (1) selection = argmax = order-preserving (linear score comparison in
z3); (2) the aggregation output is a convex combination of value vectors
— the softmax weights form a probability simplex, a LINEAR exact
constraint. We prove the downstream decode is correct for every point of
the reachable hull. No `exp` in the solver.
- On **p08_reverse** (layer-3 gather, the SOFT/saturated case the eps-
  bound gave up on): **all 8 queries PROVED** — the decode equals the
  gathered token across the full 2-key hull, so reverse is certified
  correct despite the saturated `opp_idx`.
- **Falsification-tested:** swapping two columns of the value→write path
  collapses all 8 proofs (0/8). Has teeth.
- **A false-counterexample bug found and fixed en route** (same class as
  the free-variable error earlier): the BARE simplex is sound but far too
  loose — letting a 0.01-weight runner-up reach 0.5 fabricates flips the
  real softmax never produces (reverse is 100% correct behaviorally). Fix:
  cap each non-winner weight by its reachable max.

**HARDENED — weight caps from PROVEN symbolic score gaps
(`proven_weight_caps`, `hardened=True`):** each runner-up's softmax weight
cap now comes from a score-gap lower bound proved in z3 over the real Q/K
weights, quantifying over the reachable upstream state as an INTERVAL BOX
(sound over-approximation, no exhaustive-enumeration dependency for the
cap). From the proven gap γ: `lambda_j <= e^{-γ}/(1+e^{-γ})`, with `exp`
evaluated ONCE as a numeric constant — never in the solver.
- On p08_reverse: proved gap >= 4.57 (correctly BELOW the empirical 4.583
  — a valid, slightly-conservative lower bound), cap <= 0.0103, all 8
  queries PROVED. Falsification-tested on the hardened path (corrupt
  value→write ⇒ 0/8).
- **Two soundness checks enforced:** (1) proven gap must be ≤ empirical
  gap (a bound coming out ABOVE observed = unsound — the dangerous
  direction; here 4.57 ≤ 4.583 ✓); (2) prover still catches a broken
  gather.
- **Bug found and fixed en route (unit error):** the z3 score omitted the
  `/√K` normalization, so a raw proven gap of 19.38 *looked* like it
  exceeded the empirical 4.583 (false unsoundness alarm). Dividing by √K
  gives 4.57 — sound. The missing normalization would also have produced
  a wrong (too-tight) weight cap, so it was a real bug, not just cosmetics.
  Lesson: check proven-bound vs empirical in MATCHING units before
  trusting either the alarm or the proof.

The interval box breaks input correlations (sound: it only enlarges the
state set ⇒ conservative gap), which is why the proven 4.57 sits just
under the true 4.583. Tighter relational constraints (e.g. carrying the
one-hot structure of `indices`) would close that small gap if needed.

**Built — mechanistic property layer (`mechanistic.py`), the safety
payoff: properties of the network's STRUCTURE, proved over weights, that
behavioral equivalence cannot express.**

1. **Path-independence.** `dependency_graph` builds the exact axis-level
   dependency relation over the residual bus (embed → per layer {Q/K/V →
   attn out; MLP} → unembed), separating STRUCTURAL edges (OV/MLP content)
   from POSITIONAL edges (attention routing). `prove_independence(out, in)`
   returns proved=True iff no weight path carries `in`'s axes to `out`'s
   axes — a theorem over ALL inputs. Sound + conservative: a reported
   independence is a theorem; a reported dependency may be spurious.
   - p01: `identity_1` PROVED independent of `indices`, depends on
     `tokens`. Brute-force validated (identity output constant across
     position for every token). **Falsification-tested:** wiring a
     position→identity MLP path makes the prover correctly REFUSE the
     independence claim (never misses a real dependency).
   - p08_reverse: `reverse_19` depends on both tokens and indices (gather);
     no false independence.

2. **Circuit-equality.** `used_components` proves which per-layer
   {attn, mlp} components influence the output (output-reachability in the
   dependency graph). `circuit_equality` compares that against tracr's
   ground-truth structure (each variable's birth layer). **0 discrepancies
   across p01/p03/p04/p06/p08.** On reverse it exactly recovers the circuit:
   L0.attn=length, L1/L2.mlp=mirror-index, L3.attn=gather; L1.attn,
   L2.attn, L3.mlp proved DEAD — matching ground truth.
   - `load_interp_edges` reads InterpBench `edges.pkl` ((src,dst) hook
     tuples) so the same comparison extends to trained models once the
     dependency graph is built over TL weights.

This is the qualitative goal of the whole reframe: "output j provably
cannot depend on input i" and "these are exactly the live components" are
statements about the network's mechanism, unattainable from I/O agreement.

**Built — dependency graph for TRAINED InterpBench models
(`interp_graph.py`):** trained weights are DENSE, so dependency cannot be
read off weight sparsity (every component nominally connects to every
other). The sound primitive is CAUSAL: an edge exists iff an intervention
on the source changes the output. We use **corrupt-baseline path
patching** (patch a source to its activation on a shuffled input — an
on-distribution intervention, far sharper than off-distribution mean-
ablation, which lets other components compensate and hides used ones — the
self-repair/hydra effect). `circuit_equality_interp` compares the recovered
live-source set against `edges.pkl`.
- **Recovers the circuit exactly where it is source-level and the effect
  spectrum has a cliff:** cases 8, 11, 75 match `edges.pkl` (0 missing, 0
  extra) with a clean 6–50x separation gap. `_auto_threshold` cuts at the
  largest absolute drop; `circuit_equality_interp` reports whether the
  separation was CLEAN or SMOOTH (unreliable) so the verdict is never
  over-trusted.
- **Honest limits, quantified:** of 86 cases, **59 are source-level**
  (this graph's granularity) and **27 route through specific head q/k/v
  inputs** — those need finer path-patching into `hook_{q,k,v}_input[h]`
  (demonstrated: on case 21, patching `b3.q_input[0]` gives effect 5.0 vs
  0.51 for the non-circuit `[1]`, a 10x separation the source-level ablate
  misses). Some source-level cases (20, 21) still show smooth spectra where
  single-node attribution under-separates; a full ACDC-style recursive
  path patch is the principled completion.

Contrast with the tracr path: there the labeled basis gives R and the
dependency graph is a WEIGHT-level theorem (forall inputs). Here it is a
sound but DISTRIBUTIONAL causal measurement (over sampled inputs) — the
honest cost of dense trained weights, and why `edges.pkl` itself was built
by patching, not proof.

**Head-input granularity (`head_input_edges`):** corrupt-baseline path
patching into each `hook_{q,k,v}_input[h]` — the granularity 27/86 circuits
route through. It surfaces the right components in RANK order (case 21: the
three ground-truth edges b3.{q,k,v}_input[0] are the top-3 effects 8.3/5.2/
5.0, cleanly above the 3.6 next tier), but the single-gap auto-threshold
cuts imperfectly when the circuit forms a cluster rather than a lone spike.
The honest state: ranked attribution is correct; automatic cutoff is not
robust on smooth spectra — a recursive ACDC pass (patch, threshold, recurse
on survivors) is the principled fix.

**Where this leaves the agenda (see also the top of this file):** the
tracr path gives WEIGHT-LEVEL ∀-theorems (path-independence, circuit-
equality, functional step-lemmas). The trained-model path gives sound but
DISTRIBUTIONAL causal attribution — because dense weights admit no free
forall, which is why edges.pkl itself was built by patching. Closing that
gap (provable guarantees on dense trained nets over input REGIONS via
sound abstract interpretation through real attention) is the open research
problem, deferred by choice. Consolidated here; not yet attempted.

**Next (when resumed):**
- Recursive (ACDC) path patching for robust automatic circuit cutoff.
- Compositional chaining of the tracr step-lemmas into an end-to-end
  functional theorem (network implements program).
- The hard problem: sound region-level guarantees on trained models.
- Refine R for scalar/broadcast variables (`length`, `opp_idx`: one-hot
  at p*, zero elsewhere) and the hist saturation corner.
- Chain step-lemmas into an end-to-end network theorem, then the
  mechanistic property layer (path-independence, circuit-equality vs
  `edges.pkl`).
- Refine R where the pre-check shows structure (scalar positions, the
  hist saturation corner) so the relation stated to the prover is the one
  that actually holds.
- Mechanistic property layer: once R is proven, "output j independent of
  input i" and circuit-equality vs. `edges.pkl` become structural checks
  over which axes feed which — proved about the network, not a surrogate.
