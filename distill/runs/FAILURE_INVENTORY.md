# sae_decompile failure inventory — full 84-case interp sweep (+ messy)

Mean sequence agreement **0.996** over 84 interp cases (tree control 0.883).
**81/84 cases at ≥ 0.99**, 9 exact-equivalence certificates. This file
tracks the three cases still below the bar, plus the messy residuals —
the requirements list for further work.

## Resolved since the first inventory (Approach-B mechanisms shipped)

| mechanism | fix | cases |
|---|---|---|
| B1 position-arithmetic gathers | **synthesized** per-position gather map: for each query q, probe which source position's token is decodable from the SAE code at q (out-of-sample validated), assemble admitted (q→s) into one Select table | 103 0.000→1.000, 110 0.000→1.000, 93 0.000→1.000 — learned maps recover the exact ground-truth permutations (i⊕1, ⌊i/2⌋, odd-parity swap) |
| B2 aggregates over derived predicates | count/threshold candidates over admitted binary features (`all/any/allm1/prefix_count/all-so-far`) + majority indicators (`maj_v = count_v > L/2`); models represent the thresholded boolean, not the raw count | 13 0.786→0.971 |
| seq_num readout overfit | the 10→106-feature bank made the base ridge overfit (regressed 3/4/39 to ~0); replaced with sparse forward-selection ridge (numeric frac_* first, admit a categorical block only if it cuts holdout RMSE) | 3/4/39 restored to 1.000 (3,4 exact) |

Infrastructure added: tabulation readout (CART-in-DSL `NaryMap`, chosen
only when it beats the SoftHead on holdout, CEGIS-repairable per cell),
`Coalesce` boundary node, vectorized `NaryMap`, base-rate-aware admission
(rejects degenerate near-constant candidates), structured sampling
(constant/one-flip/sorted/two-block sequences so rare-event predicates
like "all equal" are learnable at all).

## Remaining residuals (3 interp cases)

| case | agree | task | classification |
|---|---|---|---|
| 97  | 0.779 | scale sequence by its maximum | **IRREDUCIBLE — model-side, not extractor-side.** A ceiling diagnostic (unrestricted HistGradientBoosting over a rich numeric bank that *includes* `val/max` and `val*max`) tops out at **0.779 — identical to our program**. Permutation importance ranks `val_over_max` first (+0.52): the model genuinely computes value/max, but **division is not exactly representable in a 2-layer attention+MLP**, so the model approximates it lossily (input value 1 with max 10 maps to 7 distinct outputs). No feature recovers information the network already destroyed. Belongs with 124 / the messy models. Original "add tok×f(max) product tables" proposal was aimed at the wrong target and would not help. |
| 124 | 0.957 | (described "all equal", but the SIIT model computes neither that nor any clean predicate — ∃-pair 46%, mode≥4 92%, run-of-3 92%) | **IRREDUCIBLE — model artifact.** The tree control needs 411 splits for 0.994. Our compact program at 0.957 is arguably the more faithful *distillation* of a messy circuit. |
| 13  | 0.971 | trend (increasing/decreasing/constant) | **AT LOCAL CEILING.** A `(prev,cur,next)` lookup table ceilings at ~0.964–0.971 (position adds nothing); our SoftHead already matches it. Majority candidates lifted it from 0.786. The gap to 1.0 is non-local boundary behavior (no "next" at the last position — the model uses earlier context a local program structurally cannot see). SAE-capacity sweep (8×→32×, 3000→8000 steps) admits the SAME features and does NOT raise the ceiling — feature discovery was never the bottleneck. A boundary-token fix (Coalesce boundary to a fresh out-of-range symbol, not 0) is in the tabulation path and general to all local-window tasks. |

## Messy residuals (unchanged targets)

m04 multitask 0.697 (deep task-gating / mode-gather), m06 ngram mixture
0.634 (improved from 0.449 via the richer bank; the n-gram blend is
designed to be irreducible). m01/m05 ~0.89, m02/m03 0.992.

## Ceiling analysis (why the residuals are residual)

Both interp residuals were run through a **ceiling diagnostic** — the best
accuracy any function of the admitted features can reach, measured with an
unrestricted model (boosted tree / oracle-fit lookup table). If our
compact program already matches that ceiling, the gap is model-side
(information the target network destroyed), not extractor-side.

- **interp/97 (0.779):** ceiling = 0.779. A boosted tree given `val/max`
  and `val*max` explicitly cannot beat our program. The model computes
  value/max, but division isn't exactly representable in a 2-layer
  transformer, so it approximates lossily (value 1, max 10 → 7 distinct
  outputs). Irreducible.
- **interp/13 (0.971):** ceiling ≈ 0.964–0.971 for any local
  (prev,cur,next) table; our SoftHead already matches it. The remaining
  gap to 1.0 is non-local boundary behavior (no "next" at the last
  position) that a local RASP program structurally cannot represent.
  SAE-capacity sweep (8×→32× expansion, 3000→8000 steps) admits the SAME
  features and does NOT raise the ceiling — feature discovery was never
  the bottleneck here. A boundary-token fix in the tabulation path
  (Coalesce boundary to a fresh out-of-range symbol, not 0, so seq-end
  cells stay distinct from real token 0) closed the tabulation gap
  0.964→0.971 and is general to all local-window tasks, but 0.971 is the
  local ceiling.

SAE capacity helps only when feature DISCOVERY is the bottleneck
(superposed/undiscovered features) — the place to test scaling is m04
(0.697 deep gating) / m06 (0.634 ngram mixture), not the near-solved
interp cases.

## Notes

- interp/39 (26-token vocab, 60-position window) OOM fixes from the prior
  round hold: 1.000 at ~40 GB peak.
- The +cegis path is available (`--cegis`) but not part of the headline
  sweep; on the near-misses it refits with counterexamples upweighted.
  The tabulation readout is the better lever for the joint-logic cases.
