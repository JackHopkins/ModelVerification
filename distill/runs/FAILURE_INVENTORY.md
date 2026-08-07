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

| case | agree | task | mechanism still missing |
|---|---|---|---|
| 97  | 0.779 | scale sequence by its maximum | **numeric interaction with a gathered global value**: needs tok × f(max) product feeding a numeric readout. `is_max` admits at 1.0 and the max value is gatherable, but the (tok, max_val) product isn't decodable enough to trigger the composition probe. |
| 124 | 0.957 | (described "all equal", but the SIIT model computes neither that nor any clean predicate — ∃-pair 46%, mode≥4 92%, run-of-3 92%) | genuinely messy model artifact; the tree control needs 411 splits for 0.994. Our compact program at 0.957 is arguably the more faithful *distillation* of a messy circuit. |
| 13  | 0.971 | trend (increasing/decreasing/constant) | joint position-conditional logic over prev/cur/next; majority candidates lifted it from 0.786 but the last ~3% is soft-readout calibration at rare configurations. |

## Messy residuals (unchanged targets)

m04 multitask 0.697 (deep task-gating / mode-gather), m06 ngram mixture
0.634 (improved from 0.449 via the richer bank; the n-gram blend is
designed to be irreducible). m01/m05 ~0.89, m02/m03 0.992.

## Notes

- interp/39 (26-token vocab, 60-position window) OOM fixes from the prior
  round hold: 1.000 at ~40 GB peak.
- The +cegis path is available (`--cegis`) but not part of the headline
  sweep; on the near-misses it refits with counterexamples upweighted.
  The tabulation readout is the better lever for the joint-logic cases.
