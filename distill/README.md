# distill — network → program distillation pipeline

Converts the repo's transformer models into programs expressing their
logic, evaluated across all three suites (`rasp/`, `interp_bench/`,
`messy/`). We distill the **model** (oracle = model outputs), never the
task: ground-truth labels are not used anywhere.

## Extractors

| name | kind | idea | provenance |
|---|---|---|---|
| `tree` | behavioral control | CART on IO features; soft leaf distributions from oracle probs | sklearn |
| `tprogram` | imitation baseline | Friedman et al.'s discrete-constrained TransformerProgram trained on **KL to oracle logits**, frozen to argmax | [princeton-nlp/TransformerPrograms](https://github.com/princeton-nlp/TransformerPrograms) (NeurIPS 2023), cloned in `third_party/` |
| `tracr_exact` | mechanistic decompiler (rasp suite) | reads Select/Aggregate/SelectorWidth/Map tables directly out of the weights via the labeled residual basis | ours; exactness in the spirit of [lacoco-lab decompiling_transformers](https://github.com/lacoco-lab/decompiling_transformers) (ICML 2026) |
| `tl_decompile` | feature-probing decompiler (TL models) | DSL feature bank (counts, frac-prevs, positional lookups) + sparse SoftHead readout fitted to oracle probs; features grounded against the residual stream (probe R²) | ours; scoped fallback for the lacoco-lab reparametrization (their release is cluster-bound) |
| `sae_decompile` | SAE feature-discovery decompiler (TL models) | sparse autoencoder on the final residual stream; candidates from an EXTENDED bank (offset/mirror tokens, equality) selected by multivariate-probe decodability from the sparse code; base primitives always kept; same DSL readout | ours; dictionary-learning feature discovery replacing the hand-picked bank |

All extractors emit programs over one DSL (`dsl/core.py`): a RASP core
with explicit finite tables (tracr-compatible) plus exactly two soft
constructs — `SoftHead`/`MixHead` (probability readouts) and numerical
threshold nodes. `harden()`-style degradation and one interpreter serve
both hard and soft programs.

## Novel improvements

**CEGIS refinement** (`cegis/`): counterexample search (exhaustive when
|V|^L ≤ 5e6, else random + mutation fuzzing) feeding extractor-specific
`refine()`. For `tracr_exact`, refinement does *white-box fault
localization*: every tracr variable is decodable from the model's final
residual stream, so the first diverging variable pinpoints the faulty
node, whose table entry is repaired locally. Headline result: p11_dyck1's
compiled model has a genuine selector-width saturation bug at full length
(computes length 9 for 10-token inputs); one CEGIS round localizes it and
writes `table[10] = 9` into the program — which is then **exactly
equivalent on all 2,046 inputs**, bug included.

**Soft program semantics** (`runs/messy_soft_study.json`): on m02 the
recovered MixHead weights are (0.848, 0.151) against the training
objective's (0.85, 0.15) — the program "85% follow the count rule, 15%
follow the marker" is recovered from the network to 3 decimal places.
The cost-of-hardness metric (TV of hardened vs soft program) is ≈0 for
behaviorally-deterministic messy models and large exactly where the
suite was designed to be irreducible (m05 +0.08, m06 +0.35).

## Headline results

- `tracr_exact`: **11/12 rasp models exactly equivalent** over their full
  input spaces (up to 1.4M inputs); p12_dyck2 at 99.95% with the complete
  670-input disagreement set enumerated in its certificate.
- Certificates (`runs/**/certificate.json`) distinguish "verified
  equivalent on the full input space" from "0 counterexamples in N
  samples" — every rasp and messy case gets the exhaustive kind.
- The `tree` control shows IO-fitting is enough for counting-flavored
  tasks but collapses on structural ones (reverse 0.51, interp/2 0.007) —
  evidence that mechanism-aware extraction is doing real work.
- `sae_decompile` is the strongest extractor on trained models — interp
  mean **0.996** over the FULL 84-case suite, **81/84 cases at ≥ 0.99**
  and 9 exact-equivalence certificates, beating the black-box tree
  control (0.883) while emitting verifiable programs (vs 0.680 for the
  hand-banked tl_decompile), by *discovering* structural features the
  hand bank lacked (interp/2 reverse: 0.013 -> 1.000 via the
  mirrored-position feature). The extractor discovers candidates rather
  than enumerating them: (a) feature COMPOSITION — composed candidates
  (count_self, rank_lt, prefix_count_self / is_first_occurrence via a
  token-x-position pair variable, pairwise interactions), taking
  hist-style cases from 0.000 to 1.000; (b) granularity mismatch — models
  representing only a task-specific coarsening of token identity fail the
  identity probe, so base primitives are always retained and SAE
  selection only ADDS structure; (c) Approach-A composition templates
  (T1 pairwise, T2a position-gathers, T2b k-th-occurrence gathers via a
  pair variable, second-order gating) — interp/21 unique-extract
  0.107 -> 0.999; (d) SYNTHESIZED position-map gathers — for each query
  position, probe which source position's token is decodable from the
  SAE code there (out-of-sample validated) and assemble the admitted
  (q->s) map into one Select table. This fixed the three total failures
  (93/103/110 from 0.000 to 1.000) and recovers the exact ground-truth
  permutations (i XOR 1, floor(i/2), odd-parity swap) — the general cure
  for conditionally-represented features that no fixed shift can match;
  (e) counts/thresholds over derived predicates (majority indicators,
  all/any/prefix-count of admitted binary features): interp/13 trend
  0.786 -> 0.971; plus a tabulation readout (CART-in-DSL NaryMap, adopted
  only when it beats the SoftHead on holdout) for joint position-
  conditional logic. Residual limitations, inventoried by mechanism in
  `runs/FAILURE_INVENTORY.md`: numeric interaction with a gathered global
  value (97, scale-by-max, 0.779), one genuinely messy SIIT artifact
  (124, 0.957 — matches no clean predicate), and m04's mode-gather /
  deep task-gating (0.697). A latent sklearn pitfall found on the way:
  LogisticRegression omits coefficient rows for training-absent classes;
  the readout now maps rows via model.classes_.
- Oracle correction found during SAE work: SIIT interp models are
  full-window-length trained (verified on case 2: the model reverses the
  entire padded window); interp evaluation now uses full-length inputs.

## Layout / usage

```
oracles/    # uniform Oracle over JAX-tracr + TransformerLens models
dsl/        # program IR + vectorized interpreter
extractors/ # tree, tprograms/, decompile/{tracr_exact,tl_decompile}
cegis/      # counterexample search + refinement loop
metrics/    # behavioral, formal (enumeration), structural (rasp)
scripts/    # run_case.py, run_suite.py, make_table.py, messy_calibration.py
runs/       # per-(suite, case, extractor) metrics + certificates + TABLE.md
```

```bash
../.venv/bin/python scripts/run_case.py rasp p08_reverse tracr_exact --cegis
../.venv/bin/python scripts/run_suite.py --suite messy --extractor tree,tl_decompile
../.venv/bin/python scripts/make_table.py          # aggregate leaderboard
../.venv/bin/python tests/smoke_test.py            # one case per extractor
```

Tolerances: rasp compares decoded outputs at 1e-3; interp_bench at 0.05
(the SIIT models' own training atol). The two IOI cases are excluded
(GPT-2 tokenizer, no explicit vocab). `tprogram` covers categorical
outputs only (seq_num cases are rejected by design).
