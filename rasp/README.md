# RASP / TRACR suite

Twelve RASP programs of increasing complexity are
compiled with [TRACR](https://github.com/google-deepmind/tracr)
(DeepMind's RASP-to-transformer compiler) into transformers whose weights
implement *known, exactly specified* algorithms. Because the ground-truth
semantics of every circuit is the RASP source, these models serve as
verification targets where the correct answer is known: a candidate
verification method should be able to prove (or refute) that each
compiled circuit implements its program for **all** inputs up to
`max_seq_len`, not just the sampled examples.

## Layout

```
programs/          # RASP source programs (one module each), complexity-ordered
compile_all.py     # compiles every program with TRACR and validates it
compiled/<name>/   # one directory per compiled model:
  params.npz         # transformer weights (haiku params, "module||param" keys)
  model.pkl          # pickled params + config + residual labels for exact reload
  architecture.json  # layers/heads/dims + meaning of every residual direction
  validation.json    # compiled model output vs. RASP interpreter, per example
```

## The suite

| # | Program | Semantics | Circuit features | Layers | d_model | Params |
|---|---------|-----------|------------------|--------|---------|--------|
| 01 | `identity` | copy input | embedding path only | 1 | 18 | 529 |
| 02 | `increment` | x → (x+1) mod 5 | MLP lookup table | 1 | 20 | 648 |
| 03 | `length` | sequence length | selector-width attention | 1 | 24 | 1.8k |
| 04 | `shift_right` | previous token | positional offset head | 1 | 19 | 1.2k |
| 05 | `frac_prevs` | running fraction of "x" | numerical averaging attention | 2 | 15 | 1.6k |
| 06 | `hist` | per-token occurrence count | content-match + counting | 1 | 24 | 1.8k |
| 07 | `pair_balance` | running ( − ) balance | parallel numerical pipelines | 2 | 19 | 3.8k |
| 08 | `reverse` | reverse sequence | 2-stage attention composition | 4 | 59 | 53k |
| 09 | `detect_pattern` | trigram "abc" detector | shifted indicators + AND tree | 3 | 28 | 8.5k |
| 10 | `sort` | ascending sort | data-dependent routing | 3 | 71 | 55k |
| 11 | `dyck1` | balanced `()` recognizer | global safety property | 6 | 53 | 293k |
| 12 | `dyck2` | shuffle-Dyck-2 recognizer | full compositional stack | 8 | 70 | 1.03M |

Each program module documents the circuit TRACR is expected to emit and
why it is interesting as a verification target (e.g. `p04` is the
previous-token head found inside induction heads; `p11`/`p12` encode a
"never enters a bad state" global property).

## Usage

```bash
# from the repository root:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd rasp
../.venv/bin/python compile_all.py             # compile + validate everything
../.venv/bin/python compile_all.py p08_reverse # just one program
```

Reloading a compiled model for analysis:

```python
import pickle, numpy as np

with open("compiled/p08_reverse/model.pkl", "rb") as f:
    bundle = pickle.load(f)        # params, config, residual_labels, vocab, ...

weights = np.load("compiled/p08_reverse/params.npz")
print(bundle["residual_labels"])   # what each residual dimension means
```

Or recompile in-process to get a live model with `.apply()`:

```python
from tracr.compiler import compiling
from programs import p08_reverse as p

model = compiling.compile_rasp_to_model(
    p.make_program(), vocab=p.VOCAB, max_seq_len=p.MAX_SEQ_LEN,
    compiler_bos="bos", compiler_pad="pad")
model.apply(["bos", "a", "b", "c"]).decoded   # ['bos', 'c', 'b', 'a']
```

## Why this helps with verification research

TRACR models are ideal first targets for formal circuit verification:

- **Known ground truth.** The RASP program *is* the specification; the
  `residual_labels` in `architecture.json` name every residual-stream
  direction, so claims like "head 2.0 routes tokens from the mirrored
  position" are checkable against a known answer.
- **Graded difficulty.** From an embedding-only identity map up to an
  8-layer, 1M-parameter Dyck-2 recognizer with parallel numerical
  pipelines and global aggregation.
- **Both value types.** Categorical (one-hot subspaces, hard attention)
  and numerical (real-valued averaging) circuits, which stress different
  proof techniques (discrete case analysis vs. interval/linear-bound
  reasoning).
- **Exact validation harness.** `validation.json` compares the compiled
  transformer against the RASP interpreter; a verification method should
  strengthen this from finitely many samples to all inputs.
