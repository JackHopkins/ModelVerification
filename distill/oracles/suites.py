"""Registry of all distillation target cases across the three suites.

iter_cases(suite) yields (case_name, loader) lazily — oracles are only
constructed on demand (loading 86 torch models eagerly is wasteful).

The interp_bench token-encoding convention (sorted(vocab + [BOS, PAD]),
BOS at position 0) is verified once per session by verify_interp_encoding()
against case 3, whose semantics (running fraction of 'x') is simple
enough to check directly.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MESSY_CASES = [
    "m01_heuristic_vote", "m02_signal_mixture", "m03_superposed_counts",
    "m04_multitask", "m05_soft_mixture", "m06_ngram_mixture",
]


def rasp_cases():
    return sorted(p.name for p in (ROOT / "rasp" / "compiled").iterdir() if p.is_dir())


def interp_meta():
    meta = json.loads((ROOT / "interp_bench" / "benchmark_metadata.json").read_text())
    # ioi / ioi_next_token use GPT-2 tokenization over natural-language
    # prompts (no explicit vocab) — outside the uniform-vocab task set.
    return {str(c["case_id"]): c for c in meta["cases"] if "vocab" in c}


def load_oracle(suite: str, case_id: str):
    if suite == "rasp":
        from .tracr_oracle import TracrOracle
        return TracrOracle(case_id)
    if suite == "messy":
        from .tl_oracle import MessyOracle
        return MessyOracle(case_id)
    if suite == "interp":
        from .tl_oracle import InterpOracle
        return InterpOracle(case_id, interp_meta()[case_id])
    raise ValueError(suite)


def iter_cases(suite=None):
    suites = [suite] if suite else ["rasp", "interp", "messy"]
    for s in suites:
        ids = (rasp_cases() if s == "rasp"
               else MESSY_CASES if s == "messy"
               else sorted(interp_meta(), key=lambda c: (len(c), c)))
        for cid in ids:
            yield s, cid


def verify_interp_encoding():
    """Case 3 = frac_prevs of 'x' over vocab [a,b,c,x]. If our encoding
    convention is right, the model's regression output must track the
    running fraction of 'x' closely."""
    oracle = load_oracle("interp", "3")
    rng = np.random.default_rng(0)
    content = oracle.sample(rng, 500, max(oracle.spec.seq_lens))
    out = oracle.outputs(content)
    x_idx = oracle.spec.vocab.index("x")
    is_x = (content == x_idx).astype(float)
    frac = np.cumsum(is_x, axis=1) / np.arange(1, content.shape[1] + 1)
    err = np.abs(out - frac).mean()
    assert err < 0.05, f"interp encoding convention check failed: mean err {err:.3f}"
    return err
