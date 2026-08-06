"""M7 — soft-semantics study on the messy suite.

1. m02 mixture-weight recovery: express the model as the two-feature
   MixHead  p(y) = pi_rule * onehot(rule) + pi_marker * onehot(marker)
   and fit pi by least squares against the ORACLE's probabilities.
   Acceptance: recovered weights within 0.02 of the training objective's
   0.85 / 0.15.

2. Cost-of-hardness: for every messy case, the TV gap between the best
   soft readout (tl_decompile SoftHead) and its hardened argmax version —
   how much faithfulness a crisp logical program must give up.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from distill.extractors.decompile.tl_decompile import TLDecompileExtractor
from distill.oracles import suites

OUT = Path(__file__).resolve().parents[1] / "runs" / "messy_soft_study.json"


def m02_mixture_recovery():
    o = suites.load_oracle("messy", "m02_signal_mixture")
    rng = np.random.default_rng(0)
    content = o.sample(rng, 50_000, o.spec.seq_lens[0])
    p1 = o.probs(content)[:, 1]

    counts_a = (content == 0).sum(1)
    counts_b = (content == 1).sum(1)
    rule = (counts_a > counts_b).astype(float)
    marker = (content[:, -1] == 4).astype(float)  # 'x'

    # p1 = pi_r * rule + pi_m * marker  (least squares, no intercept)
    A = np.column_stack([rule, marker])
    (pi_r, pi_m), *_ = np.linalg.lstsq(A, p1, rcond=None)
    pred = A @ [pi_r, pi_m]
    tv_mix = float(np.abs(pred - p1).mean())
    tv_hard = float(np.abs(rule - p1).mean())  # hardened: follow the rule
    return {
        "recovered_pi_rule": round(float(pi_r), 4),
        "recovered_pi_marker": round(float(pi_m), 4),
        "objective_pi_rule": 0.85,
        "objective_pi_marker": 0.15,
        "abs_err_rule": round(abs(pi_r - 0.85), 4),
        "abs_err_marker": round(abs(pi_m - 0.15), 4),
        "passes_0.02": bool(abs(pi_r - 0.85) < 0.02 and abs(pi_m - 0.15) < 0.02),
        "mean_abs_p_err_mixhead": round(tv_mix, 4),
        "mean_abs_p_err_hardened": round(tv_hard, 4),
    }


def cost_of_hardness():
    ex = TLDecompileExtractor()
    out = {}
    for _, cid in suites.iter_cases("messy"):
        o = suites.load_oracle("messy", cid)
        rng = np.random.default_rng(0)
        try:
            prog = ex.extract(o, rng)
        except Exception as e:
            out[cid] = {"error": str(e)}
            continue
        tvs_soft, tvs_hard = [], []
        for content in o.sample_batches(np.random.default_rng(1), 20_000):
            op = o.probs(content)
            if op is None:
                break
            pp = prog.probs(content)
            if pp.shape != op.shape:
                break
            hard = np.eye(op.shape[-1])[pp.argmax(-1)]
            tvs_soft.append(0.5 * np.abs(pp - op).sum(-1).mean())
            tvs_hard.append(0.5 * np.abs(hard - op).sum(-1).mean())
        if tvs_soft:
            out[cid] = {
                "tv_soft_program": round(float(np.mean(tvs_soft)), 4),
                "tv_hardened_program": round(float(np.mean(tvs_hard)), 4),
                "cost_of_hardness": round(
                    float(np.mean(tvs_hard) - np.mean(tvs_soft)), 4),
            }
    return out


if __name__ == "__main__":
    report = {
        "m02_mixture_recovery": m02_mixture_recovery(),
        "cost_of_hardness": cost_of_hardness(),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
