"""The simulation relation R, read directly off the labeled residual basis.

A tracr residual stream of width D is partitioned by `residual_labels`
into disjoint blocks, one per program variable. A categorical variable v
with value-space {0..m-1} occupies m one-hot axes: the residual encodes
`v = j` iff the sub-vector on v's axes is the standard basis vector e_j.
Numerical variables occupy a single axis holding a real value.

R is therefore not learned or fitted — it is the block partition itself,
plus the tracr convention that categoricals are one-hot. This module
extracts R from architecture.json and exposes the decode/encode maps and
the per-variable axis blocks the step-lemma prover quantifies over.
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Relation:
    def __init__(self, case_id):
        arch = json.loads(
            (ROOT / "rasp" / "compiled" / case_id / "architecture.json").read_text())
        self.case_id = case_id
        self.arch = arch
        self.labels = arch["residual_labels"]
        self.D = len(self.labels)
        assert arch["layer_norm"] is False, "LN not yet handled"
        # block[var] = sorted list of residual axes; value_of[var][axis_pos]
        # = the categorical value that axis encodes (parsed from "var:val")
        self.block = OrderedDict()
        self.axis_value = {}
        for i, lab in enumerate(self.labels):
            if ":" in lab:
                var, val = lab.rsplit(":", 1)
            else:
                var, val = lab, None
            self.block.setdefault(var, []).append(i)
            self.axis_value[i] = val
        # a variable is CATEGORICAL if its axes carry distinct value tags,
        # NUMERICAL if it is a single untagged axis (e.g. a selector-width
        # count written into one dimension)
        self.categorical = {}
        for var, axes in self.block.items():
            vals = [self.axis_value[a] for a in axes]
            self.categorical[var] = len(axes) > 1 or vals[0] is not None

    def variables(self):
        return list(self.block)

    def decode(self, resid):
        """resid: (..., D). Returns {var: value} where categorical vars are
        argmax over their block (the one-hot index) and numerical vars are
        the raw scalar. Vectorized over leading dims."""
        out = {}
        for var, axes in self.block.items():
            sub = resid[..., axes]
            if self.categorical[var]:
                out[var] = sub.argmax(-1)
            else:
                out[var] = sub[..., 0]
        return out

    def is_onehot(self, resid, var, atol=1e-3):
        """Does resid's block for `var` look like a clean one-hot? (used to
        certify the relation holds empirically before the symbolic proof.)"""
        sub = resid[..., self.block[var]]
        top = np.sort(sub, -1)
        return (np.abs(top[..., -1] - 1.0) < atol) & (np.abs(top[..., -2]) < atol)
