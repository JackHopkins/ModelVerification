"""RASP-style DSL with explicit finite tables, plus soft extensions.

Hard core (RASP-compatible, tracr-exportable):
  Tokens, Indices, Const, TableMap, SeqMap, Cmp, Select, Aggregate,
  SelectorWidth

Soft extensions (exactly two families — see plan):
  SoftHead  — logits[c] = b[c] + sum_f W_f[value_f][c]; linear-softmax
              over categorical feature values
  MixHead   — p(y|x) = sum_f pi_f * onehot(feature_f(x)); convex mixture
              of hard feature predictions (directly expresses objectives
              like messy/m02's 0.85*rule + 0.15*marker)

All predicates/maps are finite tables (serializable, canonicalizable).
Values are numpy float arrays with a parallel validity mask; None (RASP
undefined) positions are invalid and treated as wildcards by metrics.

The interpreter is vectorized over (n, L) batches of content vocab
indices, matching the Oracle content convention.
"""

import dataclasses
import json
from typing import Optional

import numpy as np


class Node:
    """Base DSL node. Subclasses define children() and eval_(ctx)."""

    name: str = ""

    def children(self):
        return []

    def named(self, name):
        self.name = name
        return self


# ---------------------------------------------------------------- primitives


class Tokens(Node):
    def eval_(self, ctx, kids):
        return ctx["content"].astype(np.float64), np.ones(ctx["content"].shape, bool)


class Indices(Node):
    def eval_(self, ctx, kids):
        n, L = ctx["content"].shape
        return np.broadcast_to(np.arange(L, dtype=np.float64), (n, L)).copy(), np.ones((n, L), bool)


class Const(Node):
    def __init__(self, value):
        self.value = value

    def eval_(self, ctx, kids):
        n, L = ctx["content"].shape
        return np.full((n, L), float(self.value)), np.ones((n, L), bool)


class TableMap(Node):
    """Map over an int-valued SOp via an explicit table {int: value}."""

    def __init__(self, table: dict, x: Node, default=None):
        self.table, self.x, self.default = dict(table), x, default

    def children(self):
        return [self.x]

    def eval_(self, ctx, kids):
        (xv, xok), = kids
        out = np.full(xv.shape, np.nan)
        ok = xok.copy()
        xi = np.where(xok, xv, 0).astype(np.int64)
        lut_keys = np.array(sorted(self.table), dtype=np.int64)
        lut_vals = np.array([float(self.table[k]) for k in sorted(self.table)])
        pos = np.searchsorted(lut_keys, xi)
        pos = np.clip(pos, 0, len(lut_keys) - 1)
        hit = lut_keys[pos] == xi
        out[xok & hit] = lut_vals[pos][xok & hit]
        miss = xok & ~hit
        if self.default is not None:
            out[miss] = float(self.default)
        else:
            ok[miss] = False
        return out, ok


class SeqMap(Node):
    """Binary map over two int-valued SOps via table {(v1, v2): value}."""

    def __init__(self, table: dict, x: Node, y: Node):
        self.table, self.x, self.y = dict(table), x, y

    def children(self):
        return [self.x, self.y]

    def eval_(self, ctx, kids):
        (xv, xok), (yv, yok) = kids
        ok = xok & yok
        out = np.full(xv.shape, np.nan)
        pack = {}
        for (a, b), v in self.table.items():
            pack[(int(a), int(b))] = float(v)
        xi = np.where(ok, xv, 0).astype(np.int64)
        yi = np.where(ok, yv, 0).astype(np.int64)
        flat_ok = ok.ravel()
        flat_out = out.ravel()
        for i in np.flatnonzero(flat_ok):
            key = (int(xi.ravel()[i]), int(yi.ravel()[i]))
            if key in pack:
                flat_out[i] = pack[key]
            else:
                flat_ok[i] = False
        return flat_out.reshape(out.shape), flat_ok.reshape(ok.shape)


class NaryMap(Node):
    """N-ary map over int-valued SOps via table {(v1..vn): value}."""

    def __init__(self, table: dict, xs: list):
        self.table = {tuple(int(v) for v in k): float(val)
                      for k, val in table.items()}
        self.xs = list(xs)

    def children(self):
        return self.xs

    def eval_(self, ctx, kids):
        ok = np.ones(kids[0][0].shape, bool)
        for _, kok in kids:
            ok &= kok
        idxs = [np.where(ok, kv, 0).astype(np.int64) for kv, _ in kids]
        out = np.full(idxs[0].shape, np.nan)
        flat_ok = ok.ravel()
        flat_out = out.ravel()
        cols = [ix.ravel() for ix in idxs]
        for i in np.flatnonzero(flat_ok):
            key = tuple(int(c[i]) for c in cols)
            if key in self.table:
                flat_out[i] = self.table[key]
            else:
                flat_ok[i] = False
        return flat_out.reshape(out.shape), flat_ok.reshape(ok.shape)


class Cmp(Node):
    """Numerical comparison against a constant: (x op c) -> {0, 1}."""

    OPS = {"<": np.less, "<=": np.less_equal, "==": np.isclose,
           ">": np.greater, ">=": np.greater_equal}

    def __init__(self, op: str, const: float, x: Node):
        assert op in self.OPS
        self.op, self.const, self.x = op, float(const), x

    def children(self):
        return [self.x]

    def eval_(self, ctx, kids):
        (xv, xok), = kids
        return self.OPS[self.op](xv, self.const).astype(np.float64), xok.copy()


class Between(Node):
    """Numerical interval indicator: (lo < x < hi) -> {0, 1}."""

    def __init__(self, lo: float, hi: float, x: Node):
        self.lo, self.hi, self.x = float(lo), float(hi), x

    def children(self):
        return [self.x]

    def eval_(self, ctx, kids):
        (xv, xok), = kids
        return ((xv > self.lo) & (xv < self.hi)).astype(np.float64), xok.copy()


class LinComb(Node):
    """Linear combination of numerical SOps: sum_i w_i * x_i + c."""

    def __init__(self, weights: list, terms: list, const: float = 0.0):
        self.weights = [float(w) for w in weights]
        self.terms = list(terms)
        self.const = float(const)

    def children(self):
        return self.terms

    def eval_(self, ctx, kids):
        n, L = ctx["content"].shape
        out = np.full((n, L), self.const)
        ok = np.ones((n, L), bool)
        for w, (xv, xok) in zip(self.weights, kids):
            out = out + w * np.where(xok, xv, 0.0)
            ok &= xok
        return out, ok


class Select(Node):
    """Boolean selector: matrix[key_value, query_value] over int domains."""

    def __init__(self, keys: Node, queries: Node, matrix: np.ndarray):
        self.keys, self.queries = keys, queries
        self.matrix = np.asarray(matrix, dtype=bool)

    def children(self):
        return [self.keys, self.queries]

    def eval_sel(self, ctx, kv, ko, qv, qo):
        """Returns (n, Lq, Lk) bool selection."""
        ki = np.clip(np.where(ko, kv, 0).astype(np.int64), 0, self.matrix.shape[0] - 1)
        qi = np.clip(np.where(qo, qv, 0).astype(np.int64), 0, self.matrix.shape[1] - 1)
        sel = self.matrix[ki[:, None, :], qi[:, :, None]]  # (n, Lq, Lk)
        sel &= ko[:, None, :] & qo[:, :, None]
        return sel


class Aggregate(Node):
    """RASP aggregate: mean of selected sop values; default if none."""

    def __init__(self, sel: Select, sop: Node, default=None):
        self.sel, self.sop, self.default = sel, sop, default

    def children(self):
        return [self.sel.keys, self.sel.queries, self.sop]

    def eval_(self, ctx, kids):
        (kv, ko), (qv, qo), (sv, sok) = kids
        sel = self.sel.eval_sel(ctx, kv, ko, qv, qo)
        sel = sel & sok[:, None, :]
        counts = sel.sum(-1)
        vals = np.where(sok, sv, 0.0)
        summed = np.einsum("nqk,nk->nq", sel.astype(np.float64), vals)
        out = np.full(counts.shape, np.nan)
        ok = counts > 0
        out[ok] = summed[ok] / counts[ok]
        if self.default is not None:
            out[~ok] = float(self.default)
            ok = np.ones_like(ok)
        return out, ok


class SelectorWidth(Node):
    def __init__(self, sel: Select):
        self.sel = sel

    def children(self):
        return [self.sel.keys, self.sel.queries]

    def eval_(self, ctx, kids):
        (kv, ko), (qv, qo) = kids
        sel = self.sel.eval_sel(ctx, kv, ko, qv, qo)
        return sel.sum(-1).astype(np.float64), np.ones(qv.shape, bool)


# ------------------------------------------------------------------- soft


class SoftHead(Node):
    """Linear-softmax readout over feature values.

    logits[..., c] = b[c] + sum_f W[f][feature_f_value][c]   (categorical)
                          + sum_g w[g][c] * feature_g_value  (numerical)
    Read at the last position for 'classify', per position otherwise.
    """

    def __init__(self, features: list, tables: list, bias: np.ndarray,
                 num_features: list = (), num_weights: list = ()):
        self.features = list(features)
        self.tables = [np.asarray(t, dtype=np.float64) for t in tables]
        self.bias = np.asarray(bias, dtype=np.float64)
        self.num_features = list(num_features)
        self.num_weights = [np.asarray(w, dtype=np.float64) for w in num_weights]

    def children(self):
        return self.features + self.num_features

    def eval_probs(self, ctx, kids):
        n, L = ctx["content"].shape
        logits = np.broadcast_to(self.bias, (n, L, len(self.bias))).copy()
        cat_kids = kids[: len(self.features)]
        num_kids = kids[len(self.features):]
        for (fv, fok), table in zip(cat_kids, self.tables):
            fi = np.clip(np.where(fok, fv, 0).astype(np.int64), 0, len(table) - 1)
            logits += table[fi]
        for (fv, fok), w in zip(num_kids, self.num_weights):
            logits += np.where(fok, fv, 0.0)[..., None] * w
        logits -= logits.max(-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(-1, keepdims=True)


class MixHead(Node):
    """Convex mixture of hard feature predictions:
    p(y = c) = sum_f pi_f * [feature_f == c]."""

    def __init__(self, features: list, weights: np.ndarray, n_classes: int):
        self.features = list(features)
        self.weights = np.asarray(weights, dtype=np.float64)
        self.n_classes = n_classes

    def children(self):
        return self.features

    def eval_probs(self, ctx, kids):
        n, L = ctx["content"].shape
        p = np.zeros((n, L, self.n_classes))
        for (fv, fok), w in zip(kids, self.weights):
            fi = np.clip(np.where(fok, fv, 0).astype(np.int64), 0, self.n_classes - 1)
            onehot = np.eye(self.n_classes)[fi]
            p += w * onehot
        p = np.clip(p, 1e-12, None)
        return p / p.sum(-1, keepdims=True)


# ------------------------------------------------------------------ program


def _eval_node(node, ctx, cache):
    if id(node) in cache:
        return cache[id(node)]
    kids = [_eval_node(c, ctx, cache) for c in node.children()]
    if isinstance(node, (SoftHead, MixHead)):
        res = node.eval_probs(ctx, kids)
    else:
        res = node.eval_(ctx, kids)
    cache[id(node)] = res
    return res


def _walk(node, seen=None):
    seen = seen if seen is not None else {}
    if id(node) in seen:
        return seen
    seen[id(node)] = node
    for c in node.children():
        _walk(c, seen)
    if isinstance(node, (Aggregate, SelectorWidth)):
        seen[id(node.sel)] = node.sel
    return seen


@dataclasses.dataclass
class Program:
    """A DSL program with oracle-comparable output semantics."""

    output: Node
    kind: str  # matches CaseSpec.kind
    n_outputs: int
    decode: Optional[list] = None  # seq_cat: program value -> spec value
    name: str = "program"

    def _run(self, content):
        ctx = {"content": np.asarray(content, dtype=np.int64)}
        cache = {}
        res = _eval_node(self.output, ctx, cache)
        return res, cache

    def probs(self, content):
        res, _ = self._run(content)
        if isinstance(self.output, (SoftHead, MixHead)):
            p = res  # (n, L, C)
            return p[:, -1] if self.kind == "classify" else p
        # hard program: one-hot over its (int) output values
        vals, ok = res
        vi = np.clip(np.where(ok, vals, 0).astype(np.int64), 0, self.n_outputs - 1)
        p = np.eye(self.n_outputs)[vi]
        return p[:, -1] if self.kind == "classify" else p

    def outputs(self, content):
        """Same output space as Oracle.outputs (values decoded for seq_cat).
        Invalid (None) positions become np.nan / object None."""
        if isinstance(self.output, (SoftHead, MixHead)):
            return self.probs(content).argmax(-1)
        res, _ = self._run(content)
        vals, ok = res
        if self.kind == "classify":
            v = vals[:, -1]
            return np.where(ok[:, -1], v, np.nan).astype(np.int64)
        if self.kind == "seq_num":
            out = vals.copy()
            out[~ok] = np.nan
            return out
        # seq_cat: decode int program values to spec values (vectorized)
        vi = np.where(ok, vals, 0).astype(np.int64)
        if self.decode is not None:
            lut = np.array(self.decode, dtype=object)
            out = lut[np.clip(vi, 0, len(lut) - 1)]
        else:
            out = vi.astype(object)
        out[~ok] = None
        return out

    def intermediates(self, content):
        _, cache = self._run(content)
        nodes = _walk(self.output)
        return {
            (n.name or f"{type(n).__name__}_{i}"): cache.get(id(n))
            for i, n in enumerate(nodes.values())
            if id(n) in cache
        }

    def complexity(self):
        nodes = _walk(self.output).values()
        n_cells = 0
        for n in nodes:
            for attr in ("table", "matrix"):
                t = getattr(n, attr, None)
                if t is not None:
                    n_cells += len(t) if isinstance(t, dict) else int(np.asarray(t).size)
            if isinstance(n, SoftHead):
                n_cells += sum(int(np.count_nonzero(t)) for t in n.tables)
            if isinstance(n, MixHead):
                n_cells += len(n.weights)
        return {"n_nodes": len(list(nodes)), "n_table_cells": int(n_cells)}

    def is_hard(self):
        return not any(
            isinstance(n, (SoftHead, MixHead)) for n in _walk(self.output).values()
        )

    def to_json(self):
        return json.dumps(
            {"name": self.name, "kind": self.kind, "n_outputs": self.n_outputs,
             "complexity": self.complexity(), "is_hard": self.is_hard()},
            indent=2)
