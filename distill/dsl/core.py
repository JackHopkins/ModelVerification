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
        # vectorized fast path: dense LUT over the key grid when small
        dims = tuple(max(k[a] for k in self.table) + 1
                     for a in range(len(idxs))) if self.table else ()
        if self.table and np.prod(dims) <= 1_000_000 \
                and all(min(k[a] for k in self.table) >= 0
                        for a in range(len(idxs))):
            lut = np.full(dims, np.nan)
            for k, v in self.table.items():
                lut[k] = v
            inb = np.ones(ok.shape, bool)
            for ix, d in zip(idxs, dims):
                inb &= (ix >= 0) & (ix < d)
            safe = [np.clip(ix, 0, d - 1) for ix, d in zip(idxs, dims)]
            out = np.where(inb, lut[tuple(safe)], np.nan)
            ok = ok & inb & ~np.isnan(out)
            return np.where(ok, out, np.nan), ok
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


class Coalesce(Node):
    """Boundary default: x where defined, const where x is invalid."""

    def __init__(self, x: Node, const: float = 0.0):
        self.x, self.const = x, float(const)

    def children(self):
        return [self.x]

    def eval_(self, ctx, kids):
        (xv, xok), = kids
        return np.where(xok, xv, self.const), np.ones(xv.shape, bool)


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

    def source(self):
        return render_source(self)


# ------------------------------------------------------------------ source


def _fmt_num(v):
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:.3g}"


def _fmt_table(table, max_items=10):
    keys = sorted(table)
    vals = {float(table[k]) for k in keys}
    if vals <= {0.0, 1.0}:  # indicator table
        ones = [k for k in keys if float(table[k]) == 1.0]
        if len(ones) <= max_items:
            return f"1 if x in {{{', '.join(map(str, ones))}}} else 0"
    items = [f"{k}→{_fmt_num(table[k])}" for k in keys[:max_items]]
    tail = f", …(+{len(keys) - max_items} more)" if len(keys) > max_items else ""
    return "{" + ", ".join(items) + tail + "}"


def _fmt_matrix(m):
    """Recognize common selector predicates, else summarize true cells."""
    a, b = m.shape
    k, q = np.indices((a, b))
    for pat, desc in ((k == q, "key == query"), (k < q, "key < query"),
                      (k <= q, "key <= query"), (k > q, "key > query"),
                      (k >= q, "key >= query")):
        if a == b and np.array_equal(m, pat):
            return desc
    if m.all():
        return "always"
    cells = np.argwhere(m)
    if len(cells) <= 8:
        body = ", ".join(f"({ki},{qi})" for ki, qi in cells)
        return f"true at (key,query) ∈ {{{body}}}"
    return f"{a}×{b} matrix, {len(cells)} true cells"


def render_source(program):
    """Readable pseudo-RASP: one assignment per DAG node, topological."""
    order = list(_walk(program.output).values())
    names, lines, counter = {}, [], [0]

    def nm(node):
        if id(node) in names:
            return names[id(node)]
        counter[0] += 1
        n = node.name or f"v{counter[0]}"
        # disambiguate repeated names
        while n in names.values():
            n += "_"
        names[id(node)] = n
        return n

    def expr(node):
        t = type(node).__name__
        if isinstance(node, Tokens):
            return "tokens"
        if isinstance(node, Indices):
            return "indices"
        if isinstance(node, Const):
            return f"const({_fmt_num(node.value)})"
        if isinstance(node, TableMap):
            d = f", default={_fmt_num(node.default)}" if node.default is not None else ""
            return f"map({nm(node.x)}, {_fmt_table(node.table)}{d})"
        if isinstance(node, SeqMap):
            return f"map2({nm(node.x)}, {nm(node.y)}, {_fmt_table(node.table)})"
        if isinstance(node, NaryMap):
            xs = ", ".join(nm(x) for x in node.xs)
            return f"mapN([{xs}], {_fmt_table(node.table)})"
        if isinstance(node, Coalesce):
            return f"coalesce({nm(node.x)}, {_fmt_num(node.const)})"
        if isinstance(node, Cmp):
            return f"({nm(node.x)} {node.op} {_fmt_num(node.const)})"
        if isinstance(node, Between):
            return f"({_fmt_num(node.lo)} < {nm(node.x)} < {_fmt_num(node.hi)})"
        if isinstance(node, LinComb):
            terms = " + ".join(f"{_fmt_num(w)}*{nm(x)}"
                               for w, x in zip(node.weights, node.terms))
            c = f" + {_fmt_num(node.const)}" if node.const else ""
            return terms + c if terms else _fmt_num(node.const)
        if isinstance(node, Select):
            return (f"select(key={nm(node.keys)}, query={nm(node.queries)}, "
                    f"{_fmt_matrix(node.matrix)})")
        if isinstance(node, Aggregate):
            d = f", default={_fmt_num(node.default)}" if node.default is not None else ""
            return f"aggregate({nm(node.sel)}, {nm(node.sop)}{d})"
        if isinstance(node, SelectorWidth):
            return f"selector_width({nm(node.sel)})"
        if isinstance(node, (SoftHead, MixHead)):
            return t  # rendered in detail below
        return t

    # emit children before parents
    emitted = set()

    def emit(node):
        if id(node) in emitted:
            return
        emitted.add(id(node))
        for c in node.children():
            emit(c)
        if isinstance(node, (Aggregate, SelectorWidth)):
            emit(node.sel)
        if isinstance(node, SoftHead):
            lines.append(f"{nm(node)} = softmax_readout(")
            lines.append(f"  bias = [{', '.join(_fmt_num(b) for b in node.bias)}]")
            for f, tbl in zip(node.features, node.tables):
                votes = []
                for v in range(len(tbl)):
                    if np.abs(tbl[v]).max() < 1e-3:
                        continue
                    c = int(tbl[v].argmax())
                    votes.append(f"{v}→class{c}({tbl[v][c]:+.1f})")
                    if len(votes) >= 8 and v < len(tbl) - 1:
                        votes.append(f"…({len(tbl)} values)")
                        break
                lines.append(f"  {nm(f)}: " + " ".join(votes))
            for f, w in zip(node.num_features, node.num_weights):
                ws = ", ".join(f"{x:+.2f}" for x in w)
                lines.append(f"  {nm(f)} (numeric): w=[{ws}]")
            lines.append(")")
        elif isinstance(node, MixHead):
            body = " + ".join(f"{_fmt_num(w)}·onehot({nm(f)})"
                              for f, w in zip(node.features, node.weights))
            lines.append(f"{nm(node)} = mixture({body})")
        elif isinstance(node, Select):
            lines.append(f"{nm(node)} = {expr(node)}")
        else:
            lines.append(f"{nm(node)} = {expr(node)}")

    emit(program.output)
    head = (f"# {program.name}  kind={program.kind}  "
            f"n_outputs={program.n_outputs}  nodes={len(order)}")
    out_name = names[id(program.output)]
    return "\n".join([head] + lines + [f"return {out_name}"])
