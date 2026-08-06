"""Labeled-basis exact decompiler for TRACR-compiled models.

Exploits the fact that tracr's residual stream is a *named* basis
(architecture.json residual_labels): every dimension belongs to a
variable ("tokens:a", "indices:3", "length_3:5", "is_x_9", "one").

Per layer, each component is symbolized directly from the weights:

  attention head:
    - W_QK = Wq Wk^T restricted to (query-var, key-var) one-hot subspaces
      is a crisp {0, 100} score table -> threshold -> Select matrix
    - value path reading only tokens:bos  -> selector-width circuit;
      the following MLP (bos-weight decoder) is absorbed and the head
      becomes SelectorWidth(sel)
    - otherwise Aggregate(sel, value_var) with the OV map folded in
      (categorical: permutation table; numerical: scalar scale)
  MLP:
    - categorical inputs -> exact lookup table by probing every input
      value combination through relu(x W1 + b1) W2  -> TableMap / SeqMap
    - single numerical input -> dense-grid probe; recognized as a step
      function -> Cmp threshold (else fitted piecewise table fails the
      case loudly)
    - numerical inputs combined linearly -> least-squares -> LinComb
      (exactness verified on the probe set)

The output variable named in io_spec.output_labels becomes the program
root. No sampling from the model is used anywhere — this is pure weight
reading, so exhaustive CEGIS certificates are meaningful end-to-end.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from distill.dsl.core import (Aggregate, Between, Cmp, Indices, LinComb,
                              NaryMap, Program, Select, SelectorWidth, SeqMap,
                              TableMap, Tokens)
from distill.extractors.base import Extractor


def parse_value(s):
    if s is None:
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    if s in ("True", "False"):
        return s == "True"
    return s


class Var:
    def __init__(self, name):
        self.name = name
        self.dims = []
        self.values = []

    @property
    def categorical(self):
        return self.values and self.values[0] is not None

    def __repr__(self):
        return f"Var({self.name}, {len(self.dims)} dims, cat={self.categorical})"


def parse_vars(labels):
    out = {}
    for i, lab in enumerate(labels):
        name, _, val = lab.rpartition(":")
        if not name:
            name, val = lab, None
        v = out.setdefault(name, Var(name))
        v.dims.append(i)
        v.values.append(parse_value(val) if val is not None else None)
    return out


def var_mass(matrix, rows_of, cols_of):
    """Max |matrix| over a dim-subset rectangle."""
    return np.abs(matrix[np.ix_(rows_of, cols_of)]).max() if rows_of and cols_of else 0.0


class DecompileError(Exception):
    pass


class TracrExactExtractor(Extractor):
    name = "tracr_exact"

    def extract(self, oracle, rng=None):
        self.o = oracle
        self.labels = oracle.arch["residual_labels"]
        self.p = oracle.p
        self.vars = parse_vars(self.labels)
        self.one_dim = self.vars["one"].dims[0] if "one" in self.vars else None
        self.node = {}  # var name -> DSL node

        # primitives
        tokens_var = self.vars["tokens"]
        content = oracle.spec.vocab  # sorted content tokens
        tok_map = {}
        for dsl_idx, tok in enumerate(content):
            tok_map[dsl_idx] = tokens_var.values.index(tok)
        self.node["tokens"] = TableMap(tok_map, Tokens()).named("tokens")
        if "indices" in self.vars:
            vals = self.vars["indices"].values
            assert vals == sorted(vals) and vals[0] == 0
            self.node["indices"] = Indices().named("indices")

        H, K = oracle.n_heads, oracle.key_size
        pending_sw = None  # (sel, attn_out_var_name)
        for l in range(oracle.n_layers):
            pending_sw = self._attn(l, H, K, pending_sw)
            pending_sw = self._mlp(l, pending_sw)

        # program root
        out_label = None
        import json
        spec_io = json.loads(
            (Path(oracle.spec.meta["dir"]) / "io_spec.json").read_text())
        first = spec_io["output_labels"][0]
        out_name = first.rpartition(":")[0] or first
        if out_name not in self.node:
            raise DecompileError(f"output var {out_name} was never computed")
        var = self.vars[out_name]
        decode = [v for v in var.values] if var.categorical else None
        return Program(self.node[out_name], oracle.spec.kind,
                       oracle.spec.n_outputs, decode=decode,
                       name=f"decompiled_{oracle.spec.case_id}")

    # -- CEGIS refinement: white-box fault localization ------------------
    #
    # Every tracr variable is written exactly once, so the oracle's final
    # residual stream decodes the ground-truth value of every intermediate
    # variable. On a counterexample we walk variables in construction
    # order, find the first one whose program value diverges from the
    # oracle's, and repair that node's table/threshold entry locally.

    def refine(self, program, oracle, cex_list, rng):
        repaired = False
        for cex in cex_list:
            for row in cex[:32]:
                repaired |= self._repair_row(program, oracle, row[None])
        return program if repaired else None

    def _repair_row(self, program, oracle, c):
        self._repair_content = c
        inter = program.intermediates(c)
        for vname, node in self.node.items():
            if vname in ("tokens", "indices"):
                continue
            var = self.vars[vname]
            gt, gt_ok = oracle.decode_var(c, var.dims, var.categorical)
            pv = inter.get(vname)
            if pv is None:
                continue
            pvals, pok = pv
            for pos in range(c.shape[1]):
                if not gt_ok[0, pos]:
                    continue
                if pok[0, pos]:
                    if var.categorical:
                        bad = int(pvals[0, pos]) != int(gt[0, pos])
                    else:
                        bad = abs(pvals[0, pos] - gt[0, pos]) > 1e-3
                else:
                    bad = True
                if bad:
                    return self._repair_node(node, oracle, c, pos,
                                             gt[0, pos], inter)
        return False

    def _child_val(self, child, inter, pos):
        from distill.dsl.core import _eval_node
        vals, ok = _eval_node(child, {"content": self._repair_content}, {})
        return vals[0, pos], ok[0, pos]

    def _repair_node(self, node, oracle, c, pos, gt_val, inter):
        from distill.dsl.core import (Between, Cmp, LinComb, NaryMap, SeqMap,
                                      TableMap)
        if isinstance(node, TableMap) and not isinstance(
                node.x, (Cmp, Between, LinComb)):
            xv, ok = self._child_val(node.x, inter, pos)
            if not ok:
                return False
            node.table[int(xv)] = gt_val if not isinstance(gt_val, np.integer) \
                else int(gt_val)
            return True
        if isinstance(node, (SeqMap, NaryMap)):
            xs = node.children()
            vals = []
            for ch in xs:
                v, ok = self._child_val(ch, inter, pos)
                if not ok:
                    return False
                vals.append(int(v))
            key = tuple(vals) if isinstance(node, NaryMap) else (vals[0], vals[1])
            node.table[key] = int(gt_val) if isinstance(node, SeqMap) and \
                isinstance(gt_val, (int, np.integer)) else float(gt_val)
            return True
        # threshold chains: TableMap(Cmp/Between) or LinComb([Cmp/Between])
        chain = None
        if isinstance(node, TableMap) and isinstance(node.x, (Cmp, Between)):
            inv = {v: k for k, v in node.table.items()}
            want = inv.get(int(gt_val))
            chain = (node.x, want)
        elif isinstance(node, LinComb) and len(node.terms) == 1 and \
                isinstance(node.terms[0], (Cmp, Between)):
            w, v0 = node.weights[0], node.const
            b = (float(gt_val) - v0) / w if w else None
            if b is not None and abs(b - round(b)) < 1e-3 and round(b) in (0, 1):
                chain = (node.terms[0], int(round(b)))
        if chain is None or chain[1] is None:
            return False
        ind, want_b = chain
        xv, ok = self._child_val(ind.x, inter, pos)
        if not ok:
            return False
        x = float(xv)
        eps = 1e-4
        if isinstance(ind, Cmp) and ind.op == ">":
            if want_b == 1 and x <= ind.const:
                ind.const = x - eps
                return True
            if want_b == 0 and x > ind.const:
                ind.const = x + eps
                return True
            return False
        if isinstance(ind, Between):
            inside = ind.lo < x < ind.hi
            if want_b == 1 and not inside:
                if x <= ind.lo:
                    ind.lo = x - eps
                else:
                    ind.hi = x + eps
                return True
            if want_b == 0 and inside:
                if x - ind.lo < ind.hi - x:
                    ind.lo = x + eps
                else:
                    ind.hi = x - eps
                return True
        return False

    # -- helpers ---------------------------------------------------------

    def _pick_var(self, matrix, side_dims_axis0, exclude=("one",), min_mass=0.5):
        """Which var's dims dominate the given axis of a matrix."""
        best, best_m = None, min_mass
        for name, v in self.vars.items():
            if name in exclude:
                continue
            m = np.abs(matrix[v.dims]).max() if side_dims_axis0 else \
                np.abs(matrix[:, v.dims]).max()
            if m > best_m:
                best, best_m = name, m
        return best

    def _var_node(self, name):
        if name not in self.node:
            raise DecompileError(f"var {name} used before computed")
        return self.node[name]

    def _attn(self, l, H, K, pending_sw):
        p = self.p
        Wq, Wk = p[f"transformer/layer_{l}/attn/query||w"], p[f"transformer/layer_{l}/attn/key||w"]
        Wv, Wo = p[f"transformer/layer_{l}/attn/value||w"], p[f"transformer/layer_{l}/attn/linear||w"]
        for h in range(H):
            s = slice(h * K, (h + 1) * K)
            WQK = Wq[:, s] @ Wk[:, s].T
            WOV = Wv[:, s] @ Wo[s, :]
            if np.abs(WQK).max() < 1.0:
                continue  # inactive head
            # pick (query var, key var) jointly by rectangle mass, keeping
            # the 'one' row (BOS-attend trick) out of the attribution
            qname = kname = None
            best = 1.0
            for qn, qvv in self.vars.items():
                if qn == "one":
                    continue
                for kn, kvv in self.vars.items():
                    if kn == "one":
                        continue
                    m = np.abs(WQK[np.ix_(qvv.dims, kvv.dims)]).max()
                    if m > best:
                        qname, kname, best = qn, kn, m
            if qname is None or kname is None:
                raise DecompileError(f"L{l}h{h}: cannot identify QK vars")
            qv, kv = self.vars[qname], self.vars[kname]
            sub = WQK[np.ix_(qv.dims, kv.dims)]
            if sub.max() <= 1:
                table_qk = np.zeros(sub.shape, bool)
            elif sub.max() - sub.min() < sub.max() / 2:
                table_qk = np.ones(sub.shape, bool)  # constant high: TRUE selector
            else:
                table_qk = sub > (sub.max() + sub.min()) / 2  # [query_val, key_val]

            # value path
            vname = self._pick_var(WOV, True, exclude=("one",))
            if vname is None:
                raise DecompileError(f"L{l}h{h}: no value var")
            vvar = self.vars[vname]
            bos_read = None
            if vname == "tokens":
                reads = np.abs(WOV[vvar.dims]).max(axis=1)
                big = [vvar.values[i] for i in np.flatnonzero(reads > 0.5)]
                if big == ["bos"]:
                    bos_read = True

            sel = self._make_select(qname, kname, table_qk)
            if bos_read:
                pending_sw = (sel, l)
                continue

            oname = self._pick_var(WOV.T, True, exclude=("one", vname))
            if oname is None:
                oname = vname
            ovar = self.vars[oname]
            sub_ov = WOV[np.ix_(vvar.dims, ovar.dims)]
            value_node = self._var_node_for_agg(vname, ovar, sub_ov)
            default = None if ovar.categorical else 0
            self.node[oname] = Aggregate(sel, value_node,
                                         default=default).named(oname)
        return pending_sw

    def _make_select(self, qname, kname, table_qk):
        # DSL Select matrix is [key_val, query_val]; map var-value-index
        # spaces straight through (var nodes emit their own value indices).
        return Select(self._var_node(kname), self._var_node(qname),
                      table_qk.T)

    def _var_node_for_agg(self, vname, ovar, sub_ov):
        vvar = self.vars[vname]
        node = self._var_node(vname)
        if vvar.categorical and ovar.categorical:
            # permutation map value-var index -> out-var index
            mapping = {}
            for i in range(len(vvar.dims)):
                j = int(np.argmax(np.abs(sub_ov[i])))
                if abs(sub_ov[i, j]) > 0.5:
                    mapping[i] = j
            ident = all(k == v for k, v in mapping.items())
            return node if ident and len(mapping) == len(vvar.dims) else \
                TableMap(mapping, node)
        # numerical: scalar scale
        scale = float(sub_ov.ravel()[np.argmax(np.abs(sub_ov))])
        return node if abs(scale - 1) < 1e-6 else LinComb([scale], [node])

    def _mlp(self, l, pending_sw):
        p = self.p
        w1 = p[f"transformer/layer_{l}/mlp/linear_1||w"]
        b1 = p[f"transformer/layer_{l}/mlp/linear_1||b"]
        w2 = p[f"transformer/layer_{l}/mlp/linear_2||w"]
        b2 = p[f"transformer/layer_{l}/mlp/linear_2||b"]

        def run(x):
            return np.maximum(x @ w1 + b1, 0) @ w2 + b2

        in_names = [n for n, v in self.vars.items()
                    if n not in ("one",) and np.abs(w1[v.dims]).max() > 0.5]
        out_names = [n for n, v in self.vars.items()
                     if n not in ("one",) and np.abs(w2[:, v.dims]).max() > 0.5
                     and n not in in_names]
        if not out_names:
            return pending_sw

        # selector-width decoder MLP reads the attn scratch dim; other
        # outputs of the same MLP layer are fitted normally
        sw_scratch = [n for n in in_names if n.endswith("selector_width_attn_output")]
        fit_ins = [n for n in in_names if not n.endswith("selector_width_attn_output")]
        sw_used = False
        for oname in out_names:
            ovar = self.vars[oname]
            is_count_var = (ovar.categorical
                            and all(isinstance(v, int) for v in ovar.values)
                            and sorted(ovar.values) == list(range(len(ovar.values))))
            if sw_scratch and pending_sw is not None and is_count_var:
                sw = SelectorWidth(pending_sw[0])
                # Always wrap in a count -> value-index table: it is the
                # CEGIS repair point when the model's own width decoder
                # deviates (e.g. saturation at full sequence length).
                count_to_idx = {v: i for i, v in enumerate(ovar.values)}
                self.node[oname] = TableMap(count_to_idx, sw).named(oname)
                sw_used = True
            else:
                self._mlp_one_output(l, run, fit_ins, oname)
        return None if sw_used else pending_sw

    def _mlp_one_output(self, l, run, in_names, oname):
        ovar = self.vars[oname]
        d = len(self.labels)
        cat_ins = [n for n in in_names if self.vars[n].categorical]
        num_ins = [n for n in in_names if not self.vars[n].categorical]

        def probe(assign_dims):
            x = np.zeros((len(assign_dims), d))
            if self.one_dim is not None:
                x[:, self.one_dim] = 1.0
            for r, dims_vals in enumerate(assign_dims):
                for dim, val in dims_vals:
                    x[r, dim] = val
            out = run(x)[:, ovar.dims]
            if ovar.categorical:
                idx = out.argmax(1)
                ok = out[np.arange(len(out)), idx] > 0.2
                return np.where(ok, idx, -1)
            return out[:, 0]

        if not num_ins:
            vars_in = [self.vars[n] for n in cat_ins]
            if len(vars_in) == 1:
                v = vars_in[0]
                rows = [[(v.dims[i], 1.0)] for i in range(len(v.dims))]
                res = probe(rows)
                table = {i: r for i, r in enumerate(res) if r != -1 or not ovar.categorical}
                if ovar.categorical:
                    table = {i: int(r) for i, r in enumerate(res) if r != -1}
                node = TableMap(table, self._var_node(cat_ins[0]))
            else:
                import itertools
                sizes = [len(v.dims) for v in vars_in]
                if np.prod(sizes) > 20000:
                    raise DecompileError(
                        f"L{l} MLP {oname}: cat product too large {sizes}")
                rows, keys = [], []
                for combo in itertools.product(*(range(s) for s in sizes)):
                    rows.append([(v.dims[i], 1.0) for v, i in zip(vars_in, combo)])
                    keys.append(combo)
                res = probe(rows)
                if ovar.categorical:
                    table = {k: int(r) for k, r in zip(keys, res) if r != -1}
                else:
                    table = {k: float(r) for k, r in zip(keys, res)}
                nodes_in = [self._var_node(n) for n in cat_ins]
                if len(vars_in) == 2:
                    node = SeqMap(table, *nodes_in)
                else:
                    node = NaryMap(table, nodes_in)
        elif len(num_ins) == 1 and not cat_ins:
            node = self._fit_numeric_mlp(l, run, num_ins[0], ovar, probe)
        elif not cat_ins:
            node = self._fit_linear_mlp(l, run, num_ins, ovar)
        else:
            raise DecompileError(f"L{l} MLP {oname}: mixed cat+num inputs")
        self.node[oname] = node.named(oname)

    def _fit_numeric_mlp(self, l, run, in_name, ovar, probe):
        ivar = self.vars[in_name]
        dim = ivar.dims[0]
        # tracr's numerical MLPs are discretized over the achievable value
        # range; our numerical vars are balances/fractions in [-1, 1]
        grid = np.round(np.linspace(-1, 1, 1601), 6)
        rows = [[(dim, g)] for g in grid]
        res = probe(rows)
        node_in = self._var_node(in_name)
        if ovar.categorical:
            # drop uncertain boundary rows, then recognize step / interval
            keep = res != -1
            g, r = grid[keep], res[keep]
            change = np.flatnonzero(r[1:] != r[:-1])
            if len(set(r)) == 2 and len(change) == 1:
                c = (g[change[0]] + g[change[0] + 1]) / 2
                lo, hi = int(r[0]), int(r[-1])
                return TableMap({0: lo, 1: hi}, Cmp(">", c, node_in))
            if len(set(r)) == 2 and len(change) == 2 and r[0] == r[-1]:
                lo_b = (g[change[0]] + g[change[0] + 1]) / 2
                hi_b = (g[change[1]] + g[change[1] + 1]) / 2
                outside, inside = int(r[0]), int(r[change[0] + 1])
                return TableMap({0: outside, 1: inside},
                                Between(lo_b, hi_b, node_in))
            raise DecompileError(f"L{l} MLP {ovar.name}: non-step numeric->cat")
        # numerical->numerical: affine fit, else two-valued step/interval
        A = np.column_stack([grid, np.ones_like(grid)])
        coef, *_ = np.linalg.lstsq(A, res, rcond=None)
        if np.abs(A @ coef - res).max() <= 1e-3:
            return LinComb([coef[0]], [node_in], const=coef[1])
        # two-plateau step/interval, tolerating soft transition samples
        lo, hi = res.min(), res.max()
        near_lo = np.abs(res - lo) < 1e-3
        near_hi = np.abs(res - hi) < 1e-3
        keep = near_lo | near_hi
        if keep.mean() > 0.98 and hi - lo > 1e-3:
            g2 = grid[keep]
            v2 = near_hi[keep].astype(int)
            change = np.flatnonzero(v2[1:] != v2[:-1])
            plat = np.array([lo, hi])
            if len(change) == 1:
                c = (g2[change[0]] + g2[change[0] + 1]) / 2
                v0, v1 = plat[v2[0]], plat[v2[-1]]
                return LinComb([v1 - v0], [Cmp(">", c, node_in)], const=v0)
            if len(change) == 2 and v2[0] == v2[-1]:
                lo_b = (g2[change[0]] + g2[change[0] + 1]) / 2
                hi_b = (g2[change[1]] + g2[change[1] + 1]) / 2
                v0, v1 = plat[v2[0]], plat[v2[change[0] + 1]]
                return LinComb([v1 - v0], [Between(lo_b, hi_b, node_in)], const=v0)
        raise DecompileError(f"L{l} MLP {ovar.name}: non-affine numeric map")

    def _fit_linear_mlp(self, l, run, num_ins, ovar):
        d = len(self.labels)
        dims = [self.vars[n].dims[0] for n in num_ins]
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (400, len(dims)))
        xin = np.zeros((400, d))
        if self.one_dim is not None:
            xin[:, self.one_dim] = 1.0
        for j, dim in enumerate(dims):
            xin[:, dim] = X[:, j]
        y = (np.maximum(xin @ self.p[f"transformer/layer_{l}/mlp/linear_1||w"]
                        + self.p[f"transformer/layer_{l}/mlp/linear_1||b"], 0)
             @ self.p[f"transformer/layer_{l}/mlp/linear_2||w"]
             + self.p[f"transformer/layer_{l}/mlp/linear_2||b"])[:, ovar.dims[0]]
        A = np.column_stack([X, np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if np.abs(A @ coef - y).max() <= 1e-2:
            return LinComb(list(coef[:-1]), [self._var_node(n) for n in num_ins],
                           const=coef[-1])
        if len(num_ins) == 2:
            return self._fit_2d_threshold_table(l, num_ins, ovar)
        raise DecompileError(f"L{l} MLP {ovar.name}: non-linear multi-numeric")

    def _mlp_out(self, l, xin, ovar):
        """Evaluate the MLP on raw inputs, read variable ovar.
        Categorical: argmax index with margin-based confidence (-1 if
        ambiguous); numerical: scalar value."""
        w1 = self.p[f"transformer/layer_{l}/mlp/linear_1||w"]
        b1 = self.p[f"transformer/layer_{l}/mlp/linear_1||b"]
        w2 = self.p[f"transformer/layer_{l}/mlp/linear_2||w"]
        b2 = self.p[f"transformer/layer_{l}/mlp/linear_2||b"]
        out = (np.maximum(xin @ w1 + b1, 0) @ w2 + b2)[:, ovar.dims]
        if not ovar.categorical:
            return out[:, 0]
        if out.shape[1] == 1:
            return (out[:, 0] > 0.5).astype(int)
        srt = np.sort(out, axis=1)
        idx = out.argmax(1)
        margin = srt[:, -1] - srt[:, -2]
        return np.where(margin > 0.3, idx, -1)

    def _fit_2d_threshold_table(self, l, num_ins, ovar):
        """Fit f(x, y) as a table over threshold-bucketed inputs (covers
        boolean combinations of per-input sign tests, e.g. (x<0) | (y<0))."""
        d = len(self.labels)
        dims = [self.vars[n].dims[0] for n in num_ins]
        g = np.round(np.linspace(-1.5, 1.5, 121), 6)
        XX, YY = np.meshgrid(g, g, indexing="ij")
        xin = np.zeros((XX.size, d))
        if self.one_dim is not None:
            xin[:, self.one_dim] = 1.0
        xin[:, dims[0]] = XX.ravel()
        xin[:, dims[1]] = YY.ravel()
        Z = self._mlp_out(l, xin, ovar).reshape(XX.shape)
        if not ovar.categorical:
            Z = np.round(Z, 4)
        valid = Z != -1 if ovar.categorical else np.ones(Z.shape, bool)

        def boundaries(axis):
            bs = set()
            for s in range(0, 121, 12):
                line = Z[:, s] if axis == 0 else Z[s, :]
                lv = valid[:, s] if axis == 0 else valid[s, :]
                gi = np.flatnonzero(lv)
                seg, gseg = line[gi], g[gi]
                for c in np.flatnonzero(seg[1:] != seg[:-1]):
                    bs.add(round((gseg[c] + gseg[c + 1]) / 2, 3))
            merged = []
            for b in sorted(bs):
                if not merged or b - merged[-1] > 0.08:
                    merged.append(b)
            return merged

        bx, by = boundaries(0), boundaries(1)
        if len(bx) > 2 or len(by) > 2:
            raise DecompileError(f"L{l} MLP {ovar.name}: too many 2d thresholds")

        def bucket(vals, bs):
            idx = np.zeros(vals.shape, dtype=int)
            for b in bs:
                idx += vals > b
            return idx

        bxi, byi = bucket(XX, bx), bucket(YY, by)
        table = {}
        for i in range(len(bx) + 1):
            for j in range(len(by) + 1):
                cell = Z[(bxi == i) & (byi == j) & valid]
                if len(cell) == 0:
                    continue
                if np.ptp(cell) > 1e-3:
                    raise DecompileError(
                        f"L{l} MLP {ovar.name}: non-constant 2d cell")
                table[(i, j)] = float(cell[0]) if not ovar.categorical else int(cell[0])

        def bucket_node(name, bs):
            node = self._var_node(name)
            cmps = [Cmp(">", b, node) for b in bs]
            return cmps[0] if len(cmps) == 1 else LinComb([1.0] * len(cmps), cmps)

        return SeqMap(table, bucket_node(num_ins[0], bx),
                      bucket_node(num_ins[1], by))
