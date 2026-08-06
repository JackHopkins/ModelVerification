"""Baseline C — behavioral control: decision trees over IO features.

Fits CART on (featurized input -> oracle output), ignoring the network's
internals entirely. Soft variant: leaf distributions are the mean oracle
probabilities of training rows in the leaf, capturing calibration that
argmax targets alone would lose.

Sequence kinds are handled by flattening positions into rows with
positional context features (position, token windows, whole-sequence and
prefix counts). Structure-heavy tasks (reverse, sort) are expected to
defeat this baseline — that is what makes it the control condition.
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

from .base import Extractor


def _pad_to(arr, L, fill=-1):
    if arr.shape[1] == L:
        return arr
    pad = np.full((len(arr), L - arr.shape[1]), fill, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def seq_features(content, max_len, n_vocab):
    """Per-position rows: (n*L, F). Features: pos, L, tok@pos, tok@pos-1,
    tok@pos+1, tok@0, tok@last, all-seq counts, prefix counts <= pos."""
    n, L = content.shape
    rows = []
    counts = np.stack([(content == v).sum(1) for v in range(n_vocab)], axis=1)
    onehot = np.stack([(content == v) for v in range(n_vocab)], axis=2)  # (n,L,V)
    prefix = np.cumsum(onehot, axis=1)  # counts of v in [0..pos]
    for pos in range(L):
        prev_tok = content[:, pos - 1] if pos > 0 else np.full(n, -1)
        next_tok = content[:, pos + 1] if pos < L - 1 else np.full(n, -1)
        feat = np.column_stack([
            np.full(n, pos), np.full(n, L),
            content[:, pos], prev_tok, next_tok,
            content[:, 0], content[:, -1],
            counts, prefix[:, pos, :],
        ])
        rows.append(feat)
    return np.concatenate(rows, axis=0)  # ordered pos-major


def classify_features(content, n_vocab):
    counts = np.stack([(content == v).sum(1) for v in range(n_vocab)], axis=1)
    return np.column_stack([content, counts])


class TreeProgram:
    def __init__(self, mode, trees, leaf_probs, oracle_spec, max_depth):
        self.mode = mode  # 'classify' | 'seq' | 'lm'
        self.trees = trees  # dict or single tree
        self.leaf_probs = leaf_probs  # tree -> {leaf_id: prob vector}
        self.spec = oracle_spec
        self.max_depth = max_depth
        self.name = "tree"

    # -- feature plumbing -------------------------------------------------
    def _X(self, content):
        V = len(self.spec.vocab)
        if self.mode == "classify":
            return classify_features(content, V)
        if self.mode == "lm":
            n, L = content.shape
            prev2 = np.column_stack([np.full(n, -1), content[:, :-1]])
            return np.stack([np.arange(L)[None].repeat(n, 0),
                             content, prev2], axis=2).reshape(n * L, 3)
        return seq_features(content, max(self.spec.seq_lens), V)

    def probs(self, content):
        tree = self.trees
        X = self._X(content)
        leaves = tree.apply(X)
        C = self.spec.n_outputs
        P = np.empty((len(X), C))
        default = np.full(C, 1.0 / C)
        lp = self.leaf_probs
        uniq, inv = np.unique(leaves, return_inverse=True)
        table = np.stack([lp.get(int(u), default) for u in uniq])
        P = table[inv]
        n, L = content.shape
        if self.mode == "classify":
            return P
        P = P.reshape(L, n, C).transpose(1, 0, 2) if self.mode == "seq" else P.reshape(n, L, C)
        return P

    def outputs(self, content):
        n, L = content.shape
        kind = self.spec.kind
        if kind == "seq_num":
            X = self._X(content)
            pred = self.trees.predict(X)
            return pred.reshape(L, n).T
        if kind == "classify":
            return self.probs(content).argmax(-1)
        if kind == "lm":
            return self.probs(content).argmax(-1)
        # seq_cat: argmax over leaf distribution, decoded
        idx = self.probs(content).argmax(-1)
        vals = np.array(self.spec.output_values, dtype=object)
        return vals[idx]

    def complexity(self):
        return {"n_nodes": int(self.trees.tree_.node_count), "n_table_cells": 0}

    def is_hard(self):
        return False  # leaf distributions are soft

    def source(self):
        return export_text(self.trees, max_depth=6)


class TreeExtractor(Extractor):
    name = "tree"

    def __init__(self, n_train=60000, max_depth=12, min_leaf=5):
        self.n_train, self.max_depth, self.min_leaf = n_train, max_depth, min_leaf

    def _dataset(self, oracle, rng, n, weights=None):
        xs, ys, ps = [], [], []
        for content in oracle.sample_batches(rng, n):
            xs.append(content)
            ys.append(oracle.outputs(content))
            p = oracle.probs(content)
            ps.append(p)
        return xs, ys, ps

    def extract(self, oracle, rng, extra=None):
        kind = oracle.spec.kind
        mode = {"classify": "classify", "lm": "lm"}.get(kind, "seq")
        xs, ys, ps = self._dataset(oracle, rng, self.n_train)
        if extra is not None:
            for content in extra:
                xs.append(content)
                ys.append(oracle.outputs(content))
                ps.append(oracle.probs(content))

        prog = TreeProgram(mode, None, {}, oracle.spec, self.max_depth)
        X = np.concatenate([prog._X(c) for c in xs])
        n_pos_rows = {"classify": 1}.get(mode)

        if kind == "seq_num":
            y = np.concatenate([o.T.ravel() for o in ys])  # pos-major like seq_features
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         min_samples_leaf=self.min_leaf)
            tree.fit(X, y)
            prog.trees = tree
            return prog

        if kind == "seq_cat":
            y = np.concatenate([o.T.ravel() for o in ys])
        elif kind == "classify":
            y = np.concatenate(ys)
        else:  # lm: row for position i targets token i+1; last position has
            # no target, so drop those rows (X was built row-major (n, L)).
            L = xs[0].shape[1]
            keep = np.concatenate(
                [np.tile(np.arange(L) < L - 1, len(c)) for c in xs])
            y = np.concatenate([c[:, 1:].ravel() for c in xs])
            X = X[keep]
        tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                      min_samples_leaf=self.min_leaf)
        tree.fit(X, y)
        prog.trees = tree

        # soft leaves: mean oracle prob per leaf (or empirical target dist)
        leaves = tree.apply(X)
        C = oracle.spec.n_outputs
        if ps[0] is not None:
            if kind == "lm":
                # oracle probs at position i are the distribution of token i+1
                P = np.concatenate(
                    [p[:, :-1].reshape(-1, C) for p in ps])
            elif kind == "classify":
                P = np.concatenate(ps)
            else:
                P = None
        else:
            P = None
        leaf_probs = {}
        if P is not None and len(P) == len(leaves):
            for leaf in np.unique(leaves):
                leaf_probs[int(leaf)] = P[leaves == leaf].mean(0)
        else:
            onehot = np.eye(C)[y.astype(int)]
            for leaf in np.unique(leaves):
                leaf_probs[int(leaf)] = onehot[leaves == leaf].mean(0)
        prog.leaf_probs = leaf_probs
        return prog

    def refine(self, program, oracle, cex, rng):
        # refit with counterexample arrays replicated into the pool
        extra = [arr for arr in cex for _ in range(10)]
        return self.extract(oracle, rng, extra=extra)
