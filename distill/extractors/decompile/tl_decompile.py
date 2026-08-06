"""Feature-probing decompiler for TransformerLens models (interp + messy).

The lacoco-lab exact D-RASP reparametrization is cluster-oriented; per
plan this is the scoped reimplementation of its substance for 2-layer
models: express the model as an interpretable feature bank + fitted
readout, with every feature *grounded* against the model's residual
stream, and CEGIS supplying the fidelity check downstream.

Pipeline:
  1. Build a candidate feature bank as DSL nodes (so the result is a real
     program, not a black box): token identity, position, first/last
     token, per-token whole-sequence counts (SelectorWidth), per-token
     running fractions (numerical Aggregate over LEQ positions).
  2. Evaluate features on samples; fit a readout against the ORACLE:
       - classify / lm / seq_cat: multinomial logistic -> SoftHead
         (soft variant IS the program head; harden() available)
       - seq_num: ridge regression -> LinComb over numerical features
  3. Keep only features with meaningful weight (sparsity by pruning);
     refit. Report per-feature grounding: R^2 of a linear probe from the
     model's final residual stream to the feature value.
  4. refine(): refit the readout with counterexamples upweighted.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from distill.dsl.core import (Aggregate, Indices, LinComb, Program, Select,
                              SelectorWidth, SoftHead, TableMap, Tokens,
                              _eval_node)
from distill.extractors.base import Extractor


def build_feature_bank(spec):
    """Returns (cat_feats, num_feats): lists of (name, node, cardinality)."""
    V = len(spec.vocab)
    L = max(spec.seq_lens)
    tok = Tokens().named("tok")
    idx = Indices().named("pos")
    cat, num = [], []
    cat.append(("tok", tok, V))
    cat.append(("pos", idx, L))
    # token at fixed positions (first, last) broadcast to every position
    all_q = np.ones((L, L), bool)
    first_sel = Select(idx, idx, np.outer(np.arange(L) == 0, np.ones(L, bool)).T
                       if False else np.tile((np.arange(L) == 0)[:, None], (1, L)))
    cat.append(("first_tok", Aggregate(first_sel, tok).named("first_tok"), V))
    last_sel = Select(idx, idx, np.tile((np.arange(L) == L - 1)[:, None], (1, L)))
    cat.append(("last_tok", Aggregate(last_sel, tok).named("last_tok"), V))
    for v in range(V):
        m = np.zeros((V, V), bool)
        m[v, :] = True
        cat.append((f"count_{spec.vocab[v]}",
                    SelectorWidth(Select(tok, tok, m)).named(f"count_{v}"),
                    L + 1))
    leq = np.tril(np.ones((L, L), bool)).T  # matrix[kv, qv] = kv <= qv
    for v in range(V):
        ind = TableMap({i: float(i == v) for i in range(V)}, tok)
        num.append((f"frac_{spec.vocab[v]}",
                    Aggregate(Select(idx, idx, leq), ind, default=0).named(f"frac_{v}")))
    return cat, num


def eval_features(feats, content):
    """(n, L, F) matrix of feature values (nan-safe)."""
    ctx = {"content": np.asarray(content, dtype=np.int64)}
    cols = []
    for _, node, *_ in feats:
        vals, ok = _eval_node(node, ctx, {})
        cols.append(np.where(ok, vals, 0.0))
    return np.stack(cols, axis=2) if cols else np.zeros(content.shape + (0,))


class TLDecompileExtractor(Extractor):
    name = "tl_decompile"

    def __init__(self, n_train=30000, l1=0.02, keep_top=6,
                 penalty="l1", max_rows=None):
        self.n_train, self.l1, self.keep_top = n_train, l1, keep_top
        self.penalty, self.max_rows = penalty, max_rows

    def _dataset(self, oracle, rng, n, extra=None):
        batches = list(oracle.sample_batches(rng, n))
        if extra:
            for cex in extra:
                batches.extend([cex] * 5)  # upweight counterexamples
        return batches

    def _fit_readout(self, oracle, batches, cat, num):
        from sklearn.linear_model import LogisticRegression, Ridge

        kind = oracle.spec.kind
        C = oracle.spec.n_outputs
        Xc_l, Xn_l, y_l, w_l = [], [], [], []
        for content in batches:
            fc = eval_features(cat, content)
            fn = eval_features(num, content)
            if kind == "classify":
                fc, fn = fc[:, -1:], fn[:, -1:]
                tgt = oracle.probs(content)[:, None]  # (n, 1, C)
            elif kind == "lm":
                tgt = oracle.probs(content)
            elif kind == "seq_cat":
                tgt = np.eye(C)[oracle.outputs(content)]
            else:  # seq_num
                tgt = oracle.outputs(content)[..., None]
            Xc_l.append(fc.reshape(fc.shape[0] * fc.shape[1], fc.shape[2]))
            Xn_l.append(fn.reshape(fn.shape[0] * fn.shape[1], fn.shape[2]))
            y_l.append(tgt.reshape(-1, tgt.shape[-1]))
        Xc, Xn, Y = map(np.concatenate, (Xc_l, Xn_l, y_l))
        if self.max_rows and len(Y) > self.max_rows:
            keep = np.random.default_rng(0).choice(
                len(Y), self.max_rows, replace=False)
            Xc, Xn, Y = Xc[keep], Xn[keep], Y[keep]

        # one-hot expand categorical features
        card = [c[2] for c in cat]
        Xoh = np.concatenate(
            [np.eye(k)[np.clip(Xc[:, j].astype(int), 0, k - 1)]
             for j, k in enumerate(card)] + [Xn], axis=1)

        if kind == "seq_num":
            model = Ridge(alpha=1e-3)
            model.fit(Xoh, Y[:, 0])
            return ("ridge", model, card)
        # soft targets: sample hard labels proportionally is noisy; use
        # probability-weighted fit via duplicating with sample_weight
        labels = Y.argmax(1)
        weights = Y.max(1)
        if self.penalty == "l1":
            model = LogisticRegression(
                penalty="l1", C=1.0 / self.l1, solver="saga", max_iter=200,
                tol=1e-3)
        else:
            model = LogisticRegression(
                penalty="l2", C=10.0, solver="lbfgs", max_iter=500, tol=1e-4)
        model.fit(Xoh, labels, sample_weight=weights)
        # calibrate soft blend: refit on soft targets by expanding each row
        # into (label c, weight Y[c]) pairs for the top-2 classes
        top2 = np.argsort(-Y, axis=1)[:, :2]
        rows = np.concatenate([np.arange(len(Y))] * 2)
        labs = top2.T.reshape(-1)
        ws = Y[np.arange(len(Y))[:, None], top2].T.reshape(-1)
        keep = ws > 1e-3
        model.fit(Xoh[rows[keep]], labs[keep], sample_weight=ws[keep])
        return ("logistic", model, card)

    def _to_program(self, oracle, fit, cat, num):
        kind, model, card = fit
        spec = oracle.spec
        if kind == "ridge":
            # LinComb over numerical features + per-value tables via
            # TableMap contributions folded into weights on cat one-hots
            coefs = model.coef_
            off = int(np.sum(card))
            terms, weights = [], []
            for j, (name, node) in enumerate(num):
                w = coefs[off + j]
                if abs(w) > 1e-4:
                    terms.append(node)
                    weights.append(float(w))
            pos = 0
            for (name, node, k) in cat:
                w = coefs[pos:pos + k]
                if np.abs(w).max() > 1e-4:
                    terms.append(TableMap({i: float(w[i]) for i in range(k)}, node))
                    weights.append(1.0)
                pos += k
            out = LinComb(weights, terms, const=float(model.intercept_))
            return Program(out, spec.kind, spec.n_outputs,
                           name=f"tl_decompiled_{spec.case_id}")
        # logistic -> SoftHead. sklearn only emits rows for classes seen in
        # training; map rows to their class indices, unseen classes get a
        # large negative bias (never predicted).
        C_out = spec.n_outputs
        F_in = model.coef_.shape[1]
        classes = model.classes_.astype(int)
        coefs = np.zeros((C_out, F_in))
        intercept = np.full(C_out, -30.0)
        if model.coef_.shape[0] == 1 and len(classes) == 2:
            coefs[classes[0]] = -model.coef_[0] / 2
            coefs[classes[1]] = model.coef_[0] / 2
            intercept[classes[0]] = -model.intercept_[0] / 2
            intercept[classes[1]] = model.intercept_[0] / 2
        else:
            for r, cls in enumerate(classes):
                coefs[cls] = model.coef_[r]
                intercept[cls] = model.intercept_[r]
        tables, feats = [], []
        pos = 0
        for (name, node, k) in cat:
            t = coefs[:, pos:pos + k].T  # (k, C)
            pos += k
            if np.abs(t).max() > 1e-3:
                feats.append(node)
                tables.append(t)
        num_feats, num_ws = [], []
        for j, (name, node) in enumerate(num):
            w = coefs[:, pos + j]
            if np.abs(w).max() > 1e-3:
                num_feats.append(node)
                num_ws.append(w)
        head = SoftHead(feats, tables, intercept, num_feats, num_ws)
        return Program(head, spec.kind, spec.n_outputs,
                       decode=spec.output_values,
                       name=f"tl_decompiled_{spec.case_id}")

    def grounding(self, oracle, program, rng, n=4000):
        """R^2 of linear probes from the model's final residual stream to
        each feature used by the program — are the features mechanistically
        represented, not just IO-predictive?"""
        import torch
        content = next(oracle.sample_batches(rng, n))
        ids = oracle._encode(content) if hasattr(oracle, "_encode") else content
        with torch.no_grad():
            _, cache = oracle.model.run_with_cache(torch.from_numpy(ids).long())
        resid = cache["resid_post", -1].float().cpu().numpy()
        if resid.shape[1] != content.shape[1]:
            resid = resid[:, 1:]
        out = {}
        head = program.output
        feats = getattr(head, "features", []) + getattr(head, "num_features", [])
        for node in feats:
            vals, ok = _eval_node(node, {"content": content}, {})
            X = resid.reshape(-1, resid.shape[-1])
            y = np.where(ok, vals, 0.0).ravel()
            A = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
            out[node.name or type(node).__name__] = round(float(ss), 3)
        return out

    def extract(self, oracle, rng, extra=None):
        cat, num = build_feature_bank(oracle.spec)
        batches = self._dataset(oracle, rng, self.n_train, extra)
        fit = self._fit_readout(oracle, batches, cat, num)
        return self._to_program(oracle, fit, cat, num)

    def refine(self, program, oracle, cex_list, rng):
        return self.extract(oracle, rng, extra=cex_list)
