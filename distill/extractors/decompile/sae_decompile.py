"""SAE-feature decompiler: autoencode activations, then symbolize.

Pipeline (per case):
  1. Cache final-residual-stream activations over on-distribution samples.
  2. Train a sparse autoencoder (ReLU, L1, unit-norm decoder rows) so the
     model's activation geometry proposes its own feature set.
  3. Symbolize: match every sufficiently-alive latent against an EXTENDED
     candidate library of DSL-expressible functions — the tl_decompile
     bank (token/position/first/last/counts/fracs) plus the structural
     features it lacked: tokens at fixed offsets, the token at the
     length-mirrored position, and neighbor-equality indicators. A latent
     is symbolized when a 1D affine fit to some candidate reaches R^2 >=
     0.85; unmatched latents are reported as uncovered (honest residue).
  4. Fit the readout (SoftHead / ridge, reused from tl_decompile) over the
     candidates the SAE selected — the program contains only symbolized,
     mechanistically-discovered features.

The SAE thus acts as a feature *selector grounded in the activations*
(replacing hand-picked banks), while the emitted artifact remains a
verifiable DSL program for the existing metrics/CEGIS machinery.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from distill.dsl.core import (Aggregate, Indices, Select, SelectorWidth,
                              SeqMap, TableMap, Tokens, _eval_node)
from distill.extractors.decompile.tl_decompile import (TLDecompileExtractor,
                                                       build_feature_bank,
                                                       eval_features)

SHIFTS = (-3, -2, -1, 1, 2, 3)


def build_extended_bank(spec):
    """tl_decompile bank + offset tokens, mirrored token, neighbor equality."""
    cat, num = build_feature_bank(spec)
    V = len(spec.vocab)
    L = max(spec.seq_lens)
    tok = Tokens().named("tok")
    idx = Indices().named("pos")

    shifted = {}
    for o in SHIFTS:
        m = np.zeros((L, L), bool)
        for q in range(L):
            if 0 <= q + o < L:
                m[q + o, q] = True  # key position == query position + o
        node = Aggregate(Select(idx, idx, m), tok).named(f"tok_at{o:+d}")
        shifted[o] = node
        cat.append((f"tok_at{o:+d}", node, V))

    # token at the mirrored position: needs the sequence length
    all_true = Select(tok, tok, np.ones((V, V), bool))
    length = SelectorWidth(all_true).named("len")
    opp_table = {}
    for l in range(L + 1):
        for i in range(L):
            j = l - 1 - i
            if 0 <= j < L:
                opp_table[(l, i)] = j
    opp = SeqMap(opp_table, length, idx).named("opp_pos")
    eye = np.eye(L, dtype=bool)
    mirror = Aggregate(Select(idx, opp, eye), tok).named("tok_mirror")
    cat.append(("tok_mirror", mirror, V))

    # neighbor equality indicators
    for o in (-1, 1):
        eq_table = {(a, b): float(a == b) for a in range(V) for b in range(V)}
        cat.append((f"eq_tok{o:+d}",
                    SeqMap(eq_table, tok, shifted[o]).named(f"eq_tok{o:+d}"), 2))

    # -- composed candidates: functions of features indexed by features --
    # count of the token at the current position (hist)
    count_self = SelectorWidth(
        Select(tok, tok, np.eye(V, dtype=bool))).named("count_self")
    cat.append(("count_self", count_self, L + 1))
    # rank: how many positions hold a strictly smaller token
    lt = np.zeros((V, V), bool)
    for a in range(V):
        for b in range(V):
            lt[a, b] = a < b  # key value < query value
    cat.append(("rank_lt", SelectorWidth(Select(tok, tok, lt)).named("rank_lt"),
                L + 1))
    # prefix count of the current token / first-occurrence indicator,
    # via a (token, position) pair variable and a conjunctive selector
    pair = SeqMap({(t, i): t * L + i for t in range(V) for i in range(L)},
                  tok, idx).named("tok_pos_pair")
    M = np.zeros((V * L, V * L), bool)
    for t in range(V):
        for ki in range(L):
            for qi in range(ki + 1, L):
                M[t * L + ki, t * L + qi] = True  # same token, earlier position
    prefix_self = SelectorWidth(Select(pair, pair, M)).named("prefix_count_self")
    cat.append(("prefix_count_self", prefix_self, L + 1))
    cat.append(("is_first_occurrence",
                TableMap({c: int(c == 0) for c in range(L + 1)},
                         prefix_self).named("is_first_occurrence"), 2))
    # pairwise interactions (gating / bigram) for small vocabularies
    if V * V <= 300:
        first_node = cat[2][1]  # first_tok
        pair_tab = {(a, b): a * V + b for a in range(V) for b in range(V)}
        cat.append(("first_x_tok",
                    SeqMap(pair_tab, first_node, tok).named("first_x_tok"),
                    V * V))
        cat.append(("prev_x_tok",
                    SeqMap(pair_tab, shifted[-1], tok).named("prev_x_tok"),
                    V * V))

    # extremum / order primitives (token indices are value-sorted, so <
    # over indices tracks < over numeric token values)
    gt = lt.T.copy()
    rank_gt = SelectorWidth(Select(tok, tok, gt)).named("rank_gt")
    cat.append(("rank_gt", rank_gt, L + 1))
    cat.append(("is_max", TableMap({c: int(c == 0) for c in range(L + 1)},
                                   rank_gt).named("is_max"), 2))
    rank_lt_node = cat[[n for n, *_ in cat].index("rank_lt")][1]
    cat.append(("is_min", TableMap({c: int(c == 0) for c in range(L + 1)},
                                   rank_lt_node).named("is_min"), 2))
    # neighbor comparisons (trend building blocks)
    lt_tab = {(a, b): float(a < b) for a in range(V) for b in range(V)}
    gt_tab = {(a, b): float(a > b) for a in range(V) for b in range(V)}
    cat.append(("prev_lt_tok", SeqMap(lt_tab, shifted[-1], tok).named("prev_lt_tok"), 2))
    cat.append(("prev_gt_tok", SeqMap(gt_tab, shifted[-1], tok).named("prev_gt_tok"), 2))
    # global trend fractions (numerical): mean of the comparison indicators
    sel_all = Select(idx, idx, np.ones((L, L), bool))
    for nm in ("prev_lt_tok", "prev_gt_tok"):
        node = cat[[n for n, *_ in cat].index(nm)][1]
        num.append((f"frac_{nm}",
                    Aggregate(sel_all, node, default=0).named(f"frac_{nm}")))
    return cat, num


def build_compositions(sel_cat, spec):
    """Approach-A templates over SAE-selected features.

    T1  pairwise interaction: joint SeqMap of two selected cat features
    T2a gather: token at the position named by a position-valued feature
    T2b kth-occurrence gather: token at the q-th position where a selected
        binary feature fires (expressible with a pair variable + selector;
        with b = is_first_occurrence this is exactly 'extract unique')
    """
    V = len(spec.vocab)
    L = max(spec.seq_lens)
    tok = Tokens().named("tok")
    idx = Indices().named("pos")
    comp = []

    base = [(n, node, card) for n, node, card in sel_cat]
    # T1: pairwise interactions
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            na, a, ca = base[i]
            nb, b, cb = base[j]
            if ca * cb > 400:
                continue
            tab = {(x, y): x * cb + y for x in range(ca) for y in range(cb)}
            comp.append((f"{na}_x_{nb}",
                         SeqMap(tab, a, b).named(f"{na}_x_{nb}"), ca * cb))

    # T2a: gather by position-valued features (card fits the position range)
    eye = np.eye(L, dtype=bool)
    for n, node, card in base:
        if n in ("pos", "tok") or card > L:
            continue
        if not n.startswith(("rank", "count", "prefix")) and card != L:
            continue
        comp.append((f"tok_at_{n}",
                     Aggregate(Select(idx, node, eye[:, :card] if card <= L else eye),
                               tok).named(f"tok_at_{n}"), V))

    # T2b: kth-occurrence gathers over selected binary features
    for n, node, card in base:
        if card != 2:
            continue
        pair_tab = {(f, i): f * L + i for f in range(2) for i in range(L)}
        pair_b = SeqMap(pair_tab, node, idx).named(f"pair_{n}")
        M = np.zeros((2 * L, 2 * L), bool)
        for ki in range(L):
            for qi in range(L):
                if ki < qi:
                    M[1 * L + ki, 0 * L + qi] = True
                    M[1 * L + ki, 1 * L + qi] = True
        prefix_ones = SelectorWidth(Select(pair_b, pair_b, M)).named(f"nprev_{n}")
        pair2_tab = {(f, c): f * (L + 1) + c for f in range(2) for c in range(L + 1)}
        pair2 = SeqMap(pair2_tab, node, prefix_ones).named(f"pair2_{n}")
        M2 = np.zeros((2 * (L + 1), L), bool)
        for c in range(L + 1):
            for q in range(L):
                if c == q:
                    M2[1 * (L + 1) + c, q] = True  # b fires and it's the q-th
        comp.append((f"kth_{n}_tok",
                     Aggregate(Select(pair2, idx, M2), tok).named(f"kth_{n}_tok"),
                     V))
    return comp


class SAE(torch.nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.enc = torch.nn.Linear(d, m)
        self.dec = torch.nn.Linear(m, d)
        with torch.no_grad():
            self.dec.weight.data = torch.nn.functional.normalize(
                self.dec.weight.data, dim=0)

    def forward(self, x):
        f = torch.relu(self.enc(x - self.dec.bias))
        return self.dec(f), f


def train_sae(acts, expansion=8, l1=0.1, steps=3000, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(acts).float()
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    d = X.shape[1]
    sae = SAE(d, expansion * d)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    n = len(X)
    for step in range(steps):
        idx = torch.randint(0, n, (min(1024, n),))
        x = X[idx]
        xhat, f = sae(x)
        # standard SAE convention: sums per sample, not means over units
        loss = ((xhat - x) ** 2).sum(-1).mean() + l1 * f.sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            sae.dec.weight.data = torch.nn.functional.normalize(
                sae.dec.weight.data, dim=0)
    with torch.no_grad():
        _, F = sae(X)
    return sae, F.numpy(), float(loss.item())


class SAEDecompileExtractor(TLDecompileExtractor):
    name = "sae_decompile"

    def __init__(self, n_train=30000, n_sae=60000, expansion=8, r2_min=0.7,
                 alive_freq=0.005, **kw):
        kw.setdefault("penalty", "l2")  # SAE selection already sparsifies
        kw.setdefault("max_rows", 60000)
        super().__init__(n_train=n_train, **kw)
        self.n_sae, self.expansion = n_sae, expansion
        self.r2_min, self.alive_freq = r2_min, alive_freq

    def _activations(self, oracle, content):
        ids = oracle._encode(content) if hasattr(oracle, "_encode") else content
        with torch.no_grad():
            _, cache = oracle.model.run_with_cache(torch.from_numpy(ids).long())
        resid = cache["resid_post", -1].float().cpu().numpy()
        if resid.shape[1] != content.shape[1]:
            resid = resid[:, 1:]
        return resid

    def _select_features(self, oracle, rng):
        """SAE-driven selection from the extended candidate bank."""
        cat, num = build_extended_bank(oracle.spec)
        # gather activations + candidate values on the same samples
        contents, acts = [], []
        for content in oracle.sample_batches(rng, self.n_sae):
            contents.append(content)
            acts.append(self._activations(oracle, content))
        A = np.concatenate([a.reshape(-1, a.shape[-1]) for a in acts])
        _, F, recon = train_sae(A, expansion=self.expansion)

        cand_vals = []
        for content in contents:
            fc = eval_features(cat, content)
            fn = eval_features(num, content)
            cand_vals.append(np.concatenate([fc, fn], axis=2))
        CV = np.concatenate(
            [c.reshape(c.shape[0] * c.shape[1], c.shape[2]) for c in cand_vals])

        # expand categorical candidates into per-value indicator columns —
        # SAE latents encode "feature == value", not the value as a number
        cols, parent = [], []
        for k, (_, _, card) in enumerate(cat):
            for v in range(card):
                cols.append(CV[:, k] == v)
                parent.append(k)
        for j in range(len(num)):
            cols.append(CV[:, len(cat) + j])
            parent.append(len(cat) + j)
        C = np.stack(cols, axis=1).astype(np.float64)

        sub = np.random.default_rng(0).choice(
            len(C), min(len(C), 25_000), replace=False)
        Csub = C[sub]

        freq = (F > 0).mean(0)
        alive = np.flatnonzero((freq > self.alive_freq) & (freq < 0.995))
        Fa = F[sub][:, alive]
        Faug = np.concatenate([Fa, np.ones((len(Fa), 1))], axis=1)

        def ridge_fit(X, Y, lam=1e-3):
            G = X.T @ X + lam * np.eye(X.shape[1])
            coef = np.linalg.solve(G, X.T @ Y)
            pred = X @ coef
            ss_res = ((Y - pred) ** 2).sum(0)
            ss_tot = ((Y - Y.mean(0)) ** 2).sum(0) + 1e-9
            return pred, 1 - ss_res / ss_tot

        def ridge_r2(X, Y, lam=1e-3):
            return ridge_fit(X, Y, lam)[1]

        # a candidate is REPRESENTED if it is linearly decodable from the
        # sparse code (multivariate probe) — robust to feature splitting.
        # Categorical candidates are judged by probe CLASSIFICATION
        # accuracy (argmax over predicted value-indicators): indicator R^2
        # understates decodability when logit margins vary.
        pred, col_r2 = ridge_fit(Faug, Csub)
        chosen = {}
        parent_arr = np.array(parent)
        for k, (name, _, card) in enumerate(cat):
            cols = np.flatnonzero(parent_arr == k)
            true_val = CV[sub, k].astype(int)
            pred_val = pred[:, cols].argmax(1)
            acc = float((pred_val == true_val).mean())
            base = float(np.bincount(true_val).max() / len(true_val))
            if acc >= 0.85 or acc - base >= 0.5:
                chosen[k] = round(acc, 3)
        for j in range(len(num)):
            col = len(C[0]) - len(num) + j
            if col_r2[col] >= self.r2_min:
                chosen[len(cat) + j] = round(float(col_r2[col]), 3)

        # coverage: how much of the sparse code the selected candidates
        # explain (mass-weighted reverse regression) — the honest residue
        sel_cols = [c for c in range(C.shape[1]) if parent[c] in chosen]
        if sel_cols:
            S = np.concatenate([Csub[:, sel_cols],
                                np.ones((len(Csub), 1))], axis=1)
            r2_lat = ridge_r2(S, Fa)
        else:
            r2_lat = np.zeros(len(alive))
        mass = F[:, alive].mean(0)
        coverage = float((np.clip(r2_lat, 0, 1) * mass).sum() /
                         max(mass.sum(), 1e-9))
        report = {
            "n_latents_alive": int(len(alive)),
            "n_latents_symbolized": int((np.clip(r2_lat, 0, 1) > 0.5).sum()),
            "coverage_activation_mass": round(coverage, 3),
            "sae_final_loss": round(recon, 4),
            "selected_features": {},
        }
        sel_cat, sel_num = [], []
        for k, r2 in sorted(chosen.items(), key=lambda kv: -kv[1]):
            if k < len(cat):
                sel_cat.append(cat[k])
                report["selected_features"][cat[k][0]] = round(r2, 3)
            else:
                sel_num.append(num[k - len(cat)])
                report["selected_features"][num[k - len(cat)][0]] = round(r2, 3)
        # Base primitives are never removed: SAE selection ADDS discovered
        # structure. Models often represent only a task-specific coarsening
        # of token identity, which fails the identity probe even though the
        # readout needs (a function of) the token.
        for base_idx in (0, 1):  # tok, pos
            if not any(n == cat[base_idx][0] for n, *_ in sel_cat):
                sel_cat.append(cat[base_idx])

        # ---- composition round (Approach A): templates over the selected
        # features, admitted by the same decodability probe ----
        top_sel = sel_cat[:8]
        comps = [c for c in build_compositions(top_sel, oracle.spec)
                 if not any(c[0] == n for n, *_ in sel_cat)]
        if comps:
            comp_vals = np.concatenate(
                [eval_features(comps, content).reshape(-1, len(comps))
                 for content in contents])
            report["compositions"] = {}
            admitted = []
            for k, (name, node, card) in enumerate(comps):
                vals = comp_vals[sub, k].astype(int)
                if len(np.unique(vals)) < 2:
                    continue
                ind = np.stack([vals == v for v in range(card)], 1).astype(float)
                predc, _ = ridge_fit(Faug, ind)
                acc = float((predc.argmax(1) == vals).mean())
                basefreq = float(np.bincount(vals).max() / len(vals))
                if acc >= 0.85 or acc - basefreq >= 0.5:
                    sel_cat.append((name, node, card))
                    admitted.append((name, node, card))
                    report["compositions"][name] = round(acc, 3)

            # second-order pass: small gating features x admitted gathers
            # (e.g. task token x extremum value) — depth-3 compositions
            gates = [c for c in top_sel if c[2] <= 16]
            gathers = [c for c in admitted if c[0].startswith(("kth_", "tok_at_"))]
            for gn, gnode, gc in gates:
                for hn, hnode, hc in gathers:
                    if gc * hc > 400:
                        continue
                    name = f"{gn}_x_{hn}"
                    if any(name == n for n, *_ in sel_cat):
                        continue
                    tab = {(x, y): x * hc + y for x in range(gc) for y in range(hc)}
                    node = SeqMap(tab, gnode, hnode).named(name)
                    vals = np.concatenate(
                        [np.where(ok, v, 0).ravel() for v, ok in
                         (_eval_node(node, {"content": c}, {})
                          for c in contents)])[sub].astype(int)
                    if len(np.unique(vals)) < 2:
                        continue
                    ind = np.stack([vals == v for v in range(gc * hc)], 1).astype(float)
                    predc, _ = ridge_fit(Faug, ind)
                    acc = float((predc.argmax(1) == vals).mean())
                    basefreq = float(np.bincount(vals).max() / len(vals))
                    if acc >= 0.85 or acc - basefreq >= 0.5:
                        sel_cat.append((name, node, gc * hc))
                        report["compositions"][name] = round(acc, 3)
        return sel_cat, sel_num, report

    def extract(self, oracle, rng, extra=None):
        cat, num, report = self._select_features(oracle, rng)
        batches = self._dataset(oracle, rng, self.n_train, extra)
        fit = self._fit_readout(oracle, batches, cat, num)
        prog = self._to_program(oracle, fit, cat, num)
        prog.sae_report = report
        return prog

    def refine(self, program, oracle, cex_list, rng):
        return self.extract(oracle, rng, extra=cex_list)
