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

from distill.dsl.core import (Aggregate, Coalesce, Indices, NaryMap, Program,
                              Select, SelectorWidth, SeqMap, TableMap, Tokens,
                              _eval_node)
from distill.extractors.decompile.tl_decompile import (TLDecompileExtractor,
                                                       build_feature_bank,
                                                       eval_features)

SHIFTS = (-3, -2, -1, 1, 2, 3)


def _admit(acc, base):
    """Candidate admission: must beat the majority-class base rate, and
    near-constant candidates (base ~1) are never admitted — their raw
    accuracy is trivially high without carrying any information."""
    return base < 0.995 and acc > base + 0.01 and (acc >= 0.85
                                                   or acc - base >= 0.5)


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
    # majority indicators: models threshold counts (majority tests) far
    # more often than they expose the raw count, which probes reject
    half_tab = {c: float(c > L / 2) for c in range(L + 1)}
    for name, node, card in list(cat):
        if name.startswith("count_"):
            cat.append((f"maj_{name[6:]}",
                        TableMap(half_tab, node).named(f"maj_{name[6:]}"), 2))
    maj_self = TableMap(half_tab, count_self).named("maj_self")
    cat.append(("maj_self", maj_self, 2))
    # the majority token's value (defined when a strict majority exists);
    # targets mode-gather circuits (m04)
    m_fire = np.zeros((2, 2), bool)
    m_fire[1, :] = True
    cat.append(("mode_maj_tok",
                Aggregate(Select(maj_self, maj_self, m_fire), tok)
                .named("mode_maj_tok"), V))
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
    cat.append(("next_lt_tok", SeqMap(lt_tab, shifted[1], tok).named("next_lt_tok"), 2))
    cat.append(("next_gt_tok", SeqMap(gt_tab, shifted[1], tok).named("next_gt_tok"), 2))
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

    # counts + thresholds of binary features (B2 in the failure
    # inventory). Models rarely represent the raw count — they represent
    # the BOOLEAN the task needs ("all elements equal", "increasing so
    # far"), so alongside each count we offer thresholded and prefix
    # variants; the decodability probe picks whichever the model carries.
    m2 = np.zeros((2, 2), bool)
    m2[1, :] = True  # count keys where the feature fires, any query
    for n, node, card in base:
        if card != 2:
            continue
        cnt = SelectorWidth(Select(node, node, m2)).named(f"count_{n}")
        comp.append((f"count_{n}", cnt, L + 1))
        # global thresholds: fires-everywhere (L or L-1 valid positions)
        # and fires-anywhere
        for t, nm2 in ((L, f"all_{n}"), (L - 1, f"allm1_{n}")):
            comp.append((nm2, TableMap({c: float(c == t) for c in range(L + 1)},
                                       cnt).named(nm2), 2))
        comp.append((f"any_{n}",
                     TableMap({c: float(c >= 1) for c in range(L + 1)},
                              cnt).named(f"any_{n}"), 2))
        # prefix count (positions <= q) via a (feature, position) pair
        pair_tab = {(f, i): f * L + i for f in range(2) for i in range(L)}
        pb = SeqMap(pair_tab, node, idx).named(f"cpair_{n}")
        Mle = np.zeros((2 * L, 2 * L), bool)
        for ki in range(L):
            for qi in range(ki, L):
                Mle[L + ki, 0 * L + qi] = True
                Mle[L + ki, 1 * L + qi] = True
        pc = SelectorWidth(Select(pb, pb, Mle)).named(f"prefix_count_{n}")
        comp.append((f"prefix_count_{n}", pc, L + 1))
        # all-so-far: prefix count == number of prefix positions (offset
        # 1 for features invalid at position 0, e.g. neighbor comparisons)
        for off in (0, 1):
            tab = {(c, i): float(c == i + 1 - off)
                   for c in range(L + 1) for i in range(L)}
            comp.append((f"allpre{off}_{n}",
                         SeqMap(tab, pc, idx).named(f"allpre{off}_{n}"), 2))

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
    """Returns (sae, X_normalized, final_loss). Codes are NOT materialized
    for the full dataset here — at 3.5M rows x 8·d latents that is >10 GB;
    callers encode the rows they need in chunks via sae(Xn[rows])[1]."""
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
    return sae, X, float(loss.item())


class SAEDecompileExtractor(TLDecompileExtractor):
    name = "sae_decompile"

    def __init__(self, n_train=30000, n_sae=60000, expansion=8, r2_min=0.7,
                 alive_freq=0.005, **kw):
        kw.setdefault("penalty", "l2")  # SAE selection already sparsifies
        kw.setdefault("max_rows", 60000)
        super().__init__(n_train=n_train, **kw)
        self.n_sae, self.expansion = n_sae, expansion
        self.r2_min, self.alive_freq = r2_min, alive_freq

    def _structured(self, oracle, rng, n, length):
        """Structured sequences (constant / one-flip / sorted / two-block):
        rare-event families a uniform sampler essentially never produces,
        without which ALL/majority predicates are invisible to probes and
        readout alike (interp/124: all-equal inputs are 8^-9 rare)."""
        V = len(oracle.spec.vocab)
        out = np.empty((n, length), dtype=np.int64)
        out[:] = rng.integers(0, V, n)[:, None]
        kind = rng.integers(0, 4, n)
        for i in range(n):
            if kind[i] == 1:  # constant with one deviation
                out[i, rng.integers(0, length)] = rng.integers(0, V)
            elif kind[i] == 2:  # sorted ascending or descending
                s = np.sort(rng.integers(0, V, length))
                out[i] = s if rng.random() < 0.5 else s[::-1]
            elif kind[i] == 3:  # two constant blocks
                out[i, rng.integers(1, length):] = rng.integers(0, V)
        return out

    def _sample_mixed(self, oracle, rng, n_total):
        batches = list(oracle.sample_batches(rng, n_total))
        if oracle.spec.suite == "interp":
            per = max(n_total // 8 // len(oracle.spec.seq_lens), 1)
            for length in oracle.spec.seq_lens:
                batches.append(self._structured(oracle, rng, per, length))
        return batches

    def _dataset(self, oracle, rng, n, extra=None):
        batches = self._sample_mixed(oracle, rng, n)
        if extra:
            for cex in extra:
                batches.extend([cex] * 5)
        return batches

    def _activations(self, oracle, content, batch=4096):
        # cache ONLY the hook we read, in chunks: a full run_with_cache
        # stores every attention score/pattern tensor, which is O(n·L²)
        # per head and OOM-kills long-window cases (interp/39: ~100 GB)
        ids = oracle._encode(content) if hasattr(oracle, "_encode") else content
        name = f"blocks.{oracle.model.cfg.n_layers - 1}.hook_resid_post"
        outs = []
        with torch.no_grad():
            for i in range(0, len(ids), batch):
                t = torch.from_numpy(ids[i:i + batch]).long()
                _, cache = oracle.model.run_with_cache(t, names_filter=name)
                outs.append(cache[name].float().cpu().numpy())
        resid = np.concatenate(outs)
        if resid.shape[1] != content.shape[1]:
            resid = resid[:, 1:]
        return resid

    def _select_features(self, oracle, rng):
        """SAE-driven selection from the extended candidate bank."""
        cat, num = build_extended_bank(oracle.spec)
        # gather activations + candidate values on the same samples
        contents, acts = [], []
        for content in self._sample_mixed(oracle, rng, self.n_sae):
            contents.append(content)
            acts.append(self._activations(oracle, content))
        A = np.concatenate([a.reshape(-1, a.shape[-1]) for a in acts])
        del acts
        sae, Xn, recon = train_sae(A, expansion=self.expansion)
        del A
        n_rows = len(Xn)
        sub = np.random.default_rng(0).choice(
            n_rows, min(n_rows, 25_000), replace=False)
        # streaming code statistics; full codes only for the probe rows
        n_lat = sae.enc.out_features
        freq = np.zeros(n_lat)
        mass_all = np.zeros(n_lat)
        with torch.no_grad():
            for i in range(0, n_rows, 200_000):
                f = sae(Xn[i:i + 200_000])[1]
                freq += (f > 0).sum(0).numpy()
                mass_all += f.sum(0).numpy()
            Fsub = sae(Xn[torch.from_numpy(sub)])[1].numpy()
        freq /= n_rows
        mass_all /= n_rows

        cand_vals = []
        for content in contents:
            fc = eval_features(cat, content)
            fn = eval_features(num, content)
            cand_vals.append(np.concatenate([fc, fn], axis=2))
        CV = np.concatenate(
            [c.reshape(c.shape[0] * c.shape[1], c.shape[2]) for c in cand_vals])
        del cand_vals

        # expand categorical candidates into per-value indicator columns —
        # SAE latents encode "feature == value", not the value as a number
        # indicator expansion only on the probe subsample: it is Σ card
        # columns wide (~2,200 for interp/39's 26-token × 60-position
        # case → ~60 GB dense over all rows, which OOM-kills the process)
        CVs = CV[sub]
        cols, parent = [], []
        for k, (_, _, card) in enumerate(cat):
            for v in range(card):
                cols.append(CVs[:, k] == v)
                parent.append(k)
        for j in range(len(num)):
            cols.append(CVs[:, len(cat) + j])
            parent.append(len(cat) + j)
        Csub = np.stack(cols, axis=1).astype(np.float64)

        alive = np.flatnonzero((freq > self.alive_freq) & (freq < 0.995))
        Fa = Fsub[:, alive]
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
            true_val = CVs[:, k].astype(int)
            pred_val = pred[:, cols].argmax(1)
            acc = float((pred_val == true_val).mean())
            base = float(np.bincount(true_val).max() / len(true_val))
            if _admit(acc, base):
                chosen[k] = round(acc, 3)
        for j in range(len(num)):
            col = Csub.shape[1] - len(num) + j
            if col_r2[col] >= self.r2_min:
                chosen[len(cat) + j] = round(float(col_r2[col]), 3)

        # coverage: how much of the sparse code the selected candidates
        # explain (mass-weighted reverse regression) — the honest residue
        sel_cols = [c for c in range(Csub.shape[1]) if parent[c] in chosen]
        if sel_cols:
            S = np.concatenate([Csub[:, sel_cols],
                                np.ones((len(Csub), 1))], axis=1)
            r2_lat = ridge_r2(S, Fa)
        else:
            r2_lat = np.zeros(len(alive))
        mass = mass_all[alive]
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

        # ---- synthesized position-map gather (Approach B / B1): instead
        # of enumerating fixed shifts, learn the gather's position map.
        # For each query position q, probe which OTHER source position's
        # token is decodable from the code at q; assemble the admitted
        # (q -> s(q)) entries into one Select table. Per-position probing
        # is what handles conditionally-represented features (swap-pairs:
        # partner = i XOR 1 matches tok_at+1 on only half the positions,
        # so every global-probed fixed shift is correctly rejected).
        CONTENT = np.concatenate(contents)
        Lc = CONTENT.shape[1]
        Vc = len(oracle.spec.vocab)
        rows_seq, rows_pos = sub // Lc, sub % Lc
        smap = np.arange(Lc)  # identity = "no gather admitted here"
        gacc = np.zeros(Lc)
        for q in range(Lc):
            rq = np.flatnonzero(rows_pos == q)
            if len(rq) < 80:
                continue
            # few rows per position vs many latents: validate out-of-sample
            # or overfit probes admit spurious gathers
            cut = int(len(rq) * 0.7)
            tr, va = rq[:cut], rq[cut:]
            toks_tr = CONTENT[rows_seq[tr]]
            toks_va = CONTENT[rows_seq[va]]
            Yq = np.zeros((len(tr), Lc * Vc))
            Yq[np.arange(len(tr))[:, None],
               toks_tr + np.arange(Lc) * Vc] = 1.0
            Xtr = Faug[tr]
            G = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
            coef = np.linalg.solve(G, Xtr.T @ Yq)
            predv = Faug[va] @ coef
            for s in range(Lc):
                if s == q:  # own token is the base primitive already
                    continue
                pv = predv[:, s * Vc:(s + 1) * Vc].argmax(1)
                acc = float((pv == toks_va[:, s]).mean())
                basefreq = float(np.bincount(toks_va[:, s], minlength=Vc).max()
                                 / len(va))
                if _admit(acc, basefreq) and acc > gacc[q]:
                    smap[q], gacc[q] = s, acc
        if (gacc > 0).any():
            # unadmitted positions keep the identity gather (own token):
            # always-valid beats invalid, which the SoftHead would read as
            # value 0; pos-interactions disambiguate admitted vs fallback
            Mg = np.zeros((Lc, Lc), bool)
            Mg[smap, np.arange(Lc)] = True
            learned = Aggregate(
                Select(Indices().named("pos"), Indices().named("pos"), Mg),
                Tokens().named("tok")).named("tok_at_learned")
            score = float(gacc[gacc > 0].mean())
            sel_cat.append(("tok_at_learned", learned, Vc))
            report["selected_features"]["tok_at_learned"] = round(score, 3)
            report["learned_gather_map"] = {
                int(q): int(smap[q]) for q in np.flatnonzero(gacc > 0)}

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
                if _admit(acc, basefreq):
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
                    if _admit(acc, basefreq):
                        sel_cat.append((name, node, gc * hc))
                        report["compositions"][name] = round(acc, 3)
        return sel_cat, sel_num, report

    def _fit_readout(self, oracle, batches, cat, num):
        # seq_num with the large SAE candidate bank: the base Ridge
        # (alpha=1e-3) overfits ~100 one-hot categorical blocks, each
        # contributing weight-1 TableMaps that collectively swamp the
        # frac_* numeric signal (regressed cases 3/4/39 from 1.0 to ~0).
        # Fit numeric features with tight ridge; add a categorical block
        # only if it materially cuts residual — keeps the LinComb sparse.
        if oracle.spec.kind == "seq_num":
            return self._fit_seq_num(oracle, batches, cat, num)
        return super()._fit_readout(oracle, batches, cat, num)

    def _fit_seq_num(self, oracle, batches, cat, num):
        from sklearn.linear_model import Ridge
        Xn_l, Xc_l, y_l = [], [], []
        for content in batches:
            fn = eval_features(num, content)
            fc = eval_features(cat, content)
            tgt = oracle.outputs(content)
            Xn_l.append(fn.reshape(-1, fn.shape[2]))
            Xc_l.append(fc.reshape(-1, fc.shape[2]))
            y_l.append(tgt.reshape(-1))
        Xn, Xc, Y = np.concatenate(Xn_l), np.concatenate(Xc_l), np.concatenate(y_l)
        if self.max_rows and len(Y) > self.max_rows:
            keep = np.random.default_rng(0).choice(len(Y), self.max_rows, False)
            Xn, Xc, Y = Xn[keep], Xc[keep], Y[keep]
        card = [c[2] for c in cat]
        # start from numeric-only; greedily admit categorical blocks that
        # reduce holdout residual by a real margin (forward selection)
        n = len(Y)
        tr = np.random.default_rng(1).permutation(n)
        cut = int(n * 0.8)
        trn, val = tr[:cut], tr[cut:]

        def fit_pred(cols_oh):
            X = np.concatenate([Xn] + cols_oh, axis=1) if cols_oh else Xn
            m = Ridge(alpha=1e-3).fit(X[trn], Y[trn])
            return m, float(np.sqrt(((m.predict(X[val]) - Y[val]) ** 2).mean()))

        chosen, cols_oh = [], []
        _, best_rmse = fit_pred(cols_oh)
        onehots = [np.eye(k)[np.clip(Xc[:, j].astype(int), 0, k - 1)]
                   for j, k in enumerate(card)]
        improved = True
        while improved and len(chosen) < 6:
            improved = False
            cand = None
            for j in range(len(cat)):
                if j in chosen:
                    continue
                _, rmse = fit_pred(cols_oh + [onehots[j]])
                if rmse < best_rmse - 1e-3 and (cand is None or rmse < cand[1]):
                    cand = (j, rmse)
            if cand is not None:
                chosen.append(cand[0])
                cols_oh.append(onehots[cand[0]])
                best_rmse = cand[1]
                improved = True
        Xfull = np.concatenate([Xn] + cols_oh, axis=1) if cols_oh else Xn
        model = Ridge(alpha=1e-3).fit(Xfull, Y)
        return ("ridge_sparse", model, [card[j] for j in chosen],
                [cat[j] for j in chosen])

    def _to_program(self, oracle, fit, cat, num):
        if fit[0] == "ridge_sparse":
            _, model, chosen_card, chosen_cat = fit
            coefs = model.coef_
            terms, weights = [], []
            for j, (name, node) in enumerate(num):
                w = float(coefs[j])
                if abs(w) > 1e-4:
                    terms.append(node)
                    weights.append(w)
            pos = len(num)
            for (name, node, k) in chosen_cat:
                w = coefs[pos:pos + k]
                if np.abs(w).max() > 1e-4:
                    terms.append(TableMap({i: float(w[i]) for i in range(k)}, node))
                    weights.append(1.0)
                pos += k
            from distill.dsl.core import LinComb
            out = LinComb(weights, terms, const=float(model.intercept_))
            return Program(out, oracle.spec.kind, oracle.spec.n_outputs,
                           name=f"sae_decompiled_{oracle.spec.case_id}")
        return super()._to_program(oracle, fit, cat, num)

    def _try_tabulation(self, oracle, batches, cat, scores=None):
        """CART-in-DSL readout: exhaustive small joint tables (NaryMap)
        over top admitted features. Linear SoftHeads cannot express joint
        position-conditional logic (e.g. trend = f(prev, cur, next, pos))
        even when every needed feature is admitted; an explicit table can
        — and CEGIS can later repair single cells. Returns
        (program, holdout_accuracy) for the best table found."""
        from itertools import combinations

        spec = oracle.spec
        C = spec.n_outputs
        # pool = highest-probe-score features across selection AND
        # composition rounds (list order buries late-round composites)
        if scores:
            ranked = sorted(cat, key=lambda c: -scores.get(c[0], 0.0))
        else:
            ranked = list(cat)
        feats = ranked[:16]
        if not any(n == "pos" for n, *_ in feats):
            feats += [c for c in cat if c[0] == "pos"][:1]
        # evaluate with validity; features with boundary-invalid positions
        # are wrapped in Coalesce(x, 0) so fit == runtime semantics
        wrapped, Xcols = [], []
        contents = list(batches)
        allc = np.concatenate(contents)
        for name, node, card in feats:
            vals, ok = _eval_node(node, {"content": allc}, {})
            if not ok.all():
                node = Coalesce(node, 0).named(f"{name}0")
            wrapped.append((name, node, card))
            Xcols.append(np.where(ok, vals, 0).ravel().astype(np.int64))
        X = np.stack(Xcols, 1)
        numeric = spec.kind == "seq_num"
        tol = float(spec.meta.get("tol", 1e-3))
        y = np.concatenate([oracle.outputs(c).reshape(-1) for c in contents])
        y = y.astype(np.float64) if numeric else y.astype(np.int64)
        perm = np.random.default_rng(1).permutation(len(X))
        tr, va = perm[: int(len(X) * 0.8)], perm[int(len(X) * 0.8):]
        cards = [c[2] for c in wrapped]
        maj = 0 if numeric else int(np.bincount(y[tr], minlength=C).argmax())

        def table_for(s, rows):
            dims = [cards[i] for i in s]
            code = np.ravel_multi_index(
                [np.clip(X[:, i], 0, cards[i] - 1) for i in s], dims)
            ncell = int(np.prod(dims))
            if numeric:  # cell means
                cnt = np.bincount(code[rows], minlength=ncell)
                sm = np.bincount(code[rows], weights=y[rows], minlength=ncell)
                pred = np.where(cnt > 0, sm / np.maximum(cnt, 1),
                                y[rows].mean())
            else:  # cell majority class
                counts = np.zeros((ncell, C), np.int64)
                np.add.at(counts, (code[rows], y[rows]), 1)
                pred = np.where(counts.sum(1) > 0, counts.argmax(1), maj)
            return code, pred

        def score(pred, code, rows):
            if numeric:
                return float((np.abs(pred[code[rows]] - y[rows]) <= tol).mean())
            return float((pred[code[rows]] == y[rows]).mean())

        best_s, best_acc = None, -1.0
        for r in (1, 2, 3, 4):
            for s in combinations(range(len(wrapped)), r):
                if np.prod([cards[i] for i in s]) > 4096:
                    continue
                code, pred = table_for(s, tr)
                acc = score(pred, code, va)
                if acc > best_acc:
                    best_s, best_acc = s, acc
        if best_s is None:
            return None, 0.0
        dims = [cards[i] for i in best_s]
        _, pred = table_for(best_s, np.arange(len(X)))  # final fit: all rows
        table = {tuple(int(v) for v in np.unravel_index(ci, dims)):
                 float(pred[ci]) for ci in range(len(pred))}
        node = NaryMap(table, [wrapped[i][1] for i in best_s])
        node.named("readout_table[" +
                   ",".join(wrapped[i][0] for i in best_s) + "]")
        prog = Program(node, spec.kind, C,
                       decode=None if numeric else spec.output_values,
                       name=f"sae_tabulated_{spec.case_id}")
        return prog, best_acc

    def extract(self, oracle, rng, extra=None):
        cat, num, report = self._select_features(oracle, rng)
        batches = self._dataset(oracle, rng, self.n_train, extra)
        fit = self._fit_readout(oracle, batches, cat, num)
        prog = self._to_program(oracle, fit, cat, num)
        prog.sae_report = report
        if oracle.spec.kind == "seq_cat":
            scores = dict(report.get("selected_features", {}))
            scores.update(report.get("compositions", {}))
            tab, tab_acc = self._try_tabulation(oracle, batches, cat, scores)
            if tab is not None:
                # judge both readouts on fresh held-out sequences
                ho = oracle.sample(np.random.default_rng(7), 4000,
                                   max(oracle.spec.seq_lens))
                yo = oracle.outputs(ho)
                soft_acc = float((prog.outputs(ho) == yo).mean())
                hard_acc = float((tab.outputs(ho) == yo).mean())
                if hard_acc > soft_acc + 1e-6:
                    tab.sae_report = dict(
                        report, readout="tabulated",
                        tab_acc=round(hard_acc, 4),
                        soft_acc=round(soft_acc, 4))
                    return tab
        return prog

    def refine(self, program, oracle, cex_list, rng):
        return self.extract(oracle, rng, extra=cex_list)
