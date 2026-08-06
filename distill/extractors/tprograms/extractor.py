"""Baseline B — TransformerPrograms as an imitation student.

Trains Friedman et al.'s discrete-constrained TransformerProgramModel
(third_party/TransformerPrograms, MIT) to imitate the *oracle's* output
distribution (soft cross-entropy on oracle probs — distillation, not task
supervision), annealing the Gumbel temperature per their recipe, then
freezes it into hard argmax mode: the resulting discretized model IS the
extracted program (their code_utils can additionally render it to Python
source, attempted best-effort).

Categorical outputs only (their framework): seq_num cases are rejected.
"""

import sys
from pathlib import Path

import numpy as np
import torch

TP_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "TransformerPrograms"
if str(TP_ROOT) not in sys.path:
    sys.path.insert(0, str(TP_ROOT))

from ..base import Extractor


def _pad(content, L, pad_idx=0):
    ids = content + 1  # reserve 0 for <pad>
    if ids.shape[1] < L:
        pad = np.full((len(ids), L - ids.shape[1]), pad_idx, dtype=ids.dtype)
        ids = np.concatenate([ids, pad], axis=1)
    return ids


class TPProgram:
    def __init__(self, model, spec, max_len, autoregressive):
        self.model = model
        self.spec = spec
        self.max_len = max_len
        self.autoregressive = autoregressive
        self.name = "tprogram"

    def _logp(self, content):
        n, L = content.shape
        x = torch.from_numpy(_pad(content, self.max_len)).long()
        m = (x != 0).float()
        mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
        if self.autoregressive:
            mask = torch.tril(mask)
        with torch.no_grad():
            logp = self.model(x, mask=mask).log_softmax(-1)
        return logp[:, :L].float().cpu().numpy()

    def probs(self, content):
        p = np.exp(self._logp(content))
        if self.spec.kind == "classify":
            return p[:, content.shape[1] - 1]
        return p

    def outputs(self, content):
        if self.spec.kind == "classify":
            return self.probs(content).argmax(-1)
        idx = self.probs(content).argmax(-1)
        if self.spec.kind == "seq_cat":
            vals = np.array(self.spec.output_values, dtype=object)
            return vals[idx]
        return idx  # lm

    def complexity(self):
        cfg = self.model
        n = sum(p.numel() for p in cfg.parameters())
        return {"n_nodes": len(list(cfg.blocks)) if hasattr(cfg, "blocks") else 0,
                "n_table_cells": int(n)}

    def is_hard(self):
        return True  # frozen argmax discretization

    def source(self):
        return "(TransformerProgram: see model_to_code output if emitted)"


class TProgramExtractor(Extractor):
    name = "tprogram"

    def __init__(self, steps=6000, batch=256, lr=5e-2, tau_init=3.0, tau_end=0.01):
        self.steps, self.batch, self.lr = steps, batch, lr
        self.tau_init, self.tau_end = tau_init, tau_end

    def _build(self, oracle):
        from src.models.programs import TransformerProgramModel

        spec = oracle.spec
        if spec.kind == "seq_num":
            raise ValueError("tprogram extractor: seq_num unsupported "
                             "(categorical framework)")
        V = len(spec.vocab)
        L = max(spec.seq_lens)
        d_var = max(V + 1, L + 1, spec.n_outputs, 8)
        model = TransformerProgramModel(
            d_vocab=V + 1,
            d_vocab_out=spec.n_outputs,
            n_ctx=L,
            n_layers=2,
            n_vars_cat=4,
            n_vars_num=2,
            d_var=d_var,
            d_mlp=64,
            n_heads_cat=4,
            n_heads_num=2,
            n_cat_mlps=2,
            n_num_mlps=1,
            one_hot_embed=False,
            attention_type="cat",
            rel_pos_bias="fixed",
            mlp_vars_in=2,
            temp=self.tau_init,
        )
        return model, L

    def _targets(self, oracle, content):
        """(target_probs (n, L, C) or (n, C), position_mask (n, L))."""
        kind = oracle.spec.kind
        C = oracle.spec.n_outputs
        n, L = content.shape
        if kind == "classify":
            t = np.zeros((n, L, C))
            t[:, L - 1] = oracle.probs(content)
            m = np.zeros((n, L), bool)
            m[:, L - 1] = True
            return t, m
        if kind == "lm":
            p = oracle.probs(content)  # (n, L, C); pos i -> dist of tok i+1
            t = np.zeros((n, L, C))
            t[:, : L - 1] = p[:, : L - 1]
            m = np.zeros((n, L), bool)
            m[:, : L - 1] = True
            return t, m
        # seq_cat: hard oracle outputs as one-hot
        out = oracle.outputs(content)
        t = np.eye(C)[out]
        return t, np.ones((n, L), bool)

    def _train(self, model, oracle, rng, steps, tau_init, tau_end, extra=None):
        from src.models.programs import gumbel_soft, argmax

        model.set_temp(tau_init, gumbel_soft)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        temps = np.geomspace(tau_init, tau_end, steps)
        auto = oracle.spec.kind == "lm"
        max_len = max(oracle.spec.seq_lens)
        pool = list(extra) if extra else []
        for step in range(steps):
            model.set_temp(float(temps[step]))
            length = int(rng.choice(oracle.spec.seq_lens))
            content = oracle.sample(rng, self.batch, length)
            if pool and step % 4 == 0:
                cex = pool[rng.integers(len(pool))]
                take = cex[rng.integers(0, len(cex), min(64, len(cex)))]
                if take.shape[1] == length:
                    content = np.concatenate([content, take])
            t, m = self._targets(oracle, content)
            x = torch.from_numpy(_pad(content, max_len)).long()
            mm = (x != 0).float()
            mask = (mm.unsqueeze(-1) @ mm.unsqueeze(-2)).bool()
            if auto:
                mask = torch.tril(mask)
            logp = model(x, mask=mask).log_softmax(-1)[:, : content.shape[1]]
            tt = torch.from_numpy(t).float()
            tm = torch.from_numpy(m).float()
            loss = -((tt * logp).sum(-1) * tm).sum() / tm.sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.set_temp(tau_end, argmax)
        model.eval()
        return model

    def extract(self, oracle, rng):
        torch.manual_seed(0)
        model, L = self._build(oracle)
        model = self._train(model, oracle, rng, self.steps,
                            self.tau_init, self.tau_end)
        return TPProgram(model, oracle.spec, L, oracle.spec.kind == "lm")

    def refine(self, program, oracle, cex_list, rng):
        model = self._train(program.model, oracle, rng, steps=1500,
                            tau_init=0.3, tau_end=self.tau_end, extra=cex_list)
        return TPProgram(model, oracle.spec, program.max_len,
                         program.autoregressive)
