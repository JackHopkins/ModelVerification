"""Torch oracles for TransformerLens checkpoints (messy/ and interp_bench/).

messy models: content vocab indices ARE the model token ids (trained
without BOS); classification tasks read logits at the last position, the
LM task (m06) reads per-position next-token distributions.

interp_bench models: tracr-convention encoding — token id = rank in
sorted(vocab + ['BOS', 'PAD']) by string; BOS prepended at position 0.
This convention is verified empirically in suites.py against case 3
(whose semantics, frac_prevs of 'x', we can check directly).
Outputs: per-position regression value (d_vocab_out == 1) or categorical
argmax, read at positions 1..L.
"""

import importlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .base import CaseSpec, Oracle, softmax

ROOT = Path(__file__).resolve().parents[2]

# Some InterpBench cfg pickles reference an older module layout.
_shim = types.ModuleType("transformer_lens.HookedTransformerConfig")
_shim.HookedTransformerConfig = HookedTransformerConfig
sys.modules["transformer_lens.HookedTransformerConfig"] = _shim


def load_tl_model(model_dir: Path) -> HookedTransformer:
    cfg = pickle.load(open(model_dir / "ll_model_cfg.pkl", "rb"))
    cfg_dict = cfg if isinstance(cfg, dict) else cfg.to_dict()
    cfg_dict = dict(cfg_dict, device="cpu")
    model = HookedTransformer(HookedTransformerConfig.from_dict(cfg_dict))
    model.load_state_dict(torch.load(model_dir / "ll_model.pth", map_location="cpu"))
    model.eval()
    return model


class _TorchOracle(Oracle):
    """Shared batched-logits machinery."""

    model: HookedTransformer

    def _logits(self, ids: np.ndarray, batch: int = 8192) -> np.ndarray:
        outs = []
        with torch.no_grad():
            for i in range(0, len(ids), batch):
                t = torch.from_numpy(ids[i : i + batch]).long()
                outs.append(self.model(t).float().cpu().numpy())
        return np.concatenate(outs)


class MessyOracle(_TorchOracle):
    def __init__(self, name: str):
        if str(ROOT / "messy") not in sys.path:
            sys.path.insert(0, str(ROOT / "messy"))
        self.task = importlib.import_module(f"tasks.{name}")
        self.model = load_tl_model(ROOT / "messy" / "models" / name)
        self.lm = bool(getattr(self.task, "LM", False))
        self.spec = CaseSpec(
            suite="messy",
            case_id=name,
            vocab=list(self.task.VOCAB),
            seq_lens=[self.task.SEQ_LEN],
            kind="lm" if self.lm else "classify",
            n_outputs=self.task.N_CLASSES,
            meta={"description": self.task.DESCRIPTION},
        )

    def sample(self, rng, n, length):
        tokens, _ = self.task.gen_batch(rng, n)
        return tokens  # gen_batch already returns vocab-index arrays

    def outputs(self, content):
        return self.probs(content).argmax(-1)

    def probs(self, content):
        logits = self._logits(content)
        if self.lm:
            return softmax(logits)  # (n, L, C)
        return softmax(logits[:, -1])  # (n, C)


class InterpOracle(_TorchOracle):
    def __init__(self, case_id: str, case_meta: dict):
        self.model = load_tl_model(ROOT / "interp_bench" / "tasks" / case_id)
        cfg = self.model.cfg
        vocab = sorted(case_meta["vocab"], key=str)
        ranked = sorted(vocab + ["BOS", "PAD"], key=str)
        self.tok_id = {t: ranked.index(t) for t in ranked}
        assert cfg.d_vocab == len(ranked), (
            f"case {case_id}: d_vocab {cfg.d_vocab} != {len(ranked)}")
        self.vocab_ids = np.array([self.tok_id[t] for t in vocab])
        self.bos_id = self.tok_id["BOS"]
        self.categorical = cfg.d_vocab_out > 1
        # SIIT models were trained at full length (shorter contents are
        # padded through the window and behave off-distribution — verified
        # on case 2, whose model reverses the full padded window exactly).
        self.spec = CaseSpec(
            suite="interp",
            case_id=case_id,
            vocab=vocab,
            seq_lens=[cfg.n_ctx - 1],
            kind="seq_cat" if self.categorical else "seq_num",
            n_outputs=cfg.d_vocab_out,
            output_values=list(range(cfg.d_vocab_out)) if self.categorical else None,
            meta={"description": case_meta["task_description"], "tol": 0.05},
        )

    def _encode(self, content):
        ids = self.vocab_ids[content]
        bos = np.full((len(ids), 1), self.bos_id, dtype=ids.dtype)
        return np.concatenate([bos, ids], axis=1)

    def outputs(self, content):
        logits = self._logits(self._encode(content))[:, 1:]  # drop BOS pos
        if self.categorical:
            return logits.argmax(-1)
        return logits[..., 0]

    def probs(self, content):
        if not self.categorical:
            return None
        return softmax(self._logits(self._encode(content))[:, 1:])
