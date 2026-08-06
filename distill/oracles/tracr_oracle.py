"""Numpy oracle for TRACR-compiled models in rasp/compiled/.

Reimplements tracr's forward pass exactly (verified against every example
in validation.json at load time):
  resid = token_embed[ids] + pos_embed[positions]
  per layer: multi-head softmax attention (scores / sqrt(key_size),
             no mask), output linear; then relu MLP; residual adds
  unembed: resid @ P where P selects the output-space dims named in
           io_spec.output_labels; categorical -> argmax, numerical -> scalar

BOS sits at sequence position 0; content outputs are read at 1..L.
"""

import json
from pathlib import Path

import numpy as np

from .base import CaseSpec, Oracle

RASP_DIR = Path(__file__).resolve().parents[2] / "rasp"


class TracrOracle(Oracle):
    def __init__(self, name: str):
        d = RASP_DIR / "compiled" / name
        arch = json.loads((d / "architecture.json").read_text())
        spec_io = json.loads((d / "io_spec.json").read_text())
        params = np.load(d / "params.npz")

        self.arch = arch
        self.p = {k: params[k].astype(np.float64) for k in params.files}
        self.n_layers = arch["num_layers"]
        self.n_heads = arch["num_heads"]
        self.key_size = arch["key_size"]
        self.encoding = dict(
            (tok if not isinstance(tok, list) else tuple(tok), idx)
            for tok, idx in spec_io["input_encoding_map"]
        )
        self.bos_id = self.encoding[arch["bos_token"]]
        self.categorical = spec_io["categorical_output"]

        labels = arch["residual_labels"]
        self.out_dims = np.array(
            [labels.index(lab) for lab in spec_io["output_labels"]]
        )

        vocab = sorted(arch["vocab"], key=str)
        self.vocab_ids = np.array([self.encoding[t] for t in vocab])

        output_values = spec_io.get("output_values")
        self.spec = CaseSpec(
            suite="rasp",
            case_id=name,
            vocab=vocab,
            seq_lens=list(range(1, arch["max_seq_len"] + 1)),
            kind="seq_cat" if self.categorical else "seq_num",
            n_outputs=len(output_values) if self.categorical else 1,
            output_values=output_values,
            meta={"program_module": f"programs.{name}", "dir": str(d)},
        )
        self._validate(d)

    # -- forward ---------------------------------------------------------

    def _forward(self, ids: np.ndarray) -> np.ndarray:
        """ids: (n, T) model token ids incl. BOS. Returns final resid (n,T,D)."""
        n, T = ids.shape
        resid = self.p["token_embed||embeddings"][ids]
        resid = resid + self.p["pos_embed||embeddings"][np.arange(T)][None]
        H, K = self.n_heads, self.key_size
        for l in range(self.n_layers):
            a = f"transformer/layer_{l}/attn/"
            q = (resid @ self.p[a + "query||w"] + self.p[a + "query||b"]).reshape(n, T, H, K)
            k = (resid @ self.p[a + "key||w"] + self.p[a + "key||b"]).reshape(n, T, H, K)
            v = (resid @ self.p[a + "value||w"] + self.p[a + "value||b"]).reshape(n, T, H, K)
            scores = np.einsum("nqhk,nthk->nhqt", q, k) / np.sqrt(K)
            scores = scores - scores.max(axis=-1, keepdims=True)
            w = np.exp(scores)
            w = w / w.sum(axis=-1, keepdims=True)
            z = np.einsum("nhqt,nthk->nqhk", w, v).reshape(n, T, H * K)
            resid = resid + z @ self.p[a + "linear||w"] + self.p[a + "linear||b"]
            m = f"transformer/layer_{l}/mlp/"
            hid = np.maximum(resid @ self.p[m + "linear_1||w"] + self.p[m + "linear_1||b"], 0)
            resid = resid + hid @ self.p[m + "linear_2||w"] + self.p[m + "linear_2||b"]
        return resid

    def _encode(self, content: np.ndarray) -> np.ndarray:
        ids = self.vocab_ids[content]
        bos = np.full((len(ids), 1), self.bos_id, dtype=ids.dtype)
        return np.concatenate([bos, ids], axis=1)

    def outputs(self, content: np.ndarray) -> np.ndarray:
        resid = self._forward(self._encode(content))
        out = resid[:, 1:, :][:, :, self.out_dims]  # drop BOS position
        if self.categorical:
            return out.argmax(-1)
        return out[..., 0]

    def decode_var(self, content: np.ndarray, dims: list, categorical: bool):
        """Ground-truth value of an internal variable at every content
        position, read off the final residual stream (each tracr variable
        is written exactly once, so the final stream holds all of them)."""
        resid = self._forward(self._encode(content))[:, 1:, :]
        sub = resid[:, :, dims]
        if categorical:
            return sub.argmax(-1), sub.max(-1) > 0.5
        return sub[..., 0], np.ones(sub.shape[:2], bool)

    # -- load-time validation -------------------------------------------

    def _validate(self, d: Path):
        val = json.loads((d / "validation.json").read_text())
        tok_to_idx = {t: i for i, t in enumerate(self.spec.vocab)}
        for ex in val:
            content = np.array([[tok_to_idx[t] for t in ex["input"]]])
            got = self.outputs(content)[0]
            for e, g in zip(ex["model_decoded"], got):
                gv = self.spec.output_values[g] if self.categorical else g
                if isinstance(e, float) or isinstance(gv, float):
                    assert abs(float(e) - float(gv)) < 1e-3, (
                        f"{self.spec.case_id}: numpy forward mismatch {e} vs {gv}")
                else:
                    assert e == gv, (
                        f"{self.spec.case_id}: numpy forward mismatch {e} vs {gv}")
