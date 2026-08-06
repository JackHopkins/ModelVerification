"""Oracle protocol: a uniform interface over all target models.

Content sequences are numpy int arrays of *vocab indices* — index i means
spec.vocab[i]. Oracles translate to model-specific token ids internally.

Output kinds:
  seq_cat  — per-position categorical value: outputs (n, L) int indices
             into spec.output_values  (tracr categorical)
  seq_num  — per-position float: outputs (n, L) float  (tracr numerical,
             interp_bench regression heads)
  classify — single label read at last position: outputs (n,) int,
             probs (n, C)
  lm       — next-token distribution per position: outputs (n, L) int
             argmax, probs (n, L, C)
"""

import abc
import dataclasses
from typing import Any, Iterator, Optional

import numpy as np


@dataclasses.dataclass
class CaseSpec:
    suite: str  # 'rasp' | 'interp' | 'messy'
    case_id: str
    vocab: list  # content tokens; content arrays hold indices into this
    seq_lens: list  # valid content lengths
    kind: str  # 'seq_cat' | 'seq_num' | 'classify' | 'lm'
    n_outputs: int  # #classes (classify/lm/seq_cat) or 1 (seq_num)
    output_values: Optional[list] = None  # seq_cat: index -> value
    meta: dict = dataclasses.field(default_factory=dict)


class Oracle(abc.ABC):
    spec: CaseSpec

    @abc.abstractmethod
    def outputs(self, content: np.ndarray) -> np.ndarray:
        """Canonical comparable output for a (n, L) vocab-index array."""

    def probs(self, content: np.ndarray) -> Optional[np.ndarray]:
        """Output distributions where the model defines them, else None."""
        return None

    def sample(self, rng: np.random.Generator, n: int, length: int) -> np.ndarray:
        """On-distribution content sample of a given length. Default uniform."""
        return rng.integers(0, len(self.spec.vocab), (n, length))

    def sample_batches(self, rng, n_total: int):
        """Yield (n_i, L) samples spread across valid lengths."""
        lens = self.spec.seq_lens
        per = max(n_total // len(lens), 1)
        for length in lens:
            yield self.sample(rng, per, length)

    def enum_size(self) -> int:
        v = len(self.spec.vocab)
        return sum(v**length for length in self.spec.seq_lens)

    def enumerate_inputs(self, chunk: int = 65536) -> Iterator[np.ndarray]:
        """Yield chunks of the complete input space, grouped by length."""
        v = len(self.spec.vocab)
        for length in self.spec.seq_lens:
            total = v**length
            for start in range(0, total, chunk):
                idx = np.arange(start, min(start + chunk, total), dtype=np.int64)
                out = np.empty((len(idx), length), dtype=np.int64)
                rem = idx
                for pos in range(length - 1, -1, -1):
                    out[:, pos] = rem % v
                    rem = rem // v
                yield out


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)
