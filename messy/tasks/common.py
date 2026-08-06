"""Shared helpers for messy-task data generation and evaluation."""

import numpy as np
import torch


def batched_logits(model, tokens_np, batch_size=4096):
    """Run the model over a numpy token array, return all logits as numpy."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(tokens_np), batch_size):
            toks = torch.from_numpy(tokens_np[i : i + batch_size]).long()
            outs.append(model(toks).float().cpu().numpy())
    return np.concatenate(outs)


def predict_last(model, tokens_np, batch_size=4096):
    """Argmax prediction read at the final sequence position."""
    return batched_logits(model, tokens_np, batch_size)[:, -1].argmax(-1)


def accuracy(model, tokens_np, labels_np):
    return float((predict_last(model, tokens_np) == labels_np).mean())


def one_hot_counts(tokens, n_values):
    """Per-row counts of each value: (n, seq) int array -> (n, n_values)."""
    n = tokens.shape[0]
    counts = np.zeros((n, n_values), dtype=np.int64)
    for v in range(n_values):
        counts[:, v] = (tokens == v).sum(axis=1)
    return counts
