"""Train every messy task's model from scratch and save it in the same
format as interp_bench (ll_model.pth + ll_model_cfg.pkl + meta.json),
plus eval.json with the task's messiness diagnostics.

Run:  ../.venv/bin/python train.py [task_name ...]
"""

import json
import math
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

import tasks

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
MODELS = HERE / "models"

TRAIN_DEFAULTS = {"steps": 3000, "batch": 256, "lr": 1e-3, "weight_decay": 0.01,
                  "warmup": 100, "seed": 0}
MODEL_DEFAULTS = {"n_layers": 2, "d_model": 32, "n_heads": 4}


def build_cfg(task, seed):
    m = dict(MODEL_DEFAULTS, **task.MODEL)
    return HookedTransformerConfig(
        n_layers=m["n_layers"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        d_head=m["d_model"] // m["n_heads"],
        d_mlp=4 * m["d_model"],
        n_ctx=task.SEQ_LEN,
        d_vocab=len(task.VOCAB),
        d_vocab_out=task.N_CLASSES,
        act_fn="gelu",
        normalization_type="LN",
        attention_dir="causal",
        positional_embedding_type="standard",
        seed=seed,
        device="cpu",
    )


def loss_fn(task, logits, toks, labels):
    if getattr(task, "LM", False):
        start = task.LM_LOSS_START
        return F.cross_entropy(
            logits[:, start:-1].reshape(-1, logits.shape[-1]),
            toks[:, start + 1 :].reshape(-1),
        )
    return F.cross_entropy(logits[:, -1], labels)


def train_task(task):
    t0 = time.time()
    tcfg = dict(TRAIN_DEFAULTS, **getattr(task, "TRAIN", {}))
    torch.manual_seed(tcfg["seed"])
    rng = np.random.default_rng(tcfg["seed"])

    cfg = build_cfg(task, tcfg["seed"])
    model = HookedTransformer(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    steps, warmup = tcfg["steps"], tcfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min((s + 1) / warmup, 1.0)
        * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))),
    )

    losses = []
    for step in range(steps):
        toks_np, labels_np = task.gen_batch(rng, tcfg["batch"])
        toks = torch.from_numpy(toks_np).long()
        labels = torch.from_numpy(labels_np).long()
        logits = model(toks)
        loss = loss_fn(task, logits, toks, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if (step + 1) % max(steps // 5, 1) == 0:
            recent = float(np.mean(losses[-100:]))
            print(f"    step {step + 1:>5}/{steps}  loss {recent:.4f}", flush=True)

    model.eval()
    metrics = task.evaluate(model, np.random.default_rng(tcfg["seed"] + 1))

    outdir = MODELS / task.NAME
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), outdir / "ll_model.pth")
    with open(outdir / "ll_model_cfg.pkl", "wb") as f:
        pickle.dump(cfg.to_dict(), f)
    n_params = sum(p.numel() for p in model.parameters())
    meta = {
        "task": task.NAME,
        "description": task.DESCRIPTION,
        "vocab": list(task.VOCAB),
        "seq_len": task.SEQ_LEN,
        "n_classes": task.N_CLASSES,
        "lm": bool(getattr(task, "LM", False)),
        "train": tcfg,
        "n_params": n_params,
        "final_loss_mean100": float(np.mean(losses[-100:])),
        "train_seconds": round(time.time() - t0, 1),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    (outdir / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(f"  {task.NAME}: {n_params} params, "
          f"final loss {meta['final_loss_mean100']:.4f}, "
          f"{meta['train_seconds']}s")
    return metrics


def main():
    only = set(sys.argv[1:])
    for task in tasks.load_all():
        if only and task.NAME not in only:
            continue
        print(f"== {task.NAME}", flush=True)
        metrics = train_task(task)
        for k, v in metrics.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
