"""MLX inference sanity check: reimplement the TransformerLens forward
pass in MLX from the raw InterpBench state dicts, and compare logits
against the PyTorch CPU reference on identical inputs."""

import math
import pickle
import sys
import time
import types
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

warnings.filterwarnings("ignore")
torch.manual_seed(0)

_shim = types.ModuleType("transformer_lens.HookedTransformerConfig")
_shim.HookedTransformerConfig = HookedTransformerConfig
sys.modules["transformer_lens.HookedTransformerConfig"] = _shim

TASKS = Path("/Users/jackhopkins/ModelVerification/interp_bench/tasks")


def load_cfg_dict(path):
    cfg = pickle.load(open(path, "rb"))
    cfg_dict = cfg if isinstance(cfg, dict) else cfg.to_dict()
    return dict(cfg_dict, device="cpu")


def act_fn(name):
    if name == "relu":
        return lambda x: mx.maximum(x, 0)
    if name == "gelu":  # exact (erf) gelu, matching torch F.gelu default
        return lambda x: 0.5 * x * (1 + mx.erf(x / math.sqrt(2)))
    if name == "gelu_new":  # gpt2 tanh approximation
        return lambda x: 0.5 * x * (
            1 + mx.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3))
        )
    raise NotImplementedError(name)


def layer_norm(x, w, b, eps):
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / mx.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    return x * w + b


def mlx_forward(cfg, sd, tokens):
    """tokens: (batch, seq) int array. Returns (batch, seq, d_vocab_out)."""
    g = lambda k: mx.array(sd[k].numpy())
    norm = cfg["normalization_type"]  # None | 'LN' | 'LNPre'
    eps = cfg["eps"]
    act = act_fn(cfg["act_fn"])
    scale = math.sqrt(cfg["d_head"]) if cfg["use_attn_scale"] else 1.0
    causal = cfg["attention_dir"] == "causal"
    seq = tokens.shape[1]

    resid = g("embed.W_E")[tokens] + g("pos_embed.W_pos")[:seq]

    for l in range(cfg["n_layers"]):
        p = f"blocks.{l}."

        x = resid
        if norm == "LN":
            x = layer_norm(x, g(p + "ln1.w"), g(p + "ln1.b"), eps)
        elif norm == "LNPre":
            x = layer_norm(x, 1.0, 0.0, eps)

        # attention: W_Q/K/V (heads, d_model, d_head), W_O (heads, d_head, d_model)
        q = mx.einsum("bsd,hde->bhse", x, g(p + "attn.W_Q")) + g(p + "attn.b_Q")[None, :, None, :]
        k = mx.einsum("bsd,hde->bhse", x, g(p + "attn.W_K")) + g(p + "attn.b_K")[None, :, None, :]
        v = mx.einsum("bsd,hde->bhse", x, g(p + "attn.W_V")) + g(p + "attn.b_V")[None, :, None, :]
        scores = mx.einsum("bhqe,bhke->bhqk", q, k) / scale
        if causal:
            mask = mx.triu(mx.full((seq, seq), -mx.inf), k=1)
            scores = scores + mask
        pattern = mx.softmax(scores, axis=-1)
        z = mx.einsum("bhqk,bhke->bhqe", pattern, v)
        attn_out = mx.einsum("bhqe,hed->bqd", z, g(p + "attn.W_O")) + g(p + "attn.b_O")
        resid = resid + attn_out

        x = resid
        if norm == "LN":
            x = layer_norm(x, g(p + "ln2.w"), g(p + "ln2.b"), eps)
        elif norm == "LNPre":
            x = layer_norm(x, 1.0, 0.0, eps)
        hidden = act(x @ g(p + "mlp.W_in") + g(p + "mlp.b_in"))
        resid = resid + hidden @ g(p + "mlp.W_out") + g(p + "mlp.b_out")

    if norm == "LN":
        resid = layer_norm(resid, g("ln_final.w"), g("ln_final.b"), eps)
    elif norm == "LNPre":
        resid = layer_norm(resid, 1.0, 0.0, eps)
    return resid @ g("unembed.W_U") + g("unembed.b_U")


results = []
t0 = time.time()
for task_dir in sorted(TASKS.iterdir(), key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name)):
    if not task_dir.is_dir():
        continue
    name = task_dir.name
    try:
        cfg_dict = load_cfg_dict(task_dir / "ll_model_cfg.pkl")
        sd = torch.load(task_dir / "ll_model.pth", map_location="cpu")

        # torch reference
        model = HookedTransformer(HookedTransformerConfig.from_dict(cfg_dict))
        model.load_state_dict(sd)
        model.eval()
        seq = min(cfg_dict["n_ctx"], 16)
        tokens = torch.randint(0, cfg_dict["d_vocab"], (2, seq))
        with torch.no_grad():
            ref = model(tokens).numpy()

        out = np.asarray(mlx_forward(cfg_dict, sd, mx.array(tokens.numpy())))
        diff = float(np.max(np.abs(out - ref)))
        rel = diff / (float(np.max(np.abs(ref))) + 1e-12)
        ok = diff < 1e-3 or rel < 1e-4
        results.append((name, "OK" if ok else "MISMATCH", diff))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))

n_ok = sum(1 for r in results if r[1] == "OK")
worst = max((r[2] for r in results if isinstance(r[2], float)), default=None)
for name, status, detail in results:
    if status != "OK":
        print(f"  {name:16} {status}: {detail}")
print(f"MLX (M4 Max): {n_ok}/{len(results)} models match the torch CPU logits "
      f"(worst max-abs-diff {worst:.2e}) in {time.time()-t0:.1f}s")
