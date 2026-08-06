"""CPU inference sanity check: load every InterpBench model with
transformer_lens and run a random-token batch through it."""

import pickle
import sys
import time
import types
import warnings
from pathlib import Path

import torch
import transformer_lens
from transformer_lens import HookedTransformer, HookedTransformerConfig

warnings.filterwarnings("ignore")
torch.manual_seed(0)

# Some configs were pickled when HookedTransformerConfig lived in its own
# module; give pickle that import path.
_shim = types.ModuleType("transformer_lens.HookedTransformerConfig")
_shim.HookedTransformerConfig = HookedTransformerConfig
sys.modules["transformer_lens.HookedTransformerConfig"] = _shim


def load_cfg(path):
    cfg = pickle.load(open(path, "rb"))
    cfg_dict = cfg if isinstance(cfg, dict) else cfg.to_dict()
    cfg_dict = dict(cfg_dict, device="cpu")  # configs were saved with cuda
    return HookedTransformerConfig.from_dict(cfg_dict)

TASKS = Path("/Users/jackhopkins/ModelVerification/interp_bench/tasks")

results = []
t0 = time.time()
for task_dir in sorted(TASKS.iterdir(), key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else 0, p.name)):
    if not task_dir.is_dir():
        continue
    name = task_dir.name
    try:
        cfg = load_cfg(task_dir / "ll_model_cfg.pkl")
        model = HookedTransformer(cfg)
        sd = torch.load(task_dir / "ll_model.pth", map_location="cpu")
        model.load_state_dict(sd)
        model.eval()

        seq = min(cfg.n_ctx, 16)
        tokens = torch.randint(0, cfg.d_vocab, (2, seq))
        with torch.no_grad():
            logits = model(tokens)
        finite = bool(torch.isfinite(logits).all())
        results.append((name, "OK" if finite else "NONFINITE",
                        f"logits{tuple(logits.shape)}"))
    except Exception as e:
        results.append((name, "FAIL", f"{type(e).__name__}: {e}"))

n_ok = sum(1 for r in results if r[1] == "OK")
for name, status, detail in results:
    if status != "OK":
        print(f"  {name:16} {status}: {detail}")
print(f"CPU (torch {torch.__version__}): {n_ok}/{len(results)} models loaded and produced finite logits "
      f"in {time.time()-t0:.1f}s")
