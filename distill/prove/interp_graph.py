"""Dependency graph for TRAINED InterpBench models (TransformerLens).

Trained weights are DENSE: unlike tracr, a dependency cannot be read off
weight sparsity — every component nominally connects to every other. The
sound primitive is CAUSAL: an edge (A -> B) exists iff intervening on A's
activation changes B's activation (or the output) beyond a threshold. This
is the activation-patching / ACDC methodology `edges.pkl` was derived from.

We compute, over the model's own hooks, an edge attribution and compare
the recovered circuit to the ground-truth `edges.pkl`. This is NOT a
weight-level theorem like the tracr path (dense nets don't admit one for
free); it is a sound *causal* dependency measured on the real network over
the input distribution, with the honest caveat that it is distributional
(patched over sampled inputs), not a forall over the whole input space.

Node vocabulary matches edges.pkl:
  hook_embed, hook_pos_embed,
  blocks.L.hook_mlp_in / hook_mlp_out,
  blocks.L.hook_{q,k,v}_input[h] / attn.hook_{q,k,v}[h] / attn.hook_result[h],
  blocks.L.hook_resid_post
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from distill.oracles import suites

ROOT = Path(__file__).resolve().parents[2]


def _edge_targets(model):
    """Downstream hook points we measure the effect of a patch at, mapped to
    a readable node name. Covers MLP-in and per-head q/k/v inputs + result,
    matching edges.pkl granularity."""
    cfg = model.cfg
    nodes = {}
    for l in range(cfg.n_layers):
        nodes[f"blocks.{l}.hook_mlp_in"] = f"blocks.{l}.hook_mlp_in"
        nodes[f"blocks.{l}.hook_mlp_out"] = f"blocks.{l}.hook_mlp_out"
        for h in range(cfg.n_heads):
            for io in ("q", "k", "v"):
                nodes[(f"blocks.{l}.hook_{io}_input", h)] = \
                    f"blocks.{l}.hook_{io}_input[{h}]"
            nodes[(f"blocks.{l}.attn.hook_result", h)] = \
                f"blocks.{l}.attn.hook_result[{h}]"
        nodes[f"blocks.{l}.hook_resid_post"] = f"blocks.{l}.hook_resid_post"
    return nodes


def _sources(model):
    """Upstream nodes whose contribution we ablate: embed, pos_embed, each
    head's result, each MLP out. These are the residual-stream WRITERS whose
    removal defines an edge."""
    cfg = model.cfg
    src = ["hook_embed", "hook_pos_embed"]
    for l in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            src.append((f"blocks.{l}.attn.hook_result", h))
        src.append(f"blocks.{l}.hook_mlp_out")
    return src


def _auto_threshold(effects):
    """Set the used/dead cut at the largest ABSOLUTE drop in the sorted
    effect spectrum: SIIT circuits show a cliff between genuine-circuit
    sources (large effect) and mean-ablation leakage (small). The ratio
    gap is dominated by the near-zero tail and misses this cliff, so we use
    the absolute drop. Returns (threshold, drop-to-next-ratio) so the report
    can say how clean the separation was."""
    vals = sorted([v for v in effects.values() if v > 0], reverse=True)
    if len(vals) < 2:
        return 1e-3, 1.0
    drops = [(vals[i] - vals[i + 1], i) for i in range(len(vals) - 1)]
    _, i = max(drops)
    thresh = (vals[i] + vals[i + 1]) / 2
    ratio = vals[i] / max(vals[i + 1], 1e-9)
    return thresh, ratio


def build_dependency_graph(case_id, n=256, thresh=None, seed=0, verbose=True):
    """Causal edge attribution via mean-ablation patching.

    For each residual-writer source S, we replace S's activation with its
    MEAN over the batch (destroying the input-dependent signal it carries)
    and measure the change in the model OUTPUT (KL / logit shift). A source
    with effect > thresh is 'used'. For edge granularity we also measure the
    effect of S on each downstream head-input / mlp-in read.

    Sound but distributional: the effect is averaged over sampled inputs;
    an edge reported ABSENT means no measured effect on this distribution.
    """
    oracle = suites.load_oracle("interp", case_id)
    model = oracle.model
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    content = oracle.sample(rng, n, max(oracle.spec.seq_lens))
    ids = torch.from_numpy(oracle._encode(content)).long()
    # CORRUPT baseline: a different (shuffled) batch. Patching a source to its
    # corrupt activation is an ON-DISTRIBUTION intervention -- much sharper
    # than mean-ablation (which is off-distribution and lets other components
    # compensate, hiding used ones -- the self-repair/hydra effect).
    perm = torch.from_numpy(rng.permutation(len(ids)))
    ids_corr = ids[perm]

    with torch.no_grad():
        base_logits = model(ids)
        _, corr_cache = model.run_with_cache(ids_corr)
    base_lp = torch.log_softmax(base_logits, -1)

    sources = _sources(model)
    used = {}
    for S in sources:
        hookname = S[0] if isinstance(S, tuple) else S
        head = S[1] if isinstance(S, tuple) else None
        corr_act = corr_cache[hookname]

        def patch(t, hook, head=head, corr=corr_act):
            if head is not None:
                t[:, :, head] = corr[:, :, head]
            else:
                t[:] = corr
            return t

        with torch.no_grad():
            patched = model.run_with_hooks(ids, fwd_hooks=[(hookname, patch)])
        plp = torch.log_softmax(patched, -1)
        effect = float((plp - base_lp).abs().mean())
        name = (f"{hookname}[{head}]" if head is not None else hookname)
        used[name] = round(effect, 5)

    if thresh is None:
        thresh, gap_ratio = _auto_threshold(used)
    else:
        gap_ratio = None
    live = {k: v for k, v in used.items() if v > thresh}
    if verbose:
        gr = f", gap ratio {gap_ratio:.1f}x" if gap_ratio else ""
        print(f"  {case_id}: {len(live)}/{len(used)} sources causally used "
              f"(auto thresh {thresh:.3f}{gr})")
        for k, v in sorted(used.items(), key=lambda kv: -kv[1]):
            mark = "USED" if v > thresh else "dead"
            print(f"    {mark} {k}: effect={v}")
    return {"case": case_id, "effects": used, "live_sources": list(live)}


def head_input_edges(case_id, n=256, seed=0, verbose=True):
    """Finer granularity: corrupt-baseline path patch into each head's
    q/k/v INPUT (hook_{q,k,v}_input[h]) — the granularity 27 of 86 circuits
    route through (b3.q_input[0] etc.). Sharper than source-level ablation:
    on case 21, a circuit q-input effects ~5.0 vs ~0.5 for a non-circuit one.
    Returns per-(layer,head,io) effect + auto-threshold live set."""
    oracle = suites.load_oracle("interp", case_id)
    model = oracle.model
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    content = oracle.sample(rng, n, max(oracle.spec.seq_lens))
    ids = torch.from_numpy(oracle._encode(content)).long()
    ids_corr = ids[torch.from_numpy(rng.permutation(len(ids)))]
    with torch.no_grad():
        base_lp = torch.log_softmax(model(ids), -1)
        _, corr = model.run_with_cache(ids_corr)

    eff = {}
    for l in range(cfg.n_layers):
        for io in ("q", "k", "v"):
            hook = f"blocks.{l}.hook_{io}_input"
            if hook not in corr:
                continue
            for h in range(cfg.n_heads):
                def patch(t, hook, h=h, c=corr[hook]):
                    t[:, :, h] = c[:, :, h]
                    return t
                with torch.no_grad():
                    ph = model.run_with_hooks(ids, fwd_hooks=[(hook, patch)])
                e = float((torch.log_softmax(ph, -1) - base_lp).abs().mean())
                eff[f"blocks.{l}.hook_{io}_input[{h}]"] = round(e, 5)
    thresh, gap = _auto_threshold(eff)
    live = [k for k, v in eff.items() if v > thresh]
    if verbose:
        print(f"  {case_id}: {len(live)} head-input edges live "
              f"(gap {gap:.1f}x)")
        for k, v in sorted(eff.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {'USED' if v > thresh else 'dead'} {k}: {v}")
    return {"case": case_id, "head_input_effects": eff, "live": live}


def load_ground_truth(case_id):
    import pickle
    return pickle.load(open(ROOT / "interp_bench" / "tasks" / str(case_id)
                            / "edges.pkl", "rb"))


def gt_live_sources(case_id):
    """Ground-truth writers that appear as an edge SOURCE (embed / pos /
    head result / mlp out)."""
    edges = load_ground_truth(case_id)
    live = set()
    for s, d in edges:
        # normalize head-result sources; embed/pos; mlp_out
        if s in ("hook_embed", "hook_pos_embed"):
            live.add(s)
        elif "attn.hook_result" in s:
            live.add(s)
        elif "hook_mlp_out" in s:
            live.add(s)
    return live


def circuit_equality_interp(case_id, thresh=None, verbose=True):
    """Compare measured live sources vs edges.pkl live sources."""
    dg = build_dependency_graph(case_id, thresh=None, verbose=False)
    measured = set(dg["live_sources"])
    gt = gt_live_sources(case_id)
    # normalize: measured uses "[h]" for heads; gt uses "attn.hook_result[h]"
    def norm(s):
        return s.replace("blocks.", "b").replace(".attn.hook_result", ".res")
    m_n = {norm(x) for x in measured}
    gt_n = {norm(x) for x in gt}
    missing = gt_n - m_n     # in ground truth, we measured DEAD (false neg)
    extra = m_n - gt_n       # we measured USED, not in ground truth
    # separation quality: mean-ablation only cleanly recovers a circuit when
    # the effect spectrum has a CLIFF (single-node ablation misses components
    # that others compensate for -- the self-repair/hydra effect). Report it.
    vals = sorted(dg["effects"].values(), reverse=True)
    _, gap_ratio = _auto_threshold(dg["effects"])
    clean = gap_ratio >= 5.0
    if verbose:
        print(f"  {case_id}: measured {len(measured)} live, gt {len(gt)} live "
              f"(separation {'CLEAN' if clean else 'SMOOTH -- unreliable'}, "
              f"gap {gap_ratio:.1f}x)")
        print(f"    missing (gt says live, we say dead): {sorted(missing)}")
        print(f"    extra   (we say live, gt says dead): {sorted(extra)}")
        if not clean:
            print("    NOTE: no cliff in the effect spectrum; single-node "
                  "mean-ablation under-attributes multi-layer circuits "
                  "(self-repair). Path patching needed for these.")
    return {"case": case_id, "measured": sorted(measured), "ground_truth": sorted(gt),
            "missing": sorted(missing), "extra": sorted(extra)}


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "75"
    print(f"== interp dependency graph: case {case} ==")
    build_dependency_graph(case)
    print("== circuit-equality vs edges.pkl ==")
    circuit_equality_interp(case)
