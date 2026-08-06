"""Run one (suite, case, extractor) end to end: extract [-> CEGIS refine]
-> behavioral metrics -> formal certificate -> save under distill/runs/.

Usage: run_case.py <suite> <case_id> <extractor> [--cegis]
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from distill.metrics.behavioral import behavioral_metrics
from distill.metrics import formal
from distill.oracles import suites

RUNS = Path(__file__).resolve().parents[1] / "runs"


def get_extractor(name):
    if name == "tree":
        from distill.extractors.tree import TreeExtractor
        return TreeExtractor()
    if name == "tracr_exact":
        from distill.extractors.decompile.tracr_exact import TracrExactExtractor
        return TracrExactExtractor()
    if name == "tprogram":
        from distill.extractors.tprograms.extractor import TProgramExtractor
        return TProgramExtractor()
    if name == "tl_decompile":
        from distill.extractors.decompile.tl_decompile import TLDecompileExtractor
        return TLDecompileExtractor(max_rows=80000)
    if name == "sae_decompile":
        from distill.extractors.decompile.sae_decompile import SAEDecompileExtractor
        return SAEDecompileExtractor()
    raise ValueError(f"unknown extractor {name}")


def run_case(suite, case_id, extractor_name, cegis=False, seed=0):
    t0 = time.time()
    oracle = suites.load_oracle(suite, case_id)
    extractor = get_extractor(extractor_name)
    rng = np.random.default_rng(seed)

    program = extractor.extract(oracle, rng)
    rounds = 0
    if cegis:
        from distill.cegis.loop import cegis_refine
        program, rounds = cegis_refine(extractor, program, oracle, rng)

    metrics = behavioral_metrics(program, oracle, np.random.default_rng(seed + 1))
    cert = formal.check(program, oracle, np.random.default_rng(seed + 2))
    metrics["formal"] = {k: cert[k] for k in
                         ("mode", "n_checked", "n_disagreements",
                          "disagreement_rate", "exact_equivalent")}
    if suite == "rasp" and hasattr(program, "output"):
        try:
            from distill.metrics.structural import structural_score
            metrics["structural"] = structural_score(program, case_id)
        except Exception as e:
            metrics["structural"] = {"error": str(e)}
    metrics["complexity"] = program.complexity()
    metrics["is_hard"] = program.is_hard()
    if hasattr(program, "sae_report"):
        metrics["sae"] = program.sae_report
    metrics["cegis_rounds"] = rounds
    metrics["seconds"] = round(time.time() - t0, 1)

    outdir = RUNS / suite / case_id / (extractor_name + ("+cegis" if cegis else ""))
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (outdir / "certificate.json").write_text(json.dumps(cert, indent=2))
    try:
        (outdir / "program.txt").write_text(program.source())
    except Exception:
        pass
    return metrics


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    m = run_case(*args[:3], cegis="--cegis" in sys.argv)
    print(json.dumps(m, indent=2))
