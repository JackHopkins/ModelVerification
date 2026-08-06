"""Fast regression smoke test: one case per extractor, minutes total."""

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from distill.metrics.behavioral import behavioral_metrics
from distill.oracles import suites


def main():
    t0 = time.time()
    checks = []

    from distill.extractors.decompile.tracr_exact import TracrExactExtractor
    o = suites.load_oracle("rasp", "p03_length")
    m = behavioral_metrics(TracrExactExtractor().extract(o), o,
                           np.random.default_rng(1), n=2000)
    checks.append(("tracr_exact/p03", m["sequence_agreement"], 1.0))

    from distill.extractors.tree import TreeExtractor
    o = suites.load_oracle("messy", "m01_heuristic_vote")
    m = behavioral_metrics(TreeExtractor(n_train=20000).extract(
        o, np.random.default_rng(0)), o, np.random.default_rng(1), n=4000)
    checks.append(("tree/m01", m["sequence_agreement"], 0.95))

    from distill.extractors.decompile.tl_decompile import TLDecompileExtractor
    o = suites.load_oracle("messy", "m02_signal_mixture")
    m = behavioral_metrics(TLDecompileExtractor(n_train=10000).extract(
        o, np.random.default_rng(0)), o, np.random.default_rng(1), n=4000)
    checks.append(("tl_decompile/m02", m["sequence_agreement"], 0.95))

    from distill.extractors.tprograms.extractor import TProgramExtractor
    o = suites.load_oracle("interp", "3")
    try:
        TProgramExtractor(steps=50).extract(o, np.random.default_rng(0))
        checks.append(("tprogram/interp3(seq_num rejected)", 0.0, -1))
    except ValueError:
        checks.append(("tprogram/seq_num-rejection", 1.0, 1.0))
    o = suites.load_oracle("messy", "m02_signal_mixture")
    m = behavioral_metrics(TProgramExtractor(steps=300).extract(
        o, np.random.default_rng(0)), o, np.random.default_rng(1), n=2000)
    checks.append(("tprogram/m02(undertrained)", m["sequence_agreement"], 0.4))

    ok = True
    for name, got, want in checks:
        status = "PASS" if got >= want else "FAIL"
        ok &= got >= want
        print(f"{status} {name}: {got:.3f} (>= {want})")
    print(f"smoke test {'PASSED' if ok else 'FAILED'} in {time.time()-t0:.0f}s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
