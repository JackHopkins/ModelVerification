"""Sweep extractors over suites; resumable (skips existing runs).

Usage: run_suite.py --suite rasp,messy --extractor tree [--cegis] [--redo]
       run_suite.py --suite interp --extractor tree --limit 20
"""

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

from distill.oracles import suites
from distill.scripts.run_case import RUNS, run_case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="rasp,interp,messy")
    ap.add_argument("--extractor", default="tree")
    ap.add_argument("--cegis", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="max cases per suite")
    args = ap.parse_args()

    for suite in args.suite.split(","):
        n = 0
        for s, cid in suites.iter_cases(suite):
            if args.limit and n >= args.limit:
                break
            n += 1
            for ex in args.extractor.split(","):
                tag = ex + ("+cegis" if args.cegis else "")
                out = RUNS / s / cid / tag / "metrics.json"
                if out.exists() and not args.redo:
                    continue
                try:
                    m = run_case(s, cid, ex, cegis=args.cegis)
                    print(f"{s}/{cid}/{tag}: "
                          f"agree={m.get('sequence_agreement', 0):.3f} "
                          f"tv={m.get('mean_tv', float('nan')):.4f} "
                          f"exact={m['formal']['exact_equivalent']} "
                          f"({m['seconds']}s)", flush=True)
                except Exception as e:
                    print(f"{s}/{cid}/{tag}: FAIL {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()


if __name__ == "__main__":
    main()
