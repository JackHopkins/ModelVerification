"""Aggregate distill/runs/** into a comparison table (markdown + csv)."""

import json
import sys
from collections import defaultdict
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def rows():
    for mfile in sorted(RUNS.glob("*/*/*/metrics.json")):
        extractor = mfile.parent.name
        case = mfile.parent.parent.name
        suite = mfile.parent.parent.parent.name
        m = json.loads(mfile.read_text())
        yield suite, case, extractor, m


def main():
    by_ext_suite = defaultdict(list)
    lines = ["| suite | case | extractor | seq_agree | pos_agree | TV | exact | cex_rate | nodes | s |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for suite, case, ext, m in rows():
        f = m["formal"]
        by_ext_suite[(ext, suite)].append(m)
        lines.append(
            f"| {suite} | {case} | {ext} "
            f"| {m.get('sequence_agreement', float('nan')):.3f} "
            f"| {m.get('position_agreement', float('nan')):.3f} "
            f"| {m.get('mean_tv', float('nan')):.4f} "
            f"| {'YES' if f['exact_equivalent'] else ''} "
            f"| {f['disagreement_rate']:.2e} "
            f"| {m['complexity']['n_nodes']} | {m['seconds']} |")

    summary = ["", "## Summary (mean per extractor x suite)", "",
               "| extractor | suite | n | seq_agree | exact_count |", "|---|---|---|---|---|"]
    for (ext, suite), ms in sorted(by_ext_suite.items()):
        agree = sum(m.get("sequence_agreement", 0) for m in ms) / len(ms)
        exact = sum(m["formal"]["exact_equivalent"] for m in ms)
        summary.append(f"| {ext} | {suite} | {len(ms)} | {agree:.3f} | {exact}/{len(ms)} |")

    out = "\n".join(summary + [""] + lines) + "\n"
    (RUNS / "TABLE.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
