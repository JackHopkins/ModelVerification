"""Re-download the full InterpBench model set into interp_bench/tasks.

InterpBench (Gupta et al., NeurIPS 2024) is a set of semi-synthetic
transformers trained with Strict Interchange Intervention Training (SIIT)
to implement known circuits — 84 numbered tasks from circuits-benchmark
plus two IOI models. Repo-level metadata files are moved up one level so
that tasks/ contains only task directories.

Run:  ../.venv/bin/python download.py
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

HERE = Path(__file__).parent
TASKS = HERE / "tasks"

METADATA_FILES = [
    "README.md",
    "benchmark_cases_metadata.csv",
    "benchmark_cases_metadata.parquet",
    "benchmark_metadata.json",
    "benchmark_metadata_croissant.json",
]


def main():
    snapshot_download(repo_id="cybershiptrooper/InterpBench", local_dir=TASKS)
    for name in METADATA_FILES:
        src = TASKS / name
        if src.exists():
            shutil.move(src, HERE / name)
    for junk in [TASKS / ".cache", TASKS / ".gitattributes"]:
        if junk.is_dir():
            shutil.rmtree(junk)
        elif junk.exists():
            junk.unlink()
    n = sum(1 for p in TASKS.iterdir() if p.is_dir())
    print(f"{n} task directories in {TASKS}")


if __name__ == "__main__":
    main()
