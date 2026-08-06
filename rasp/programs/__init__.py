"""Registry of RASP programs, ordered by increasing complexity.

Each module exposes:
  NAME          — unique identifier, also the output directory name
  DESCRIPTION   — one-line semantics
  VOCAB         — set of input tokens (excluding BOS/pad)
  MAX_SEQ_LEN   — maximum input length the compiled model supports
  EXAMPLES      — list of example inputs used for validation
  make_program()— returns the rasp.SOp to compile
"""

import importlib

PROGRAM_MODULES = [
    "p01_identity",
    "p02_increment",
    "p03_length",
    "p04_shift_right",
    "p05_frac_prevs",
    "p06_hist",
    "p07_pair_balance",
    "p08_reverse",
    "p09_detect_pattern",
    "p10_sort",
    "p11_dyck1",
    "p12_dyck2",
]


def load_all():
    """Import and return all program modules in complexity order."""
    return [
        importlib.import_module(f"programs.{name}") for name in PROGRAM_MODULES
    ]
