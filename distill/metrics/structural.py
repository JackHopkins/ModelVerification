"""Structural program-recovery scoring (rasp suite only).

Compares the extracted DSL program's node-type multiset against the
ground-truth RASP program (walked from rasp/programs/<case>.py and mapped
onto DSL node types). Reports multiset precision/recall/F1. Exact
semantic equivalence is scored separately by the formal certificate —
this metric measures whether the *shape* of the algorithm was recovered.
"""

import importlib
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def gt_node_types(case_id):
    if str(ROOT / "rasp") not in sys.path:
        sys.path.insert(0, str(ROOT / "rasp"))
    module = importlib.import_module(f"programs.{case_id}")
    program = module.make_program()

    from tracr.rasp import rasp
    counts = Counter()
    seen = set()

    def walk(expr):
        if id(expr) in seen:
            return
        seen.add(id(expr))
        if isinstance(expr, rasp.TokensType):
            counts["Tokens"] += 1
        elif isinstance(expr, rasp.IndicesType):
            counts["Indices"] += 1
        elif isinstance(expr, rasp.Select):
            counts["Select"] += 1
        elif isinstance(expr, rasp.Aggregate):
            counts["Aggregate"] += 1
        elif isinstance(expr, rasp.SelectorWidth):
            counts["SelectorWidth"] += 1
        elif isinstance(expr, rasp.LinearSequenceMap):
            counts["LinComb"] += 1
        elif isinstance(expr, rasp.SequenceMap):
            counts["SeqMap"] += 1
        elif isinstance(expr, rasp.Map):
            counts["Map"] += 1
        for child in expr.children:
            walk(child)

    walk(program)
    return counts


def program_node_types(program):
    from distill.dsl.core import (Aggregate, Between, Cmp, Indices, LinComb,
                                  NaryMap, Select, SelectorWidth, SeqMap,
                                  TableMap, Tokens, _walk)
    counts = Counter()
    for node in _walk(program.output).values():
        t = type(node).__name__
        # DSL refinements of RASP Map all count as Map
        if t in ("TableMap", "Cmp", "Between"):
            t = "Map"
        elif t == "NaryMap":
            t = "SeqMap"
        if t in ("Tokens", "Indices", "Select", "Aggregate", "SelectorWidth",
                 "Map", "SeqMap", "LinComb"):
            counts[t] += 1
    return counts


def structural_score(program, case_id):
    gt = gt_node_types(case_id)
    got = program_node_types(program)
    tp = sum((gt & got).values())
    precision = tp / max(sum(got.values()), 1)
    recall = tp / max(sum(gt.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "node_f1": round(f1, 3),
        "node_precision": round(precision, 3),
        "node_recall": round(recall, 3),
        "gt_nodes": dict(gt),
        "extracted_nodes": dict(got),
    }
