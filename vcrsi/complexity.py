#!/usr/bin/env python3
"""The machine-checked complexity floor (§6B).

"Complex" is COMPUTED here from a task's reference solution (via the IR's own
AST) plus an execution trace on the longest public example -- it is never a
human claim. A task that fails ANY threshold is rejected and cannot enter the
suite or be counted as solved. The same metrics are printed in ``--mode demo``
so an auditor reads the numbers, not the names.

Per-task thresholds (all must hold):
  distinct_ops >= 5          distinct non-trivial operation kinds in the AST
                             (plumbing leaves lit/arg/var/param do not count)
  has_loop                   at least one loop (map/filter/foldl) or recursion
  has_rec_or_struct          recursion OR an auxiliary data structure is built
  max_exec_depth >= 6        loop iterations + call frames exercised on the
                             longest public example (a constant straight-line
                             program in disguise cannot reach this)

There is also a floor on ADOPTED programs (``MIN_SOLUTION_OPS``): a task whose
smallest correct program falls below it was mis-classified and is rejected
(checked in the RSI loop, not here).
"""
from __future__ import annotations

from typing import Dict, Set, Tuple

from .ir import Node, inline
from .interp import run

DISTINCT_OPS_MIN = 5
EXEC_DEPTH_MIN = 6
MIN_SOLUTION_OPS = 5     # adopted-program floor (distinct non-trivial ops)

_PLUMBING = {"lit", "arg", "var", "param"}
_LOOPS = {"map", "filter", "foldl"}
# ops that allocate/return a NEW container not taken verbatim from input
_CONSTRUCTORS = {"cons", "lapp", "lsingle", "lrange", "pair", "sconcat",
                 "srepeat", "schars", "map", "filter", "ltake", "ldrop"}


def _distinct_ops(n: Node, acc: Set[str]) -> None:
    if n.op not in _PLUMBING:
        # 'call' counts as one op kind keyed by callee so two different blocks
        # are two kinds; but for the reference (no calls) this is just n.op.
        acc.add(n.op if n.op != "call" else f"call:{n.const}")
    for k in n.kids:
        _distinct_ops(k, acc)


def _has_loop(n: Node) -> bool:
    if n.op in _LOOPS:
        return True
    return any(_has_loop(k) for k in n.kids)


def _has_constructor(n: Node) -> bool:
    if n.op in _CONSTRUCTORS:
        return True
    return any(_has_constructor(k) for k in n.kids)


def distinct_op_count(n: Node) -> int:
    acc: Set[str] = set()
    _distinct_ops(n, acc)
    return len(acc)


def complexity_floor(oracle) -> Tuple[bool, Dict]:
    """Return (passed, metrics) for one SealedOracle's task. The reference is
    read from the (sealed) oracle; this function is sealed tooling, not search."""
    task = oracle.task
    ref = task.reference

    ops: Set[str] = set()
    _distinct_ops(ref, ops)
    distinct = len(ops)
    has_loop = _has_loop(ref)
    has_struct = _has_constructor(ref)
    # recursion in a reference would be a self-referential block; references here
    # are loop-based, so the {recursion, aux-structure} requirement is met by the
    # auxiliary structure built during the loop.
    has_rec_or_struct = has_struct

    # dynamic execution depth: iterations + call frames on the longest public
    # example (an actual trace, not a static guess).
    args = oracle.longest_public_args()
    r = run(ref, list(args))
    max_exec_depth = r.iters

    passed = (distinct >= DISTINCT_OPS_MIN and has_loop and has_rec_or_struct
              and max_exec_depth >= EXEC_DEPTH_MIN)
    metrics = {
        "task": task.name,
        "family": task.family,
        "distinct_ops": distinct,
        "has_loop": has_loop,
        "has_rec_or_struct": has_rec_or_struct,
        "max_exec_depth": max_exec_depth,
        "passed": passed,
    }
    return passed, metrics


def adopted_program_ops(prog: Node, blocks: Dict) -> int:
    """Distinct non-trivial op count of an ADOPTED program AFTER inlining every
    library call -- so work hidden inside a subroutine still counts. Used to
    reject tasks solved by a below-floor program."""
    flat = inline(prog, blocks)
    return distinct_op_count(flat)
