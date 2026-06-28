#!/usr/bin/env python3
"""Cost-aware program optimization across the correctness-equivalence class.

This is the INNER self-improvement loop of the cheap-verifier experiment. For one
target it:

  1. finds a verified correct SEED via the existing correctness-blind portfolio
     (bottom-up OE / memetic search), gated ONLY by the sealed oracle's ``verify``
     callback -- it never reads the reference solution;
  2. forms the cost-UNAWARE BASELINE E(S): the seed guarded by a cold conditional
     ``ifx(C_false, DEAD, S)`` whose predicate is always false on the data (so DEAD
     never runs and the program is exactly as correct as S), plus some out-of-loop
     identity wrapping. This is the kind of correct-but-unlean program a synthesizer
     with no cost signal ships and has no pressure to prune;
  3. minimizes PROXY-predicted cost over the rewrite-neighborhood of E(S), gated by
     the sealed oracle at EVERY step (a rewrite is kept only if the oracle still
     accepts it). A TRAINED proxy descends -- it strips the cold guard and the wraps,
     because it counts their nodes -- to the lean program; a FROZEN (untrained)
     proxy has no gradient and stays at E(S).

The proxy cost is a real OBJECTIVE (not a tiebreak): the search follows the proxy
wherever it leads -- including into its blind spot. A static, execution-frequency-
blind proxy cannot tell a node that runs n times per input from a node behind a
false guard that never runs at all: both are "one node". So the optimizer's
proxy-measured win from pruning DEAD is almost entirely illusory at audit scale --
the phenomenon ``audit.py`` measures.

SEALING: imports neither ``cost.py`` (the expensive held-out audit) nor any task
reference; correctness is read only through the ``verify`` callback. Control
``inner_loop_cost_blind`` proves the no-``cost`` import by source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .ir import Node, pp
from .search import synthesize
from .search_oe import oe_solve
from .library import broad_policy, stateful_policy
from .proxy import CostProxy

# Deterministic portfolio budgets for the SEED (correctness-blind, oracle-gated).
SEED_OE_SIZE = 14
SEED_OE_BUDGET = 90_000
SEED_MEMETIC_BUDGET = 45_000
SEED_MEMETIC_SEED = 7
DEAD_COPIES = 3            # how many copies of S the cold dead branch stacks


def _b(op: str, *kids: Node, const=None, rtype="V") -> Node:
    from .ir import PRIMS as P, COMB_RTYPE as C
    rt = C.get(op) or (P[op][0] if op in P else rtype)
    return Node(op, rt, tuple(kids), const)


def _lit(v, t) -> Node:
    return Node("lit", t, const=v)


# --------------------------------------------------------------------------- #
# Step 1: the verified seed (correctness-blind portfolio, oracle-gated)        #
# --------------------------------------------------------------------------- #
def seed_program(view, verify: Callable[[Node], bool]) -> Optional[Node]:
    """A correct seed for the target, found WITHOUT reading the reference: bottom-up
    OE first, then memetic search under family-neutral priors. Gated by ``verify``."""
    p = oe_solve(view, blocks=[], max_size=SEED_OE_SIZE, eval_budget=SEED_OE_BUDGET)
    if p is not None and verify(p):
        return p
    for pol in (stateful_policy(), broad_policy()):
        p, _st = synthesize(view, pol, SEED_MEMETIC_BUDGET, SEED_MEMETIC_SEED)
        if p is not None and verify(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# Helpers: a provably-false guard and a chunky dead subtree                    #
# --------------------------------------------------------------------------- #
def _false_guard(arg_types) -> Node:
    """A predicate that is ALWAYS false on the data but looks input-dependent. For a
    list argument: ``len(list) < 0`` (a length is never negative). Otherwise a bare
    ``0 == 1``."""
    for i, t in enumerate(arg_types):
        if t == "L":
            return _b("lt", _b("llen", Node("arg", "L", const=i)), _lit(0, "I"))
        if t == "S":
            return _b("lt", _b("slen", Node("arg", "S", const=i)), _lit(0, "I"))
    return _b("eqi", _lit(0, "I"), _lit(1, "I"))


def _dead_subtree(seed: Node, out_type: str) -> Node:
    """A chunky, type-correct subtree that is NEVER executed (it sits in the false
    branch of the cold guard). It stacks copies of S, so it carries many NODES the
    proxy will price -- yet costs ZERO executed steps because it never runs."""
    if out_type == "S":
        d = seed
        for _ in range(DEAD_COPIES):
            d = _b("sconcat", _b("lapp", _b("lsingle", seed), _b("lsingle", d)))
        return d
    if out_type == "L":
        d = seed
        for _ in range(DEAD_COPIES):
            d = _b("lapp", seed, d)
        return d
    # scalar / other: stack arithmetic on the seed
    d = seed
    for _ in range(DEAD_COPIES):
        d = _b("add", seed, d)
    return d


# --------------------------------------------------------------------------- #
# Step 2a: the cost-unaware baseline E(S) = cold guard + out-of-loop wraps      #
# --------------------------------------------------------------------------- #
def baseline_elaboration(seed: Node, out_type: str, arg_types,
                         verify: Callable[[Node], bool]) -> Node:
    """E(S): ``seed`` guarded by a cold (always-false) conditional carrying a chunky
    dead branch, plus a couple of out-of-loop identity wraps. Every layer is
    oracle-verified, so E(S) is exactly as correct as S; it just carries structure a
    cost-unaware synthesizer would not bother to prune."""
    cur = seed

    def try_apply(node: Node) -> None:
        nonlocal cur
        if verify(node):
            cur = node

    # the cold guard: if (impossible) then DEAD else cur  ==  cur, DEAD never runs
    dead = _dead_subtree(seed, out_type)
    try_apply(Node("ifx", "V", (_false_guard(arg_types), dead, cur)))
    # a little live out-of-loop wrapping on top (these DO run, cheaply)
    if out_type == "L":
        try_apply(_b("lrev", _b("lrev", cur)))
    elif out_type == "S":
        try_apply(_b("sconcat", _b("lsingle", cur)))
    return cur


# --------------------------------------------------------------------------- #
# Step 2b: a LIVE elaboration (for the proxy's training corpus only)           #
# --------------------------------------------------------------------------- #
def live_elaboration(seed: Node, out_type: str,
                     verify: Callable[[Node], bool]) -> Node:
    """A bigger but fully-LIVE correct program (every node executes). Used only to
    train the proxy, so the proxy's per-node cost is calibrated on programs whose
    nodes all run -- which is exactly why it later MISPRICES the dead branch."""
    cur = seed
    if out_type == "L":
        for w in (_b("lrev", _b("lrev", cur)), _b("lapp", cur, _lit([], "L"))):
            if verify(w):
                cur = w
    elif out_type == "S":
        for w in (_b("sconcat", _b("lsingle", cur)),
                  _b("sconcat", _b("schars", cur))):
            if verify(w):
                cur = w
    return cur


# --------------------------------------------------------------------------- #
# Step 3: the rewrite-neighborhood + proxy-guided descent                      #
# --------------------------------------------------------------------------- #
# Candidate rewrites. NOT assumed semantics-preserving: every candidate is
# independently oracle-verified by the caller, so an unsafe rewrite is simply
# rejected. This lets the optimizer DISCOVER that dropping the cold guard is safe
# (the then-branch never fired) without having to prove it statically.
def _rewrite_candidates(n: Node) -> List[Node]:
    out: List[Node] = []
    op = n.op
    k = n.kids
    if op == "ifx" and len(k) == 3:
        out.append(k[2])                                        # drop guard+then -> else
        out.append(k[1])                                        # (else-elimination; oracle filters)
    if op == "lrev" and k and k[0].op == "lrev":
        out.append(k[0].kids[0])                                # lrev(lrev x) -> x
    if op == "lapp" and len(k) == 2:
        if k[1].op == "lit" and k[1].const == []:
            out.append(k[0])                                    # x ++ [] -> x
        if k[0].op == "lit" and k[0].const == []:
            out.append(k[1])                                    # [] ++ x -> x
    if op == "ldrop" and len(k) == 2 and k[1].op == "lit" and k[1].const == 0:
        out.append(k[0])                                        # drop 0 -> x
    if op == "sconcat" and k and k[0].op == "lsingle":
        out.append(k[0].kids[0])                                # concat [x] -> x
    if op == "sconcat" and k and k[0].op == "schars":
        out.append(k[0].kids[0])                                # rejoin chars -> x
    if op == "srepeat" and len(k) == 2 and k[1].op == "lit" and k[1].const == 1:
        out.append(k[0])                                        # x repeated once -> x
    if op in ("add", "sub") and len(k) == 2 and k[1].op == "lit" and k[1].const == 0:
        out.append(k[0])                                        # x +/- 0 -> x
    if op == "mul" and len(k) == 2 and k[1].op == "lit" and k[1].const == 1:
        out.append(k[0])                                        # x * 1 -> x
    return out


def _all_paths(n: Node, path=()):
    yield path, n
    for i, kid in enumerate(n.kids):
        yield from _all_paths(kid, path + (i,))


def _replace_at(root: Node, path, new: Node) -> Node:
    if not path:
        return new
    i = path[0]
    kids = list(root.kids)
    kids[i] = _replace_at(kids[i], path[1:], new)
    return Node(root.op, root.rtype, tuple(kids), root.const)


def _neighbors(prog: Node) -> List[Node]:
    """All programs reachable by ONE rewrite, in a fixed deterministic order."""
    out: List[Node] = []
    seen = set()
    for path, node in _all_paths(prog):
        for repl in _rewrite_candidates(node):
            cand = _replace_at(prog, path, repl)
            key = pp(cand)
            if key not in seen and key != pp(prog):
                seen.add(key)
                out.append(cand)
    return out


@dataclass
class OptResult:
    seed: Node
    baseline: Node                 # E(S): the cost-unaware program
    optimized: Node                # proxy-minimized program for THIS arm
    proxy_trained: bool
    steps: int                     # verify calls spent (cost of the inner search)
    trace: List[str]


def cost_aware_arm(view, verify: Callable[[Node], bool], proxy: CostProxy,
                   budget: int, seed_prog: Optional[Node] = None) -> Optional[OptResult]:
    """One arm: build E(S), then greedily minimise ``proxy`` over the rewrite
    neighborhood (each move oracle-verified). A TRAINED proxy descends to the lean
    program; a FROZEN proxy (constant 0) finds no improving move and returns E(S)."""
    S = seed_prog if seed_prog is not None else seed_program(view, verify)
    if S is None:
        return None
    base = baseline_elaboration(S, view.out_type, view.arg_types, verify)
    cur = base
    cur_score = proxy.predict(cur)
    spent = 0
    trace: List[str] = [f"E(S) size={base.size()} proxy={cur_score:.2f}"]
    while spent < budget:
        best: Optional[Node] = None
        best_score = cur_score
        best_key = None
        for c in _neighbors(cur):
            spent += 1                       # a verify call -- the inner-loop cost
            if not verify(c):
                continue
            sc = proxy.predict(c)
            key = pp(c)
            # strict improvement; deterministic tie-break by canonical text
            if sc < best_score - 1e-9 or (best is not None and abs(sc - best_score) <= 1e-9
                                          and key < best_key):
                best, best_score, best_key = c, sc, key
            if spent >= budget:
                break
        if best is None:
            break
        cur, cur_score = best, best_score
        trace.append(f"-> size={cur.size()} proxy={cur_score:.2f}")
    return OptResult(S, base, cur, proxy.is_trained(), spent, trace)
