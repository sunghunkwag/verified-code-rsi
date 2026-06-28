#!/usr/bin/env python3
"""Cost-aware program optimization across the correctness-equivalence class.

PHASE H -- the honest measurement. Phase G fabricated its baseline (it wrapped a
correct seed in an always-false guard carrying a chunky dead branch, then "saved"
the proxy-counted node-cost of structure that never ran). That made the proxy<->real
gap a property of the PLANTED structure, not of the optimization landscape -- a
dial, not a measurement. All of that machinery is DELETED here. Nothing sits
between "the synthesizer emits a program" and "that program's cost is measured".

For one target this module now:

  1. finds a verified correct SEED via the existing correctness-blind portfolio
     (bottom-up OE / memetic), gated ONLY by the sealed oracle's ``verify`` callback
     -- it never reads the reference. The baseline ``p0`` IS this raw output,
     byte-for-byte; NO post-processing of any kind (control ``no_planted_strawman``).
  2. produces ``p1`` by a proxy-GUIDED SEARCH over GENUINE programs: a greedy descent
     of the oracle-gated rewrite neighborhood of ``p0``. Same budget and seed. The
     proxy is used to guide the search -- to rank the real correct candidates the
     search generates -- NEVER to strip a planted baseline. Both ``p0`` and ``p1``
     are raw search outputs.

The rewrite neighborhood contains only GENERAL, correctness-preserving structural
moves that genuinely trade static node count against execution frequency -- hoist a
node to a type-compatible child (drop a real redundant wrapper), fuse two passes of
a loop into one, and the standard algebraic identities (x+0, x*1, lrev(lrev x),
x++[], x/0==0, ...). EVERY candidate is independently re-checked by the sealed
oracle, so an unsafe move is simply rejected. NO move introduces a never-executed
branch, an always-false guard, or no-op padding. If the static proxy has a real
blind spot on these programs, the proxy-guided search walks into it on a REAL
program -- discovered, not fabricated. If it does not, the natural delta is ~0 and
that honest negative is the headline.

SEALING: imports neither ``cost.py`` (the expensive held-out audit) nor any task
reference; correctness is read only through the ``verify`` callback. Controls
``inner_loop_cost_blind`` and ``no_planted_strawman`` prove these by source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .ir import Node, pp, tmatch
from .search import synthesize
from .search_oe import oe_solve
from .library import broad_policy, stateful_policy
from .proxy import CostProxy

# Deterministic portfolio budgets for the SEED (correctness-blind, oracle-gated).
SEED_OE_SIZE = 14
SEED_OE_BUDGET = 90_000
SEED_MEMETIC_BUDGET = 45_000
SEED_MEMETIC_SEED = 7


def _b(op: str, *kids: Node, const=None, rtype="V") -> Node:
    from .ir import PRIMS as P, COMB_RTYPE as C
    rt = C.get(op) or (P[op][0] if op in P else rtype)
    return Node(op, rt, tuple(kids), const)


# --------------------------------------------------------------------------- #
# Step 1: the verified seed (correctness-blind portfolio, oracle-gated)        #
# This raw output IS the baseline p0 -- nothing transforms it before cost.     #
# --------------------------------------------------------------------------- #
def seed_program(view, verify: Callable[[Node], bool]) -> Optional[Node]:
    """A correct seed for the target, found WITHOUT reading the reference: bottom-up
    OE first, then memetic search under family-neutral priors. Gated by ``verify``.
    Whatever it ships is the baseline, byte-for-byte."""
    p = oe_solve(view, blocks=[], max_size=SEED_OE_SIZE, eval_budget=SEED_OE_BUDGET)
    if p is not None and verify(p):
        return p
    for pol in (stateful_policy(), broad_policy()):
        p, _st = synthesize(view, pol, SEED_MEMETIC_BUDGET, SEED_MEMETIC_SEED)
        if p is not None and verify(p):
            return p
    return None


# --------------------------------------------------------------------------- #
# Step 2: the GENUINE rewrite neighborhood (general, node-vs-frequency moves)   #
# --------------------------------------------------------------------------- #
def _algebraic_rules(n: Node) -> List[Node]:
    """Standard correctness-preserving algebraic simplifications. Each only proposes
    a SMALLER equivalent; the sealed oracle re-checks it before use."""
    out: List[Node] = []
    op, k = n.op, n.kids
    if op == "lrev" and k and k[0].op == "lrev":
        out.append(k[0].kids[0])                                 # lrev(lrev x) -> x
    if op == "lapp" and len(k) == 2:
        if k[1].op == "lit" and k[1].const == []:
            out.append(k[0])                                     # x ++ [] -> x
        if k[0].op == "lit" and k[0].const == []:
            out.append(k[1])                                     # [] ++ x -> x
    if op == "ldrop" and len(k) == 2 and k[1].op == "lit" and k[1].const == 0:
        out.append(k[0])                                         # drop 0 -> x
    if op == "ltake" and len(k) == 2 and k[1].op == "llen" and pp(k[1].kids[0]) == pp(k[0]):
        out.append(k[0])                                         # take (len x) x -> x
    if op == "sconcat" and k and k[0].op == "lsingle":
        out.append(k[0].kids[0])                                 # concat [x] -> x
    if op == "srepeat" and len(k) == 2 and k[1].op == "lit" and k[1].const == 1:
        out.append(k[0])                                         # x repeated once -> x
    if op in ("add", "sub") and len(k) == 2 and k[1].op == "lit" and k[1].const == 0:
        out.append(k[0])                                         # x +/- 0 -> x
    if op == "add" and len(k) == 2 and k[0].op == "lit" and k[0].const == 0:
        out.append(k[1])                                         # 0 + x -> x
    if op == "mul" and len(k) == 2 and k[1].op == "lit" and k[1].const == 1:
        out.append(k[0])                                         # x * 1 -> x
    if op == "mul" and len(k) == 2 and k[0].op == "lit" and k[0].const == 1:
        out.append(k[1])                                         # 1 * x -> x
    if op in ("sdiv", "smod") and len(k) == 2 and k[1].op == "lit" and k[1].const == 0:
        out.append(Node("lit", "I", const=0))                   # x / 0 == 0 (IR semantics)
    return out


def _hoist_candidates(n: Node) -> List[Node]:
    """GENERAL move: replace a node with one of its type-compatible children -- i.e.
    drop a wrapper that the data make redundant (the oracle decides). This is what
    discovers genuine redundancy a cost-blind synthesizer shipped, e.g.
    ``imax(fst(it), add(snd(it),k)) -> add(snd(it),k)`` when ``fst<=snd+k`` always,
    or ``sub(x, sdiv(x,0)) -> x``. It removes nodes that may be IN a loop body (run
    n times) -- exactly the static-node-count-vs-execution-frequency trade the proxy
    is blind to. Strictly reduces size, so the descent terminates."""
    out: List[Node] = []
    if n.op in ("lit", "arg", "var", "param", "call"):
        return out
    for kid in n.kids:
        if tmatch(kid.rtype, n.rtype) and kid.size() < n.size():
            out.append(kid)
    return out


def _subst_var(node: Node, name: str, repl: Node) -> Node:
    """Capture-avoiding substitution of ``var name`` by ``repl``; stops descending
    where a combinator re-binds the variable."""
    if node.op == "var" and node.const == name:
        return repl
    rebinds = ((node.op in ("map", "filter")) or
               (node.op in ("foldl", "scan", "iterate")))
    if not node.kids:
        return node
    new_kids = []
    for i, kid in enumerate(node.kids):
        # map/filter bind `it` in kid 1; foldl/scan/iterate bind it+acc in kid 2
        binds_here = ((node.op in ("map", "filter") and i == 1 and name == "it") or
                      (node.op in ("foldl", "scan", "iterate") and i == 2
                       and name in ("it", "acc")))
        new_kids.append(kid if binds_here else _subst_var(kid, name, repl))
    return Node(node.op, node.rtype, tuple(new_kids), node.const)


def _references_var(n: Node, name: str) -> bool:
    if n.op == "var" and n.const == name:
        return True
    return any(_references_var(k, name) for k in n.kids)


def _fusion_candidates(n: Node) -> List[Node]:
    """GENERAL fold/unfold move: fuse two passes of a map into one --
    ``map(map(s, f), g) -> map(s, g[it := f])`` -- collapsing a node that runs the
    list twice into one that runs it once. The honest node-count-vs-frequency move
    §1d asks for. Only fired when ``f`` is loop-body-pure (no free acc) so the
    substitution cannot capture; the oracle re-checks regardless."""
    out: List[Node] = []
    if n.op == "map" and len(n.kids) == 2 and n.kids[0].op == "map":
        inner, g = n.kids[0], n.kids[1]
        s, f = inner.kids
        # f references the inner element `it`; after fusion the element is s's, so f
        # stays valid. Avoid fusing if g binds its own `it` via a nested loop.
        if not _references_var(f, "acc"):
            out.append(Node("map", "L", (s, _subst_var(g, "it", f))))
    return out


def _rewrite_candidates(n: Node) -> List[Node]:
    """All genuine one-step rewrites at node ``n`` (algebraic + hoist + fusion).
    NONE introduces a dead branch, an always-false guard, or no-op padding."""
    return _algebraic_rules(n) + _hoist_candidates(n) + _fusion_candidates(n)


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
    """All programs reachable by ONE genuine rewrite, in a fixed deterministic order.
    Every candidate is SMALLER than (or restructured from) ``prog``; the caller
    oracle-gates each before accepting it."""
    out: List[Node] = []
    seen = {pp(prog)}
    for path, node in _all_paths(prog):
        for repl in _rewrite_candidates(node):
            cand = _replace_at(prog, path, repl)
            key = pp(cand)
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def neighborhood_sample(seed: Node, verify: Callable[[Node], bool],
                        depth: int = 3, cap: int = 40) -> List[Node]:
    """BFS the oracle-gated genuine neighborhood of ``seed`` (no proxy) up to
    ``depth`` rewrites / ``cap`` programs. Used to build the proxy's training corpus
    from REAL programs only -- never any planted structure."""
    frontier = [seed]
    found = [seed]
    seen = {pp(seed)}
    for _ in range(depth):
        nxt = []
        for prog in frontier:
            for cand in _neighbors(prog):
                key = pp(cand)
                if key in seen:
                    continue
                if not verify(cand):
                    continue
                seen.add(key)
                found.append(cand)
                nxt.append(cand)
                if len(found) >= cap:
                    return found
        frontier = nxt
        if not frontier:
            break
    return found


# --------------------------------------------------------------------------- #
# Step 3: the proxy-guided descent (both p0 and p1 are raw search outputs)      #
# --------------------------------------------------------------------------- #
@dataclass
class OptResult:
    seed: Node
    baseline: Node                 # == seed (the raw synthesizer output, no edits)
    optimized: Node                # proxy-minimized program for THIS arm
    proxy_trained: bool
    steps: int                     # verify calls spent (cost of the inner search)
    trace: List[str]


def cost_aware_arm(view, verify: Callable[[Node], bool], proxy: CostProxy,
                   budget: int, seed_prog: Optional[Node] = None) -> Optional[OptResult]:
    """One arm: start from the RAW seed S (no elaboration) and greedily minimise
    ``proxy`` over the genuine oracle-gated rewrite neighborhood. A TRAINED proxy
    descends to a leaner correct program; a FROZEN proxy (constant 0) finds no
    improving move and returns S unchanged. Both outputs are real programs."""
    S = seed_prog if seed_prog is not None else seed_program(view, verify)
    if S is None:
        return None
    cur = S                                        # the baseline is the seed itself
    cur_score = proxy.predict(cur)
    spent = 0
    trace: List[str] = [f"seed size={S.size()} proxy={cur_score:.2f}"]
    while spent < budget:
        best: Optional[Node] = None
        best_score = cur_score
        best_key = None
        for c in _neighbors(cur):
            spent += 1                             # a verify call -- the inner cost
            if not verify(c):
                continue
            sc = proxy.predict(c)
            key = pp(c)
            if sc < best_score - 1e-9 or (best is not None
                                          and abs(sc - best_score) <= 1e-9
                                          and key < best_key):
                best, best_score, best_key = c, sc, key
            if spent >= budget:
                break
        if best is None:
            break
        cur, cur_score = best, best_score
        trace.append(f"-> size={cur.size()} proxy={cur_score:.2f}")
    return OptResult(S, S, cur, proxy.is_trained(), spent, trace)
