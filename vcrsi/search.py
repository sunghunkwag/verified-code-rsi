#!/usr/bin/env python3
"""The LLM-free synthesizer: budgeted stochastic / genetic search over the IR.

Synthesis is a pure search procedure -- there is NO language-model call anywhere.
A candidate program is built by weighted typed sampling under the policy genome
(``library.Policy``), refined by a memetic local search (leaf swaps, fresh
subtrees and "wrapping" edits that grow structure around a correct fragment) and
light crossover. The policy's operator weights and mined subroutines are the ONLY
thing search behaviour depends on, so a better policy = a better search = the
measured self-improvement.

Element typing + accessor terminals. The loop variables ``it`` / ``acc`` carry
their CONCRETE type (inferred from the public examples). When the iterated
element is a pair (t0, t1), the generator does not offer the raw pair where a t0
is needed -- it offers ``fst(it)``/``snd(it)`` as first-class typed leaves. This
is what makes multi-step accessor programs (the only kind that clears the
complexity floor) reachable by stochastic search at all.

Sealing (§4.9): this module receives a ``PublicView`` (spec + public examples
only). It never imports tasks, reference solutions, the oracle, or the held-out
battery; the no-leakage control inspects this source to prove it.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ir import (Block, Node, PRIMS, COMB_RTYPE, MAX_LEN, tmatch, all_nodes,
                 replace_at, pp)
from .interp import run
from .library import Policy

INT_CONSTS = (0, 1, 2, -1)


# --------------------------------------------------------------------------- #
# Problem description (public only) + type inference                           #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Problem:
    arg_types: Tuple[str, ...]
    out_type: str
    examples: Tuple[Tuple[Tuple[Any, ...], Any], ...]
    arg_elem: Tuple[str, ...]    # for list args: element type; else the arg type
    pair_comps: Tuple[str, str]  # component types of pairs seen in the inputs


def _value_type(v: Any) -> str:
    if isinstance(v, bool):
        return "B"
    if isinstance(v, int):
        return "I"
    if isinstance(v, str):
        return "S"
    if isinstance(v, tuple):
        return "P"
    if isinstance(v, list):
        return "L"
    return "V"


def _infer_arg_elem(arg_types, examples) -> Tuple[str, ...]:
    out = []
    for i, t in enumerate(arg_types):
        if t != "L":
            out.append(t)
            continue
        et = "V"
        for args, _exp in examples:
            lst = args[i]
            if isinstance(lst, list) and lst:
                et = _value_type(lst[0])
                break
        out.append(et)
    return tuple(out)


def _infer_pair_comps(examples) -> Tuple[str, str]:
    """Component types of the first pair found anywhere in the inputs."""
    def find(v):
        if isinstance(v, tuple) and len(v) == 2:
            return (_value_type(v[0]), _value_type(v[1]))
        if isinstance(v, list):
            for e in v:
                r = find(e)
                if r:
                    return r
        return None
    for args, _exp in examples:
        for a in args:
            r = find(a)
            if r:
                return r
    return ("V", "V")


def problem_from_public(view) -> Problem:
    ex = tuple(view.public_examples)
    return Problem(view.arg_types, view.out_type, ex,
                   _infer_arg_elem(view.arg_types, ex), _infer_pair_comps(ex))


def node_type(n: Node, scope: Dict[str, str], arg_types: Tuple[str, ...]) -> str:
    op = n.op
    if op == "lit":
        return n.rtype
    if op == "arg":
        return arg_types[n.const]
    if op == "var":
        return scope.get(n.const, "V")
    if op in ("param", "call"):
        return n.rtype
    if op == "ifx":
        return node_type(n.kids[1], scope, arg_types)
    if op in ("map", "filter", "scan"):
        return "L"
    if op == "foldl":
        return node_type(n.kids[1], scope, arg_types)
    if op == "iterate":
        return node_type(n.kids[0], scope, arg_types)
    if op in PRIMS:
        return PRIMS[op][0]
    return n.rtype


def elem_type(n: Node, scope, arg_elem, arg_is_list) -> str:
    op = n.op
    if op == "arg":
        return arg_elem[n.const] if arg_is_list[n.const] else "V"
    if op == "schars":
        return "S"
    if op == "lrange":
        return "I"
    if op in ("lrev", "tail", "linit", "lsort", "ltake", "ldrop", "filter"):
        return elem_type(n.kids[0], scope, arg_elem, arg_is_list)
    if op == "map":
        it_t = elem_type(n.kids[0], scope, arg_elem, arg_is_list)
        return node_type(n.kids[1], {**scope, "it": it_t}, arg_elem)
    if op == "scan":
        # scan returns the list of intermediate accumulators -> element = acc type
        return node_type(n.kids[1], scope, arg_elem)
    if op == "lsingle":
        return node_type(n.kids[0], scope, arg_elem)
    if op == "cons":
        return node_type(n.kids[0], scope, arg_elem)
    if op == "lapp":
        return elem_type(n.kids[0], scope, arg_elem, arg_is_list)
    return "V"


# --------------------------------------------------------------------------- #
# Bound-variable terminals: it / acc expanded into typed accessor expressions   #
# --------------------------------------------------------------------------- #
def _expand_var(name: str, t: str, comps: Tuple[str, str]) -> List[Tuple[Node, str]]:
    v = Node("var", "V", const=name)
    if t == "P":
        return [(v, "P"),
                (Node("fst", "V", (v,)), comps[0]),
                (Node("snd", "V", (v,)), comps[1])]
    return [(v, t)]


def _term_key(node: Node) -> str:
    return node.const if node.op == "var" else node.op


# --------------------------------------------------------------------------- #
# Weighted typed generation                                                     #
# --------------------------------------------------------------------------- #
class _Gen:
    def __init__(self, prob: Problem, policy: Policy, rng: random.Random):
        self.prob = prob
        self.policy = policy
        self.blocks = policy.blocks
        self.rng = rng
        self.arg_elem = prob.arg_elem
        self.arg_is_list = tuple(t == "L" for t in prob.arg_types)
        self.comps = prob.pair_comps

    def _w(self, key: str) -> float:
        return self.policy.w(key)

    # leaf candidates as concrete (node, weight) pairs ---------------------- #
    def _leaf_cands(self, rtype: str, scope: Tuple[Tuple[Node, str], ...]
                    ) -> List[Tuple[Node, float]]:
        out: List[Tuple[Node, float]] = []
        if tmatch("I", rtype):
            out.append((Node("lit", "I", const=self.rng.choice(INT_CONSTS)),
                        self._w("lit_int")))
        if tmatch("B", rtype):
            out.append((Node("lit", "B", const=bool(self.rng.getrandbits(1))),
                        self._w("lit_bool")))
        if tmatch("L", rtype):
            out.append((Node("lit", "L", const=[]), self._w("lit_nil")))
        if tmatch("S", rtype):
            out.append((Node("lit", "S", const=""), self._w("lit_estr")))
        if tmatch("P", rtype):
            out.append((Node("lit", "P", const=(0, 0)), self._w("lit_pair")))
        for i, t in enumerate(self.prob.arg_types):
            if tmatch(t, rtype):
                out.append((Node("arg", t, const=i), self._w("arg")))
        for node, t in scope:
            if tmatch(t, rtype):
                out.append((node, self._w(_term_key(node))))
        return out

    def _producer_ops(self, rtype: str) -> List[str]:
        out: List[str] = []
        if tmatch("L", rtype):
            out += ["map", "filter"]
        out += ["foldl", "ifx"]
        # scan / iterate are the new stateful building blocks (Unlock A). They are
        # offered ONLY when the policy explicitly enables them (a weight key is
        # present), so every policy that predates them -- including the Phase-A
        # default_policy -- generates byte-identically to before.
        if tmatch("L", rtype) and self.policy.weights.get("scan", 0.0) > 0:
            out.append("scan")
        if self.policy.weights.get("iterate", 0.0) > 0:
            out.append("iterate")
        for name, (rt, ats, _fn) in PRIMS.items():
            if tmatch(rt, rtype):
                out.append(name)
        return out

    def _pick(self, cands: List[Tuple[Any, float]]):
        tot = sum(w for _, w in cands)
        r = self.rng.random() * tot
        for c, w in cands:
            r -= w
            if r <= 0:
                return c
        return cands[-1][0]

    def gen(self, rtype: str, depth: int,
            scope: Tuple[Tuple[Node, str], ...] = ()) -> Node:
        if self.blocks and depth > 0 and self.rng.random() < self.policy.block_prob:
            cands = [b for b in self.blocks if tmatch(b.rtype, rtype)]
            if cands:
                # M1 -- abstraction-first: prefer the NEWEST blocks (a block built
                # on an earlier block is enumerated as one unit, so nested structure
                # accumulates instead of being re-derived from primitives).
                cands.sort(key=lambda b: (b.created_round, b.name), reverse=True)
                idx = min(int(self.rng.expovariate(1.0)), len(cands) - 1)
                blk = cands[idx]
                kids = tuple(self.gen(pt, depth - 1, scope) for pt in blk.ptypes)
                return Node("call", blk.rtype, kids, blk.name)
        leafc = self._leaf_cands(rtype, scope)
        if depth <= 0:
            return self._pick(leafc) if leafc else \
                Node("lit", "I", const=self.rng.choice(INT_CONSTS))
        ops = self._producer_ops(rtype)
        if not ops:
            return self._pick(leafc) if leafc else \
                Node("lit", "I", const=self.rng.choice(INT_CONSTS))
        if leafc and self.rng.random() < 0.18:
            return self._pick(leafc)
        opname = self._pick([(o, self._w(o)) for o in ops])
        return self._mknode(opname, rtype, depth, scope)

    def _mknode(self, name: str, rtype: str, depth: int,
                scope: Tuple[Tuple[Node, str], ...]) -> Node:
        if name == "ifx":
            return Node("ifx", rtype, (self.gen("B", depth - 1, scope),
                                       self.gen(rtype, depth - 1, scope),
                                       self.gen(rtype, depth - 1, scope)))
        if name == "map":
            src = self.gen("L", depth - 1, scope)
            it_t = elem_type(src, _scope_types(scope), self.arg_elem, self.arg_is_list)
            body = self.gen("V", depth - 1, scope + tuple(_expand_var("it", it_t, self.comps)))
            return Node("map", "L", (src, body))
        if name == "filter":
            src = self.gen("L", depth - 1, scope)
            it_t = elem_type(src, _scope_types(scope), self.arg_elem, self.arg_is_list)
            body = self.gen("B", depth - 1, scope + tuple(_expand_var("it", it_t, self.comps)))
            return Node("filter", "L", (src, body))
        if name == "foldl":
            src = self.gen("L", depth - 1, scope)
            init = self.gen(rtype, depth - 1, scope)
            it_t = elem_type(src, _scope_types(scope), self.arg_elem, self.arg_is_list)
            acc_t = node_type(init, _scope_types(scope), self.prob.arg_types)
            sc2 = scope + tuple(_expand_var("it", it_t, self.comps)) \
                        + tuple(_expand_var("acc", acc_t, self.comps))
            return Node("foldl", rtype, (src, init, self.gen(rtype, depth - 1, sc2)))
        if name == "scan":
            src = self.gen("L", depth - 1, scope)
            init = self.gen("V", max(0, depth - 2), scope)
            it_t = elem_type(src, _scope_types(scope), self.arg_elem, self.arg_is_list)
            acc_t = node_type(init, _scope_types(scope), self.prob.arg_types)
            sc2 = scope + tuple(_expand_var("it", it_t, self.comps)) \
                        + tuple(_expand_var("acc", acc_t, self.comps))
            return Node("scan", "L", (src, init, self.gen(acc_t, depth - 1, sc2)))
        if name == "iterate":
            init = self.gen(rtype, depth - 1, scope)
            count = self.gen("I", max(0, depth - 2), scope)
            acc_t = node_type(init, _scope_types(scope), self.prob.arg_types)
            sc2 = scope + tuple(_expand_var("it", "I", self.comps)) \
                        + tuple(_expand_var("acc", acc_t, self.comps))
            return Node("iterate", acc_t, (init, count, self.gen(acc_t, depth - 1, sc2)))
        rt, ats, _fn = PRIMS[name]
        return Node(name, rt, tuple(self.gen(at, depth - 1, scope) for at in ats))


def _scope_types(scope: Tuple[Tuple[Node, str], ...]) -> Dict[str, str]:
    """Map var-name -> type for node_type/elem_type (only raw vars matter)."""
    d: Dict[str, str] = {}
    for node, t in scope:
        if node.op == "var":
            d[node.const] = t
    return d


# --------------------------------------------------------------------------- #
# Fitness                                                                       #
# --------------------------------------------------------------------------- #
def _seqsim(a, b) -> float:
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    if la * lb > 120_000:
        common = sum(1 for x, y in zip(a, b) if x == y)
        return 0.9 * common / max(la, lb)
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        row_prev = prev
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = row_prev[j - 1] + 1
            else:
                cur[j] = row_prev[j] if row_prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return 0.97 * prev[lb] / max(la, lb)


def _similarity(out, exp) -> float:
    if out == exp:
        return 1.0
    if out is None:
        return 0.0
    if (isinstance(exp, str) and isinstance(out, str)) or \
       (isinstance(exp, list) and isinstance(out, list)):
        content = _seqsim(out, exp)
        m = max(len(out), len(exp), 1)
        length = 1.0 - abs(len(out) - len(exp)) / m
        return min(0.98, 0.7 * content + 0.3 * length)
    return 0.0


def _case_scores(prog: Node, examples, blocks: Dict[str, Block]) -> List[float]:
    res = []
    for args, exp in examples:
        r = run(prog, list(args), blocks, max_steps=40_000)
        res.append(_similarity(r.value if r.ok else None, exp))
    return res


def _exact(prog: Node, examples, blocks: Dict[str, Block]) -> bool:
    for args, exp in examples:
        r = run(prog, list(args), blocks, max_steps=120_000)
        if not r.ok or r.value != exp:
            return False
    return True


# --------------------------------------------------------------------------- #
# Typed scope reconstruction + genetic operators                               #
# --------------------------------------------------------------------------- #
def _scope_at(root: Node, path, prob: Problem) -> Tuple[Tuple[Node, str], ...]:
    arg_elem = prob.arg_elem
    arg_is_list = tuple(t == "L" for t in prob.arg_types)
    comps = prob.pair_comps
    scope: Tuple[Tuple[Node, str], ...] = ()
    cur = root
    for idx in path:
        st = _scope_types(scope)
        if cur.op in ("map", "filter") and idx == 1:
            it_t = elem_type(cur.kids[0], st, arg_elem, arg_is_list)
            scope = scope + tuple(_expand_var("it", it_t, comps))
        elif cur.op in ("foldl", "scan") and idx == 2:
            it_t = elem_type(cur.kids[0], st, arg_elem, arg_is_list)
            acc_t = node_type(cur.kids[1], st, prob.arg_types)
            scope = scope + tuple(_expand_var("it", it_t, comps)) \
                          + tuple(_expand_var("acc", acc_t, comps))
        elif cur.op == "iterate" and idx == 2:
            acc_t = node_type(cur.kids[0], st, prob.arg_types)
            scope = scope + tuple(_expand_var("it", "I", comps)) \
                          + tuple(_expand_var("acc", acc_t, comps))
        cur = cur.kids[idx]
    return scope


def _bound_vars(scope) -> set:
    return {node.const for node, _t in scope if node.op == "var"}


def _vars_ok(n: Node, allowed: set) -> bool:
    if n.op == "var" and n.const not in allowed:
        return False
    return all(_vars_ok(k, allowed) for k in n.kids)


def _crossover(a: Node, b: Node, rng: random.Random, prob: Problem) -> Node:
    na = all_nodes(a)
    path_a, node_a = rng.choice(na)
    allowed = _bound_vars(_scope_at(a, path_a, prob))
    nb = [(p, n) for (p, n) in all_nodes(b)
          if tmatch(n.rtype, node_a.rtype) and _vars_ok(n, allowed)]
    if not nb:
        return a
    _, node_b = rng.choice(nb)
    return replace_at(a, path_a, node_b)


def _wrap_candidates(gen: _Gen, node: Node, scope, rng: random.Random,
                     k: int) -> List[Node]:
    """Grow structure AROUND ``node`` (fst(it) -> srepeat(fst(it), <gen int>)).
    Op choice is policy-weighted."""
    want = node.rtype
    options: List[Tuple[str, int]] = []
    weights: List[float] = []
    for name, (rt, ats, _fn) in PRIMS.items():
        if not tmatch(rt, want):
            continue
        for j, at in enumerate(ats):
            if tmatch(at, node.rtype):
                options.append((name, j))
                weights.append(gen._w(name))
    if not options:
        return []
    tot = sum(weights)
    out: List[Node] = []
    for _ in range(k):
        r = rng.random() * tot
        idx = 0
        for idx, w in enumerate(weights):
            r -= w
            if r <= 0:
                break
        name, slot = options[idx]
        rt, ats, _fn = PRIMS[name]
        kids = [node if j == slot else gen.gen(at, rng.randint(0, 2), scope)
                for j, at in enumerate(ats)]
        out.append(Node(name, rt, tuple(kids)))
    return out


# --------------------------------------------------------------------------- #
# Top-level synthesizer                                                         #
# --------------------------------------------------------------------------- #
def shrink(prog: Node, examples, blocks: Dict[str, Block]) -> Node:
    """Reduce an adopted program to a smaller behaviourally-identical one, so the
    blocks mined from it are clean ATOMS rather than bloated fragments. Two
    moves, applied to fixpoint: hoist a node up to one of its children (drops
    dead wrappers) and replace a node with a constant leaf -- each kept only if
    the FULL public battery still matches exactly."""
    cur = prog
    changed = True
    guard = 0
    while changed and guard < 200:
        changed = False
        guard += 1
        for path, node in all_nodes(cur):
            if not node.kids:
                continue
            repl = [k for k in node.kids if tmatch(k.rtype, node.rtype)]
            for cand_sub in repl:
                cand = replace_at(cur, path, cand_sub)
                if cand.size() < cur.size() and _exact(cand, examples, blocks):
                    cur = cand
                    changed = True
                    break
            if changed:
                break
    return cur


@dataclass
class SearchStats:
    evals: int
    solved: bool
    best_fitness: float


def synthesize(view, policy: Policy, budget: int, seed: int,
               restarts: int = 1, batch: int = 500
               ) -> Tuple[Optional[Node], SearchStats]:
    """Search for a program that exactly matches ALL public examples. Identical
    (policy, budget, seed) give identical results -> reproducible counterfactual."""
    prob = problem_from_public(view)
    blocks = policy.block_map()
    spent = [0]
    best_f = [-1.0]
    nfit = min(4, len(prob.examples))
    fit_examples = prob.examples[:nfit]

    def run_once(rng: random.Random, my_cap: int) -> Optional[Node]:
        gen = _Gen(prob, policy, rng)
        seen: set = set()
        hof: List[Tuple[float, int, Node]] = []
        HOF_CAP = 120

        def consider(p: Node) -> Optional[Node]:
            if p.height() > 16 or p.size() > 220:
                return None
            key = pp(p)
            if key in seen:
                return None
            seen.add(key)
            spent[0] += 1
            cs = _case_scores(p, fit_examples, blocks)
            f = sum(cs) / len(cs) - min(0.0006 * p.size(), 0.04)
            if f > best_f[0]:
                best_f[0] = f
            if all(s >= 1.0 for s in cs) and _exact(p, prob.examples, blocks):
                return p
            if len(hof) < HOF_CAP or f > hof[-1][0]:
                hof.append((f, p.size(), p))
                hof.sort(key=lambda x: (-x[0], x[1]))
                del hof[HOF_CAP:]
            return None

        def local_search(prog: Node) -> Optional[Node]:
            nodes = all_nodes(prog)
            rng.shuffle(nodes)
            for path, node in nodes:
                if spent[0] >= my_cap:
                    return None
                scope = _scope_at(prog, path, prob)
                cands: List[Node] = [c for c, _w in gen._leaf_cands(node.rtype, scope)]
                cands += _wrap_candidates(gen, node, scope, rng, 6)
                cands += [gen.gen(node.rtype, rng.randint(0, 3), scope)
                          for _ in range(4)]
                for sub in cands:
                    if spent[0] >= my_cap:
                        return None
                    r = consider(replace_at(prog, path, sub))
                    if r is not None:
                        return r
            return None

        while spent[0] < my_cap:
            for _ in range(batch):
                if spent[0] >= my_cap:
                    break
                p = gen.gen(prob.out_type, rng.randint(2, 6))
                r = consider(p)
                if r is not None:
                    return r
            targets = [p for _f, sz, p in hof if sz <= 16][:60]
            for partial in targets:
                if spent[0] >= my_cap:
                    break
                sol = local_search(partial)
                if sol is not None:
                    return sol
            if len(hof) >= 2:
                for _ in range(batch // 4):
                    if spent[0] >= my_cap:
                        break
                    a = hof[rng.randrange(len(hof))][2]
                    bb = hof[rng.randrange(len(hof))][2]
                    r = consider(_crossover(a, bb, rng, prob))
                    if r is not None:
                        return r
        return None

    for i in range(restarts):
        cap = int(budget * (i + 1) / restarts)
        rng = random.Random(seed * 1_000_003 + i)
        sol = run_once(rng, cap)
        if sol is not None:
            return sol, SearchStats(spent[0], True, 1.0)
        if spent[0] >= budget:
            break
    return None, SearchStats(spent[0], False, best_f[0])
