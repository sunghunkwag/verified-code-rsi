#!/usr/bin/env python3
"""M1 -- bottom-up, observational-equivalence program synthesis.

A SECOND, deterministic solver run as a portfolio alongside the stochastic
search. It enumerates IR expressions by increasing size, keeps exactly ONE
representative per distinct behaviour (the observational-equivalence key) on the
PUBLIC training inputs, and returns the smallest program matching all public
examples. Minimal programs are clean transfer candidates. Library blocks are
available as extra operators, so the OE solver can DISCOVER that a transferred
block makes a held-out task reachable.

Coverage: ground expressions + ``map``/``filter`` over the list argument (the
single-pass shapes). It does NOT enumerate ``foldl`` bodies (a threaded
accumulator defeats per-element observational equivalence), so genuinely stateful
tasks stay OPEN here -- an honest, reported limitation.

Anti-leakage (§5 oe_no_leakage): every equivalence key and match test uses ONLY
the public training examples; the held-out battery is never read here.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .ir import Block, Node, PRIMS
from .interp import run

INT_CONSTS = (0, 1, 2, -1)
_TYPES = ("I", "B", "S", "L", "P")


def _vt(v: Any) -> str:
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


def _pair_comps(exs, list_arg) -> Tuple[str, str]:
    for args, _e in exs:
        for el in args[list_arg]:
            if isinstance(el, tuple) and len(el) == 2:
                return (_vt(el[0]), _vt(el[1]))
    return ("V", "V")


class _Bank:
    def __init__(self):
        self.by_type: Dict[str, List[Tuple[Node, Tuple]]] = {t: [] for t in _TYPES}
        self.seen: set = set()

    def add(self, node: Node, rtype: str, sig: Tuple) -> bool:
        if sig is None or any(v == "ERR" for v in sig):
            return False
        key = (rtype, sig)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.by_type.setdefault(rtype, []).append((node, sig))
        return True

    def pool(self, want: str, cap: int) -> List[Tuple[Node, Tuple]]:
        if want == "V":
            out = []
            for t in _TYPES:
                out += self.by_type.get(t, [])
            return out[:cap]
        return (self.by_type.get(want, []) + self.by_type.get("V", []))[:cap]


def oe_solve(view, blocks: Optional[List[Block]] = None, max_size: int = 10,
             cap: int = 130, eval_budget: int = 35_000) -> Optional[Node]:
    """Return the smallest program matching ALL public examples, or None.

    Bounded by ``eval_budget`` candidate evaluations so a miss fails fast."""
    blocks = blocks or []
    bm = {b.name: b for b in blocks}
    spent = [0]
    exs = list(view.public_examples)
    arg_types = view.arg_types
    out_type = view.out_type
    target = tuple(repr(e) for _a, e in exs)

    list_arg = next((i for i, t in enumerate(arg_types) if t == "L"), None)
    # it environments grouped by example, so map/filter signatures can be
    # reconstructed from a body's per-element outputs WITHOUT re-running.
    groups: List[List[Any]] = []
    it_type = "V"
    comps = ("V", "V")
    if list_arg is not None:
        for args, _e in exs:
            groups.append(list(args[list_arg]))
        flat = [el for g in groups for el in g]
        if flat:
            it_type = _vt(flat[0])
            comps = _pair_comps(exs, list_arg)
    have_it = list_arg is not None and any(groups)

    class _Budget(Exception):
        pass

    def gsig(node: Node) -> Optional[Tuple]:
        spent[0] += 1
        if spent[0] > eval_budget:
            raise _Budget()
        out = []
        for args, _e in exs:
            r = run(node, list(args), bm, max_steps=40_000)
            out.append(repr(r.value) if r.ok else "ERR")
        return tuple(out)

    def isig(node: Node) -> Optional[Tuple]:
        spent[0] += 1
        if spent[0] > eval_budget:
            raise _Budget()
        out = []
        for gi, g in enumerate(groups):
            for el in g:
                r = run(node, list(exs[gi][0]), bm, max_steps=20_000,
                        env={"it": el})
                out.append(repr(r.value) if r.ok else "ERR")
        return tuple(out)

    # raw VALUE per (group,element) for combinator reconstruction
    def ivals(node: Node):
        spent[0] += 1
        if spent[0] > eval_budget:
            raise _Budget()
        out = []
        for gi, g in enumerate(groups):
            row = []
            for el in g:
                r = run(node, list(exs[gi][0]), bm, max_steps=20_000,
                        env={"it": el})
                row.append((True, r.value) if r.ok else (False, None))
            out.append(row)
        return out

    root, itb = _Bank(), _Bank()

    def seed(bank: _Bank, sigfn, with_it: bool):
        for c in INT_CONSTS:
            n = Node("lit", "I", const=c); bank.add(n, "I", sigfn(n))
        n = Node("lit", "L", const=[]); bank.add(n, "L", sigfn(n))
        n = Node("lit", "S", const=""); bank.add(n, "S", sigfn(n))
        for i, t in enumerate(arg_types):
            n = Node("arg", t, const=i); bank.add(n, t, sigfn(n))
        if with_it:
            n = Node("var", it_type, const="it"); bank.add(n, it_type, sigfn(n))
            if it_type == "P":
                f = Node("fst", comps[0], (Node("var", "P", const="it"),))
                s = Node("snd", comps[1], (Node("var", "P", const="it"),))
                bank.add(f, comps[0], sigfn(f)); bank.add(s, comps[1], sigfn(s))

    seed(root, gsig, False)
    if have_it:
        seed(itb, isig, True)

    def solution() -> Optional[Node]:
        for node, sig in root.by_type.get(out_type, []):
            if sig == target:
                return node
        return None

    def grow(bank: _Bank, sigfn):
        new = []
        for name, (rt, ats, _fn) in PRIMS.items():
            pools = [bank.pool(at, cap) for at in ats]
            if any(not p for p in pools):
                continue
            import itertools
            cnt = 0
            for combo in itertools.product(*pools):
                node = Node(name, rt, tuple(c[0] for c in combo))
                if node.size() <= max_size:
                    new.append((node, rt, sigfn(node)))
                cnt += 1
                if cnt >= cap * 6:
                    break
        for b in blocks:
            pools = [bank.pool(pt, cap) for pt in b.ptypes]
            if any(not p for p in pools):
                continue
            import itertools
            for combo in itertools.product(*pools):
                node = Node("call", b.rtype, tuple(c[0] for c in combo), b.name)
                if node.size() <= max_size:
                    new.append((node, b.rtype, sigfn(node)))
        for node, rt, s in new:
            bank.add(node, rt, s)

    def add_combinators():
        # map(arg0, body) and filter(arg0, predbody): reconstruct ground sigs
        # from per-element body outputs (fast; no re-running of the whole node).
        src = Node("arg", "L", const=list_arg)
        bodies = []
        for t in _TYPES:
            bodies += [(n, t) for n, _s in itb.by_type.get(t, [])]
        for body, _t in bodies[:cap]:
            if body.op == "var" and body.const == "it":
                continue
            iv = ivals(body)
            ok = all(c[0] for row in iv for c in row)
            if ok:
                node = Node("map", "L", (src, body))
                if node.size() <= max_size:
                    sig = tuple(repr([c[1] for c in row]) for row in iv)
                    root.add(node, "L", sig)
        for body, _bs in itb.by_type.get("B", [])[:cap]:
            iv = ivals(body)
            ok = all(c[0] for row in iv for c in row)
            if ok:
                node = Node("filter", "L", (src, body))
                if node.size() <= max_size:
                    sig = tuple(repr([g[j] for j, c in enumerate(row) if c[1]])
                                for g, row in zip(groups, iv))
                    root.add(node, "L", sig)

    sol = solution()
    if sol:
        return sol
    try:
        for _ in range(2, max_size + 1):
            if have_it:
                grow(itb, isig)
                add_combinators()
            grow(root, gsig)
            sol = solution()
            if sol:
                return sol
    except _Budget:
        return solution()
    return None
